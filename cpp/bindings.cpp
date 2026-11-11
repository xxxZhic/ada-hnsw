// Adaptive cosine HNSW bindings. See THIRD_PARTY_NOTICES.md.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "b_only_policy.h"
namespace py = pybind11;
using namespace pybind11::literals;

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);
                    if (id >= end) break;
                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


// A small reusable worker pool for native construction paths that need a
// synchronization barrier between controller blocks.  ParallelFor is still
// appropriate for one-shot batches, but creating and joining 24 threads for
// every 128K/256K controller block becomes visible at large N.
class PersistentParallelFor {
 public:
    explicit PersistentParallelFor(size_t num_threads)
        : num_threads_(std::max<size_t>(1, num_threads)),
          current_(0), end_(0), generation_(0), workers_left_(0),
          stopping_(false) {
        if (num_threads_ == 1) return;
        workers_.reserve(num_threads_);
        for (size_t thread_id = 0; thread_id < num_threads_; ++thread_id) {
            workers_.emplace_back([this, thread_id] { workerLoop(thread_id); });
        }
    }

    PersistentParallelFor(const PersistentParallelFor &) = delete;
    PersistentParallelFor &operator=(const PersistentParallelFor &) = delete;

    ~PersistentParallelFor() {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto &worker : workers_) worker.join();
    }

    template<class Function>
    void run(size_t start, size_t end, Function fn) {
        if (end <= start) return;
        if (num_threads_ == 1) {
            for (size_t row = start; row < end; ++row) fn(row, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = fn;
            current_.store(start, std::memory_order_relaxed);
            end_ = end;
            workers_left_ = num_threads_;
            exception_ = nullptr;
            ++generation_;
        }
        start_cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return workers_left_ == 0; });
        if (exception_) std::rethrow_exception(exception_);
    }

 private:
    void workerLoop(size_t thread_id) {
        size_t observed_generation = 0;
        while (true) {
            std::function<void(size_t, size_t)> task;
            size_t local_end = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [this, observed_generation] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                task = task_;
                local_end = end_;
            }
            while (true) {
                const size_t row = current_.fetch_add(
                    1, std::memory_order_relaxed);
                if (row >= local_end) break;
                try {
                    task(row, thread_id);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!exception_) exception_ = std::current_exception();
                    current_.store(local_end, std::memory_order_relaxed);
                    break;
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--workers_left_ == 0) done_cv_.notify_one();
            }
        }
    }

    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::atomic<size_t> current_;
    size_t end_;
    size_t generation_;
    size_t workers_left_;
    bool stopping_;
    std::function<void(size_t, size_t)> task_;
    std::exception_ptr exception_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
};


using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Graph = hnswlib::HierarchicalNSW<float>;
class Index {
    size_t dim_;
    std::unique_ptr<hnswlib::InnerProductSpace> space_;
    std::unique_ptr<Graph> graph_;
    int threads_{1};
    void require_index() const {
        if (!graph_) throw std::runtime_error("Initialize or load the index first");
    }
    void check_shape(const Array &data) const {
        if (data.ndim()!=2 || static_cast<size_t>(data.shape(1))!=dim_)
            throw std::invalid_argument("Expected an N x dim float32 matrix");
    }
    void normalize(const float *data, float *out) const {
        // Use hnswlib's float32 normalization order.
        float norm=0.0f;
        for(size_t j=0;j<dim_;++j) norm+=data[j]*data[j];
        if(!std::isfinite(norm)) throw std::invalid_argument("Non-finite vector norm");
        norm=1.0f/(sqrtf(norm)+1e-30f);
        for(size_t j=0;j<dim_;++j) out[j]=data[j]*norm;
    }
    std::vector<int> build_levels(const Array &data,py::object supplied) {
        require_index();check_shape(data);
        const size_t n=data.shape(0);
        if(!n || graph_->getCurrentElementCount())
            throw std::invalid_argument("Build requires nonempty data and an empty index");
        if(n>graph_->max_elements_) throw std::invalid_argument("Input exceeds max_elements");
        if(supplied.is_none()) return graph_->previewRandomLevels(n);
        py::array_t<int,py::array::c_style|py::array::forcecast> array(supplied);
        if(array.ndim()!=1 || static_cast<size_t>(array.size())!=n)
            throw std::invalid_argument("level_values must have shape (N,)");
        std::vector<int> levels(array.data(),array.data()+n);
        for(int value:levels)
            if(value<0 || value>63) throw std::invalid_argument("Levels must be in [0,63]");
        return levels;
    }
    size_t thread_count(int requested,size_t n) const {
        const int t=requested>0?requested:threads_;
        return n<=static_cast<size_t>(t)*4?1:static_cast<size_t>(t);
    }
 public:
    Index(const std::string &space,int dim):dim_(dim) {
        if(space!="cosine" || dim<=0)
            throw std::invalid_argument("Require cosine distance and dim > 0");
        space_=std::make_unique<hnswlib::InnerProductSpace>(dim_);
        threads_=std::max(1u,std::thread::hardware_concurrency());
    }
    void init_index(size_t n,size_t M,size_t ef,size_t seed) {
        if(graph_) throw std::runtime_error("Index already initialized");
        if(!n || M<2 || M>10000 || ef<M || ef>1000000)
            throw std::invalid_argument("Require N>0, 2<=M<=10000, M<=ef<=1000000");
        graph_=std::make_unique<Graph>(space_.get(),n,M,ef,seed);
    }
    void set_num_threads(int n) {
        if(n<1) throw std::invalid_argument("num_threads must be positive");
        threads_=n;
    }
    void set_ef(size_t ef) {
        require_index();
        if(!ef) throw std::invalid_argument("ef must be positive");
        graph_->setEf(ef);
    }
    size_t get_current_count() const {require_index();return graph_->getCurrentElementCount();}
    py::array_t<int> preview_levels(size_t n) const {
        require_index();const auto values=graph_->previewRandomLevels(n);
        py::array_t<int> out(n);std::copy(values.begin(),values.end(),out.mutable_data());return out;
    }
    void save_index(const std::string &path) const {require_index();graph_->saveIndex(path);}
    void load_index(const std::string &path) {
        if(graph_) throw std::runtime_error("Load requires an uninitialized index");
        graph_=std::make_unique<Graph>(space_.get(),path);
    }
    py::dict build_fixed(Array data,size_t ef,size_t upper_ef,int requested,py::object supplied) {
        const auto levels=build_levels(data,supplied);
        const size_t n=data.shape(0),nt=thread_count(requested,n);
        if(!ef) ef=graph_->ef_construction_;
        if(!upper_ef) upper_ef=graph_->ef_construction_;
        if(ef<graph_->M_ || upper_ef<graph_->M_)
            throw std::invalid_argument("Fixed budgets must be >= M");
        std::vector<float> normalized(nt*dim_);
        std::vector<size_t> calls(nt,0);
        {
            py::gil_scoped_release release;
            auto insert=[&](size_t row,size_t tid) {
                float *point=normalized.data()+tid*dim_;normalize(data.data(row),point);
                const size_t before=graph_->getThreadDistanceComputations();
                graph_->addPointFixed(point,row,ef,levels[row],upper_ef);
                calls[tid]+=graph_->getThreadDistanceComputations()-before;
            };
            insert(0,0);ParallelFor(1,n,nt,insert);
        }
        size_t total=0;for(size_t c:calls)total+=c;
        py::dict histogram;histogram[py::str(std::to_string(ef))]=n;
        return py::dict("method"_a="fixed","count"_a=n,"distance_computation_sum"_a=total,
            "ef_sum"_a=n*ef,"ef_mean"_a=double(ef),"ef_min"_a=ef,"ef_max"_a=ef,
            "upper_ef"_a=upper_ef,"ef_histogram"_a=histogram);
    }

    py::dict build_b_only(Array data,int requested,py::object supplied,
            double low_factor,double high_factor,double checkpoint_factor,
            size_t n_ref,size_t block_size,double update_weight,double z_clip) {
        if (!std::isfinite(low_factor) || low_factor<=0 || low_factor>1 ||
            !std::isfinite(high_factor) || high_factor<1 ||
            !std::isfinite(checkpoint_factor) || checkpoint_factor<=0 || checkpoint_factor>1 ||
            !n_ref || !block_size || !std::isfinite(update_weight) ||
            update_weight<0 || update_weight>1 || !std::isfinite(z_clip) || z_clip<=0)
            throw std::invalid_argument("Invalid adaptive construction parameters");
        require_index();
        if (high_factor*graph_->ef_construction_>1000000)
            throw std::invalid_argument("Maximum capacity exceeds 1000000");
        const auto levels=build_levels(data,supplied);
        const size_t n=data.shape(0),nt=thread_count(requested,n),e0=graph_->ef_construction_;
        // Round capacities to nearest-even and the checkpoint to nearest, ties upward.
        const size_t low=std::max(graph_->M_,static_cast<size_t>(std::nearbyint(low_factor*e0)));
        const size_t high=std::max(e0,static_cast<size_t>(std::nearbyint(high_factor*e0)));
        const size_t checkpoint=std::max<size_t>(1,std::llround(checkpoint_factor*e0));
        const size_t warmup=std::min(n,n_ref);
        const size_t prefix=std::min(n,warmup+std::min(n-warmup,block_size));
        std::vector<float> mean(dim_),variance(dim_),normalized(nt*dim_),temp(dim_);
        std::vector<double> sum(dim_,0.0),sq(dim_,0.0);
        for(size_t row=0;row<warmup;++row) {
            normalize(data.data(row),temp.data());
            for(size_t j=0;j<dim_;++j) {
                const double v=temp[j];sum[j]+=v;sq[j]+=v*v;
            }
        }
        const double inverse=1.0/static_cast<double>(warmup);
        for(size_t j=0;j<dim_;++j) {
            const double m=sum[j]*inverse;
            mean[j]=static_cast<float>(m);
            variance[j]=static_cast<float>(std::max(sq[j]*inverse-m*m,0.0));
        }
        struct Worker {
            size_t calls{0},ef_sum{0},reached{0},allocated{0};
            ada_b_only::Moments moments;
        };
        struct Block {
            size_t begin,end,reached;
            bool calibration;
            double ef_mean;
            ada_b_only::Calibration used;
        };
        std::vector<Worker> stats(nt);
        std::vector<std::vector<size_t>> hist(nt,std::vector<size_t>(high+1,0));
        std::vector<Block> blocks;
        ada_b_only::Calibration calibration;
        size_t calls=0,ef_sum=0,reached=0;
        {
            py::gil_scoped_release release;
            PersistentParallelFor workers(nt);
            auto insert=[&](size_t row,size_t tid,bool fixed) {
                float *point=normalized.data()+tid*dim_;
                normalize(data.data(row),point);
                double similarity=0.0,similarity_variance=0.0;
                for(size_t j=0;j<dim_;++j) {
                    const double v=point[j];
                    similarity+=v*mean[j];similarity_variance+=v*v*variance[j];
                }
                const double expected=1.0-similarity;
                const double sigma=std::sqrt(std::max(similarity_variance,1e-12));
                const auto affine=calibration.affine(expected,sigma);
                const size_t before=graph_->getThreadDistanceComputations();
                const auto result=graph_->addPointLiveEfAllocationNoStats(
                    point,row,checkpoint,low,high,z_clip,
                    affine.first,affine.second,fixed,e0,levels[row]);
                auto &s=stats[tid];
                s.calls+=graph_->getThreadDistanceComputations()-before;
                s.ef_sum+=result.efc_used;
                ++s.allocated;++hist[tid][result.efc_used];
                if(result.checkpoint_reached) {
                    ++s.reached;
                    s.moments.observe(
                        (result.frontier_best_distance-expected)/std::max(sigma,1e-9));
                }
            };
            auto collect=[&](size_t begin,size_t end,bool fixed,bool update) {
                Worker total;
                for(const auto &s:stats) {
                    total.calls+=s.calls;total.ef_sum+=s.ef_sum;
                    total.reached+=s.reached;total.allocated+=s.allocated;
                    total.moments.merge(s.moments);
                }
                calls+=total.calls;ef_sum+=total.ef_sum;reached+=total.reached;
                blocks.push_back({begin,end,total.reached,fixed,
                    static_cast<double>(total.ef_sum)/total.allocated,calibration});
                if(update)calibration.update(total.moments,fixed,update_weight);
                std::fill(stats.begin(),stats.end(),Worker{});
            };
            insert(0,0,true);
            workers.run(1,warmup,[&](size_t row,size_t tid){insert(row,tid,true);});
            // Initialize search calibration from the following reference block.
            collect(0,warmup,true,false);
            for(size_t begin=warmup;begin<n;begin+=block_size) {
                const size_t end=std::min(n,begin+block_size);
                const bool fixed=begin==warmup;
                workers.run(begin,end,[&](size_t row,size_t tid){insert(row,tid,fixed);});
                collect(begin,end,fixed,true);
            }
        }
        py::dict histogram;
        size_t minimum=high,maximum=0,count=0,weighted=0;
        for(size_t ef=0;ef<=high;++ef) {
            size_t value=0;for(size_t t=0;t<nt;++t)value+=hist[t][ef];
            if(value) {
                histogram[py::str(std::to_string(ef))]=value;
                minimum=std::min(minimum,ef);maximum=std::max(maximum,ef);
                count+=value;weighted+=ef*value;
            }
        }
        if(count!=n || weighted!=ef_sum)throw std::runtime_error("Allocation accounting mismatch");
        py::list telemetry;
        for(const auto &b:blocks)
            telemetry.append(py::dict("begin"_a=b.begin,"end"_a=b.end,"fixed_reference"_a=b.calibration,
                "checkpoint_reached"_a=b.reached,"ef_mean"_a=b.ef_mean,
                "normalized_mean_used"_a=b.used.mean,"normalized_std_used"_a=b.used.std));
        return py::dict("method"_a="b_only","count"_a=n,"distance_computation_sum"_a=calls,
            "ef_sum"_a=ef_sum,"ef_mean"_a=static_cast<double>(ef_sum)/n,
            "ef_min"_a=minimum,"ef_max"_a=maximum,"ef_low"_a=low,"ef_high"_a=high,
            "upper_ef"_a=e0,"fixed_prefix_count"_a=prefix,"allocation_count"_a=n-prefix,
            "checkpoint_reached"_a=reached,"ef_histogram"_a=histogram,"blocks"_a=telemetry,
            "early_stopping"_a=false);
    }

    py::object knn_query(Array data,size_t k,int requested,bool counted) {
        require_index();check_shape(data);
        const size_t n=data.shape(0),nt=thread_count(requested,n);
        if(!k || k>graph_->getCurrentElementCount())
            throw std::invalid_argument("Require 1 <= k <= index size");
        py::array_t<hnswlib::labeltype> labels({n,k});
        py::array_t<float> distances({n,k});
        auto *out=labels.mutable_data();auto *dist=distances.mutable_data();
        std::vector<float> normalized(nt*dim_);
        std::vector<size_t> calls(nt,0);
        {
            py::gil_scoped_release release;
            ParallelFor(0,n,nt,[&](size_t row,size_t tid) {
                float *point=normalized.data()+tid*dim_;normalize(data.data(row),point);
                const size_t before=graph_->getThreadDistanceComputations();
                auto found=graph_->searchKnn(point,k);
                calls[tid]+=graph_->getThreadDistanceComputations()-before;
                if(found.size()!=k)throw std::runtime_error("Fewer than k reachable results");
                for(size_t j=k;j>0;--j) {
                    out[row*k+j-1]=found.top().second;dist[row*k+j-1]=found.top().first;found.pop();
                }
            });
        }
        if(counted) {
            size_t total=0;for(size_t c:calls)total+=c;
            return py::make_tuple(labels,distances,total);
        }
        return py::make_tuple(labels,distances);
    }
};
PYBIND11_MODULE(ada_hnswlib,m) {
    m.doc()="Adaptive cosine HNSW construction and counted query reader";
    m.attr("__version__")="2.0.0";
    py::class_<Index>(m,"Index")
      .def(py::init<const std::string&,int>(),"space"_a,"dim"_a)
      .def("init_index",&Index::init_index,"max_elements"_a,"M"_a=16,
           "ef_construction"_a=200,"random_seed"_a=100)
      .def("set_num_threads",&Index::set_num_threads)
      .def("set_ef",&Index::set_ef)
      .def("get_current_count",&Index::get_current_count)
      .def("preview_levels",&Index::preview_levels,"count"_a)
      .def("save_index",&Index::save_index)
      .def("load_index",&Index::load_index)
      .def("build_b_only",&Index::build_b_only,"data"_a,"num_threads"_a=-1,"level_values"_a=py::none(),
           "low_factor"_a=0.50,"high_factor"_a=1.15,"checkpoint_factor"_a=0.50,
           "n_ref"_a=500,"block_size"_a=32768,"update_weight"_a=0.75,"z_clip"_a=2.5)
      .def("build_fixed",&Index::build_fixed,"data"_a,"ef"_a=0,"upper_ef"_a=0,
           "num_threads"_a=-1,"level_values"_a=py::none())
      .def("knn_query",&Index::knn_query,"data"_a,"k"_a=10,"num_threads"_a=-1,
           "return_distance_computations"_a=false);
}
