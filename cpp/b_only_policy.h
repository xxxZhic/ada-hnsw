#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace ada_b_only {

// Moments of h = (frontier_distance - estimated_distance_mean) / estimated_std.
struct Moments {
    size_t count{0};
    double sum{0}, squared_sum{0};

    void observe(double h) {
        if (!std::isfinite(h)) return;
        ++count;
        sum += h;
        squared_sum += h * h;
    }
    void merge(const Moments &other) {
        count += other.count;
        sum += other.sum;
        squared_sum += other.squared_sum;
    }
};

struct Calibration {
    double mean{0}, std{1};

    // The reference block initializes the state. Later blocks update the
    // variance with the between-block mean shift included.
    void update(const Moments &s, bool replace, double weight) {
        if (!s.count) return;
        const double n = static_cast<double>(s.count);
        const double m = s.sum / n;
        const double variance = std::max(s.squared_sum / n - m * m, 1e-12);
        if (replace) {
            mean = m;
            std = std::sqrt(variance);
        } else {
            const double shift = m - mean;
            const double old_variance = std * std;
            mean = (1.0 - weight) * mean + weight * m;
            std = std::sqrt(std::max((1.0 - weight) * old_variance
                + weight * variance + weight * (1.0 - weight) * shift * shift,
                1e-12));
        }
    }

    // z = ((delta - expected) / sigma - mean) / std = (delta - offset) / scale.
    // Precompute offset and scale before traversing the graph.
    std::pair<double, double> affine(double expected, double sigma) const {
        const double s = std::max(std, 1e-9);
        sigma = std::max(sigma, 1e-9);
        const double coefficient = 1.0 / (sigma * s);
        const double intercept = (expected / sigma + mean) / s;
        return {intercept / coefficient, 1.0 / coefficient};
    }
};
}  // namespace ada_b_only
