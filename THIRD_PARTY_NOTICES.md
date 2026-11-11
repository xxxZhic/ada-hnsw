# Third-party code

`cpp/hnswlib/` derives from [nmslib/hnswlib](https://github.com/nmslib/hnswlib),
distributed under the Apache License 2.0, reproduced in
[cpp/hnswlib/LICENSE](cpp/hnswlib/LICENSE). The worker-loop and normalization
patterns in `cpp/bindings.cpp` also derive from hnswlib's Python bindings.

Local modifications include thread-local distance counting, preassigned levels,
fixed layer-specific construction capacities, and adaptive level-0 search
using normalized frontier distances. Compatibility tests use the upstream
hnswlib 0.8.0 query reader.

No license is newly assigned here to the project's original research code.
