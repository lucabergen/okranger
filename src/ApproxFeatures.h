#ifndef APPROXFEATURES_H_
#define APPROXFEATURES_H_

#include <cstddef>
#include <vector>

namespace ranger {

class Data;

// Z is an n-by-rank matrix stored as one row-major vector.
// Z(i, k) is values[i * rank + k]; i is the original training sample ID.
// n comes from the forest's training data and is not stored separately.
struct ApproxFeatures {
  size_t rank;
  std::vector<double> values;

  ApproxFeatures() : rank(0) {}

  double get(size_t sample_id, size_t feature) const {
    return values[sample_id * rank + feature];
  }
};

// The input is read through Data::get_y(); the returned projected features are
// aligned with original training-row IDs and cover all n training rows.
ApproxFeatures approximateFromGram(
    const Data& data,
    size_t n,
    double approx_tolerance);

ApproxFeatures approximateFromFeatures(
    const Data& data,
    size_t n,
    size_t d,
    double approx_tolerance);

} // namespace ranger

#endif /* APPROXFEATURES_H_ */
