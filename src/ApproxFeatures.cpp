#include "ApproxFeatures.h"
#include "Data.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif

#include <Rconfig.h>
#include <R_ext/RS.h>
#include <R_ext/Lapack.h>

#ifndef FCONE
#define FCONE
#endif

namespace ranger {
namespace {

size_t checkedProduct(size_t rows, size_t cols) {
  if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) {
    throw std::runtime_error("Output feature matrix dimensions overflow.");
  }
  return rows * cols;
}

La_INT lapackSize(size_t n) {
  if (n > static_cast<size_t>(std::numeric_limits<La_INT>::max())) {
    throw std::runtime_error("Matrix dimension exceeds LAPACK integer range.");
  }
  return static_cast<La_INT>(n);
}

} // namespace

ApproxFeatures approximateFromGram(
    const Data& data,
    size_t n,
    double approx_tolerance) {
  if (n == 0 || n != data.getNumRows() || data.getNumColsY() != n) {
    throw std::runtime_error("K must be a non-empty n-by-n matrix.");
  }

  const La_INT order = lapackSize(n);
  const La_INT lda = std::max<La_INT>(1, order);
  const char uplo = 'U';

  // DPSTRF modifies its input, so copy K into the local vector a first.
  // Flatten (vectorise) K in column-major order, as LAPACK expects.
  std::vector<double> a(checkedProduct(n, n));
  for (size_t col = 0; col < n; ++col) {
    for (size_t row = 0; row < n; ++row) {
      a[row + col * n] = data.get_y(row, col);
    }
  }
  std::vector<La_INT> pivot(n);
  std::vector<double> work(std::max(static_cast<size_t>(1), 2 * n));
  La_INT rank_lapack = 0;
  La_INT info = 0;
  double lapack_tolerance = approx_tolerance;

  F77_CALL(dpstrf)(
      &uplo, &order, a.data(), &lda, pivot.data(),
      &rank_lapack, &lapack_tolerance, work.data(), &info FCONE);

  if (info < 0) {
    throw std::runtime_error("DPSTRF argument error.");
  }

  const size_t rank = static_cast<size_t>(rank_lapack);
  ApproxFeatures out;
  out.rank = rank;
  out.values.assign(checkedProduct(n, rank), 0.0);

  // DPSTRF's pivot array orders the observations; reorder the factor columns
  // so output row i always refers to original training sample i.
  for (size_t j = 0; j < n; ++j) {
    const size_t sample_id = static_cast<size_t>(pivot[j] - 1);
    const size_t num_features = std::min(rank, j + 1);
    for (size_t k = 0; k < num_features; ++k) {
      out.values[sample_id * rank + k] = a[k + j * n];
    }
  }

  return out;
}

ApproxFeatures approximateFromFeatures(
    const Data& data,
    size_t n,
    size_t d,
    double approx_tolerance) {
  if (n == 0 || n != data.getNumRows() || d == 0 || data.getNumColsY() != d) {
    throw std::runtime_error("Phi must be a non-empty n-by-d matrix.");
  }

  // DGEQP3 is applied to Phi^T (d by n), with observations as pivoted columns.
  const La_INT rows = lapackSize(d);
  const La_INT cols = lapackSize(n);
  const La_INT lda = std::max<La_INT>(1, rows);
  const La_INT kmax = std::min(rows, cols);

  std::vector<double> a(checkedProduct(d, n));
  for (size_t sample_id = 0; sample_id < n; ++sample_id) {
    for (size_t feature = 0; feature < d; ++feature) {
      a[feature + sample_id * d] = data.get_y(sample_id, feature);
    }
  }

  std::vector<La_INT> pivot(n, 0);
  std::vector<double> tau_qr(static_cast<size_t>(kmax));
  La_INT info = 0;
  La_INT lwork = -1;
  double work_query = 0.0;

  F77_CALL(dgeqp3)(
      &rows, &cols, a.data(), &lda, pivot.data(),
      tau_qr.data(), &work_query, &lwork, &info);

  if (info != 0 || !std::isfinite(work_query) || work_query < 1.0 ||
      work_query > static_cast<double>(std::numeric_limits<La_INT>::max())) {
    throw std::runtime_error("DGEQP3 workspace query failed.");
  }

  lwork = static_cast<La_INT>(std::ceil(work_query));
  std::vector<double> work(static_cast<size_t>(lwork));

  F77_CALL(dgeqp3)(
      &rows, &cols, a.data(), &lda, pivot.data(),
      tau_qr.data(), work.data(), &lwork, &info);

  if (info != 0) {
    throw std::runtime_error("DGEQP3 failed.");
  }

  // With column pivoting, the next diagonal of R is the largest remaining
  // residual norm. Truncate when its squared value is within the requested
  // absolute approximation tolerance.
  size_t rank = 0;
  for (; rank < static_cast<size_t>(kmax); ++rank) {
    const double diagonal = a[rank + rank * d];
    if (std::abs(diagonal) <= std::sqrt(approx_tolerance)) {
      break;
    }
  }

  ApproxFeatures out;
  out.rank = rank;
  out.values.assign(checkedProduct(n, rank), 0.0);

  // DGEQP3's upper-triangular R factor contains coordinates in pivot order.
  // Undo the QR pivoting so rows match the original sample order.
  // QR may choose either sign for a feature coordinate. Match Cholesky's
  // non-negative diagonal convention by flipping that coordinate for all samples.
  for (size_t pivoted_col = 0; pivoted_col < n; ++pivoted_col) {
    const size_t sample_id = static_cast<size_t>(pivot[pivoted_col] - 1);
    const size_t num_features = std::min(rank, pivoted_col + 1);
    for (size_t feature = 0; feature < num_features; ++feature) {
      const double diagonal = a[feature + feature * d];
      const double sign = diagonal < 0.0 ? -1.0 : 1.0;
      out.values[sample_id * rank + feature] =
        sign * a[feature + pivoted_col * d];
    }
  }

  return out;
}

} // namespace ranger
