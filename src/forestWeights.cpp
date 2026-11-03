#include <Rcpp.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

// [[Rcpp::export]]
Rcpp::List forestWeightsCpp(Rcpp::NumericMatrix train_leaf_ids,
    Rcpp::NumericMatrix query_leaf_ids, Rcpp::NumericMatrix inbag_counts, bool oob) {

  const R_xlen_t n_train = train_leaf_ids.nrow();
  const R_xlen_t n_query = query_leaf_ids.nrow();
  const R_xlen_t n_trees = train_leaf_ids.ncol();

  if (train_leaf_ids.ncol() != query_leaf_ids.ncol() ||
      train_leaf_ids.ncol() != inbag_counts.ncol() || inbag_counts.nrow() != n_train) {
    Rcpp::stop("Input dimensions for leaf IDs and inbag counts do not match.");
  }
  if (oob && n_query != n_train) {
    Rcpp::stop("OOB weights require the training observations as query data.");
  }
  if (n_query > std::numeric_limits<int>::max() ||
      n_train > std::numeric_limits<int>::max()) {
    Rcpp::stop("Matrix dimensions exceed the sparse-matrix index range.");
  }

  // weight_totals[i][j] is the sum of tree-wise weights for query i and
  // training sample j. Dividing by trees_used[i] gives w_j(x_i).
  std::vector<std::unordered_map<R_xlen_t, double>> weight_totals(n_query);

  // Number of trees that contribute weights to each query
  std::vector<int> trees_used(n_query, 0);

  for (R_xlen_t t = 0; t < n_trees; ++t) {
    // Group training samples by the leaf they reach in tree t (indexed by leaf ID)
    std::unordered_map<std::size_t, std::vector<R_xlen_t>> train_by_leaf;
    train_by_leaf.reserve(static_cast<std::size_t>(n_train));

    for (R_xlen_t j = 0; j < n_train; ++j) {
      // inbag_counts(j, t) is how many times training sample j was drawn in tree t's bootstrap sample
      const double inbag_count = inbag_counts(j, t);
      if (!std::isfinite(inbag_count) || inbag_count < 0) {
        Rcpp::stop("Inbag counts must be finite and non-negative.");
      }

      if (inbag_count > 0) {
        const double leaf_id = train_leaf_ids(j, t);
        if (!std::isfinite(leaf_id) || leaf_id < 0 || std::floor(leaf_id) != leaf_id) {
          Rcpp::stop("Training terminal-node IDs must be finite non-negative integers.");
        }
        train_by_leaf[static_cast<std::size_t>(leaf_id)].push_back(j);
      }
    }

    // Group query samples by the leaf they reach in tree t (indexed by leaf id)
    std::unordered_map<std::size_t, std::vector<R_xlen_t>> query_by_leaf;
    query_by_leaf.reserve(static_cast<std::size_t>(n_query));

    for (R_xlen_t i = 0; i < n_query; ++i) {
      const double leaf_id = query_leaf_ids(i, t);
      if (!std::isfinite(leaf_id) || leaf_id < 0 || std::floor(leaf_id) != leaf_id) {
        Rcpp::stop("Query terminal-node IDs must be finite non-negative integers.");
      }
      query_by_leaf[static_cast<std::size_t>(leaf_id)].push_back(i);
    }

    // For each query sample, find the training samples in the same leaf
    for (auto query_group = query_by_leaf.begin();
         query_group != query_by_leaf.end(); ++query_group) {
      auto train_group = train_by_leaf.find(query_group->first);
      if (train_group == train_by_leaf.end()) {
        continue;
      }

      const std::vector<R_xlen_t>& train_ids = train_group->second;

      // Sum the bootstrap counts of the in-bag training rows in this leaf.
      // This is the denominator for the tree weights below.
      double leaf_count = 0;
      for (R_xlen_t j : train_ids) {
        leaf_count += inbag_counts(j, t);
      }
      if (leaf_count <= 0) {
        continue;
      }

      for (R_xlen_t i : query_group->second) {
        // OOB weights use only trees that did not sample query i
        if (oob && inbag_counts(i, t) > 0) {
          continue;
        }

        ++trees_used[i];
        std::unordered_map<R_xlen_t, double>& query_weights = weight_totals[i];

        // Normalize j's in-bag count by the total in-bag count in the leaf
        // shared with query i. This is j's weight from tree t.
        for (R_xlen_t j : train_ids) {
          query_weights[j] += inbag_counts(j, t) / leaf_count;
        }
      }
    }
  }

  // Give errors for queries with no contributing trees
  for (R_xlen_t i = 0; i < n_query; ++i) {
    if (trees_used[i] == 0) {
      if (oob) {
        Rcpp::stop(
          "No OOB trees are available for a training observation. "
          "Increase `num.trees` or adjust the sampling settings."
        );
      }
      Rcpp::stop(
        "No in-bag training samples found in the query's terminal leaf. "
        "Check that `training.data` and in-bag counts match the fitted forest."
      );
    }
  }

  // Store each nonzero weight as a 1-based triplet for Matrix::sparseMatrix()
  // (query row, training-sample column, and weight value)
  std::vector<int> i_index;
  std::vector<int> j_index;
  std::vector<double> x_value;

  R_xlen_t nnz = 0;
  for (R_xlen_t i = 0; i < n_query; ++i) {
    nnz += static_cast<R_xlen_t>(weight_totals[i].size());
  }
  if (nnz > static_cast<R_xlen_t>(std::numeric_limits<int>::max())) {
    Rcpp::stop("Number of non-zero weights exceeds sparse-matrix index range.");
  }

  i_index.reserve(static_cast<std::size_t>(nnz));
  j_index.reserve(static_cast<std::size_t>(nnz));
  x_value.reserve(static_cast<std::size_t>(nnz));

  for (R_xlen_t i = 0; i < n_query; ++i) {
    // Average each training samples summed weight over trees used for query i
    const double tree_scale = 1.0 / static_cast<double>(trees_used[i]);

    for (auto entry = weight_totals[i].begin();
         entry != weight_totals[i].end(); ++entry) {
      const double weight = entry->second * tree_scale;
      if (weight != 0) {
        i_index.push_back(static_cast<int>(i + 1));
        j_index.push_back(static_cast<int>(entry->first + 1));
        x_value.push_back(weight);
      }
    }
  }

  return Rcpp::List::create(
      Rcpp::_["i"] = i_index,
      Rcpp::_["j"] = j_index,
      Rcpp::_["x"] = x_value,
      Rcpp::_["trees.used"] = trees_used);
}
