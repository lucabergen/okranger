#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TreeKernel.h"
#include "ApproxFeatures.h"
#include "Data.h"

namespace ranger {

TreeKernel::TreeKernel(OutputRepresentation output_representation) :
    output_representation(output_representation) {
  if (output_representation != OUTPUT_REPRESENTATION_FEATURES &&
      output_representation != OUTPUT_REPRESENTATION_GRAM) {
    throw std::invalid_argument(
        "TreeKernel requires Phi or K output representation.");
  }
}

TreeKernel::TreeKernel(OutputRepresentation output_representation, std::vector<std::vector<size_t>>& child_nodeIDs,
    std::vector<size_t>& split_varIDs, std::vector<double>& split_values, 
    std::vector<std::vector<double>>& terminal_output_means) :
    Tree(child_nodeIDs, split_varIDs, split_values), output_representation(output_representation), 
    terminal_output_means(terminal_output_means) {
}

void TreeKernel::allocateMemory() {
  if (!memory_saving_splitting) {
    const size_t max_num_splits = data->getMaxNumUniqueValues();
    const size_t rank = approx_features->rank;

    if (rank > 0 && max_num_splits > std::numeric_limits<size_t>::max() / rank) {
      throw std::length_error("Large-Q split workspace size overflow.");
    }

    counter.resize(max_num_splits);
    sums.resize(max_num_splits * rank);
  }
}

void TreeKernel::addToTerminalNodes(size_t nodeID) {
  if (!storesLeafPredictions()) {
    return;
  }

  const size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  if (num_samples_node == 0) {
    throw std::logic_error("Cannot compute an output mean for an empty terminal node.");
  }

  const size_t num_features = data->getNumColsY();
  std::vector<double>& output_mean = terminal_output_means.at(nodeID);
  output_mean.assign(num_features, 0.0);

  // Iterate over bootstrap positions so duplicated sample IDs retain multiplicity
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    const size_t sample_id = sampleIDs[pos];
    for (size_t feature = 0; feature < num_features; ++feature) {
      output_mean[feature] += data->get_y(sample_id, feature);
    }
  }

  const double inverse_node_size = 1.0 / static_cast<double>(num_samples_node);

  for (double& value : output_mean) {
    value *= inverse_node_size;
  }
}

void TreeKernel::appendToFileInternal(std::ofstream& file) {
  (void) file;
  throw std::logic_error("TreeKernel serialization is not implemented yet.");
}

bool TreeKernel::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  const size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  if (save_node_stats) {
    num_samples_nodes[nodeID] = num_samples_node;
    node_predictions[nodeID] = std::numeric_limits<double>::quiet_NaN();

    // Store an output mean for every node when returning node statistics
    addToTerminalNodes(nodeID);
  }

  if (num_samples_node <= (*min_node_size)[0] || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    split_values[nodeID] = 0.0;
    if (!save_node_stats) {
      addToTerminalNodes(nodeID);
    }
    return true;
  }

  const bool stop = findBestSplit(nodeID, possible_split_varIDs);
  if (stop) {
    split_values[nodeID] = 0.0;
    if (!save_node_stats) {
      addToTerminalNodes(nodeID);
    }
  }
  return stop;
}

void TreeKernel::createEmptyNodeInternal() {
  if (storesLeafPredictions()) {
    terminal_output_means.emplace_back();
  }

  if (save_node_stats) {
    node_predictions.push_back(std::numeric_limits<double>::quiet_NaN());
  }
}

double TreeKernel::computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) {
  (void) prediction_error_casewise;
  throw std::logic_error("TreeKernel prediction accuracy is not implemented yet.");
}

void TreeKernel::cleanUpInternal() {
  counter.clear();
  counter.shrink_to_fit();
  sums.clear();
  sums.shrink_to_fit();
}

double TreeKernel::kernelScore(const std::vector<double>& sum, size_t n) const {
  if (n == 0) {
    return 0.0;
  }

  double squared_norm = 0.0;
  for (double value : sum) {
    squared_norm += value * value;
  }

  return squared_norm / static_cast<double>(n);
}

double TreeKernel::splitScore(const std::vector<double>& sum_left,
    const std::vector<double>& sum_total, size_t n_left, size_t n_right) const {
  double squared_norm_left = 0.0;
  double squared_norm_right = 0.0;

  for (size_t feature = 0; feature < sum_left.size(); ++feature) {
    const double left_value = sum_left[feature];
    const double right_value = sum_total[feature] - left_value;

    squared_norm_left += left_value * left_value;
    squared_norm_right += right_value * right_value;
  }

  const double left_score = n_left == 0
      ? 0.0
      : squared_norm_left / static_cast<double>(n_left);
  const double right_score = n_right == 0
      ? 0.0
      : squared_norm_right / static_cast<double>(n_right);

  return left_score + right_score;
}

bool TreeKernel::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  const size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  // Rank-equivalent child score; parent_score is constant across candidates
  double best_score = -1.0;
  size_t best_varID = 0;
  double best_value = 0.0;

  // Stop early if no split posssible
  if (num_samples_node < 2 * (*min_bucket)[0]) {
    return true;
  }

  if (approx_features == nullptr) {
    throw std::logic_error("TreeKernel requires forest-wide output features.");
  }

  const size_t rank = approx_features->rank;
  if (rank == 0) {
    return true;
  }

  std::vector<double> sum_total(rank, 0.0);
  std::vector<double> first_output(rank, 0.0);
  bool first_sample = true;
  bool pure = true;

  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    const size_t sample_id = sampleIDs[pos];

    for (size_t feature = 0; feature < rank; ++feature) {
      const double value = approx_features->get(sample_id, feature);
      sum_total[feature] += value;

      if (first_sample) {
        first_output[feature] = value;
      } else if (value != first_output[feature]) {
        pure = false;
      }
    }
    first_sample = false;
  }

  // Stop when every sample has the same projected output vector.
  if (pure) {
    return true;
  }

  // For all possible split variables
  for (size_t varID : possible_split_varIDs) {
    const size_t num_unique = data->getNumUniqueDataValues(varID);
    if (num_unique == 0) {
      continue;
    }

    const double q = static_cast<double>(num_samples_node) / static_cast<double>(num_unique);

    if (memory_saving_splitting || q < Q_THRESHOLD) {
      findBestSplitValueSmallQ(nodeID, varID, num_samples_node, sum_total, best_value, best_varID, best_score);
    } else {
      findBestSplitValueLargeQ(nodeID, varID, num_samples_node, sum_total, best_value, best_varID, best_score);
    }
  }

  // Stop if no good split found
  if (best_score < 0.0) {
    return true;
  }

  // Save best values
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Save split statistics
  if (save_node_stats) {
    split_stats[nodeID] = best_score;
  }

  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addImpurityImportance(nodeID, best_varID, best_score, sum_total);
  }

  saveSplitVarID(best_varID);
  return false;
}

void TreeKernel::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_samples_node,
  const std::vector<double>& sum_total, double& best_value, size_t& best_varID, double& best_score) {

  const size_t rank = approx_features->rank;

  std::vector<double> possible_split_values;
  data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  if (possible_split_values.size() < 2) {
    return;
  }

  const size_t num_splits = possible_split_values.size();

  // Use local buffers in memory-saving mode; otherwise reuse the tree workspace.
  std::vector<size_t> local_counter;
  std::vector<double> local_sums;
  std::vector<size_t>* counts = &counter;
  std::vector<double>* bucket_sums = &sums;

  if (memory_saving_splitting) {
    if (rank > 0 && num_splits > std::numeric_limits<size_t>::max() / rank) {
      throw std::length_error("Small-Q split workspace size overflow.");
    }

    local_counter.resize(num_splits);
    local_sums.resize(num_splits * rank);
    counts = &local_counter;
    bucket_sums = &local_sums;
  } else {
    if (counter.size() < num_splits || sums.size() < num_splits * rank) {
      throw std::logic_error("Small-Q split workspace is too small.");
    }

    std::fill_n(counter.begin(), num_splits, 0);
    std::fill_n(sums.begin(), num_splits * rank, 0.0);
  }

  // Accumulate output-feature sums in bootstrap order within each local x bucket.
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    const size_t sample_id = sampleIDs[pos];
    const double x_value = data->get_x(sample_id, varID);

    const auto value_it = std::lower_bound(possible_split_values.begin(), possible_split_values.end(), x_value);
    if (value_it == possible_split_values.end() || *value_it != x_value) {
      throw std::logic_error("Observed predictor value is missing from the Small-Q buckets.");
    }

    const size_t index = static_cast<size_t>(value_it - possible_split_values.begin());
    ++(*counts)[index];

    const size_t offset = index * rank;
    for (size_t feature = 0; feature < rank; ++feature) {
      (*bucket_sums)[offset + feature] += approx_features->get(sample_id, feature);
    }
  }

  std::vector<double> sum_left(rank, 0.0);
  size_t n_left = 0;

  // Iterate over predictor values observed in this node 
  for (size_t i = 0; i + 1 < num_splits; ++i) {
    if ((*counts)[i] == 0) {
      continue;
    }

    n_left += (*counts)[i];
    const size_t offset = i * rank;

    for (size_t feature = 0; feature < rank; ++feature) {
      sum_left[feature] += (*bucket_sums)[offset + feature];
    }

    const size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    if (n_left < (*min_bucket)[0] || n_right < (*min_bucket)[0]) {
      continue;
    }

    double candidate_score = splitScore(sum_left, sum_total, n_left, n_right);
    regularize(candidate_score, varID);

    if (candidate_score > best_score) {
      const double x_left = possible_split_values[i];
      const double x_right = possible_split_values[i + 1];

      best_value = (x_left + x_right) / 2.0;
      if (best_value == x_right) {
        best_value = x_left;
      }

      best_varID = varID;
      best_score = candidate_score;
    }
  }
}

void TreeKernel::findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_samples_node, 
  const std::vector<double>& sum_total, double& best_value, size_t& best_varID, double& best_score) {

  const size_t rank = approx_features->rank;
  const size_t num_unique = data->getNumUniqueDataValues(varID);

  if (num_unique < 2) {
    return;
  }

  if (counter.size() < num_unique || sums.size() < num_unique * rank) {
    throw std::logic_error("Large-Q split workspace is too small.");
  }

  std::fill_n(counter.begin(), num_unique, 0);
  std::fill_n(sums.begin(), num_unique * rank, 0.0);

  // Group the bootstrap indices by their predictor values
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    const size_t sample_id = sampleIDs[pos];
    const double x_value = data->get_x(sample_id, varID);

    const size_t index = data->getIndex(sample_id, varID);
    if (index >= num_unique) {
      throw std::logic_error("Predictor index exceeds the Large-Q workspace.");
    }

    ++counter[index];
    const size_t offset = index * rank;

    for (size_t feature = 0; feature < rank; ++feature) {
      sums[offset + feature] += approx_features->get(sample_id, feature);
    }
  }

  std::vector<double> sum_left(rank, 0.0);
  size_t n_left = 0;

  // Iterate over unique predictor values
  for (size_t i = 0; i + 1 < num_unique; ++i) {
    if (counter[i] == 0) {
      continue;
    }

    n_left += counter[i];
    const size_t offset = i * rank;

    for (size_t feature = 0; feature < rank; ++feature) {
      sum_left[feature] += sums[offset + feature];
    }

    const size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    if (n_left < (*min_bucket)[0] || n_right < (*min_bucket)[0]) {
      continue;
    }

    double candidate_score = splitScore(sum_left, sum_total, n_left, n_right);
    regularize(candidate_score, varID);

    if (candidate_score > best_score) {
      size_t next_index = i + 1;
      while (next_index < num_unique && counter[next_index] == 0) {
        ++next_index;
      }

      if (next_index == num_unique) {
        break;
      }

      const double x_left = data->getUniqueDataValue(varID, i);
      const double x_right = data->getUniqueDataValue(varID, next_index);

      best_value = (x_left + x_right) / 2.0;
      if (best_value == x_right) {
        best_value = x_left;
      }

      best_varID = varID;
      best_score = candidate_score;
    }
  }
}

void TreeKernel::addImpurityImportance(size_t nodeID, size_t varID, double best_score, const std::vector<double>& sum_total) {

  const size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  double parent_score = kernelScore(sum_total, num_samples_node);
  regularize(parent_score, varID);

  const double gain = best_score - parent_score;
  const size_t original_varID = data->getUnpermutedVarID(varID);

  if (importance_mode == IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
    (*variable_importance)[original_varID] -= gain;
  } else {
    (*variable_importance)[original_varID] += gain;
  }
}

} // namespace ranger
