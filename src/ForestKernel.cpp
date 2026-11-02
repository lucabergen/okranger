#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "ForestKernel.h"
#include "TreeKernel.h"
#include "Data.h"
#include "utility.h"

namespace ranger {

void ForestKernel::initInternal() {
  if (output_representation != OUTPUT_REPRESENTATION_FEATURES && output_representation != OUTPUT_REPRESENTATION_GRAM) {
    throw std::runtime_error("ForestKernel requires Phi or K output data.");
  }

  if (prediction_mode) {
    const bool terminal_node_prediction = prediction_type == TERMINALNODES;
     if (output_representation != OUTPUT_REPRESENTATION_FEATURES && 
      !(output_representation == OUTPUT_REPRESENTATION_GRAM && terminal_node_prediction)) {
        throw std::runtime_error(
          "Response prediction with 'K' output representation is not supported. "
          "Use getForestWeights() to obtain forest weights."
        );
      }
    return;
  }

  num_output_features = data->getNumColsY();

  // Use the regression forest defaults for tree-growth parameters
  if (mtry == 0) {
    unsigned long temp = sqrt((double) num_independent_variables);
    mtry = std::max((unsigned long) 1, temp);
  }

  if (min_node_size.size() == 1 && min_node_size[0] == 0) {
    min_node_size[0] = DEFAULT_MIN_NODE_SIZE_REGRESSION;
  }

  if (min_bucket.size() == 1 && min_bucket[0] == 0) {
    min_bucket[0] = DEFAULT_MIN_BUCKET;
  }

  // Use the regression forests predictor preprocessing
  if (!memory_saving_splitting) {
    data->sort();
  }

  // Construct a forest-level feature approximation on the full training data
  initializeApproxFeatures();
}

void ForestKernel::initializeApproxFeatures() {
  const size_t n = data->getNumRows();
  const size_t output_cols = data->getNumColsY();

  if (output_representation == OUTPUT_REPRESENTATION_FEATURES) {
    approx_features = approximateFromFeatures(*data, n, output_cols, approx_tolerance);
  } else if (output_representation == OUTPUT_REPRESENTATION_GRAM) {
    approx_features = approximateFromGram(*data, n, approx_tolerance);
  } else {
    // This indicates an internal dispatch/configuration error; R validates the
    // user-facing arguments before constructing a ForestKernel
    throw std::logic_error("ForestKernel requires Phi or K output data.");
  }
}

void ForestKernel::growInternal() {
  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i) {
    trees.push_back(std::make_unique<TreeKernel>(output_representation));
  }
}

void ForestKernel::loadForest(size_t num_trees, std::vector<std::vector<std::vector<size_t>>>& forest_child_nodeIDs,
  std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
  std::vector<std::vector<std::vector<double>>>& forest_terminal_output_means, std::vector<bool>& is_ordered_variable) {
    const bool terminal_node_prediction = prediction_type == TERMINALNODES;
    
    if (output_representation != OUTPUT_REPRESENTATION_FEATURES &&
      !(output_representation == OUTPUT_REPRESENTATION_GRAM && terminal_node_prediction)) {
        throw std::runtime_error(
          "Response prediction with 'K' output representation is not supported. "
          "Use getForestWeights() to obtain forest weights."
        );
    }

    if (forest_child_nodeIDs.size() < num_trees || forest_split_varIDs.size() < num_trees ||
        forest_split_values.size() < num_trees || forest_terminal_output_means.size() < num_trees) {
      throw std::runtime_error("Loaded OKranger forest has inconsistent tree counts.");
    }

  this->num_trees = num_trees;
  data->setIsOrderedVariable(is_ordered_variable);
  num_output_features = 0;

  for (const auto& tree_means : forest_terminal_output_means) {
    for (const auto& node_mean : tree_means) {
      if (!node_mean.empty()) {
        if (num_output_features == 0) {
          num_output_features = node_mean.size();
        } else if (node_mean.size() != num_output_features) {
          throw std::runtime_error("Loaded OKranger forest has inconsistent output dimensions.");
        }
      }
    }
  }
  if (num_output_features == 0 &&
    !(output_representation == OUTPUT_REPRESENTATION_GRAM && terminal_node_prediction)) {
      throw std::runtime_error("Loaded OKranger forest has no terminal Phi means.");
  }

  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i) {
    trees.push_back(std::make_unique<TreeKernel>(output_representation, forest_child_nodeIDs[i],
      forest_split_varIDs[i], forest_split_values[i], forest_terminal_output_means[i]));
  }
  equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

void ForestKernel::allocatePredictMemory() {
  const size_t num_prediction_samples = data->getNumRows();
  if (prediction_type == TERMINALNODES) {
    predictions = std::vector<std::vector<std::vector<double>>>(
        1, std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees))
    );
    return;
  }
  if (output_representation != OUTPUT_REPRESENTATION_FEATURES || num_output_features == 0) {
    throw std::logic_error("Kernel prediction requires loaded Phi terminal means.");
  }
   if (predict_all) {
    predictions = std::vector<std::vector<std::vector<double>>>(
      1, std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees * num_output_features))
    );
  } else {
    predictions = std::vector<std::vector<std::vector<double>>>(
      1, std::vector<std::vector<double>>(num_output_features, std::vector<double>(num_prediction_samples))
    );
  }
}

void ForestKernel::predictInternal(size_t sample_idx) {
  if (prediction_type == TERMINALNODES) {
    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
    }
    return;
  }

  if (prediction_type != RESPONSE) {
    throw std::runtime_error("Output kernel regression supports response and terminal-node predictions only.");
  }

  if (predict_all) {
    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      const std::vector<double>& tree_prediction = getTreePrediction(tree_idx, sample_idx);
      for (size_t feature = 0; feature < num_output_features; ++feature) {
        predictions[0][sample_idx][tree_idx * num_output_features + feature] = tree_prediction[feature];
      }
    }
  } else {
    for (size_t feature = 0; feature < num_output_features; ++feature) {
      double prediction_sum = 0.0;
      for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        prediction_sum += getTreePrediction(tree_idx, sample_idx)[feature];
      }
      predictions[0][feature][sample_idx] = prediction_sum / static_cast<double>(num_trees);
    }
  }
}

void ForestKernel::computePredictionErrorInternal() {
  if (output_representation != OUTPUT_REPRESENTATION_FEATURES) {
    throw std::runtime_error("OOB predictions and OOB error are not computed for K inputs");
  }

  const size_t num_features = data->getNumColsY();
  std::vector<size_t> samples_oob_count(num_samples, 0);
  predictions = std::vector<std::vector<std::vector<double>>>(
    1, std::vector<std::vector<double>>(num_samples, std::vector<double>(num_features, 0.0))
  );

  // Sum each trees Phi leaf mean only for samples that were OOB in that tree
  for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
    const auto& tree = dynamic_cast<const TreeKernel&>(*trees[tree_idx]);

    for (size_t sample_idx = 0; sample_idx < tree.getNumSamplesOob(); ++sample_idx) {
      const size_t sample_id = tree.getOobSampleIDs()[sample_idx];
      const std::vector<double>& tree_prediction = tree.getPrediction(sample_idx);

      if (tree_prediction.size() != num_features) {
        throw std::logic_error("TreeKernel returned a leaf prediction with an unexpected dimension.");
      }

      for (size_t feature = 0; feature < num_features; ++feature) {
        predictions[0][sample_id][feature] += tree_prediction[feature];
      }
      ++samples_oob_count[sample_id];
    }
  }

  // Average OOB predictions and compute MSE
  size_t num_predictions = 0;
  overall_prediction_error = 0.0;

  for (size_t sample_id = 0; sample_id < num_samples; ++sample_id) {
    if (samples_oob_count[sample_id] == 0) {
      std::fill(
        predictions[0][sample_id].begin(), predictions[0][sample_id].end(), std::numeric_limits<double>::quiet_NaN()
      );
      continue;
    }

    ++num_predictions;
    double sample_error = 0.0;
    const double inverse_oob_count = 1.0 / static_cast<double>(samples_oob_count[sample_id]);

    for (size_t feature = 0; feature < num_features; ++feature) {
      double& predicted_value = predictions[0][sample_id][feature];
      predicted_value *= inverse_oob_count;

      const double difference = predicted_value - data->get_y(sample_id, feature);
      sample_error += difference * difference;
    }

    overall_prediction_error += sample_error;
  }

  if (num_predictions == 0) {
    overall_prediction_error = std::numeric_limits<double>::quiet_NaN();
  } else {
    overall_prediction_error /= static_cast<double>(num_predictions);
  }
}

const std::vector<double>& ForestKernel::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
  const auto& tree = dynamic_cast<const TreeKernel&>(*trees[tree_idx]);
  return tree.getPrediction(sample_idx);
}

size_t ForestKernel::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
  const auto& tree = dynamic_cast<const TreeKernel&>(*trees[tree_idx]);
  return tree.getPredictionTerminalNodeID(sample_idx);
}

std::vector<std::vector<std::vector<double>>>
ForestKernel::getTerminalOutputMeans() const {
  std::vector<std::vector<std::vector<double>>> result;
  result.reserve(trees.size());

  for (const auto& tree : trees) {
    const auto& kernel_tree = dynamic_cast<const TreeKernel&>(*tree);
    result.push_back(kernel_tree.getTerminalOutputMeans());
  }

  return result;
}

void ForestKernel::writeOutputInternal() {
  if (verbose_out) {
    *verbose_out << "Tree type:                         Output kernel regression" << std::endl;
  }
}

void ForestKernel::writeConfusionFile() {
  // Kernel regression has no confusion matrix
}

void ForestKernel::writePredictionFile() {
  throw std::logic_error("ForestKernel prediction output is not implemented yet.");
}

void ForestKernel::saveToFileInternal(std::ofstream& outfile) {
  (void) outfile;
  throw std::logic_error("ForestKernel serialization is not implemented yet.");
}

void ForestKernel::loadFromFileInternal(std::ifstream& infile) {
  (void) infile;
  throw std::logic_error("ForestKernel loading is not implemented yet.");
}

} // namespace ranger
