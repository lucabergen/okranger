#ifndef TREEKERNEL_H_
#define TREEKERNEL_H_

#include <stdexcept>
#include <vector>

#include "globals.h"
#include "Tree.h"

namespace ranger {

class TreeKernel: public Tree {
public:
  explicit TreeKernel(OutputRepresentation output_representation);

  // Create from loaded forest
  TreeKernel(OutputRepresentation output_representation, std::vector<std::vector<size_t>>& child_nodeIDs, 
    std::vector<size_t>& split_varIDs, std::vector<double>& split_values, std::vector<std::vector<double>>& terminal_output_means);

  TreeKernel(const TreeKernel&) = delete;
  TreeKernel& operator=(const TreeKernel&) = delete;

  virtual ~TreeKernel() override = default;

  void allocateMemory() override;

  void appendToFileInternal(std::ofstream& file) override;

  // Returns the mean output features given by Phi in the terminal node reached by the sample
  const std::vector<double>& getPrediction(size_t sampleID) const {
    // Only defined for explicit feature representations
    if (output_representation != OUTPUT_REPRESENTATION_FEATURES) {
      throw std::logic_error("Leaf predictions are only available for explicit Phi outputs.");
    }
    const size_t terminal_nodeID = prediction_terminal_nodeIDs.at(sampleID);
    return terminal_output_means.at(terminal_nodeID);
  }

  // Returns the ID of the terminal node reached by the sample
  size_t getPredictionTerminalNodeID(size_t sampleID) const {
    return prediction_terminal_nodeIDs.at(sampleID);
  }

  // Returns output means indexed by node ID; non-empty only when given Phi
  // Entries of non-terminal nodes are empty, except when saving node statistics
  const std::vector<std::vector<double>>& getTerminalOutputMeans() const {
    return terminal_output_means;
  }

private:
  bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) override;
  void createEmptyNodeInternal() override;

  double computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) override;

  // Called by splitNodeInternal(). Sets split_varIDs and split_values.
  bool findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  // Small-Q search using predictor values observed in the current node
  void findBestSplitValueSmallQ(size_t nodeID, size_t varID,
    size_t num_samples_node, const std::vector<double>& sum_total,
    double& best_value, size_t& best_varID, double& best_score);
  // Large-Q search using ranger's global unique-value indices
  void findBestSplitValueLargeQ(size_t nodeID, size_t varID,
    size_t num_samples_node, const std::vector<double>& sum_total,
    double& best_value, size_t& best_varID, double& best_score);
  // Helper function; computes the projected-feature score of one node
  double kernelScore(const std::vector<double>& sum, size_t n) const;
  // Returns sum of projected-feature scores of left and right child nodes
  double splitScore(const std::vector<double>& sum_left, const std::vector<double>& sum_total, 
    size_t n_left, size_t n_right) const;

  // Returns true when explicit output features are available
  bool storesLeafPredictions() const {
    return output_representation == OUTPUT_REPRESENTATION_FEATURES;
  }

  // Computes and stores the explicit output feature mean in a terminal node
  void addToTerminalNodes(size_t nodeID);

  void addImpurityImportance(size_t nodeID, size_t varID, double best_score, const std::vector<double>& sum_total);

  void cleanUpInternal() override;

  const OutputRepresentation output_representation;

  // Indexed by node ID; non-empty only when Phi is given
  // Stores only means of terminal nodes, except when saving node statistics
  std::vector<std::vector<double>> terminal_output_means;

  std::vector<size_t> counter;
  std::vector<double> sums;
};

} // namespace ranger

#endif /* TREEKERNEL_H_ */
