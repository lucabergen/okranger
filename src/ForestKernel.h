#ifndef FORESTKERNEL_H_
#define FORESTKERNEL_H_

#include <limits>

#include "Forest.h"
#include "ApproxFeatures.h"

namespace ranger {

class ForestKernel: public Forest {
public:
  ForestKernel() = default;

  ForestKernel(const ForestKernel&) = delete;
  ForestKernel& operator=(const ForestKernel&) = delete;

  ~ForestKernel() override = default;

  void loadForest(size_t num_trees, std::vector<std::vector<std::vector<size_t>>>& forest_child_nodeIDs,
    std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
    std::vector<std::vector<std::vector<double>>>& forest_terminal_output_means, std::vector<bool>& is_ordered_variable);

  std::vector<std::vector<std::vector<double>>> getTerminalOutputMeans() const;

  size_t getApproximationRank() const {
    return approx_features.rank;
  }
  
private:
  void initInternal() override;
  void growInternal() override;
  void allocatePredictMemory() override;
  void predictInternal(size_t sample_idx) override;
  void computePredictionErrorInternal() override;
  void writeOutputInternal() override;
  void writeConfusionFile() override;
  void writePredictionFile() override;
  void saveToFileInternal(std::ofstream& outfile) override;
  void loadFromFileInternal(std::ifstream& infile) override;

  void initializeApproxFeatures();

  const std::vector<double>& getTreePrediction(size_t tree_idx, size_t sample_idx) const;

  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;

  const ApproxFeatures* approxFeaturesForTree() const override {
    return &approx_features;
  }

  // Forest-owned approximation initialized before tree growth
  ApproxFeatures approx_features;
  size_t num_output_features = 0;
};

} // namespace ranger

#endif /* FORESTKERNEL_H_ */
