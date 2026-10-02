
getForestWeights <- function(
    object,
    training.data, # Needed for predict(type = "terminalNodes"); needs to be passed in the same row-order as when used to train object
    new.data = NULL, # Evaluation points; NULL uses training data
    oob = FALSE
) {
  
  if (!inherits(object, "ranger")) {
    stop("`object` must be a fitted ranger model.")
  }
  
  if (is.null(object$forest)) {
    stop("`object` must contain a saved forest; fit with `write.forest = TRUE`.")
  }
  
  if (!is.logical(oob) || length(oob) != 1L || is.na(oob)) {
    stop("`oob` must be TRUE or FALSE.")
  }
  
  # OOB weights are defined here only for the training observations.
  # With no training_indices, their original row order is required.
  if (oob && !is.null(new.data)) {
    stop(
      "For OOB weights, omit `new.data`; the query points are ",
      "`training.data` in its original training row order."
    )
  }
  
  if (is.null(object$inbag.counts)) {
    stop("`object` needs to be trained with `keep.inbag = TRUE`.")
  }
  
  no.new.data <- is.null(new.data)
  if (no.new.data) {
    new.data <- training.data
  }
  
  n.train <- object$num.samples
  n.trees <- object$num.trees
  
  if (is.null(n.train) || length(n.train) != 1L || is.na(n.train)) {
    stop("Could not determine the number of training observations from `object`.")
  }
  
  if (NROW(training.data) != n.train) {
    stop(
      "`training.data` must contain all observations used to train `object`, ",
      "in their original row order."
    )
  }
  
  if (!is.list(object$inbag.counts) ||
      length(object$inbag.counts) != n.trees) {
    stop("`object$inbag.counts` must contain one count vector per tree.")
  }
  
  if (any(lengths(object$inbag.counts) != n.train)) {
    stop("Each inbag-count vector must have one entry per training observation.")
  }
  
  no.new.data <- is.null(new.data)
  
  if (no.new.data == TRUE) {
    new.data <- training.data
  } 
  
  # inbag.counts has one column per tree, one row per training observation
  inbag.counts <- do.call(cbind, object$inbag.counts)
  
  # Both leafIDs have one column per tree, one row per observation
  train.leafIDs <- predict(
    object = object, 
    data = training.data, 
    type = "terminalNodes"
  )$predictions
  
  if (no.new.data) {
    new.leafIDs <- train.leafIDs
  } else {
    new.leafIDs <- predict(
      object = object,
      data = new.data,
      type = "terminalNodes"
    )$predictions
  }
  
  n.new <- nrow(new.data)
  n.train <- nrow(training.data)
  num.trees <- ncol(train.leafIDs)
  
  weights <- matrix(0, nrow = n.new, ncol = n.train)
  trees.used <- integer(n.new)
  
  for (t in seq_len(num.trees)) {
    
    # Save count of each training observation in bootstrap sample of tree t
    boot.counts <- inbag.counts[, t]
    
    for (i in seq_len(n.new)) {
      
      # In OOB mode, keep only trees where query i was not in-bag.
      if (oob && boot.counts[i] > 0) {
        next
      }
      
      # is TRUE for each training obs. that is in the same leaf as new obs. i
      same.leaf <- train.leafIDs[, t] == new.leafIDs[i, t]
      # Multiply by the number of times the training obs. were in the bootstrap sample
      leaf.weights <- same.leaf * boot.counts
      leaf.mass <- sum(leaf.weights)
      
      if (leaf.mass > 0) {
        weights[i, ] <- weights[i, ] + leaf.weights / leaf.mass
        trees.used[i] <- trees.used[i] + 1L
      }
    }
  }
  
  # Average tree-specific normalized weights. A row with no usable OOB trees
  # is undefined.
  for (i in seq_len(n.new)) {
    if (trees.used[i] > 0L) {
      weights[i, ] <- weights[i, ] / trees.used[i]
    } else {
      weights[i, ] <- NA_real_
    }
  }
  
  weights
}
  
