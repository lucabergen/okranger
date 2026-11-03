##' Compute forest weights
##'
##' Compute the forest weight matrix for query observations and the training
##' observations of a fitted ranger forest. For query observations
##' \eqn{x_1, \ldots, x_m}, the function returns an \eqn{m \times n} matrix
##' \eqn{W}, where \eqn{W_{ij} = w_j(x_i)} is the weight assigned to training
##' observation \eqn{j} for query observation \eqn{i}. Rows correspond to
##' query observations and columns to training observations. The returned
##' matrix is sparse by default.
##'
##' If the training outputs are represented by an explicit feature matrix
##' \eqn{\Phi \in \mathbb{R}^{n \times d}}, the predicted feature vectors are
##' \eqn{\hat{\Phi} = W \Phi}.
##'
##' With \code{oob = TRUE}, weights are computed using only trees in which
##' each query observation was out of bag. This mode is restricted to the
##' training observations.
##'
##' @param object A fitted object of class \code{ranger}. It must contain a
##'   saved forest and in-bag counts (fit with \code{write.forest = TRUE} and
##'   \code{keep.inbag = TRUE}).
##' @param training.data The predictor data used to train \code{object}, in the
##'   same row order.
##' @param new.data Optional query predictor data. If \code{NULL}, the training
##'   observations are used as queries. Must be \code{NULL} when \code{oob = TRUE}.
##' @param oob If \code{TRUE}, compute OOB weights for the training
##'   observations. Each training row must have at least one OOB tree.
##' @return A sparse matrix with one row per query observation and one column
##'   per training observation. The matrix is a \code{dgCMatrix}.
##' @export
getForestWeights <- function(object, training.data, 
                             new.data = NULL, oob = FALSE) {
  if (!inherits(object, "ranger")) {
    stop("`object` must be a fitted ranger model.")
  }
  
  if (is.null(object$forest)) {
    stop("`object` must contain a saved forest; fit with `write.forest = TRUE`.")
  }
  
  if (!is.logical(oob) || length(oob) != 1L || is.na(oob)) {
    stop("`oob` must be TRUE or FALSE.")
  }
  
  if (oob && !is.null(new.data)) {
    stop(
      "For OOB weights, omit `new.data`; the query points are `training.data` ",
      "in its original training row order."
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
  n.new <- NROW(new.data)
  
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
  
  # One column per tree, one row per training observation.
  inbag.counts <- do.call(cbind, object$inbag.counts)
  storage.mode(inbag.counts) <- "double"
  
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
  
  as.leaf.matrix <- function(x, n.rows, n.cols, arg) {
    if (is.null(dim(x))) {
      if (length(x) != n.rows * n.cols) {
        stop("Unexpected number of terminal-node IDs in `", arg, "`.")
      }
      x <- matrix(x, nrow = n.rows, ncol = n.cols)
    }
    
    if (length(dim(x)) != 2L ||
        nrow(x) != n.rows ||
        ncol(x) != n.cols) {
      stop("Unexpected dimensions for terminal-node IDs in `", arg, "`.")
    }
    
    storage.mode(x) <- "double"
    x
  }
  
  train.leafIDs <- as.leaf.matrix(
    train.leafIDs, n.train, n.trees, "training.data"
  )
  new.leafIDs <- as.leaf.matrix(
    new.leafIDs, n.new, n.trees, "new.data"
  )
  
  result <- forestWeightsCpp(
    train_leaf_ids = train.leafIDs,
    query_leaf_ids = new.leafIDs,
    inbag_counts = inbag.counts,
    oob = oob
  )
  
  weights <- Matrix::sparseMatrix(
    i = result$i,
    j = result$j,
    x = result$x,
    dims = c(n.new, n.train),
    dimnames = list(rownames(new.data), rownames(training.data))
  )
  
  weights
}
