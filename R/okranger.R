#' Fit an Output Kernel Random Forest
#'
#' Trains an Output Kernel Random Forest using predictor data and either an
#' output feature matrix or a Gram matrix.
#'
#' @details
#' If possible, provide a feature matrix \code{Phi}. This allows out-of-bag 
#' (OOB) predictions and OOB error to be computed when \code{oob.error = TRUE}. 
#' If only a Gram matrix \code{K} is provided, OOB predictions and OOB error are 
#' skipped; use \code{getForestWeights()} to obtain forest weights. Set 
#' \code{keep.inbag = TRUE} if OOB forest weights are needed.
#'
#' In \code{Phi}, rows represent observations and columns represent output
#' features. \code{K} is a square Gram matrix whose rows and columns represent
#' observations. If both \code{Phi} and \code{K} are supplied, only
#' \code{Phi} is used and a warning is issued.
#'
#' The predictor data must be a non-empty data frame without missing or
#' non-finite values. Character predictors are converted to factors.
#' Unordered factors are currently supported only with
#' \code{respect.unordered.factors = NULL}, \code{FALSE}, or \code{"ignore"}.
#'
#' @param Phi Optional numeric matrix of output features, with one row per
#'   observation and one column per output feature.
#' @param K Optional numeric Gram matrix with one row and column per
#'   observation in \code{x}. Used only when \code{Phi} is \code{NULL}.
#'   Symmetry is checked, but positive semidefiniteness (PSD) is not.
#' @param x Non-empty data frame of predictor variables.
#' @param approx.tolerance Positive scalar tolerance specifying the allowed
#'   deviation of the approximated score from the true score.
#' @param num.trees Number of trees to grow.
#' @param mtry Number of candidate predictors considered at each split.
#'   Defaults to the floor of the square root of the number of predictors,
#'   with a minimum of 1.
#' @param write.forest If \code{TRUE}, retain the fitted forest in the result.
#' @param min.node.size Minimum number of observations required in a node for
#'   attempting a split. Defaults to 5.
#' @param min.bucket Minimum number of observations required in each child node
#'   created by a split. Defaults to 1.
#' @param max.depth Maximum tree depth. Defaults to 0, which imposes no
#'   maximum depth.
#' @param replace If \code{TRUE}, sample observations with replacement.
#' @param sample.fraction Fraction of observations to sample for each tree.
#'   Defaults to \code{1} when sampling with replacement and \code{0.632}
#'   otherwise.
#' @param split.select.weights Optional numeric vector of split-selection
#'   weights, with one value per predictor, or a list with one vector per tree.
#' @param always.split.variables Optional names of predictors that should
#'   always be considered for splitting.
#' @param respect.unordered.factors How to handle unordered factors. Defaults
#'   to \code{"ignore"}; currently only \code{"ignore"} is supported.
#' @param keep.inbag If \code{TRUE}, retain in-bag counts for each tree.
#' @param oob.error If \code{TRUE} and \code{Phi} is supplied, compute OOB
#'   predictions and OOB error. These are skipped when only \code{K} is
#'   supplied.
#' @param num.threads Number of threads to use. Defaults to the value of
#'   \code{R_RANGER_NUM_THREADS} if set; otherwise to the
#'   \code{ranger.num.threads} option, then the \code{Ncpus} option, and
#'   finally to 2.
#' @param verbose If \code{TRUE}, print status information during fitting.
#' @param node.stats If \code{TRUE}, calculate and retain node statistics.
#' @param seed Random seed. If \code{NULL}, a seed is generated.
#'
#' @return An object of class \code{"ranger"} containing the fitted model
#'   information. When \code{write.forest = TRUE}, the fitted forest is
#'   included. If \code{Phi} is supplied and \code{oob.error = TRUE}, the
#'   result also contains OOB predictions and the OOB prediction error.
#'
#' @examples
#' n <- 30
#' x <- data.frame(
#'   x1 = seq_len(n),
#'   x2 = sin(seq_len(n) / 4)
#' )
#'
#' # Rows are observations; columns are output features.
#' Phi <- cbind(
#'   feature1 = sin(seq_len(n) / 5),
#'   feature2 = cos(seq_len(n) / 5)
#' )
#'
#' # Recommended: fit using the output feature matrix.
#' fit_phi <- okranger(
#'   Phi = Phi, x = x, approx.tolerance = 1e-3,
#'   num.trees = 20, num.threads = 1, verbose = FALSE, seed = 123
#' )
#'
#' # Alternative: fit using the corresponding Gram matrix.
#' K <- tcrossprod(Phi)
#' fit_K <- okranger(
#'   K = K, x = x, approx.tolerance = 1e-3,
#'   num.trees = 20, num.threads = 1, verbose = FALSE,
#'   oob.error = FALSE, seed = 123
#' )
#' 
#' @seealso \code{\link{ranger}}
#' @export
okranger <- function(Phi = NULL, K = NULL, x, approx.tolerance, num.trees = 500, 
                     mtry = NULL, write.forest = TRUE, min.node.size = NULL, 
                     min.bucket = NULL, max.depth = NULL, replace = TRUE, 
                     sample.fraction = ifelse(replace, 1, 0.632), 
                     split.select.weights = NULL, always.split.variables = NULL,
                     respect.unordered.factors = NULL, keep.inbag = FALSE, 
                     oob.error = TRUE, num.threads = NULL, verbose = TRUE, 
                     node.stats = FALSE, seed = NULL, ...) {
  
  ## Check inputs
  
  if (length(list(...)) > 0) {
    warning(paste("Unused arguments:", paste(names(list(...)), collapse = ", ")))
  }
  
  if (is.null(Phi) && is.null(K)) {
    stop("Either 'Phi' or 'K' must be provided.")
  }
  
  if (!is.null(Phi) && !is.null(K)) {
    warning("Both 'Phi' and 'K' were provided; only 'Phi' will be used.")
  }
  
  if (!is.numeric(approx.tolerance) || length(approx.tolerance) != 1L ||
      !is.finite(approx.tolerance) || approx.tolerance <= 0) {
    stop("'approx.tolerance' must be a single finite number greater than 0.")
  }
  
  if (!is.data.frame(x)) {
    stop("'x' must be a data.frame.")
  }
  
  if (nrow(x) < 1L) {
    stop("'x' must contain at least one row.")
  }
  
  if (ncol(x) < 1L) {
    stop("'x' must contain at least one column.")
  }
  
  if (!all(vapply(
    x,
    function(col) {is.atomic(col) && is.null(dim(col)) && !is.complex(col)},
    logical(1)
  ))) {
    stop("Each column of 'x' must be a one-dimensional, non-complex vector.")
  }
  
  bad_x <- vapply(x, function(col) {
      if (is.numeric(col) || is.logical(col)) {
        any(!is.finite(col))
      } else {
        anyNA(col)
      }
    }, logical(1))
  
  if (any(bad_x)) {
    if (anyNA(x)) {
      stop("'x' must not contain NA or NaN values.")
    }
    stop("'x' must not contain non-finite numeric values.")
  }
  
  if (!is.null(Phi)) {
    if (!is.matrix(Phi) || !is.numeric(Phi) || is.complex(Phi)) {
      stop("'Phi' must be a numeric matrix.")
    }
    
    if (nrow(Phi) != nrow(x) || ncol(Phi) < 1L) {
      stop("'Phi' must have nrow(x) rows and at least one column.")
    }
    
    if (any(!is.finite(Phi))) {
      stop("'Phi' must not contain missing or infinite values.")
    }
  }
  
  if (is.null(Phi) && !is.null(K)) {
    if (!is.matrix(K) || !is.numeric(K) || is.complex(K)) {
      stop("'K' must be a numeric matrix.")
    }
    
    if (nrow(K) != nrow(x) || ncol(K) != nrow(x)) {
      stop("'K' must be an n-by-n matrix, where n equals nrow(x).")
    }
    
    if (any(!is.finite(K))) {
      stop("'K' must not contain missing or infinite values.")
    }

    if (!isSymmetric(K)) {
      stop("'K' must be symmetric.")
    }
  }
  
  if (is.null(respect.unordered.factors)) {
    respect <- "ignore"
  } else if (is.logical(respect.unordered.factors) &&
             length(respect.unordered.factors) == 1L &&
             !is.na(respect.unordered.factors)) {
    if (respect.unordered.factors) {
      stop("'respect.unordered.factors = TRUE' ('order') is not supported by okranger.")
    }
    
    respect <- "ignore"
  } else if (is.character(respect.unordered.factors) &&
             length(respect.unordered.factors) == 1L &&
             !is.na(respect.unordered.factors)) {
    if (respect.unordered.factors == "order") {
      stop("'respect.unordered.factors = 'order'' is not supported by okranger.")
    }
    
    # TODO: Add support for missing values
    if (respect.unordered.factors == "partition") {
      stop("'respect.unordered.factors = 'partition'' is currently not supported by okranger.")
    }
    
    if (respect.unordered.factors != "ignore") {
      stop("'respect.unordered.factors' must be NULL, FALSE, or 'ignore'.")
    }
    
    respect <- "ignore"
  } else {
    stop("'respect.unordered.factors' must be NULL, FALSE, or 'ignore'.")
  }
  
  
  ## Prepare data and rangerCpp arguments
  
  independent.variable.names <- colnames(x)
  
  if (is.null(independent.variable.names) ||
      length(independent.variable.names) != ncol(x) ||
      anyNA(independent.variable.names) ||
      any(!nzchar(trimws(independent.variable.names))) ||
      anyDuplicated(independent.variable.names) > 0L) {
    independent.variable.names <- paste0("x", seq_len(ncol(x)))
    colnames(x) <- independent.variable.names
  }
  
  # Recode character predictors as factors and retain levels for prediction
  if (any(vapply(x, is.character, logical(1)))) {
    character.idx <- vapply(x, is.character, logical(1))
    x[character.idx] <- lapply(x[character.idx], factor)
  }
  
  covariate.levels <- NULL
  if (any(vapply(x, is.factor, logical(1)))) {
    covariate.levels <- lapply(x, levels)
  }
  
  # Ranger's C++ interface receives numeric predictor and response matrices
  x <- data.matrix(x)
  
  if (!is.null(Phi)) {
    y.mat <- as.matrix(Phi)
    output.representation <- "FEATURES"
  } else {
    y.mat <- as.matrix(K)
    output.representation <- "GRAM"
  }
  
  if (identical(output.representation, "GRAM") && isTRUE(oob.error)) {
    warning(paste0(
      "OOB predictions and OOB error are skipped when 'K' is provided. ",
      "Use getForestWeights() to obtain OOB forest weights."
    ))
  }
  
  compute.oob.error <- isTRUE(oob.error) && 
    identical(output.representation, "FEATURES")
  
  # okranger supports data.frame predictors only, so use the dense matrix
  sparse.x <- Matrix::Matrix(matrix(c(0, 0)))
  use.sparse.data <- FALSE
  
  if (is.null(mtry)) {
    mtry <- 0L
  } else {
    mtry <- as.integer(mtry)
  }
  
  if (is.null(seed)) {
    seed <- as.integer(runif(1, 0, .Machine$integer.max))
  } else {
    seed <- as.integer(seed)
  }
  
  if (is.null(num.threads)) {
    num.threads <- as.integer(Sys.getenv(
      "R_RANGER_NUM_THREADS",
      getOption("ranger.num.threads", getOption("Ncpus", 2L))
    ))
  } else {
    num.threads <- as.integer(num.threads)
  }
  
  if (is.null(min.node.size)) {
    min.node.size <- 0L
  } else {
    min.node.size <- as.integer(min.node.size)
  }
  
  if (is.null(min.bucket)) {
    min.bucket <- 0L
  } else {
    min.bucket <- as.integer(min.bucket)
  }
  
  if (is.null(max.depth)) {
    max.depth <- 0L
  } else {
    max.depth <- as.integer(max.depth)
  }
  
  if (is.null(split.select.weights)) {
    split.select.weights <- list(c(0, 0))
    use.split.select.weights <- FALSE
  } else if (is.numeric(split.select.weights)) {
    if (length(split.select.weights) != ncol(x)) {
      stop("'split.select.weights' must have one value per predictor.")
    }
    
    split.select.weights <- list(as.numeric(split.select.weights))
    use.split.select.weights <- TRUE
  } else if (is.list(split.select.weights)) {
    if (length(split.select.weights) != num.trees) {
      stop("'split.select.weights' must contain one vector per tree.")
    }
    
    split.select.weights <- lapply(split.select.weights, as.numeric)
    use.split.select.weights <- TRUE
  } else {
    stop("'split.select.weights' must be NULL, a numeric vector, or a list.")
  }
  
  if (is.null(always.split.variables)) {
    always.split.variables <- c("0", "0")
    use.always.split.variables <- FALSE
  } else {
    always.split.variables <- as.character(always.split.variables)
    use.always.split.variables <- TRUE
  }
  
  result <- rangerCpp(
    treetype = 10L,
    input_x = x,
    input_y = y.mat,
    variable_names = independent.variable.names,
    mtry = mtry,
    num_trees = as.integer(num.trees),
    verbose = verbose,
    seed = seed,
    num_threads = num.threads,
    write_forest = write.forest,
    importance_mode_r = 0L,
    min_node_size = min.node.size,
    min_bucket = min.bucket,
    split_select_weights = split.select.weights,
    use_split_select_weights = use.split.select.weights,
    always_split_variable_names = always.split.variables,
    use_always_split_variable_names = use.always.split.variables,
    prediction_mode = FALSE,
    loaded_forest = list(),
    snp_data = as.matrix(0),
    sample_with_replacement = replace,
    probability = FALSE,
    unordered_variable_names = c("0", "0"),
    use_unordered_variable_names = FALSE,
    save_memory = FALSE,
    splitrule_r = 9L,
    case_weights = c(0, 0),
    use_case_weights = FALSE,
    class_weights = numeric(0),
    predict_all = FALSE,
    keep_inbag = keep.inbag,
    sample_fraction = as.numeric(sample.fraction),
    alpha = 0.5,
    minprop = 0.1,
    poisson_tau = 1,
    holdout = FALSE,
    prediction_type_r = 1L,
    num_random_splits = 1L,
    sparse_x = sparse.x,
    use_sparse_data = use.sparse.data,
    order_snps = FALSE,
    oob_error = compute.oob.error,
    max_depth = max.depth,
    inbag = list(c(0, 0)),
    use_inbag = FALSE,
    regularization_factor = c(0, 0),
    use_regularization_factor = FALSE,
    regularization_usedepth = FALSE,
    node_stats = node.stats,
    time_interest = c(0, 0),
    use_time_interest = FALSE,
    any_na = FALSE,
    approx_tolerance = approx.tolerance,
    output_representation = output.representation
  )
  
  
  if (length(result) == 0L) {
    stop("User interrupt or internal error.")
  }
  
  # Convert explicit-feature OOB predictions to an n-by-d matrix.
  if (compute.oob.error) {
    oob.predictions <- result$predictions
    
    if (is.list(oob.predictions)) {
      oob.predictions <- do.call(rbind, oob.predictions)
    } else if (is.null(dim(oob.predictions))) {
      oob.predictions <- matrix(
        oob.predictions,
        nrow = nrow(Phi),
        ncol = ncol(Phi),
        byrow = TRUE
      )
    }
    
    result$predictions <- oob.predictions
    
    if (!is.null(rownames(x))) {
      rownames(result$predictions) <- rownames(x)
    }
    
    if (!is.null(colnames(Phi))) {
      colnames(result$predictions) <- colnames(Phi)
    }
    
  } else {
    # K inputs do not have OOB predictions or an OOB error.
    result$predictions <- NULL
    result$prediction.error <- NULL
  }
  
  result$treetype <- "Output kernel regression"
  result$output.representation <- output.representation
  result$num.samples <- nrow(x)
  
  if (isTRUE(write.forest)) {
    result$forest$independent.variable.names <- independent.variable.names
    result$forest$treetype <- result$treetype
    result$forest$output.representation <- output.representation
    result$forest$output.feature.names <-
      if (identical(output.representation, "FEATURES")) colnames(Phi) else NULL
    result$forest$num.output.features <-
      if (identical(output.representation, "FEATURES")) ncol(Phi) else 0L
    result$forest$covariate.levels <- covariate.levels
    class(result$forest) <- "ranger.forest"
  }
  
  result$call <- match.call()
  result$importance.mode <- "none"
  class(result) <- "ranger"
  return(result)
}

