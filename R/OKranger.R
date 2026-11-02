
##' @export
OKranger <- function(Phi = NULL, K = NULL, x, approx.tolerance, num.trees = 500, 
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
  
  if (anyNA(x)) {
    stop("'x' must not contain NA or NaN values.")
  }
  
  if (any(vapply(
    x,
    function(col) {(is.numeric(col) || is.logical(col)) && any(!is.finite(col))},
    logical(1)
  ))) {
    stop("'x' must not contain non-finite numeric values.")
  }
  
  if (!is.null(Phi)) {
    if (!is.matrix(Phi) || !is.numeric(Phi) || is.complex(Phi)) {
      stop("'Phi' must be a numeric matrix.")
    }
    
    if (nrow(Phi) != nrow(x) || ncol(Phi) < 1L) {
      stop("'Phi' must have nrow(x) rows and at least one column.")
    }
    
    if (anyNA(Phi)) {
      stop("'Phi' must not contain NA or NaN values.")
    }
    
    if (any(!is.finite(Phi))) {
      stop("'Phi' must contain only finite values.")
    }
  }
  
  if (!is.null(K)) {
    if (!is.matrix(K) || !is.numeric(K) || is.complex(K)) {
      stop("'K' must be a numeric matrix.")
    }
    
    if (nrow(K) != nrow(x) || ncol(K) != nrow(x)) {
      stop("'K' must be an n-by-n matrix, where n equals nrow(x).")
    }
    
    if (anyNA(K)) {
      stop("'K' must not contain NA or NaN values.")
    }
    
    if (any(!is.finite(K))) {
      stop("'K' must contain only finite values.")
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
      stop("'respect.unordered.factors = TRUE' ('order') is not supported by OKranger.")
    }
    
    respect <- "ignore"
  } else if (is.character(respect.unordered.factors) &&
             length(respect.unordered.factors) == 1L &&
             !is.na(respect.unordered.factors)) {
    if (respect.unordered.factors == "order") {
      stop("'respect.unordered.factors = 'order'' is not supported by OKranger.")
    }
    
    # TODO: Add support for missing values
    if (respect.unordered.factors == "partition") {
      stop("'respect.unordered.factors = 'partition'' is currently not supported by OKranger.")
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
  
  # OKranger supports data.frame predictors only, so use the dense matrix
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

