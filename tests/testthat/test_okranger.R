library(ranger)
context("okranger")

test_that("Phi OOB predictions are a matrix and match the reported error", {
  
  n <- 60
  x <- data.frame(x1 = runif(n), x2 = runif(n))
  Phi <- cbind(
    feature1 = x$x1 + x$x2,
    feature2 = x$x1^2,
    feature3 = sin(x$x2)
  )
  
  fit <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-8,
                           oob.error = TRUE)
  
  expect_true(is.matrix(fit$predictions))
  expect_equal(dim(fit$predictions), c(n, ncol(Phi)))
  expect_equal(colnames(fit$predictions), colnames(Phi))
  
  valid <- apply(is.finite(fit$predictions), 1, all)
  expect_true(any(valid))
  
  expected_error <- mean(rowSums(
    (fit$predictions[valid, , drop = FALSE] -Phi[valid, , drop = FALSE])^2
  ))
  
  expect_equal(fit$prediction.error, expected_error, tolerance = 1e-10)
})

test_that("K output rejects response prediction but allows terminal-node prediction", {
  
  n <- 30
  x <- data.frame(x1 = runif(n), x2 = runif(n))
  
  Phi <- cbind(feature1 = x$x1^2, feature2 = x$x2^3)
  K <- tcrossprod(Phi)
  
  fit_K <- expect_warning(
    OKranger(K = K, x = x, approx.tolerance = 1e-16, 
             num.trees = 5, seed = 123, verbose = FALSE, 
             oob.error = TRUE, write.forest = TRUE),
    "OOB predictions and OOB error are skipped"
  )
  
  expect_null(fit_K$predictions)
  expect_null(fit_K$prediction.error)
  
  expect_error(
    predict(fit_K, data = x, type = "response"),
    "Response prediction with 'K' output representation is not supported"
  )
  
  terminal_nodes <- predict(fit_K, data = x, type = "terminalNodes")
  
  expect_equal(
    dim(terminal_nodes$predictions),
    c(nrow(x), fit_K$num.trees)
  )
  expect_true(all(terminal_nodes$predictions >= 0))
})

test_that("oob.error = FALSE skips OOB outputs", {

  n <- 30
  x <- data.frame(x1 = runif(n), x2 = runif(n))
  Phi <- cbind(feature1 = x$x1, feature2 = x$x2)
  
  fit <- OKranger(
    Phi = Phi,
    x = x,
    approx.tolerance = 1e-8,
    num.trees = 5,
    seed = 31,
    verbose = FALSE,
    write.forest = FALSE,
    oob.error = FALSE
  )
  
  expect_null(fit$predictions)
  expect_null(fit$prediction.error)
})

test_that("OKranger produces the same splits as ranger", {
  
  n <- 30
  x <- data.frame(x1 = runif(n), x2 = runif(n))
  y <- 3*x$x1 + sin(x$x2) + rnorm(n)
  Phi <- matrix(y, ncol = 1)
  
  fit_Phi <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-8,
                      node.stats = TRUE, keep.inbag = TRUE, seed = 123,
                      num.trees = 30)
  
  fit_y <- ranger(y = y, x = x, 
                  node.stats = TRUE, keep.inbag = TRUE, seed = 123,
                  num.trees = 30)
  
  # TreeKernel stores 0 in split.values for terminal nodes; ranger stores
  # leaf predictions there. OKranger stores leaf means separately, so compare
  # split.values only for non-terminal nodes.
  for (tree in seq_len(fit_Phi$forest$num.trees)) {
    internal_Phi <- fit_Phi$forest$child.nodeIDs[[tree]][[1]] != 0
    internal_y   <- fit_y$forest$child.nodeIDs[[tree]][[1]] != 0
    
    expect_identical(internal_Phi, internal_y)
    
    expect_equal(
      fit_Phi$forest$split.varIDs[[tree]][internal_Phi],
      fit_y$forest$split.varIDs[[tree]][internal_y]
    )
    
    expect_equal(
      fit_Phi$forest$split.values[[tree]][internal_Phi],
      fit_y$forest$split.values[[tree]][internal_y],
      tolerance = 1e-6
    )
  }
})

test_that("Gram input skips OOB outputs and predict refers to getForestWeights()", {
  n <- 20
  x <- data.frame(x1 = runif(n), x2 = runif(n))
  Phi <- cbind(x$x1^2, x$x2^3)
  K <- tcrossprod(Phi)
  
  fit <- OKranger(
    K = K, x = x, approx.tolerance = 1e-8,
    num.trees = 3, verbose = FALSE, oob.error = TRUE
  )
  
  expect_null(fit$predictions)
  expect_null(fit$prediction.error)
  expect_error(predict(fit, x), "getForestWeights()")
})

test_that("Phi predictions equal regression forest predictions", {
  
  n <- 100
  x <- matrix(rnorm(n*7,0,1),ncol = 7)
  y <- 0.6*x[,1] - 0.2*x[,2] + 0.3*x[,4] + 0.9*x[,5] - 0.5*x[,6] + rnorm(n)
  x <- as.data.frame(x)
  Phi <- matrix(y, ncol = 1)
  
  fit_y <- ranger(y = y, x = x, 
                  write.forest = TRUE, keep.inbag = TRUE,
                  oob.error = TRUE, seed = 123)
  
  fit_Phi <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-16,
                      write.forest = TRUE, keep.inbag = TRUE,
                      oob.error = TRUE, seed = 123)
  
  # OOB predictions
  expect_equal(fit_y$predictions, as.vector(fit_Phi$predictions),
               tolerance = 1e-10)
  
  # predictions on training data
  pred_y <- predict(fit_y, x)
  pred_Phi <- predict(fit_Phi, x)
  
  expect_equal(pred_y$predictions, as.vector(pred_Phi$predictions),
               tolerance = 1e-10)
  
  # predictions on new data
  n_new <- 350
  x_new <- matrix(rnorm(n_new*7,0,1),ncol = 7)
  x_new <- as.data.frame(x_new)
  
  pred_y_new <- predict(fit_y, x_new)
  pred_Phi_new <- predict(fit_Phi, x_new)
  
  expect_equal(pred_y_new$predictions, as.vector(pred_Phi_new$predictions),
               tolerance = 1e-10)
})

test_that("Phi predictions equal probability forest predictions", {
  
  n <- 100
  x <- matrix(rnorm(n * 7, 0, 1), ncol = 7)
  eta <- 0.8 * x[, 1] - 0.5 * x[, 3] + 0.6 * x[, 5] - 0.3 * x[, 6]
  cuts <- quantile(eta + rnorm(n), probs = c(0, 1/3, 2/3, 1), names = FALSE)
  labels <- c("low", "middle", "high")
  y <- cut(eta, breaks = cuts, include.lowest = TRUE, labels = labels)
  x <- as.data.frame(x)
  class_order <- unique(as.character(y))
  # One-got encode y
  Phi <- vapply(class_order, function(cl) as.numeric(y == cl), 
                numeric(length(y)))
  colnames(Phi) <- class_order
  
  fit_prob <- ranger(y = y, x = x, probability = TRUE,
                     write.forest = TRUE, keep.inbag = TRUE,
                     oob.error = TRUE, seed = 123,
                     mtry = 7, min.node.size = 3, min.bucket = 1)  

  fit_Phi <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-16, 
                      write.forest = TRUE, keep.inbag = TRUE,
                      oob.error = TRUE, seed = 123, 
                      mtry = 7, min.node.size = 3, min.bucket = 1)
  
  # OOB predictions
  expect_equal(fit_prob$predictions[, labels, drop = FALSE],
               fit_Phi$predictions[, labels, drop = FALSE],
               tolerance = 1e-10)
  
  # predictions on training data
  pred_prob <- predict(fit_prob, x)
  pred_Phi <- predict(fit_Phi, x)
  
  expect_equal(pred_prob$predictions[, labels, drop = FALSE], 
               pred_Phi$predictions[, labels, drop = FALSE],
               tolerance = 1e-10)
  
  # predictions on new data
  n_new <- 150
  x_new <- matrix(rnorm(n_new*7,0,1),ncol = 7)
  x_new <- as.data.frame(x_new)
  
  pred_prob_new <- predict(fit_prob, x_new)
  pred_Phi_new <- predict(fit_Phi, x_new)
  
  expect_equal(pred_prob_new$predictions[, labels, drop = FALSE], 
               pred_Phi_new$predictions[, labels, drop = FALSE],
               tolerance = 1e-10)
  
})

test_that("Phi terminal nodes equal regression forest terminal nodes", {
  
  n <- 100
  x <- matrix(rnorm(n*7,0,1),ncol = 7)
  y <- 0.6*x[,1] - 0.2*x[,2] + 0.3*x[,4] + 0.9*x[,5] - 0.5*x[,6] + rnorm(n)
  x <- as.data.frame(x)
  Phi <- matrix(y, ncol = 1)
  K <- tcrossprod(Phi)
  
  fit_y <- ranger(y = y, x = x, 
                  write.forest = TRUE, keep.inbag = TRUE,
                  seed = 123)
  
  fit_Phi <- OKranger(Phi = Phi, x = x, 
                      approx.tolerance = 1e-10,
                      seed = 123)
  
  # Prediction on training data
  pred_y <- predict(fit_y, x, type = "terminalNodes")
  pred_Phi <- predict(fit_Phi, x, type = "terminalNodes")

  expect_equal(pred_y$predictions, pred_Phi$predictions,
               tolerance = 1e-10)
  
  # Prediction on new data
  n_new <- 150
  x_new <- matrix(rnorm(n_new*7,0,1),ncol = 7)
  x_new <- as.data.frame(x_new)
  
  pred_y_new <- predict(fit_y, x_new, type = "terminalNodes")
  pred_Phi_new <- predict(fit_Phi, x_new, type = "terminalNodes")

  expect_equal(pred_y_new$predictions, pred_Phi_new$predictions,
               tolerance = 1e-10)
})

test_that("Phi terminal nodes equal probability forest terminal nodes", {
  
  n <- 100
  x <- matrix(rnorm(n * 7, 0, 1), ncol = 7)
  eta <- 0.8 * x[, 1] - 0.5 * x[, 3] + 0.6 * x[, 5] - 0.3 * x[, 6]
  cuts <- quantile(eta + rnorm(n), probs = c(0, 1/3, 2/3, 1), names = FALSE)
  labels <- c("low", "middle", "high")
  y <- cut(eta, breaks = cuts, include.lowest = TRUE, labels = labels)
  x <- as.data.frame(x)
  class_order <- unique(as.character(y))
  # One-got encode y
  Phi <- vapply(class_order, function(cl) as.numeric(y == cl), 
                numeric(length(y)))
  colnames(Phi) <- class_order
  
  fit_prob <- ranger(y = y, x = x, probability = TRUE,
                     mtry = 7, min.node.size = 3, min.bucket = 1, 
                     seed = 123)  
  
  fit_Phi <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-16, 
                      mtry = 7, min.node.size = 3, min.bucket = 1, 
                      seed = 123)
  
  # Prediction on training data
  pred_prob <- predict(fit_prob, x)
  pred_Phi <- predict(fit_Phi, x)
  
  n_new <- 350
  x_new <- matrix(rnorm(n_new*7,0,1),ncol = 7)
  x_new <- as.data.frame(x_new)
  
  expect_equal(pred_prob$predictions[, labels, drop = FALSE], 
               pred_Phi$predictions[, labels, drop = FALSE],
               tolerance = 1e-10)
  
  # Prediction on new data
  pred_prob_new <- predict(fit_prob, x_new)
  pred_Phi_new <- predict(fit_Phi, x_new)
  
  expect_equal(pred_prob_new$predictions[, labels, drop = FALSE], 
               pred_Phi_new$predictions[, labels, drop = FALSE],
               tolerance = 1e-10)
  
})

test_that("Phi predictions equal weight predictions", {
  
  n <- 100
  x <- matrix(rnorm(n*7,0,1),ncol = 7)
  y <- 0.6*x[,1] - 0.2*x[,2] + 0.3*x[,4] + 0.9*x[,5] - 0.5*x[,6] + rnorm(n)
  x <- as.data.frame(x)
  Phi <- matrix(y, ncol = 1)
  
  fit <- OKranger(Phi = Phi, x = x, approx.tolerance = 1e-16,
                  write.forest = TRUE, keep.inbag = TRUE,
                  oob.error = TRUE, seed = 123)
  
  # OOB predictions
  W_oob <- getForestWeights(fit, training.data = x, oob = TRUE)
  pred_weight_oob <- unname(as.matrix(W_oob %*% Phi))
  
  expect_equal(fit$predictions, pred_weight_oob,
               tolerance = 1e-10)
  
  # (in-bag) predictions on training data
  W_train <- getForestWeights(fit, training.data = x, oob = FALSE)
  pred_weight_train <- unname(as.matrix(W_train %*% Phi))
  pred_train <- unname(predict(fit, data = x)$predictions)
  expect_equal(pred_train, pred_weight_train,
               tolerance = 1e-10)
  
  # predictions on new data
  n_new <- 150
  x_new <- matrix(rnorm(n_new*7,0,1),ncol = 7)
  x_new <- as.data.frame(x_new)
  
  W_new <- getForestWeights(fit, training.data = x, 
                            new.data = x_new, oob = FALSE)
  pred_weight_new <- unname(as.matrix(W_new %*% Phi))
  pred_new <- unname(predict(fit, data = x_new)$predictions)
  
  expect_equal(pred_weight_new, pred_new,
               tolerance = 1e-10)
})
