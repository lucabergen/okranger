# -------------------------------------------------------------------------------
#   This file is part of Ranger.
#
# Ranger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ranger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ranger. If not, see <http://www.gnu.org/licenses/>.
#
# Written by:
#
#   Marvin N. Wright
# Institut fuer Medizinische Biometrie und Statistik
# Universitaet zu Luebeck
# Ratzeburger Allee 160
# 23562 Luebeck
# Germany
#
# http://www.imbs-luebeck.de
# -------------------------------------------------------------------------------

.is_bool_scalar <- function(x, name) {
  
  call <- sys.call(sys.parent())
  
  fail <- function(message) {
    stop(simpleError(message, call = call))
  }
  
  if (!is.logical(x)) {
    fail(sprintf("'%s' must be TRUE or FALSE.", name))
  }
  
  if (length(x) != 1L || !is.null(dim(x))) {
    fail(sprintf("'%s' must be a single TRUE or FALSE value.", name))
  }
  
  if (is.na(x)) {
    fail(sprintf("'%s' must not be NA.", name))
  }
  
  x
}

# Check if the input is an integer scalar (may be type double) 
# and convert to integer
.as_int_scalar <- function(x, name, kind = c("positive", "nonnegative")) {
  
  call <- sys.call(sys.parent())
  
  kind <- match.arg(kind)
  
  fail <- function(message) {
    stop(simpleError(message, call = call))
  }
  
  if (length(x) != 1L || !is.null(dim(x))) {
    fail(sprintf("'%s' must be a single numeric scalar.", name))
  }
  
  if (!is.numeric(x) || is.complex(x)) {
    fail(sprintf("'%s' must be a real numeric scalar.", name))
  }
  
  if (is.na(x)) {
    fail(sprintf("'%s' must not be NA or NaN.", name))
  }
  
  if (!is.finite(x)) {
    fail(sprintf("'%s' must be finite.", name))
  }
  
  if (kind == "positive" && x < 1) {
    fail(sprintf("'%s' must be a positive integer.", name))
  }
  
  if (kind == "nonnegative" && x < 0) {
    fail(sprintf("'%s' must be a non-negative integer.", name))
  }
  
  if (x %% 1 != 0) {
    fail(sprintf("'%s' must be integer-valued.", name))
  }
  
  if (x > .Machine$integer.max) {
    fail(sprintf("'%s' must not exceed R's maximum integer.", name))
  }
  
  as.integer(round(x))
}

# Convert integer to factor
integer.to.factor <- function(x, labels) {
  factor(x, levels = seq_along(labels), labels = labels)
}

# Save version of sample() for length(x) == 1
# See help(sample)
save.sample <- function(x, ...) {
  x[sample.int(length(x), ...)]
}

# Order factor levels with PCA approach 
# Reference: Coppersmith, D., Hong, S.J. & Hosking, J.R. (1999) Partitioning Nominal Attributes in Decision Trees. Data Min Knowl Discov 3:197. \doi{10.1023/A:1009869804967}.
pca.order <- function(y, x) {
  x <- droplevels(x)
  if (nlevels(x) < 2) {
    return(as.character(levels(x)))
  }
  
  ## Create contingency table of the nominal outcome with the nominal covariate
  N <- table(x, droplevels(y))
  
  ## PCA of weighted covariance matrix of class probabilites
  P <- N/rowSums(N)
  S <- cov.wt(P, wt = rowSums(N))$cov
  pc1 <- prcomp(S, rank. = 1)$rotation
  score <- P %*% pc1
  
  ## Return ordered factor levels
  as.character(levels(x)[order(score)])
}

# Compute median survival if available or largest quantile available in all strata if median not available.
largest.quantile <- function(formula) {
  ## Fit survival model
  fit <- survival::survfit(formula)
  smry <- summary(fit)
  
  ## Use median survival if available or largest quantile available in all strata if median not available
  max_quant <- max(aggregate(smry$surv ~ smry$strata, FUN = min)[, "smry$surv"])
  quantiles <- quantile(fit, conf.int = FALSE, prob = min(0.5, 1 - max_quant))[, 1]
  names(quantiles) <- gsub(".+=", "", names(quantiles))
  
  ## Return ordered levels
  names(sort(quantiles))
}

# Convert ranger object from version <0.11.5 (without x/y interface)
convert.pre.xy <- function(forest, trees = 1:forest$num.trees) {
  if (is.null(forest$status.varID)) {
    # Not survival
    for (i in 1:forest$num.trees) {
      idx <- forest$split.varIDs[[i]] > forest$dependent.varID
      forest$split.varIDs[[i]][idx] <- forest$split.varIDs[[i]][idx] - 1
    }
  } else {
    # Survival
    for (i in 1:forest$num.trees) {
      idx1 <- forest$split.varIDs[[i]] > forest$dependent.varID
      idx2 <- forest$split.varIDs[[i]] > forest$status.varID
      forest$split.varIDs[[i]][idx1] <- forest$split.varIDs[[i]][idx1] - 1
      forest$split.varIDs[[i]][idx2] <- forest$split.varIDs[[i]][idx2] - 1
    }
  }
  return(forest)
}


