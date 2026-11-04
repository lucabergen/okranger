Package: okranger

Type: Package

Title: Output Kernel Random Forests

Version: 0.1.0

Authors@R: c(

&#x20;   person("Luca", "Bergen", email = "bergen@leibniz-bips.de", role = c("aut", "cre")))

Description: Implements output kernel random forests (OKRFs), which extend

&#x20;   regression forests to RKHS-valued responses. Supports output representations

&#x20;   based on either an explicit feature matrix or a Gram matrix. Uses an approximate

&#x20;   feature representation for split selection and C++ code for computationally

&#x20;   efficient training. Builds on the 'ranger' R package.

License: GPL-3

Imports: Rcpp (>= 0.11.2), Matrix

LinkingTo: Rcpp, RcppEigen

Depends: R (>= 3.1)

Suggests: 

&#x20;   survival,

&#x20;   testthat

Encoding: UTF-8

URL: https://github.com/lucabergen/okranger

BugReports: https://github.com/lucabergen/okranger/issues

Config/roxygen2/version: 8.1.0



