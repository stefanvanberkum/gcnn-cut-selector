# This module plots the node counts for each problem type.

library("ggplot2")
library("robustbase")
library("MASS")

# Path to working directory.
path <- "~/gcnn-cut-selector"
dir.create(paste(path, "/plots", sep=""))

# Evaluation results.
data <- data.frame(read.csv(paste(path, "/results/eval.csv", sep="")))

# Filter stats.
data <- data[(data$status == "optimal" | data$status == "timelimit"), ]

# Split by selector and match instances.
hybrid_stats <- data[data$selector == "hybrid", ]
gcnn_stats <- data[data$selector == "gcnn", ]
data <- merge(hybrid_stats, gcnn_stats, by=c("problem", "difficulty", "instance", "seed"), all=FALSE)

# Consider only instances that were solved by both selectors.
data <- data[(data$status.x == "optimal" & data$status.y == "optimal"), ]

# Plot results for each problem.
problems <- c("setcov", "combauc", "capfac", "indset")
for (problem in problems) {
  problem_data <- data[data$problem == problem, ]
  plot <- ggplot(data=problem_data, mapping=aes(x=n_nodes.x, y=n_nodes.y)) +
    geom_point(color = 'slategrey') +
    # Linear.
    # geom_smooth(method="lm", formula=y ~ x) +
    # robustbase MM estimator.
    # geom_smooth(method=function(formula, data, weights=weight) lmrob(formula, data, setting="KS2014"), formula=y ~ x) +
    # MASS MM estimator.
    geom_smooth(method=function(formula, data, weights=weight) rlm(formula, data, weights=weight, method="MM", maxit=100), formula=y ~ x, color="black") +
    geom_abline(aes(intercept=0, slope=1), linetype = 'dashed') +
    labs(x="Hybrid", y="GCNN") +
    theme(text = element_text(size = 20))
  plot(plot)
  ggsave(paste(path, "/plots/", problem, "_eval.png", sep=""), width=10, height=6, dpi=300)
}

# Benchmarking results.
data <- data.frame(read.csv(paste(path, "/results/benchmark.csv", sep="")))

# Filter stats.
data <- data[(data$status == "optimal" | data$status == "timelimit"), ]

# Split by selector and match instances.
hybrid_stats <- data[data$selector == "hybrid", ]
gcnn_stats <- data[data$selector == "gcnn", ]
data <- merge(hybrid_stats, gcnn_stats, by=c("problem", "instance"), all=FALSE)

# Consider only instances that were solved by both selectors.
data <- data[(data$status.x == "optimal" & data$status.y == "optimal"), ]

# Plot results for each problem.
problems <- c("setcov", "combauc", "capfac", "indset")
for (problem in problems) {
  problem_data <- data[data$problem == problem, ]
  plot <- ggplot(data=problem_data, mapping=aes(x=n_nodes.x, y=n_nodes.y)) +
    geom_point(color = 'slategrey') +
    # Linear.
    # geom_smooth(method="lm", formula=y ~ x) +
    # robustbase MM estimator.
    # geom_smooth(method=function(formula, data, weights=weight) lmrob(formula, data, setting="KS2014"), formula=y ~ x) +
    # MASS MM estimator.
    geom_smooth(method=function(formula, data, weights=weight) rlm(formula, data, weights=weight, method="MM", maxit=100), formula=y ~ x, color="black") +
    geom_abline(aes(intercept=0, slope=1), linetype = 'dashed') +
    labs(x="Hybrid", y="GCNN") +
    theme(text = element_text(size = 20))
  plot(plot)
  ggsave(paste(path, "/plots/", problem, "_benchmark.png", sep=""), width=10, height=6, dpi=300)
}
