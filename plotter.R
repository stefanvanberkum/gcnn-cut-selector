# This module plots the node counts or IGC for each problem type.

library("ggplot2")
library("robustbase")
library("MASS")
library("Rmisc")

plot_nodes <- TRUE
plot_igc <- TRUE

# Path to working directory.
path <- "~/gcnn-cut-selector"
dir.create(paste(path, "/plots", sep=""), showWarnings = FALSE)

if (plot_nodes) {
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
}

if (plot_igc) {
  ### TODO ###
  # IGC results.
  data <- data.frame(read.csv(paste(path, "/results/eval_igc.csv", sep="")))
  data['selector'][data$selector == 'hybrid', ] <- 'Hybrid'
  data['selector'][data$selector == 'gcnn', ] <- 'GCNN'
  
  # Summarize the data to get means and standard errors.
  summarized_data <- summarySE(data, measurevar='igc', groupvars=c('problem', 'selector', 'cut'))
  
  # Add origin.
  problems <- c("setcov", "combauc", "capfac", "indset")
  selectors <- c("Hybrid", "GCNN")
  for (problem in problems) {
    for (selector in selectors) {
      point <- data.frame(problem=problem, selector=selector, cut=0, N=1, igc=0, sd=0, se=0, ci=0)
      summarized_data <- rbind(summarized_data, point)
    }
  }
  
  # Plot results for each problem.
  for (problem in problems) {
    problem_data <- summarized_data[summarized_data$problem == problem, ]
    plot <- ggplot(data=problem_data, mapping=aes(x=cut, y=igc, color=selector)) +
      geom_point() +
      geom_line() +
      geom_errorbar(aes(ymin=igc - ci, ymax=igc + ci)) +
      labs(x="Cut", y="IGC") +
      guides(fill=guide_legend(title="Selector")) +
      theme(text = element_text(size = 20))
    plot(plot)
    ggsave(paste(path, "/plots/", problem, "_eval_igc.png", sep=""), width=10, height=6, dpi=300)
  }
}
