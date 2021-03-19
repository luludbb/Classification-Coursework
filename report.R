
#load data
heart_failure_data <- readr::read_csv("C:\\Users\\HaoLu\\Documents\\Classification\\Classification\\heart_failure.csv")
#the summary of data
library("skimr")
skim(heart_failure_data)
#plot the simple visualisations of data
install.packages("DataExplorer")
DataExplorer::plot_bar(heart_failure_data, ncol = 3)
DataExplorer::plot_histogram(heart_failure_data, ncol = 3)
DataExplorer::plot_boxplot(heart_failure_data, by = "fatal_mi", ncol = 3)

install.packages("mlr3verse")
install.packages("xgboost")
library("data.table")
library("mlr3verse")
library("ggplot2")

heart_failure_data$fatal_mi <-as.factor(heart_failure_data$fatal_mi)

#logistic regression
fit.lr <- glm(heart_failure_data$fatal_mi ~ ., binomial, heart_failure_data)
summary(fit.lr)
pred.lr <- predict(fit.lr, heart_failure_data, type = "response")

ggplot(data.frame(x = pred.lr), aes(x = x)) + geom_histogram()
conf.mat <- table(`suffer fatal mi` = heart_failure_data$fatal_mi, `predict suffer` = pred.lr > 0.5)
conf.mat
conf.mat/rowSums(conf.mat)*100

#use MLR3,define a task
set.seed(212) # set seed for reproducibility
heart_failure_task <- TaskClassif$new(id = "suffer_fatal_mi",
                               backend = heart_failure_data, # <- NB: no na.omit() this time
                               target = "fatal_mi",
                               positive = "1")



#go for cross validation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_failure_task)

# baseline classifier and classification trees
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_baseline$param_set
lrn_cart$param_set

res_baseline <- resample(heart_failure_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(heart_failure_task, lrn_cart, cv5, store_models = TRUE)


res <- benchmark(data.table(
  task       = list(heart_failure_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)
res
res$aggregate()
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

#trees
trees <- res$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[2]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[2]]$model, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[3]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[3]]$model, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[4]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[4]]$model, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(heart_failure_task, lrn_cart_cv, cv5, store_models = TRUE)

rpart::plotcp(res_cart_cv$learners[[1]]$model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.021)

res <- benchmark(data.table(
  task       = list(heart_failure_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))



lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

res <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)
res <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


library("data.table")
library("mlr3verse")

set.seed(212) # set seed for reproducibility

# Load data
data("heart_failure_data", package = "modeldata")

# Define task
heart_failure_data$fatal_mi <-factor(heart_failure_data$fatal_mi)
heart_failure_task2 <- TaskClassif$new(id = "suffer_fatal_mi",
                               backend = heart_failure_data,
                               target = "fatal_mi",
                               positive = "1")

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_failure_task2)

# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.020, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()

install.packages("ranger")
res_spr <- resample(credit_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))



