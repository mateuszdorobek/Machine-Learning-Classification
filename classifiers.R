library(ROCR)
library(ipred)
library(randomForest)
library(gbm)
library(extraTrees)
library(xgboost)
library(lift)

# set seed for reproducibility
set.seed(123)
data_file_name = "data.csv"
# load data
data_raw <- read.table(data_file_name,h = T,row.names = 1,sep = ",")
data <- data_raw
head(data)

# split data
data_train <- data[data$train == 1, ]
data_train$class <- as.factor(data_train$class)
test <- data[data$train == 0, ]
test$class <- NULL
smp_size <- floor(0.8 * nrow(data_train))
train_ind <- sample(seq_len(nrow(data_train)), size = smp_size)
train <- data_train[train_ind,]
valid <- data_train[-train_ind,]

#---------------RandomForest------------------

m1 <- randomForest(class ~ ., data = train)
pred1 <- predict(m1, valid, type = 'prob')[, 2]
pred_all1 <- prediction(pred1, valid$class)
perf1 <- performance(pred_all1, measure = "lift",x.measure="rpp")
plot(perf1, main = "Random Forest Lift")
unlist(perf1@y.values)[44]
perf2 <- performance(pred_all2, measure = "auc")
tail(unlist(perf2@y.values),n=1)

roc.ROCR1 <- performance(pred_all1, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR1, main = "Random Forest - ROC Curve", col = "red")
abline(0, 1)

#------------------GBM-----------------------------

m2 <- gbm(class ~ ., data = train, distribution = "multinomial")
pred2 = matrix(predict(m2, valid, n.trees = 100, type = 'response'),ncol=2)[,2]
pred_all2 <- prediction(pred2, valid$class)
perf2 <- performance(pred_all2, measure = "lift",x.measure="rpp")
plot(perf2, main = "Generalized Boosted Regression Model Lift")
unlist(perf2@x.values)[81]
unlist(perf2@y.values)[81]
perf2 <- performance(pred_all2, measure = "auc")
tail(unlist(perf2@y.values),n=1)

roc.ROCR2 <- performance(pred_all2, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR2, main = "Generalized Boosted Regression Model - ROC Curve", col = "green")
abline(0, 1)
# ----------------XGBoost------------------------

train_no_class <- train[,!(names(train) %in% "class")]
valid_no_class <- valid[,!(names(valid) %in% "class")]

m3 <- xgboost(
  data = data.matrix(train_no_class),
  label = as.numeric(as.vector(train$class)),
  nrounds = 2000,
  max.depth = 4,
  eta = 0.07,
  nthreads = 8,
  objective = "binary:logistic"
)
pred3 <- predict(m3, data.matrix(valid_no_class))
pred_all3 <- prediction(pred3, valid$class)
perf3 <- performance(pred_all3, measure = "lift",x.measure="rpp")
plot(perf3, main = "XGBoost Lift")
unlist(perf3@y.values)[44]
perf3 <- performance(pred_all3, measure = "auc")
tail(unlist(perf3@y.values),n=1)

roc.ROCR3 <- performance(pred_all3, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR3, main = "XGBoost - ROC Curve", col = "blue")
abline(0, 1)

# ----------------ExtraTrees---------------------

m4 <- extraTrees(x = train_no_class, y = train$class,  ntree=500, numThreads = 8)
pred4 <- predict(m4, as.matrix(valid_no_class), probability = TRUE)[, 2]
pred_all4 <- prediction(pred4, valid$class)
perf4 <- performance(pred_all4, measure = "lift",x.measure="rpp")
plot(perf4, main = "Extra Trees Lift")
unlist(perf4@y.values)[44]
perf4 <- performance(pred_all4, measure = "auc")
tail(unlist(perf4@y.values),n=1)

roc.ROCR4 <-performance(pred_all4, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR4, main = "Extra Trees - ROC Curve", col = "orange")
abline(0, 1)

# -----------------Plotting---------------------------

plot(roc.ROCR1, main = "ROC Curves", col = "red")
plot(roc.ROCR2, add = TRUE, col = "green")
plot(roc.ROCR3, add = TRUE, col = "navyblue")
plot(roc.ROCR4, add = TRUE, col = "orange")
abline(0, 1)
legend(0,1, legend=c("XGBoost", "ExtraTrees", "GBRM", "RandomFores"),
       col=c("red", "green", "blue", "orange"), lty=1, cex=0.8)

plot(perf1, col = "red", main = "Lift Curves")
plot(perf2, col = "green", add=TRUE)
plot(perf3, col = "navyblue", add=TRUE)
plot(perf4, col = "orange", add=TRUE)
legend(0.85,14, legend=c("XGBoost", "ExtraTrees", "GBRM", "RandomFores"),
       col=c("red", "green", "blue", "orange"), lty=1, cex=0.8)

# -----------------SavingResults-----------------------
m <- gbm(class ~ ., data = train, distribution = "multinomial")
pred = matrix(predict(m, test, n.trees = 100, type = 'response'),ncol=2)[,1]
write.table(pred, "MATDOR.txt", append = FALSE, sep = "\n", dec = ".",
            row.names = FALSE, col.names = FALSE)
