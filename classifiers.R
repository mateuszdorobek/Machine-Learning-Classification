library(ROCR)
library(ipred)
library(randomForest)
library(gbm)
library(extraTrees)
library(xgboost)

## set the seed to make your partition reproducible
set.seed(123)
data_file_name = "data.csv"

data_raw <- read.table(data_file_name,
                       h = T,
                       row.names = 1,
                       sep = ",")
data <- data_raw
head(data)

data_train <- data[data$train == 1, ]
data_train$class <- as.factor(data_train$class)

#-----smaller case -----
# smp_size <- floor(0.05 * nrow(data_train))
# train_ind <- sample(seq_len(nrow(data_train)), size = smp_size)
# data_train <- data_train[train_ind, ]
#-----------------------

test <- data[data$train == 0, ]
test$class <- NULL
## 75% of the sample size
smp_size <- floor(0.8 * nrow(data_train))


train_ind <- sample(seq_len(nrow(data_train)), size = smp_size)
train <- data_train[train_ind,]
valid <- data_train[-train_ind,]

#--------------------------------------------------

m1 <- randomForest(class ~ ., data = train)
pred1 <- predict(m1, valid, type = 'prob')[, 2]
pred_all1 <- prediction(pred1, valid$class)
perf1 <- performance(pred_all1, measure = "auc")
unlist(perf1@y.values)

roc.ROCR1 <- performance(pred_all1, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR1, main = "Random Forest - ROC Curve", col = "red")
abline(0, 1)

#---------------------------------------------------

m2 <- gbm(class ~ ., data = train, distribution = "gaussian")
pred2 <- predict(m2, valid, n.trees = 100, type = 'response')
pred_all2 <- prediction(pred2, valid$class)
perf2 <- performance(pred_all2, measure = "auc")
unlist(perf2@y.values)

roc.ROCR2 <- performance(pred_all2, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR2, main = "Generalized Boosted Regression Model - ROC Curve", col = "green")

# -----------------------------------------------

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
perf3 <- performance(pred_all3, measure = "auc")
unlist(perf3@y.values)

roc.ROCR3 <- performance(pred_all3, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR3, main = "XGBoost - ROC Curve", col = "blue")

# -------------------------------------------------

m4 <- extraTrees(x = train_no_class, y = train$class,  ntree=500, numThreads = 8)
pred4 <- predict(m4, as.matrix(valid_no_class), probability = TRUE)[, 2]
pred_all4 <- prediction(pred4, valid$class)
perf4 <- performance(pred_all4, measure = "auc")
unlist(perf4@y.values)
 
roc.ROCR4 <-performance(pred_all4, measure = "tpr", x.measure = "fpr")
plot(roc.ROCR4, add=TRUE, main = "Extra Trees - ROC Curve", col = "orange")

# -------------------------------------------------

plot(roc.ROCR1, main = "ROC Curves", col = "red")
plot(roc.ROCR2, add = TRUE, col = "green")
plot(roc.ROCR3, add = TRUE, col = "navyblue")
plot(roc.ROCR4, add = TRUE, col = "orange")
abline(0, 1)
legend(0,1, legend=c("XGBoost", "ExtraTrees", "GBRM", "RandomFores"),
       col=c("red", "green", "blue", "orange"), lty=1, cex=0.8)
