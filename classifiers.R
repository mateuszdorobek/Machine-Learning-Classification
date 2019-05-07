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
perf1 <- performance(pred_all1, measure = "auc")
auc1 <- tail(unlist(perf1@y.values),n=1)
perf1 <- performance(pred_all1, measure = "lift",x.measure="rpp")


png(filename="images/RF_Lift.png")
    unlist(perf1@x.values)[194]
    lift10_1 <- unlist(perf1@y.values)[194]
    plot(perf1, main = paste("Random Forest Lift10 = ",toString(round(lift10_1,2))))
    abline(v=0.1, lty = 2)
    abline(h=lift10_1, lty = 2)
dev.off()
png(filename="images/RF_ROC.png")
    roc.ROCR1 <- performance(pred_all1, measure = "tpr", x.measure = "fpr")
    plot(roc.ROCR1, main = paste("Random Forest - ROC Curve - AUC = ",toString(round(lift10_1,2))))
    abline(0, 1)
dev.off()

#------------------GBM-----------------------------

m2 <- gbm(class ~ ., data = train, distribution = "multinomial")
pred2 = matrix(predict(m2, valid, n.trees = 100, type = 'response'),ncol=2)[,2]
pred_all2 <- prediction(pred2, valid$class)
perf2 <- performance(pred_all2, measure = "auc")
auc2 <- tail(unlist(perf2@y.values),n=1)
perf2 <- performance(pred_all2, measure = "lift",x.measure="rpp")

png(filename="images/GBM_Lift.png")
    unlist(perf2@x.values)[791]
    lift10_2 <- unlist(perf2@y.values)[791]
    plot(perf2, main = paste("GBM Lift10 = ",toString(round(lift10_2,2))))
    abline(v=0.1, lty = 2)
    abline(h=lift10_2, lty = 2)
dev.off()
png(filename="images/GBM_ROC.png")
    roc.ROCR2 <- performance(pred_all2, measure = "tpr", x.measure = "fpr")
    plot(roc.ROCR2, main = paste("GBM - ROC Curve - AUC = ",toString(round(lift10_2,2))))
    abline(0, 1)
dev.off()
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
perf3 <- performance(pred_all3, measure = "auc")
auc3 <- tail(unlist(perf3@y.values),n=1)
perf3 <- performance(pred_all3, measure = "lift",x.measure="rpp")

png(filename="images/XGB_Lift.png")
    unlist(perf3@x.values)[800]
    lift10_3 <- unlist(perf3@y.values)[800]
    plot(perf3, main = paste("XGBoost Lift10 = ",toString(round(lift10_3,2))))
    abline(v=0.1, lty = 2)
    abline(h=lift10_3, lty = 2)
dev.off()
png(filename="images/XGB_ROC.png")
    roc.ROCR3 <- performance(pred_all3, measure = "tpr", x.measure = "fpr")
    plot(roc.ROCR3, main = paste("XGBoost - ROC Curve - AUC = ",toString(round(lift10_3,2))))
    abline(0, 1)
dev.off()


# ----------------ExtraTrees---------------------

m4 <- extraTrees(x = train_no_class, y = train$class,  ntree=500, numThreads = 8)
pred4 <- predict(m4, as.matrix(valid_no_class), probability = TRUE)[, 2]
pred_all4 <- prediction(pred4, valid$class)
perf4 <- performance(pred_all4, measure = "auc")
auc4 <- tail(unlist(perf4@y.values),n=1)
perf4 <- performance(pred_all4, measure = "lift",x.measure="rpp")

png(filename="images/ET_Lift.png")
    unlist(perf4@x.values)[100]
    lift10_4 <- unlist(perf4@y.values)[100]
    plot(perf4, main = paste("Extra Trees Lift10 = ",toString(round(lift10_4,2))))
    abline(v=0.1, lty = 2)
    abline(h=lift10_4, lty = 2)
dev.off()
png(filename="images/ET_ROC.png")
    roc.ROCR4 <-performance(pred_all4, measure = "tpr", x.measure = "fpr")
    plot(roc.ROCR4, main = paste("Extra Trees - ROC Curve - AUC = ",toString(round(auc4,2))))
    abline(0, 1)
dev.off()


# -----------------Plotting---------------------------


png(filename="images/ALL_ROC.png")
    plot(roc.ROCR1, main = "ROC Curves", col = "red")
    plot(roc.ROCR2, add = TRUE, col = "green")
    plot(roc.ROCR3, add = TRUE, col = "navyblue")
    plot(roc.ROCR4, add = TRUE, col = "purple")
    abline(0, 1, lty = 2)
    legend(
      0.6,
      0.3,
      legend = c(
        paste("RandomForest - ", toString(round(auc1, 2))),
        paste("GBM - ", toString(round(auc2, 2))),
        paste("XGBoost - ", toString(round(auc3, 2))),
        paste("ExtraTrees - ", toString(round(auc4, 2)))
      ),
      col = c("red", "green", "blue", "purple"),
      lty = 1,
      cex = 0.8
    )
dev.off()
png(filename="images/ALL_Lift.png")
    plot(perf1, col = "red", main = "Lift Curves")
    plot(perf2, col = "green", add=TRUE)
    plot(perf3, col = "navyblue", add=TRUE)
    plot(perf4, col = "purple", add=TRUE)
    legend(
      0.65,
      14,
      legend = c(
        paste("RandomForest - ", toString(round(lift10_1, 2))),
        paste("GBM - ", toString(round(lift10_2, 2))),
        paste("XGBoost - ", toString(round(lift10_3, 2))),
        paste("ExtraTrees - ",toString(round(lift10_4, 2)))
      ),
      col = c("red", "green", "blue", "purple"),
      lty = 1,
      cex = 0.8
    )
    abline(v=0.1, lty = 2)
    abline(h=lift10_1, lty = 2, col="red")
    abline(h=lift10_2, lty = 2, col="green")
    abline(h=lift10_3, lty = 2, col="blue")
    abline(h=lift10_4, lty = 2, col="purple")
dev.off()

# -----------------SavingResults-----------------------
m <- gbm(class ~ ., data = train, distribution = "multinomial")
pred = matrix(predict(m, test, n.trees = 100, type = 'response'),ncol=2)[,2]
write.table(pred, "MATDOR.txt", append = FALSE, sep = "\n", dec = ".",
            row.names = FALSE, col.names = FALSE)
