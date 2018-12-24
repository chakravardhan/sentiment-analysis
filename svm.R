library(caret)
library(RTextTools)

df <- mydata[which(mydata$HelpfulnessDenominator>10),]
df$sentiment <- ifelse(df$sentiment=="positive","1","0")
df$sentiment <- as.numeric(df$sentiment)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)
df_train <- df[1:11000,]
df_test <- df[11001:19845,]
dtmatrix <- create_matrix(df_train$Summary)
container <- create_container(dtmatrix, df$sentiment, trainSize=1:11000, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)

predmatrix <- create_matrix(df_test$Summary,originalMatrix = dtmatrix)
predsize<-nrow(predmatrix)
predictionContainer <- create_container(predmatrix, labels=rep(0,predsize), testSize=1:predsize, virgin=FALSE)
results <- classify_model(predictionContainer, model)
results <- results$SVM_LABEL
results <- as.numeric(results)
glmnet::auc(df_test$sentiment,results)
cm2<-confusionMatrix(df_test$sentiment,results)
