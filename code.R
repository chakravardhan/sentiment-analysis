library(readr)
library(text2vec)
library(glmnet)
library(quanteda)
library(caret)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(SnowballC) 
library(RTextTools)
library(ROCR)
library(wordcloud)
data <- read_csv("Reviews.csv")
mydata <- data[,names(data) %in% c("Score","Summary","HelpfulnessNumerator","HelpfulnessDenominator")]
mydata <- mydata[which(mydata$Score!=3),]
mydata["sentiment"]<- ifelse(mydata$Score>3,"positive","negative")
mydata["usefulness"]<-ifelse((mydata$HelpfulnessNumerator/mydata$HelpfulnessDenominator)>0.8 & mydata$HelpfulnessDenominator!=0,"useful","useless")
mydata$Summary <- tolower(mydata$Summary)
pattern <- "[^a-z]+"
mydata$Summary <- gsub(pattern,"",mydata$Summary)
set.seed(123)
temp <- sample(2, nrow(mydata), replace = T, prob = c(0.7, 0.3))
train <- mydata[temp == 1,]
test <- mydata[temp == 2,]

#tfidf
it_train <- itoken(train$Summary,preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train,vectorizer)
tfidf <- TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
it_test <- itoken(test$Summary,preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)


#y_train <- train["sentiment"]
y_train <- ifelse(train["sentiment"]=="positive","1","0")
y_train <- as.numeric(y_train)
#y_test <- test["sentiment"]
y_test <- ifelse(test["sentiment"]=="positive","1","0")
y_test <- as.numeric(y_test)


# multinomial naive bayes 
nb_model <- cv.glmnet(dtm_train_tfidf,y_train,family = "multinomial",type.multinomial = "grouped", parallel = TRUE)
preds<-predict(nb_model, dtm_test_tfidf, s = "lambda.min", type = "class")
glmnet::auc(y_test,preds)

cm <- confusionMatrix(preds,y_test)


# logistic classification
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = y_train, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 4,
                              thresh = 1e-3,
                              maxit = 1e3)
preds1 <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
pred <- predict(glmnet_classifier, newx = dtm_test_tfidf, 
               s = "lambda.min", type = "class")
glmnet::auc(y_test,preds1)



#evaluation metrics
p<-prediction(preds1,y_test)
perf <- performance(p,"tpr","fpr")
perf1 <- performance(p,"prec","rec")
precision<-sum(pred & y_test)/sum(pred)
recall<-sum(pred & y_test)/sum(y_test)
Fmeasure <- 2 * precision * recall / (precision + recall)
cm1 <- confusionMatrix(pred,y_test)



draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'positive', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'negative', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'positive', cex=1.2, srt=90)
  text(140, 335, 'negative', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 





