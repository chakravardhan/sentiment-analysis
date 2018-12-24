library(text2vec)
library(data.table)
data("movie_review")
setDT(movie_review)
setkey(movie_review, id)
set.seed(2017L)
all_ids = movie_review$id
train_ids = sample(all_ids, 4000)
test_ids = setdiff(all_ids, train_ids)
train = movie_review[J(train_ids)]
test = movie_review[J(test_ids)]
prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  progressbar = FALSE)
it_test = test$review %>% 
  prep_fun %>% 
  tok_fun %>% 
  itoken( 
         # turn off progressbar because it won't look nice in rmd
         progressbar = FALSE)

vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['sentiment']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 4,
                              thresh = 1e-3,
                              maxit = 1e3)

nb_model <- cv.glmnet(dtm_train_tfidf,train[['sentiment']],family = "multinomial",type.multinomial = "grouped", parallel = TRUE)
preds<-predict(nb_model, dtm_test_tfidf, s = "lambda.min", type = "class")

preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')

glmnet:::auc(test$sentiment, preds)
