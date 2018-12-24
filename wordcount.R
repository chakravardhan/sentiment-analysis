library(ngram)
library(tm)
library(qdap)

dat <- data
dat <- dat[,names(data) %in% c("Score","Text")]
dat <-dat[which(dat$Score!=3),]
dat$Text <- tolower(dat$Text)
dat$Text <- removePunctuation(dat$Text)
dat$Text <- removeNumbers(dat$Text)

docs <- Corpus(VectorSource(dat$Text))
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
