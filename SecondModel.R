library(RTextTools)

sentiment.train <-
  read.delim("~/Course Documents/DA 485/Capstone/sentiment.train.txt",
  header=FALSE, stringsAsFactors=FALSE)
colnames(sentiment.train) <- c("Text", "Sentiment")
n.train <- nrow(sentiment.train)
sentiment.test <-
  read.delim("~/Course Documents/DA 485/Capstone/sentiment.test.txt",
  header=FALSE, stringsAsFactors=FALSE)
colnames(sentiment.test) <- c("Text", "Sentiment")
n.test <- nrow(sentiment.test)

# ---------------------
# recreating SVM model from last week
# Cost = 1, kernel = linear, keep all sparse words
# ---------------------

# create_matrix in RTextTools
unigrams_matrix <- create_matrix(sentiment.train$Text,
  ngramLength = 1,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  removeSparseTerms = 0,
  removeStopwords = TRUE, #lots of possibly useful stopwords in dic.
  stemWords = TRUE,
  stripWhitespace=TRUE)
# create container in RTextTools
container <- create_container(unigrams_matrix,
  sentiment.train$Sentiment, trainSize = 1:n.train, virgin = FALSE)
# training models:
model.SVM <- train_model(container,
  algorithm = "SVM", #see doc for other options
  method = "eps-regression", #for SVM
  cross = 0, # SVM: cross=k for k-fold cross valid. of training data
  cost = 1, # SVM cost parameter default = 100
  kernel = "linear", #SVM kernel type default = radial
  size = 1, #size for NN algorithm
  rang = 0.1, # NN parameter
  decay = 5e-04, # NN parameter
  ntree = 200, # RF no. of trees
  l1_regularizer = 0, # MAXENT parameter
  l2_regularizer = 0, # MAXENT parameter
  verbose = TRUE)
# fixing the "Acronym" error in create_matrix
# see https://github.com/timjurka/RTextTools/issues/4
trace("create_matrix", edit=T)
# need to fix line 42: Acronym -> acronym
# create matrix for test data
unigrams_matrix.test <- create_matrix(sentiment.test$Text,
  originalMatrix = unigrams_matrix,
  minDocFreq = 1,
  minWordLength = 3,
  ngramLength = 1,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  removeSparseTerms = 0,
  removeStopwords = TRUE,
  stemWords = FALSE,
  stripWhitespace=TRUE)
# create prediction container
container.pred <- create_container(unigrams_matrix.test,
  labels = rep(0, n.test), testSize = 1:n.test, virgin=FALSE)
# make predictions
results.SVM <- classify_model(container.pred, model.SVM)

# assess results
correct.SVM <- sentiment.test$Sentiment == results.SVM$SVM_LABEL
accuracy.SVM <- sum(correct.SVM) / n.test
prob.accuracy.SVM <- (sum(results.SVM$SVM_PROB[correct.SVM]) +
  sum(1 - results.SVM$SVM_PROB[!correct.SVM])) / n.test

# ------------------
# Other models
# ------------------

# LogitBoost algorithm

# train the model
model.boost <- train_model(container,
  algorithm = "BOOSTING", #see doc for other options
  verbose = TRUE)
# make predictions
results.boost <- classify_model(container.pred, model.boost)
# assess results
correct.boost <-
  sentiment.test$Sentiment == results.boost$LOGITBOOST_LABEL
accuracy.boost <- sum(correct.boost) / n.test
prob.accuracy.boost <-
  (sum(results.boost$LOGITBOOST_PROB[correct.boost]) +
  sum(1 - results.boost$LOGITBOOST_PROB[!correct.boost])) / n.test

# Bagging classification and regression trees

# Results in error:
# 'Error: cannot allocate vector of size 62.8 Mb'

# train the model
model.bag <- train_model(container,
  algorithm = "BAGGING", #see doc for other options
  verbose = TRUE)
# make predictions
results.bag <- classify_model(container.pred, model.bag)
# assess results
# NEEDS TO BE FIXED IF COPIED
# NEEDS TO BE FIXED IF COPIED
# NEEDS TO BE FIXED IF COPIED
correct.bag <-
  sentiment.test$Sentiment == results.boost$LOGITBOOST_LABEL
accuracy.boost <- sum(correct.boost) / n.test
prob.accuracy.boost <-
  (sum(results.boost$LOGITBOOST_PROB[correct.boost]) +
  sum(1 - results.boost$LOGITBOOST_PROB[!correct.boost])) / n.test

# Randon forest

# train the model
model.RF <- train_model(container,
  algorithm = "RF", #see doc for other options
  ntree = 200, # RF no. of trees
  verbose = TRUE)
# make predictions
results.RF <- classify_model(container.pred, model.RF)
# assess results
correct.RF <-
  sentiment.test$Sentiment == results.RF$FORESTS_LABEL
accuracy.RF <- sum(correct.RF) / n.test
prob.accuracy.RF <-
  (sum(results.RF$FORESTS_PROB[correct.RF]) +
  sum(1 - results.RF$FORESTS_PROB[!correct.RF])) / n.test

# GLMNET (Lasso)

# train the model
model.GLMNET <- train_model(container,
  algorithm = "GLMNET", #see doc for other options
  verbose = TRUE)
# make predictions
results.GLMNET <- classify_model(container.pred, model.GLMNET)
# assess results
correct.GLMNET <-
  sentiment.test$Sentiment == results.GLMNET$GLMNET_LABEL
accuracy.GLMNET <- sum(correct.GLMNET) / n.test
prob.accuracy.GLMNET <-
  (sum(results.GLMNET$GLMNET_PROB[correct.GLMNET]) +
  sum(1 - results.GLMNET$GLMNET_PROB[!correct.GLMNET])) / n.test

# tree

# train the model
model.TREE <- train_model(container,
                            algorithm = "TREE", #see doc for other options
                            verbose = TRUE)
# make predictions
results.TREE <- classify_model(container.pred, model.TREE)
# assess results
correct.TREE <-
  sentiment.test$Sentiment == results.TREE$TREE_LABEL
accuracy.TREE <- sum(correct.TREE) / n.test
prob.accuracy.TREE <-
  (sum(results.TREE$TREE_PROB[correct.TREE]) +
     sum(1 - results.TREE$TREE_PROB[!correct.TREE])) / n.test

# Neural network

# train the model
model.NN <- train_model(container,
  algorithm = "NNET", #see doc for other options
  size = 1200, # no. of nodes in hidden layer
  MaxNWts = 5000000,
  verbose = TRUE)
# make predictions
results.NN <- classify_model(container.pred, model.NN)
# assess results
correct.TREE <-
  sentiment.test$Sentiment == results.TREE$TREE_LABEL
accuracy.TREE <- sum(correct.TREE) / n.test
prob.accuracy.TREE <-
  (sum(results.TREE$TREE_PROB[correct.TREE]) +
     sum(1 - results.TREE$TREE_PROB[!correct.TREE])) / n.test

# -------------------------
# bigram models
# -------------------------

# creating matrix
bigrams_matrix <- create_matrix(sentiment.train$Text,
  ngramLength = 2,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  removeSparseTerms = 0,
  removeStopwords = TRUE, #lots of possibly useful stopwords in dic.
  stemWords = FALSE,
  stripWhitespace=TRUE)

# ngramLength parameter does not work.  my workaround:
# (from user user3631991 on
# http://stackoverflow.com/questions/25054617/
# rtexttools-create-matrix-returns-non-character-argument-error)

# creating create_matrix by hand :(

#creating bigrams tokenizer:
library(RWeka)
library(tm)
my2tokenizer <-
  function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

# establishing all the controls: HAVE NOT ADDED ALL YET
control <- list(
  bounds=list(local=c(1,Inf)), #minDocFreq,maxDocFreq
  language="english",
  tolower=TRUE, #lowercase
  removeNumbers=TRUE,
  tokenize=my2tokenizer,
  removePunctuation=TRUE,
  stopwords=TRUE,
  stripWhitespace=TRUE,
  wordLengths=c(3,Inf),
  weighting=weightTf)
bigrams_matrix <-
  DocumentTermMatrix(Corpus(VectorSource(sentiment.train$Text)),
  control=control)

# testing bigrams

# create container in RTextTools
testcontainer <- create_container(bigrams_matrix,
                              sentiment.train$Sentiment, trainSize = 1:n.train, virgin = FALSE)
# training models:
testmodel.SVM <- train_model(testcontainer,
                         algorithm = "SVM", #see doc for other options
                         method = "eps-regression", #for SVM
                         cross = 0, # SVM: cross=k for k-fold cross valid. of training data
                         cost = 1, # SVM cost parameter default = 100
                         kernel = "linear", #SVM kernel type default = radial
                         size = 1, #size for NN algorithm
                         rang = 0.1, # NN parameter
                         decay = 5e-04, # NN parameter
                         ntree = 200, # RF no. of trees
                         l1_regularizer = 0, # MAXENT parameter
                         l2_regularizer = 0, # MAXENT parameter
                         verbose = TRUE)

# create matrix for test data
bigrams_matrix.test <- create_matrix(sentiment.test$Text,
                                      originalMatrix = bigrams_matrix,
                                      minDocFreq = 1,
                                      minWordLength = 3,
                                      ngramLength = 2,
                                      removeNumbers = TRUE,
                                      removePunctuation = TRUE,
                                      removeSparseTerms = 0,
                                      removeStopwords = TRUE,
                                      stemWords = FALSE,
                                      stripWhitespace=TRUE)
# create prediction container
testcontainer.pred <- create_container(bigrams_matrix.test,
                                   labels = rep(0, n.test), testSize = 1:n.test, virgin=FALSE)
# make predictions
testresults.SVM <- classify_model(testcontainer.pred, testmodel.SVM)

# assess results
testcorrect.SVM <- sentiment.test$Sentiment == testresults.SVM$SVM_LABEL
testaccuracy.SVM <- sum(testcorrect.SVM) / n.test
prob.accuracy.SVM <- (sum(results.SVM$SVM_PROB[correct.SVM]) +
                        sum(1 - results.SVM$SVM_PROB[!correct.SVM])) / n.test
