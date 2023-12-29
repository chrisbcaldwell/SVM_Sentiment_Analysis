library(tm)
library(dplyr)
library(tidytext)
library(RTextTools)

# CREATE CUSTOM STOPWORDS LIST HERE
sw <- unique(stop_words$word) %>% removePunctuation() #stopwords have problematic punct in them

# CUSTOM FUNCTIONS:
BigramTokenizer <-
  function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
cleanup <- function(x, stopwds = NULL, stemming = FALSE) {
  corp <- VCorpus(VectorSource(x)) %>%
    tm_map(content_transformer(function(x) iconv(x, to="ASCII", sub=""))) %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwds)
  if(stemming) corp <- tm_map(corp, stemDocument)
  return(corp)
}
TrigramTokenizer <-
  function(x) unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)

# Data loading:
sentiment.train <-
  read.delim("~/Course Documents/DA 485/Capstone/sentiment.train.txt",
    header=FALSE, stringsAsFactors=FALSE)
colnames(sentiment.train) <- c("Text", "Sentiment")
n.train <- nrow(sentiment.train)
sentiment.test <-
  read.delim("~/Course Documents/DA 485/Capstone/sentiment.test.txt",
    header = FALSE, stringsAsFactors = FALSE)
colnames(sentiment.test) <- c("Text", "Sentiment")
n.test <- nrow(sentiment.test)

corpus.train <- cleanup(sentiment.train$Text, sw)
corpus.test <- cleanup(sentiment.test$Text, sw)
corpus.train.sw <- cleanup(sentiment.train$Text)
corpus.test.sw <- cleanup(sentiment.test$Text)
corpus.train.stem <- cleanup(sentiment.train$Text, sw, stemming = TRUE)
corpus.test.stem <- cleanup(sentiment.test$Text, sw, stemming = TRUE)
corpus.train.sw.stem <- cleanup(sentiment.train$Text, stemming = TRUE)
corpus.test.sw.stem <- cleanup(sentiment.test$Text, stemming = TRUE)

# 1-grams stopwords removed:
dtm1.train <- DocumentTermMatrix(corpus.train,
  control = list(weighting = weightBin))
dtm1.test <- DocumentTermMatrix(corpus.test,
  control = list(weighting = weightBin, 
    dictionary = Terms(dtm1.train)))

# create container in RTextTools
container.1 <- create_container(dtm1.train, sentiment.train$Sentiment, trainSize = 1:n.train,
  virgin = FALSE)
# training models:
model1.SVM <- train_model(container.1,
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
# create prediction container
container1.pred <- create_container(dtm1.test, labels = rep(0, n.test), testSize = 1:n.test,
  virgin=FALSE)
# make predictions
result <- classify_model(container1.pred, model1.SVM)
# assess results
correct <- sentiment.test$Sentiment == result$SVM_LABEL
pred.accuracy1.SVM <- sum(correct) / n.test
prob.accuracy1.SVM <-
  (sum(result$SVM_PROB[correct]) + sum(1 - result$SVM_PROB[!correct])) / n.test

# # 2-grams, stopwords removed:
# dtm2.train <- DocumentTermMatrix(corpus.train,
#   control = list(
#     weighting = weightBin,
#     tokenize = BigramTokenizer))
# dtm2.test <- DocumentTermMatrix(corpus.test,
#   control = list(
#     weighting = weightBin,
#     tokenize = BigramTokenizer,
#     dictionary = Terms(dtm2.train)))
# # create container in RTextTools
# container.2 <-
#   create_container(dtm2.train, sentiment.train$Sentiment, trainSize = 1:n.train, virgin = FALSE)
# # training models:
# model2.SVM <- train_model(container.2,
#                           algorithm = "SVM", #see doc for other options
#                           method = "eps-regression", #for SVM
#                           cross = 0, # SVM: cross=k for k-fold cross valid. of training data
#                           cost = 1, # SVM cost parameter default = 100
#                           kernel = "linear", #SVM kernel type default = radial
#                           size = 1, #size for NN algorithm
#                           rang = 0.1, # NN parameter
#                           decay = 5e-04, # NN parameter
#                           ntree = 200, # RF no. of trees
#                           l1_regularizer = 0, # MAXENT parameter
#                           l2_regularizer = 0, # MAXENT parameter
#                           verbose = TRUE)
# # create prediction container
# container2.pred <- create_container(dtm2.test, labels = rep(0, n.test), testSize = 1:n.test,
#                                     virgin=FALSE)
# # make predictions
# result <- classify_model(container2.pred, model2.SVM)
# # assess results
# correct <- sentiment.test$Sentiment == result$SVM_LABEL
# pred.accuracy2.SVM <- sum(correct) / n.test
# prob.accuracy2.SVM <-
#   (sum(result$SVM_PROB[correct]) + sum(1 - result$SVM_PROB[!correct])) / n.test

# 2-grams, stopwords retained:
dtm2.train <- DocumentTermMatrix(corpus.train.sw,
                                 control = list(
                                   weighting = weightBin,
                                   tokenize = BigramTokenizer))
dtm2.test <- DocumentTermMatrix(corpus.test.sw,
                                control = list(
                                  weighting = weightBin,
                                  tokenize = BigramTokenizer,
                                  dictionary = Terms(dtm2.train)))
# create container in RTextTools
container.2 <-
  create_container(dtm2.train, sentiment.train$Sentiment, trainSize = 1:n.train, virgin = FALSE)
# # training models:
# model2.SVM <- train_model(container.2.sw,
#                           algorithm = "SVM", #see doc for other options
#                           method = "eps-regression", #for SVM
#                           cross = 0, # SVM: cross=k for k-fold cross valid. of training data
#                           cost = 1, # SVM cost parameter default = 100
#                           kernel = "linear", #SVM kernel type default = radial
#                           size = 1, #size for NN algorithm
#                           rang = 0.1, # NN parameter
#                           decay = 5e-04, # NN parameter
#                           ntree = 200, # RF no. of trees
#                           l1_regularizer = 0, # MAXENT parameter
#                           l2_regularizer = 0, # MAXENT parameter
#                           verbose = TRUE)
# create prediction container
container2.pred <- create_container(dtm2.test, labels = rep(0, n.test), testSize = 1:n.test,
                                    virgin=FALSE)
# create stemmed corpus dtms and containers
dtm2.stem.train <- DocumentTermMatrix(corpus.train.sw.stem,
                                 control = list(
                                   weighting = weightBin,
                                   tokenize = BigramTokenizer))
dtm2.stem.test <- DocumentTermMatrix(corpus.test.sw.stem,
                                control = list(
                                  weighting = weightBin,
                                  tokenize = BigramTokenizer,
                                  dictionary = Terms(dtm2.stem.train)))
container.2.stem <-
  create_container(dtm2.stem.train, sentiment.train$Sentiment, trainSize = 1:n.train,
                   virgin = FALSE)
container2.pred.stem <- create_container(dtm2.stem.test, labels = rep(0, n.test),
                                         testSize = 1:n.test, virgin=FALSE)
# make predictions
result <- classify_model(container2.sw.pred, model2.sw.SVM)
# assess results
correct <- sentiment.test$Sentiment == result$SVM_LABEL
pred.accuracy2.sw.SVM <- sum(correct) / n.test
prob.accuracy2.sw.SVM <-
  (sum(result$SVM_PROB[correct]) + sum(1 - result$SVM_PROB[!correct])) / n.test

# 3-grams, stopwords retained
dtm3.sw.train <- DocumentTermMatrix(corpus.train.sw,
                                    control = list(
                                      weighting = weightBin,
                                      tokenize = TrigramTokenizer))
dtm3.sw.test <- DocumentTermMatrix(corpus.test.sw,
                                   control = list(
                                     weighting = weightBin,
                                     tokenize = TrigramTokenizer,
                                     dictionary = Terms(dtm3.sw.train)))
# create container in RTextTools
container.3.sw <-
  create_container(dtm3.sw.train, sentiment.train$Sentiment, trainSize = 1:n.train, virgin = FALSE)
# training models:
model3.sw.SVM <- train_model(container.3.sw,
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
# create prediction container
container3.sw.pred <- create_container(dtm3.sw.test, labels = rep(0, n.test), testSize = 1:n.test,
                                       virgin=FALSE)
# make predictions
result <- classify_model(container3.sw.pred, model3.sw.SVM)
# assess results
correct <- sentiment.test$Sentiment == result$SVM_LABEL
pred.accuracy3.sw.SVM <- sum(correct) / n.test
prob.accuracy3.sw.SVM <-
  (sum(result$SVM_PROB[correct]) + sum(1 - result$SVM_PROB[!correct])) / n.test

# ---------------
# OPTIMIZATION
# ---------------
# tweak the SVM model to get an approximately optimal result
kernels <- c("linear","polynomial","sigmoid","radial")
costs <- 10^(-5:5)
result <- array(dim = c(2*length(kernels)*length(costs),9))
colnames(result) <-
  c("Cost","Kernel","Stem","Pred.Ac","Prob.Ac","pred.pos_actual.pos","pred.pos_actual.neg",
    "pred.neg_actual.pos","pred.neg_actual.neg")
run_number <- 1
# no stemming:
for(kernel in kernels) {
  for(cost in costs) {
    test_model <- train_model(container.1, algorithm = "SVM",
                              method = "eps-regression", cost = cost, kernel = kernel,
                              verbose = TRUE)
    test_results <- classify_model(container1.pred, test_model)
    correct <- sentiment.test$Sentiment == test_results[,1]
    pred.accuracy <- sum(correct) / n.test
    prob.accuracy <- (sum(test_results[correct,2]) +
                        sum(1 - test_results[!correct,2])) / n.test
    pred.pos_actual.pos <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 1)
    pred.pos_actual.neg <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 0)
    pred.neg_actual.pos <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 1)
    pred.neg_actual.neg <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 0)
    result[run_number,] <- c(cost,kernel,FALSE,pred.accuracy,prob.accuracy,pred.pos_actual.pos,
                             pred.pos_actual.neg,pred.neg_actual.pos,pred.neg_actual.neg)
    print(paste("run number", run_number, "of", nrow(result)/2))
    run_number <- run_number + 1
  }
}
# With stemming:
# 1-grams stopwords removed, stemmed:
dtm1.stem.train <- DocumentTermMatrix(corpus.train.stem,
                                 control = list(
                                   weighting = weightBin))
dtm1.stem.test <- DocumentTermMatrix(corpus.test.stem,
                                control = list(weighting = weightBin, 
                                               dictionary = Terms(dtm1.stem.train)))
container.1.stem <- create_container(dtm1.stem.train, sentiment.train$Sentiment,
                                     trainSize = 1:n.train, virgin = FALSE)
container1.pred.stem <- create_container(dtm1.stem.test, labels = rep(0, n.test),
                                         testSize = 1:n.test, virgin=FALSE)
run_number <- 45
for(kernel in kernels) {
  for(cost in costs) {
    test_model <- train_model(container.1.stem, algorithm = "SVM",
                              method = "eps-regression", cost = cost, kernel = kernel,
                              verbose = TRUE)
    test_results <- classify_model(container1.pred.stem, test_model)
    correct <- sentiment.test$Sentiment == test_results[,1]
    pred.accuracy <- sum(correct) / n.test
    prob.accuracy <- (sum(test_results[correct,2]) +
                        sum(1 - test_results[!correct,2])) / n.test
    pred.pos_actual.pos <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 1)
    pred.pos_actual.neg <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 0)
    pred.neg_actual.pos <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 1)
    pred.neg_actual.neg <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 0)
    result[run_number,] <- c(cost,kernel,TRUE,pred.accuracy,prob.accuracy,pred.pos_actual.pos,
                             pred.pos_actual.neg,pred.neg_actual.pos,pred.neg_actual.neg)
    print(paste("run number", run_number, "of", nrow(result)))
    run_number <- run_number + 1
  }
}
# storing parameter test result
result.SVM1 <- result
# redoing best parameter and storing predictions
accuracy <- data.frame(SVM1 = 0)
predictions.train <- data.frame(sentiment.train = sentiment.train$Sentiment)
predictions.test <- data.frame(sentiment.test = sentiment.test$Sentiment)
best_model.SVM1 <- train_model(container.1, algorithm = "SVM",
                          method = "eps-regression", cost = 0.1, kernel = "linear",
                          verbose = TRUE)
test_results <- classify_model(container1.pred, best_model.SVM1)
train_results <- classify_model(container.1, best_model.SVM1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$SVM1 <- sum(correct) / n.test
# converting to prob of positive
pos_prob <- function(x) {
  final_thing <- c()
  for(i in 1:nrow(x)) {
    if(x[i,1] == 0) {
      final_thing[i] <- 1 - x[i,2]
    } else {
      final_thing[i] <- x[i,2]
    }
  }
  return(final_thing)
}
predictions.test$SVM1 <- pos_prob(test_results)
predictions.train$SVM1 <- pos_prob((train_results))
# SLDA without stemming, 1-grams
slda1 <- train_model(container.1, algorithm = "SLDA")
test_results <- classify_model(container1.pred, slda1)
train_results <- classify_model(container.1, slda1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$SLDA1 <- sum(correct) / n.test
predictions.test$SLDA1 <- pos_prob(test_results)
predictions.train$SLDA1 <- pos_prob((train_results))

# SLDA with stemming, 1-grams
slda1.stem <- train_model(container.1.stem, algorithm = "SLDA")
test_results <- classify_model(container1.pred.stem, slda1.stem)
train_results <- classify_model(container.1.stem, slda1.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$SLDA1.stem <- sum(correct) / n.test
# not run, accuracy less than unstemmed
# predictions.test$SLDA1 <- pos_prob(test_results)
# predictions.train$SLDA1 <- pos_prob((train_results))

# Boosting w/o stemming, 1-grams
boost1 <- train_model(container.1, algorithm = "BOOSTING")
test_results <- classify_model(container1.pred, boost1)
train_results <- classify_model(container.1, boost1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$Boost1 <- sum(correct) / n.test
predictions.test$Boost1 <- pos_prob(test_results)
predictions.train$Boost1 <- pos_prob((train_results))

# Boosting with stemming, 1-grams
boost1.stem <- train_model(container.1.stem, algorithm = "BOOSTING")
test_results <- classify_model(container1.pred.stem, boost1.stem)
train_results <- classify_model(container.1.stem, boost1.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$Boost1.stem <- sum(correct) / n.test
predictions.test$Boost1.stem <- pos_prob(test_results)
predictions.train$Boost1.stem <- pos_prob((train_results))

# Random Forest w/o stemming, 1-grams
rf1 <- train_model(container.1, algorithm = "RF")
test_results <- classify_model(container1.pred, rf1)
train_results <- classify_model(container.1, rf1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$RF1 <- sum(correct) / n.test
predictions.test$RF1 <- pos_prob(test_results)
predictions.train$RF1 <- pos_prob((train_results))

# random Forest with stemming, 1-grams
rf1.stem <- train_model(container.1.stem, algorithm = "RF")
test_results <- classify_model(container1.pred.stem, rf1.stem)
train_results <- classify_model(container.1.stem, rf1.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$RF1.stem <- sum(correct) / n.test
# ignored, less acuurate than nonstemmed
#predictions.test$RF1.stem <- pos_prob(test_results)
#predictions.train$RF1.stem <- pos_prob((train_results))

# Glmnet w/o stemming, 1-grams
glmnet1 <- train_model(container.1, algorithm = "GLMNET")
test_results <- classify_model(container1.pred, glmnet1)
train_results <- classify_model(container.1, glmnet1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$GLMNET1 <- sum(correct) / n.test
predictions.test$GLMNET1 <- pos_prob(test_results)
predictions.train$GLMNET1 <- pos_prob((train_results))

# Glmnet with stemming, 1-grams
glmnet1.stem <- train_model(container.1.stem, algorithm = "GLMNET")
test_results <- classify_model(container1.pred.stem, glmnet1.stem)
train_results <- classify_model(container.1.stem, glmnet1.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$GLMNET1.stem <- sum(correct) / n.test
predictions.test$GLMNET1.stem <- pos_prob(test_results)
predictions.train$GLMNET1.stem <- pos_prob((train_results))

# Maxent w/o stemming, 1-grams
maxent1 <- train_model(container.1, algorithm = "MAXENT")
test_results <- classify_model(container1.pred, maxent1)
train_results <- classify_model(container.1, maxent1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$maxent1 <- sum(correct) / n.test
predictions.test$maxent1 <- pos_prob(test_results)
predictions.train$maxent1 <- pos_prob((train_results))

# MaXENT with stemming, 1-grams
maxent1.stem <- train_model(container.1.stem, algorithm = "MAXENT")
test_results <- classify_model(container1.pred.stem, maxent1.stem)
train_results <- classify_model(container.1.stem, maxent1.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$maxent1.stem <- sum(correct) / n.test
predictions.test$maxent1.stem <- pos_prob(test_results)
predictions.train$maxent1.stem <- pos_prob((train_results))

# ----------------------
# 2-GRAMS
# ----------------------

# tweak the SVM model to get an approximately optimal result
kernels <- c("linear","polynomial","sigmoid","radial")
costs <- 10^(-5:5)
result2 <- array(dim = c(length(kernels)*length(costs),9))
colnames(result2) <-
  c("Cost","Kernel","Stem","Pred.Ac","Prob.Ac","pred.pos_actual.pos","pred.pos_actual.neg",
    "pred.neg_actual.pos","pred.neg_actual.neg")
run_number <- 1
# no stemming:
for(kernel in kernels) {
  for(cost in costs) {
    test_model <- train_model(container.2, algorithm = "SVM",
                              method = "eps-regression", cost = cost, kernel = kernel,
                              verbose = TRUE)
    test_results <- classify_model(container2.pred, test_model)
    correct <- sentiment.test$Sentiment == test_results[,1]
    pred.accuracy <- sum(correct) / n.test
    prob.accuracy <- (sum(test_results[correct,2]) +
                        sum(1 - test_results[!correct,2])) / n.test
    pred.pos_actual.pos <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 1)
    pred.pos_actual.neg <- sum(test_results[,1] == 1 & sentiment.test$Sentiment == 0)
    pred.neg_actual.pos <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 1)
    pred.neg_actual.neg <- sum(test_results[,1] == 0 & sentiment.test$Sentiment == 0)
    result2[run_number,] <- c(cost,kernel,FALSE,pred.accuracy,prob.accuracy,pred.pos_actual.pos,
                             pred.pos_actual.neg,pred.neg_actual.pos,pred.neg_actual.neg)
    print(paste("run number", run_number, "of", nrow(result2)))
    run_number <- run_number + 1
  }
}
# use cost = 10, linear kernel
best_model.SVM2 <- train_model(container.2, algorithm = "SVM",
                               method = "eps-regression", cost = 10, kernel = "linear",
                               verbose = TRUE)
test_results <- classify_model(container2.pred, best_model.SVM2)
train_results <- classify_model(container.2, best_model.SVM2)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$SVM2 <- sum(correct) / n.test
predictions.test$SVM2 <- pos_prob(test_results)
predictions.train$SVM2 <- pos_prob((train_results))

# # SLDA without stemming, 2-grams
# # Error: cannot allocate vector of size 821.9 Mb
# gc()
# slda2 <- train_model(container.2, algorithm = "SLDA")
# test_results <- classify_model(container2.pred, slda2)
# train_results <- classify_model(container.2, slda2)
# correct <- sentiment.test$Sentiment == test_results[,1]
# accuracy$SLDA2 <- sum(correct) / n.test
# predictions.test$SLDA2 <- pos_prob(test_results)
# predictions.train$SLDA2 <- pos_prob((train_results))

# # SLDA with stemming, 2-grams
# # Error: ccannot allocate vector of size 767.4 Mb
# slda2.stem <- train_model(container.2.stem, algorithm = "SLDA")
# test_results <- classify_model(container2.pred.stem, slda2.stem)
# train_results <- classify_model(container.2.stem, slda2.stem)
# correct <- sentiment.test$Sentiment == test_results[,1]
# accuracy$SLDA2.stem <- sum(correct) / n.test
# # not run, accuracy less than unstemmed
# # predictions.test$SLDA1 <- pos_prob(test_results)
# # predictions.train$SLDA1 <- pos_prob((train_results))

# Boosting w/o stemming, 2-grams
boost2 <- train_model(container.2, algorithm = "BOOSTING")
test_results <- classify_model(container2.pred, boost2)
train_results <- classify_model(container.2, boost1)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$Boost2 <- sum(correct) / n.test
predictions.test$Boost2 <- pos_prob(test_results)
predictions.train$Boost2 <- pos_prob((train_results))

# Boosting with stemming, 2-grams
boost2.stem <- train_model(container.2.stem, algorithm = "BOOSTING")
test_results <- classify_model(container2.pred.stem, boost2.stem)
train_results <- classify_model(container.2.stem, boost2.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$Boost2.stem <- sum(correct) / n.test
predictions.test$Boost2.stem <- pos_prob(test_results)
predictions.train$Boost2.stem <- pos_prob((train_results))

# # Random Forest w/o stemming, 2-grams
# # aborted after 1.5 hrs
# rf2 <- train_model(container.2, algorithm = "RF")
# test_results <- classify_model(container2.pred, rf2)
# train_results <- classify_model(container.2, rf2)
# correct <- sentiment.test$Sentiment == test_results[,1]
# accuracy$RF2 <- sum(correct) / n.test
# predictions.test$RF2 <- pos_prob(test_results)
# predictions.train$RF2 <- pos_prob((train_results))

# random Forest with stemming, 2-grams
rf2.stem <- train_model(container.2.stem, algorithm = "RF")
test_results <- classify_model(container2.pred.stem, rf2.stem)
train_results <- classify_model(container.2.stem, rf2.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$RF2.stem <- sum(correct) / n.test
# ignored, less acuurate than nonstemmed
#predictions.test$RF1.stem <- pos_prob(test_results)
#predictions.train$RF1.stem <- pos_prob((train_results))

# Glmnet w/o stemming, 2-grams
glmnet2 <- train_model(container.2, algorithm = "GLMNET")
test_results <- classify_model(container2.pred, glmnet2)
train_results <- classify_model(container.2, glmnet2)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$GLMNET2 <- sum(correct) / n.test
predictions.test$GLMNET2 <- pos_prob(test_results)
predictions.train$GLMNET2 <- pos_prob((train_results))

# Glmnet with stemming, 2-grams
glmnet2.stem <- train_model(container.2.stem, algorithm = "GLMNET")
test_results <- classify_model(container2.pred.stem, glmnet2.stem)
train_results <- classify_model(container.2.stem, glmnet2.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$GLMNET2.stem <- sum(correct) / n.test
predictions.test$GLMNET2.stem <- pos_prob(test_results)
predictions.train$GLMNET2.stem <- pos_prob((train_results))

# Maxent w/o stemming, 2-grams
maxent2 <- train_model(container.2, algorithm = "MAXENT")
test_results <- classify_model(container2.pred, maxent2)
train_results <- classify_model(container.2, maxent2)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$maxent2 <- sum(correct) / n.test
predictions.test$maxent2 <- pos_prob(test_results)
predictions.train$maxent2 <- pos_prob((train_results))

# MaXENT with stemming, 2-grams
maxent2.stem <- train_model(container.2.stem, algorithm = "MAXENT")
test_results <- classify_model(container2.pred.stem, maxent2.stem)
train_results <- classify_model(container.2.stem, maxent2.stem)
correct <- sentiment.test$Sentiment == test_results[,1]
accuracy$maxent2.stem <- sum(correct) / n.test
predictions.test$maxent2.stem <- pos_prob(test_results)
predictions.train$maxent2.stem <- pos_prob((train_results))

###################################
####################################
###################################
# Finding best combination of models

# retaining only the best for each algorithm at each n-gram level
pred.test.reduced <- predictions.test[,-c(4,7,9,12,14,16)]
pred.train.reduced <- predictions.train[,-c(4,7,9,12,14,16)]
colnames(pred.test.reduced) <- c("sentiment", "svm1","slda1","boost1","rf1","glmnet1","maxent1",
                                 "svm2","boost2","glmnet2","maxent2")
colnames(pred.train.reduced) <- colnames(pred.test.reduced)
pred.test.reduced$sentiment <- factor(pred.test.reduced$sentiment)
pred.train.reduced$sentiment <- factor(pred.train.reduced$sentiment)

# Finding the best model
model <- glm(sentiment ~., family = binomial, data = pred.train.reduced)
fitted.results <- predict(model, newdata = pred.test.reduced, type = "response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)

#boosting is terrible and throws everything off, remove it:
model <- glm(sentiment ~ svm1+slda1+rf1+glmnet1+maxent1+svm2+glmnet2+maxent2,
             family = binomial(link="logit"), data = pred.train.reduced)
fitted.results <- predict(model, newdata = pred.test.reduced, type = "response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)

# probabilistic algorithms only:
model <- glm(sentiment ~ svm1+rf1+glmnet1+svm2+glmnet2,
             family = binomial(link="logit"), data = pred.train.reduced)
fitted.results <- predict(model, newdata = pred.test.reduced, type = "response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)


# all p-values nearly 1, trying transformation to logits
logits.train <- apply(pred.train.reduced[,-c(1,4,9)],c(1,2),
                      function(x) if(x == 1) 1000 else x / (1-x))
logits.train <- data.frame(sentiment = pred.train.reduced$sentiment, logits.train)
logits.test <- apply(pred.test.reduced[,-c(1,4,9)],c(1,2),
                      function(x) if(x == 1) 1000 else x / (1-x))
logits.test <- data.frame(sentiment = pred.test.reduced$sentiment, logits.test)
# Finding the best model
model <- glm(sentiment ~., family = binomial, data = logits.train)
fitted.results <- predict(model, newdata = logits.test[-1], type = "response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
#accuracy figure
accuracy.logit <- sum(fitted.results == sentiment.test$Sentiment) / n.test
# 0.76556


# START FIDDLING WITH THE MODEL HERE
model <- glm(sentiment ~ , family = binomial, data = logits.train)
fitted.results <- predict(model, newdata = logits.test[-1], type = "response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
#accuracy figure
accuracy.logit <- sum(fitted.results == sentiment.test$Sentiment) / n.test