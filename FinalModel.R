library(RTextTools)
library(dplyr)
library(twitteR)
library(ROAuth)
# library(caret)
library(MASS)
library(e1071)

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

# Best tuning of SVM model
result <- array(dim = c(168,5))
colnames(result) <-
  c("Cost","SparsesRemoved","Kernel","Stemwords","Pred.Ac")
kernels <- c("linear","polynomial","sigmoid","radial")
costs <- 10^(-1:5)
sparse <- c(0, .9995, .999)
run_number <- 1
for(i in 1:3) {
  for(stem in c(TRUE,FALSE)) {
    test_matrix <- create_matrix(sentiment.train$Text, removeNumbers = TRUE, stemWords = stem,
                                 removeSparseTerms = sparse[i])
    test_container <- create_container(test_matrix,
                                       sentiment.train$Sentiment, trainSize = 1:n.train,
                                       virgin = FALSE)
    for(kernel in kernels) {
      for(cost in costs) {
        test_model <- train_model(test_container, algorithm = "SVM",
                                  method = "eps-regression", cost = cost, kernel = kernel,
                                  verbose = TRUE)
        prediction_matrix <- create_matrix(sentiment.test$Text,
                                           removeNumbers = TRUE,
                                           stemWords = stem,
                                           originalMatrix = test_matrix)
        test_container.pred <- create_container(prediction_matrix,
                                                labels = rep(0, n.test), testSize = 1:n.test,
                                                virgin=FALSE)
        test_results <- classify_model(test_container.pred, test_model)
        correct <- sentiment.test$Sentiment == test_results$SVM_LABEL
        pred.accuracy <- sum(correct) / n.test
        result[run_number,] <-
          c(cost,i-1,kernel,stem,pred.accuracy)
        print(run_number)
        run_number <- run_number + 1
      }
    }
  }
}  
result <- data.frame(result)
result$Pred.Ac <- as.numeric(as.character(result$Pred.Ac))

# Plotting for report
boxplot(result$Pred.Ac ~ result$Kernel,
        xlab = "-- Kernel Type --",
        ylab = "Test Set Prediction Accuracy")
abline(h = 0.5, lty = 2)

levels(result$Stemwords)[levels(result$Stemwords)=="TRUE"] <- "Stemmed"
levels(result$Stemwords)[levels(result$Stemwords)=="FALSE"] <- "Unstemmed"
boxplot(result$Pred.Ac ~ result$Stemwords,
        xlab = "-- Word Stemming --",
        ylab = "Test Set Prediction Accuracy")
abline(h = 0.5, lty = 2)

boxplot(result$Pred.Ac[] ~ result$Cost,
        xlab = "-- SVM Cost Parameter --",
        ylab = "Test Set Prediction Accuracy")
abline(h = 0.5, lty = 2)

linearcosts <- factor(result$Cost, levels=levels(result$Cost)[c(10,9,1,2,3,4,5,6,7,8,11)])
boxplot(result$Pred.Ac[result$Kernel == "linear"] ~ result$Cost[result$Kernel == "linear"],
        xlab = "-- Cost Parameter for Linear Kernel --",
        ylab = "Test Set Prediction Accuracy")
abline(h = 0.5, lty = 2)


# returning to best model for demonstration
unigrams_matrix <- create_matrix(sentiment.train$Text, removeNumbers = TRUE, stemWords = TRUE)
unigrams_container <- create_container(unigrams_matrix,
                                   sentiment.train$Sentiment, trainSize = 1:n.train,
                                   virgin = FALSE)
model.SVM <- train_model(unigrams_container, algorithm = "SVM",
                         method = "eps-regression", cost = 1, kernel = "linear")
unigrams_matrix.test <- create_matrix(sentiment.train$Text,
                                      removeNumbers = TRUE,
                                      stemWords = TRUE,
                                      originalMatrix = unigrams_matrix)

# finding most important features
source('msvmRFE.R')
large_matrix <- cbind(factor(sentiment.train$Sentiment), as.matrix(unigrams_matrix))
train_df <- as.data.frame(large_matrix)
top_features <- svmRFE(train_df, k=10, halve.above=100)
top25 <- unigrams_matrix$dimnames$Terms[head(top_features,25)]
last25 <- unigrams_matrix$dimnames$Terms[tail(top_features,25)]





# predict_text <- function(x) {
#   create_matrix(x, originalMatrix = unigrams_matrix, removeNumbers = TRUE, stemWords = TRUE) %>%
#     create_container(labels = rep(0, length(x)), testSize = 1:length(x), virgin=FALSE) %>%
#     classify_model(model.SVM)
# }
# 
# # enabling twitter loading
# source("twitterauth.R")
# 
# tweet_text <- function(x) {
#   tweets <- searchTwitter(x)
#   tweets <- do.call("rbind", lapply(tweets, as.data.frame)) %>% subset(select = c(text))
#   tweets <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", " ",  tweets)
#   tweets <- gsub("@\\w+", "", tweets)
#   tweets <- gsub("http\\w+", "", tweets)
#   return(tweets)
# }
# 
# predict_tweets <- function(x) {
#   predict_text(tweet_text(x))
# }

# creation of function to give details of predicting new data
# input is df w columns: 1) sentiment (as 1 or 0). 2) text
# returns list
predict_new <- function(x, model) {
  dtm.temp <- create_matrix(x[,2],
                            removeNumbers = TRUE,
                            stemWords = TRUE,
                            originalMatrix = unigrams_matrix)
  container.tmp <- create_container(dtm.temp,
                                    labels = rep(0, nrow(x)),
                                    testSize = 1:nrow(x),
                                    virgin=FALSE)
  results <- classify_model(container.tmp, model = model)
  correct.tmp <- x[,1] == results$SVM_LABEL
  accuracy.tmp <- sum(correct.tmp) / nrow(x)
  # confusion matrix
  confusion <- matrix(nrow = 2, ncol = 2)
  rownames(confusion) <- c("pred.pos","pred.neg")
  colnames(confusion) <- c("label.pos","label.neg")
  confusion[1,1] <- sum(results$SVM_LABEL == 1 & x[,1] == 1)
  confusion[1,2] <- sum(results$SVM_LABEL == 1 & x[,1] == 0)
  confusion[2,1] <- sum(results$SVM_LABEL == 0 & x[,1] == 1)
  confusion[2,2] <- sum(results$SVM_LABEL == 0 & x[,1] == 0)
  # returning a list
  list(predictions = results,
       correct = correct.tmp,
       accuracy = accuracy.tmp,
       confusion = confusion)
}


# creation of corpus to look at for validation
GOP <- read.csv("GOP_REL_ONLY.csv")
GOP <- GOP[GOP$sentiment == "Positive" | GOP$sentiment == "Negative",]
GOP$sentiment <- sapply(GOP$sentiment, function(x) if(x=="Positive") 1 else 0) 
SADS <- read.csv("Sentiment Analysis Dataset.csv")
SADS <- SADS[SADS$Sentiment == 1 | SADS$Sentiment == 0,c(2,4)]
SADS.sample.i <- sample(1:nrow(SADS),300)
SADS.sample <- SADS[SADS.sample.i,]
sentiment140 <- read.csv("s140.csv")
sentiment140 <- sentiment140[,c(1,6)]
colnames(sentiment140) <- c("sentiment","text")
sentiment140 <- subset(sentiment140, sentiment140$sentiment != 2)
sentiment140$sentiment <- sapply(sentiment140$sentiment, function(x) if(x==4) 1 else 0)
sentiment140.sample.i <- sample(1:nrow(sentiment140),25000)
sentiment140.sample <- sentiment140[sentiment140.sample.i,]

GOP.pred <- predict_new(GOP, model.SVM)
sentiment140.pred <- predict_new(sentiment140.sample, model.SVM)
