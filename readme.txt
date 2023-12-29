SENTIMENT ANALYSIS DATA SETS

Chris Caldwell
DA 485, Bellevue College
4/23/17

These data come from the Sentiment Labelled Sentences Data Set hosted by the University of California, Irvine Machine Learning Repository.  A full description can be found at https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

User reviews were sampled from Amazon, imbd.com, and Yelp for various products/film/businesses and individually graded sentence-by-sentence for sentiment (0 = negative, 1 = positive).  For each of the three sources, 500 positive and 500 negative sentences were included in the data.  Data cleaning and processing R code is included in the appendix.

REFERENCES

Kotzias, Dimitrios et. al.  "From Group to Individual Labels using Deep Features."  KDD 2015

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

APPENDIX

R code for data cleanup:

# ----------------------
# reading Amazon reviews
# ----------------------
amazon_cells_labelled <-
  read.delim("~/reviews/amazon_cells_labelled.txt", header=FALSE,
  stringsAsFactors=FALSE)
# finding rows where data did not properly separate:
unrecognized_tabs <- grep("\t", amazon_cells_labelled$V1)
# set of properly formatted data:
clean_amazon <- amazon_cells_labelled[-unrecognized_tabs,]
# using no. of characters as red flag for improperly separated rows:
nchars <- sapply(clean_amazon$V1, nchar)
# second check: finding text containing "0" or "1"(sentiment marker)
amazon_with_0 <- grep("0", clean_amazon$V1)
amazon_with_1 <- grep("1", clean_amazon$V1)
# separating the data that ran together
amazon_bad_data <- amazon_cells_labelled[unrecognized_tabs,]

# ----------------------
# reading imdb.com reviews
# ----------------------
imdb_labelled <- read.delim("~/reviews/imbd_labelled.txt",
  header=FALSE, stringsAsFactors=FALSE)
# finding rows where data did not properly separate:
unrecognized_tabs <- grep("\t", imdb_labelled$V1)
# set of properly formatted data:
clean_imdb <- imdb_labelled[-unrecognized_tabs,]
# using no. of characters as red flag for improperly separated rows:
nchars <- sapply(clean_imdb$V1, nchar)
# second check: finding text containing "0" or "1"(sentiment marker)
imdb_with_0 <- grep("0", clean_imdb$V1)
imdb_with_1 <- grep("1", clean_imdb$V1)
# separating the data that ran together
imdb_bad_data <- imdb_labelled[unrecognized_tabs,]

# ----------------------
# reading Yelp reviews
# ----------------------
yelp_labelled <- read.delim("~/reviews/yelp_labelled.txt", header=FALSE,
    stringsAsFactors=FALSE)
# finding rows where data did not properly separate:
unrecognized_tabs <- grep("\t", yelp_labelled$V1)
# set of properly formatted data:
clean_yelp <- yelp_labelled[-unrecognized_tabs,]
# using no. of characters as red flag for improperly separated rows:
nchars <- sapply(clean_yelp$V1, nchar)
# second check: finding reviews containing "0" or "1"(sentiment marker)
yelp_with_0 <- grep("0", clean_yelp$V1)
yelp_with_1 <- grep("1", clean_yelp$V1)
# separating the data that ran together
yelp_bad_data <- yelp_labelled[unrecognized_tabs,]

# ----------------------
# writing the good data to a single file sentiment.txt
# ----------------------
write.table(clean_amazon, file = "sentiment.txt", sep = "\t",
  row.names = FALSE, col.names = FALSE)
write.table(clean_imdb, file = "sentiment.txt", append = TRUE,
  sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(clean_yelp, file = "sentiment.txt", append = TRUE,
  sep = "\t", row.names = FALSE, col.names = FALSE)

# ----------------------
# creating a data frame for the problem data
# (tabs in data came out as "\t" instead of delimiters)
# ----------------------
bad_data <- rbind(amazon_bad_data, imdb_bad_data, yelp_bad_data)
# writing bad data to file
write.table(bad_data, file = "baddata.txt", row.names = FALSE,
  col.names = FALSE)

# ----------------------
# Fixing the baddata.txt file
# ----------------------
# run-on rows were enclosed with quotes: "<rows of run on data>"
# removing quotes should fix it
bad_data_file <- readLines("baddata.txt")
bad_data_file <- gsub("\\\"", "", bad_data_file)
write.table(bad_data_file, file = "fixedbaddata.txt", quote = FALSE,
  row.names = FALSE, col.names = FALSE)

# ----------------------
# merging newly fixed data with the others
# ----------------------
bad_data_fixed <- read.delim("fixedbaddata.txt", header=FALSE,
  stringsAsFactors=FALSE)
write.table(bad_data_fixed, file = "sentiment.txt", append = TRUE,
  sep = "\t", row.names = FALSE, col.names = FALSE)

# ----------------------
# Opening full data set, checking for NAs
# ----------------------
sentiment <- read.delim("~/sentiment.txt", header=FALSE,
  stringsAsFactors=FALSE)
colnames(sentiment) <- c("text", "sentiment")
bad_rows <- sentiment[!complete.cases(sentiment),]
# all mistakes are failures to assign sentiment = 1, so:
sentiment$sentiment[!complete.cases(sentiment)] <- 1
# removing two NA rows
sentiment <- sentiment[-c(2288, 3002),]

# ----------------------
# saving sentiment data to file
# ----------------------
write.table(sentiment, file = "sentiment.txt", sep = "\t",
  row.names = FALSE, col.names = FALSE)

# ----------------------
# creating train and test sets (70% train/30% test)
# ----------------------
smp_size <- .7 * nrow(sentiment)
set.seed(51) #Ichiro!
train_ind <- sample(seq(1, nrow(sentiment)), size = smp_size)
sentiment.train <- sentiment[train_ind, ]
sentiment.test <- sentiment[-train_ind, ]
write.table(sentiment.train, file = "sentiment.train.txt",
  sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(sentiment.test, file = "sentiment.test.txt", sep = "\t",
  row.names = FALSE, col.names = FALSE)
