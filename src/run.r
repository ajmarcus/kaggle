require(stringr)
require(ggplot2)
require(RTextTools)
require(randomForest)
require(e1071)
require(gbm)
require(Boruta)

## Load Data

sampleSub <- read.csv(file="data/sample_submission_file.csv",
                      stringsAsFactors=FALSE,
                      fill=FALSE)

train <- read.delim(file="data/train.tsv",
                    stringsAsFactors=FALSE,
                    fill=FALSE)

test <- read.delim(file="data/test.tsv",
                   stringsAsFactors=FALSE,
                   fill=FALSE)


# 1. The number of spelling mistakes in each essay

train.spell <- read.delim(file="data/spell_train.out",
                          stringsAsFactors=FALSE,
                          fill=FALSE)

train <- data.frame(train, train.spell)

test.spell <- read.delim(file="data/spell_test.out",
                         stringsAsFactors=FALSE,
                         fill=FALSE)

test <- data.frame(test, test.spell)


# 2. Count of a keyword derived from the essay prompt

keyword <- read.delim(file="data/prompt_keyword.tsv",
                      stringsAsFactors=FALSE,
                      fill=FALSE)

train <- merge(train, keyword, by="set")

train$num_keyword <- str_count(train$essay, ignore.case(train$keyword))

ggplot(train, aes(x=num_keyword, y=grade, colour=set)) +
   geom_jitter() +
   facet_grid(~set) +
   geom_smooth(method=lm) +
   ggtitle("Goodness of fit for linear model") 

# 3. The term document matrix of the essay

matrix.train <- create_matrix(train$essay, stemWords=TRUE, weighting=weightTfIdf,
                              removeSparseTerms=0.2, ngramLength=4)

matrix.test <- create_matrix(test$essay, stemWords=TRUE, weighting=weightTfIdf,
                             removeSparseTerms=0.2, ngramLength=4)


# 4. Number of characters and numbers in each essay

train$num_char <- str_count(train$essay, "[A-Za-z0-9]")
test$num_char <- str_count(test$essay, "[A-Za-z0-9]")


# 5. Number of exclamations points in each essay

train$num_exclamation <- str_count(train$essay, "!")
test$num_exclamation <- str_count(test$essay, "!")

train$set <- as.factor(train$set)
test$set <- as.factor(test$set)

ggplot(train, aes(x=num_char, y=grade, colour=set)) +
   geom_jitter() +
   facet_grid(~set) +
   geom_smooth(method=lm) +
   ggtitle("Goodness of fit for linear model") 


# Build and validate models





# Parameters

set.seed(3249872)
algos <- c("SVM", "RF")
nfold <- 5
mean_accuracy <- rep(NA, length(algos))
algos.cv <- data.frame(algos, mean_accuracy)



model.boruta <- Boruta(grade ~ spelling_errors + num_char + num_exclamation + num_keyword, data=train, doTrace=2)
model.boruta$finalDecision

# Support Vector Machine

svm.model <- svm(formula=spelling_errors + num_char + num_exclamation + num_keyword ~ grade, data=train, cross=nfold)

svm.accuracy <- svm.model$tot.MSE

predict.svm <- predict(svm.model, newData=test)

# Generalized Boosted Regression 

model.gbm <- gbm(formula=num_char + num_keyword ~ grade, data=train, distribution="gaussian")

predict.gbm <- predict(model.gbm, test)

# Random Forest

rf.model <- rfcv(data.frame(train$spelling_errors, train$num_char, train$num_keyword, train$num_exclamation),
                 train$grade, cv.fold=nfold)

rf.accuracy <- mean(rf.model$error.cv)


# Term Document Matrix

container.train <- create_container(matrix.train,train$grade,trainSize=1:nrow(train), virgin=FALSE)
container.test <- create_container(matrix.test,test$grade,testSize=1:nrow(test), virgin=FALSE)

models <- train_models(container.train, algorithms=algos)

for (i in 1:nrow(algos.cv)) {
   algos.cv[[2]][i] <- cross_validate(container.train,nfold,algorithm=algos.cv[[1]][i])$meanAccuracy
}

results <- classify_models(container.test, models)
score_summary <- create_scoreSummary(container.train, results)



## Test using quadratic weighted kappa from https://raw.github.com/benhamner/ASAP-AES/master/Evaluation_Metrics/R/quadratic_weighted_kappa.R

ScoreQuadraticWeightedKappa = function (rater.a , rater.b, 
                                        min.rating,
                                        max.rating) {
   
   if (missing(min.rating)) {
      min.rating = min(min(rater.a),min(rater.b))
   }
   if (missing(max.rating)) {
      max.rating = max(max(rater.a),max(rater.b))
   }
   
   rater.a = factor(rater.a, levels=min.rating:max.rating)
   rater.b = factor(rater.b, levels=min.rating:max.rating)
   
   #pairwise frequencies
   confusion.mat = table(data.frame(rater.a, rater.b))
   confusion.mat = confusion.mat / sum(confusion.mat)
   
   #get expected pairwise frequencies under independence
   histogram.a = table(rater.a) / length(table(rater.a))
   histogram.b = table(rater.b) / length(table(rater.b))
   expected.mat = histogram.a %*% t(histogram.b)
   expected.mat = expected.mat / sum(expected.mat)
   
   #get weights
   labels = as.numeric( as.vector (names(table(rater.a))))
   weights = outer(labels, labels, FUN = function(x,y) (x-y)^2 )
   
   #calculate kappa
   kappa = 1 - sum(weights*confusion.mat)/sum(weights*expected.mat)
   kappa
}

MeanQuadraticWeightedKappa = function (kappas, weights) {
   
   if (missing(weights)) {
      weights = rep(1, length(kappas))
   } else {
      weights = weights / mean(weights)
   }
   
   max999 <- function(x) sign(x)*min(0.999,abs(x))
   min001 <- function(x) sign(x)*max(0.001,abs(x))
   kappas = sapply(kappas, max999)
   kappas = sapply(kappas, min001)
   
   r2z = function(x) 0.5*log((1+x)/(1-x))
   z2r = function(x) (exp(2*x)-1) / (exp(2*x)+1)
   kappas = sapply(kappas, r2z)
   kappas = kappas * weights
   kappas = mean(kappas)
   kappas = z2r(kappas)
   kappas
}


## Write out results

submission1 <- data.frame(test$id, test$set, sampleSub$weight, score_summary$BEST_LABEL)
names(submission1) <- names(sampleSub)

write.table(submission1, file="submission1.csv", , sep=",", row.names=FALSE, na="NA", eol="\n", quote=FALSE)

