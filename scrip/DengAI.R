####### Datathon - DengAI
####### Predict number of dengue cases per week, year and city

#### Install and load packages and set seed ####
if (require('pacman') == 'FALSE') {
  install.packages('pacman')
}

pacman::p_load(readr, 
               caret, 
               dplyr,
               tidyr,
               ggplot2, 
               corrplot,
               Hmisc,
               mice,
               forecast,
               BBmisc)
set.seed(123)

#### Upload files ####
dengueFeaturesTrain <- read_csv("dengue_features_train.csv",
                        col_types = cols(city = col_factor(levels = c("sj",
                                                                      "iq")),
                                         week_start_date = col_date(format = "%Y-%m-%d")))
dengueFeaturesTest <- read_csv("dengue_features_test.csv",
                        col_types = cols(city = col_factor(levels = c("sj",
                                                                      "iq")),
                                         week_start_date = col_date(format = "%Y-%m-%d")))
dengueLabelsTrain <- read_csv("dengue_labels_train.csv",
                              col_types = cols(city = col_factor(levels = c("sj",
                                                                            "iq"))))
submissionFormat <- read_csv("submission_format.csv",
                              col_types = cols(city = col_factor(levels = c("sj",
                                                                            "iq"))))

#### Get to know the data ####
str(dengueFeaturesTest)
summary(dengueFeaturesTest)

str(dengueFeaturesTrain)
summary(dengueFeaturesTrain)

str(dengueLabelsTrain)
summary(dengueLabelsTrain)

#### Pre processing ####
# Create new data with features and labels
trainSet <- cbind(dengueFeaturesTrain, dengueLabelsTrain$total_cases)
colnames(trainSet)[ncol(trainSet)] <- "total_cases"

testSet <- cbind(dengueFeaturesTest, submissionFormat$total_cases)
colnames(testSet)[ncol(testSet)] <- "total_cases"

# Correlation Matrix
trainDummy <- dummyVars(" ~ .",
                        data = trainSet)
trainReady <- data.frame(predict(trainDummy,
                                 newdata = trainSet))
View(trainReady)
corrplot(cor(trainReady,
             use = 'pairwise.complete.obs'),
         type = "upper", 
         order = "hclust", 
         tl.srt = 45)

# Find and remove attributes that are highly correlated
highlyCorrelated <- findCorrelation(cor(trainReady,
                                        use = 'pairwise.complete.obs'),
                                    cutoff=0.75)
colnames(trainReady)[highlyCorrelated]
trainCorrelated <- trainReady[,-highlyCorrelated]
View(trainCorrelated)

# Treat missing values
md.pattern(trainCorrelated)
imputedTrainData <- mice(trainCorrelated, 
                         m=1, 
                         maxit = 50, 
                         method = 'pmm')
View(imputedTrainData)
completeTrainData <- complete(imputedTrainData)
View(completeTrainData)
sum(is.na(completeTrainData))

#### Modeling ####
# Split the data in 75%/25%
index <- createDataPartition(completeTrainData$total_cases,
                             p = 0.75, 
                             list = F)
trainModelSet <- completeTrainData[index,]
testModelSet <- completeTrainData[-index,]

normalizeData <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

normalizeComplete <- normalizeData(completeTrainData)
normalizeComplete$total_cases <- completeTrainData$total_cases
normalizeTrain <- normalizeData(trainModelSet)
normalizeTrain$total_cases <- trainModelSet$total_cases
normalizeTest <- normalizeData(testModelSet)
normalizeTest$total_cases <- testModelSet$total_cases

# Try regression models
models <- c("lm", "rf","knn", "svmLinear", "svmRadial")
compare_model <- c()

for(i in models) {
  model <- train(total_cases ~., 
                 data = normalizeTrain, 
                 method = i)
  predictions <- predict(model, newdata = normalizeTest)
  pred_metric <- postResample(pred = predictions, obs =  normalizeTest$total_cases)
  compare_model <- cbind(compare_model , pred_metric)
}

colnames(compare_model) <- models
compare_model

# Treat test data
testDummy <- dummyVars(" ~ .",
                        data = testSet)
testReady <- data.frame(predict(testDummy,
                                 newdata = testSet))
testCorrelated <- testReady[,-highlyCorrelated]
View(testCorrelated)
imputedTestData <- mice(testCorrelated, 
                         m=1, 
                         maxit = 50, 
                         method = 'pmm')
completeTestData <- complete(imputedTestData)
View(completeTestData)
sum(is.na(completeTestData))
normalizeTestData <- normalizeData(completeTestData)

# Use the choosed model
model <- train(total_cases ~., 
               data = normalizeComplete, 
               method = 'rf')
predictions <- predict(model, newdata = normalizeTestData)
plot(predictions)
submissionFormat$total_cases <- round(predictions)
View(submissionFormat)

write.csv(submissionFormat, file = 'submission.csv')
write.csv(submissionFormat, file = 'submissionNormalize.csv')
write.csv(submissionFormat, file = 'submissionNormalizeP.csv')

#### Time series Forecast ####
completeTS <- ts(completeTrainData$total_cases,
                frequency = 53,
                start = c(1990,18),
                end = c(2008,17))

# Holt Winter
hw <- HoltWinters(completeTS,
                      beta = F,
                      gamma = T)
forecastHwWeek <- forecast(hw, 
                           h=416,
                           level = c(10,25))

# Auto Arima
autoArima <- auto.arima(completeTS)
forecastArimaWeek <- forecast(autoArima,
                                       h=416,
                                       level = c(10,25))

forecastHwDF <- as.data.frame(forecastHwWeek)

submissionFormat$total_cases <- round(forecastHwDF$`Point Forecast`)
write.csv(submissionFormat, file = 'submissionForecast.csv')
