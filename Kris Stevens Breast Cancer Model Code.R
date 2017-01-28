
print("Please take the time to install the following packages: ggplot2, e1071, Hmisc, ROCR, caret, pscl")
library(ggplot2)
library(e1071)
library(Hmisc)
library(ROCR)
library(caret)
library(pscl)

#Read in your data, code was copied and pasted directly from assignment.
data<-read.csv("BreastCancerData.txt")

#Change the 999 to NA's
data$Perimeter_Mean[(data$Perimeter_Mean== 999)] <- NA
data$Smoothness_Mean[(data$Smoothness_Mean== 999)] <- NA

#Made Zeros minimal values for log testing
#     These Zeroes exist in the original data set
data$Concavity_Mean[(data$Concavity_Mean==0)]<-0.00001
data$Concave_pts_Mean[(data$Concave_pts_Mean==0)]<-0.00001

##Cleaning all NA via an equation involving the Worst and the StdErr.
#   Due to right skewness of data on this particular data I subtracted two StdErr 
#     then subtracted a random percentage of 3 StdErr
    data$Perimeter_Mean[is.na(data$Perimeter_Mean)] <- 
                data$Perimeter_Worst - 2*data$Perimeter_StdErr - runif(1)*(3*data$Perimeter_StdErr)

#Subtracted one StdErr from the worst, then a random percentage of 4 StdErr in an attempt to give a random 
#value with the intent of placing the data in a range where there is a 95% chance of being close to true mean
    data$Smoothness_Mean[is.na(data$Smoothness_Mean)] <- 
              data$Smoothness_Worst - data$Smoothness_StdErr - runif(1)*(4*data$Smoothness_StdErr)
    data$Texture_Mean[is.na(data$Texture_Mean)] <- 
              data$Texture_Worst - data$Texture_StdErr - runif(1)*(4*data$Texture_StdErr)

#Create a data frame with only means and diagnosis
    use.data <- data[,2:12]   
    
#Create column to identify train and test data
    set.seed(11152008)  #For replication purposes
    rand <- rbinom(nrow(use.data), 1, .7)
    
#Identify training and test cases:
    #Set everything in partitiion to test
    use.data$partition <- "test"
    #set everything in partition with a 1 value in Rand to Train.
    use.data$partition[rand==1] <- "train"  
    # make partition a factor
    use.data$partition <- as.factor(use.data$partition)

## There seems to be a sweet spot for model performance at 10 bins.
    # There is also a sweet spot at 36 where only 16 results are misclassified total of the 400, 
    # However when applied to the test data it is more over fit that the 10 bin model.
          
use.data.NB<-as.data.frame(lapply(use.data[,2:11], cut2, g=10) )        

use.data.NB$Diagnosis<-use.data$Diagnosis
use.data.NB$partition<-use.data$partition

##NAIVE BAYES

#Build the model
model <- naiveBayes(Diagnosis ~ ., data=use.data.NB[use.data.NB$partition=="train",])

#Predictions given by
print("Naive Bayes Predictions")
predict(model, use.data.NB[use.data.NB$partition=="train",])

#Conditioinal Posterior Probabilities Given by
print("probabilities of first 6 rows")
head(predict(model, use.data.NB[use.data.NB$partition=="train",], type = "raw"))

train.pred <- predict(model, use.data.NB[use.data.NB$partition=="train",])
#Confusion Matrix of values
print("Confusion Matrix of Values given by model compared to actual")
table(train.pred, use.data.NB$Diagnosis[use.data.NB$partition=="train"] )

train.table <- table(train.pred, use.data.NB$Diagnosis[use.data.NB$partition=="train"] )

prop.table(train.table)
#Misclassification Rate:
1-sum(diag(prop.table(train.table)))


#Sensitivity & Specificity:
# Sensitivity = P(Y-hat = 1 | Y = 1) = True.Positives / (True.Positives + False.Negatives)
#   and Specificity = P(Y-hat = 0 | Y=0) = True.Negatives / (True.Negatives + False.Positives)

colnames(train.table) <- c("B","M") 
rownames(train.table) <- c("B","M")
#Sensitivity
train.table["B","B"]/(train.table["B","B"] + train.table["M","B"])
#Specificity
train.table["M","M"]/(train.table["M","M"] + train.table["B","M"])
train.table
#Now we try the testing Data

test.pred <- predict(model, use.data.NB[use.data.NB$partition=="test",])
test.table <- table(test.pred, use.data.NB$Diagnosis[use.data.NB$partition=="test"] )
prop.table(test.table)
1-sum(diag(prop.table(test.table)))

test.table
colnames(test.table) <- c("B","M") 
rownames(test.table) <- c("B","M")
#Sensitivity
test.table["B","B"]/(test.table["B","B"] + test.table["M","B"])
#Specificity
test.table["M","M"]/(test.table["M","M"] + test.table["B","M"])

### Logistic Regression Modle
#Create A Dataframe for testing with
use.data.log<-use.data

#Create a dataframe for the training data.
use.data.train<- use.data.log[use.data$partition=="train",]
#remove prediction variable
use.data.train<-use.data.train[,1:(length(use.data.train)-1)]

#Create a data frame for the testing data.
use.data.test<- use.data.log[use.data$partition=="test",]
#remove prediction variable
use.data.test<-use.data.test[,1:(length(use.data.test)-1)]

#Build the model
logre <- train(Diagnosis ~ .,  data=use.data.train, method="glm", family="binomial")

#View results
summary(logre)
print("notice the importance of Texture_mean, Area_mean, and Concave_pts_Mean")

#view coeffecients
format(exp(coef(logre$finalModel)), digits = 5)

#Store Predictions
logre.results.train.prob <- predict(logre,newdata=use.data.train,type='prob')[,1]
logre.results.train <- predict(logre,newdata=use.data.train,type='raw')

#Confusion Matrix of Model from test
confusionMatrix(logre.results.train, use.data.train$Diagnosis)

# Create my own values of B and M based on a probability threshold of my own choosing
logre.results.train.prob <- ifelse (logre.results.train.prob > 0.9,1,0)
plot(roc(Diagnosis~logre.results.train.prob, data=use.data.train))

#Change the values to B and M
logre.results.train.prob <- ifelse (logre.results.train.prob == 1,"B","M")

#Confusion Matrix of Model from test at custom split
print("Confusion Matrix of Model from Logistic Regression at 90/10 split")
confusionMatrix(logre.results.train.prob, use.data.train$Diagnosis)

#Store Predictions Test
logre.results.test.prob <- predict(logre,newdata=use.data.test,type='prob')[,1]
logre.results.test<- predict(logre,newdata=use.data.test,type='raw')

#Confusion Matrix of Model from test at 50/50 split
print("Confusion Matrix of Model from Logistic Regression at 50/50 split")
confusionMatrix(logre.results.test, use.data.test$Diagnosis)

# Create my own values of B and M based on a probability threshold of my own choosing
logre.results.test.prob <- ifelse (logre.results.test.prob > 0.9,1,0)
plot(roc(Diagnosis~logre.results.test.prob, data=use.data.test))

#Change the values to B and M
logre.results.test.prob <- ifelse (logre.results.test.prob == 1,"B","M")

#Confusion Matrix of Model from test at custom split
print("Confusion Matrix of Model from Logistic Regression at 86.5/13.5 split")
confusionMatrix(logre.results.test.prob, use.data.test$Diagnosis)

#With this precentage we do not tell anyone who has cancer that their tumor is not cancerous.
