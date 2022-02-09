# Credit Card Fraud Detection using  Machine Learning Techniques in R

# Install the required packages and import the libraries

install.packages("tidyverse")
install.packages("dplyr")         
install.packages("ggplot2")       
install.packages("caret")         
install.packages("data.table")
install.packages("plotly")
install.packages("corrplot")      
install.packages("kernlab")
install.packages("e1071")
install.packages("class")
install.packages("randomForest")
install.packages("pROC")


library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(data.table)
library(plotly)
library(corrplot)
library(kernlab)
library(e1071)
library(class)
library(randomForest)
library(pROC)


# Load the dataset
credit <- read.csv("C:/Users/b1202001/OneDrive - Teesside University/Data  Science Foundations/MY ICA/creditcard.csv", sep =',', header = TRUE)

# Data Description
# View the structure of the dataset
str(credit)
# View the summary
summary(credit)
# View the head and tail
head(credit, 5)
tail(credit, 5)

# Attributes of the dataset
names(credit)

#check for null and N/A values
sum(is.na(credit))
sum(is.null(credit))

# Summary of Amount
summary(credit$Amount)
sd(credit$Amount)
var(credit$Amount)

# Data Exploration

# Visualize the class of fraud and non-fraud transactions
ggplot(credit, aes(x = Class, fill = Class)) +
  geom_bar() +
  ggtitle("Class Distribution")

# Determine the proportion of fraud & non-fraud transactions
# 0 = Non-fraud and 1 = Fraud
table(credit$Class)
round(prop.table(table(credit$Class))*100, 1) # shows an imbalanced data set

# Distribution of transaction time and amount
ggplot(credit, aes(Time)) +
  scale_x_continuous() +
  geom_density(fill = 'Green', alpha = 0.2) +
  labs(title = "Distribution of Time")
  
# Distribution of Amount
ggplot(credit, aes(Amount)) +
  scale_x_continuous() +
  geom_density(fill = 'Blue', alpha = 0.2) +
  labs(title = "Distribution of Amount") +
  xlim(0,500)


credit %>% 
  filter(Amount < 10, Class == 1) %>%
  ggplot(aes(x = Amount, fill = Class)) +
  geom_histogram() + 
  ggtitle('Count of Fraud under 10')

credit %>% 
  filter(Amount >= 10, Class == 1) %>%
  ggplot(aes(x = Amount, fill = Class)) +
  geom_histogram() + 
  ggtitle('Count of Fraud over 10 Amount')


# Correlation Plot
credit$Class <- as.numeric(credit$Class)
M <- cor(credit)
corrplot(M, method = 'circle')
# The correlation plot shows that the amount  attribute is insignificant
# This can be due to the imbalanced dataset or encrypted data

credit <- select(credit, -Time)   # removing time from the data



# Data Modeling
# The data is imbalanced and this is common in fraud detection. This factor would affect the prediction of the model
# This would be balanced using the sampling method 'sample'
# We first split the data set into their unique fraud and non-fraud transactions

sample_credit = split(credit, credit$Class)
non_f = sample_credit$`0`
fraud = sample_credit$`1`

non_f = non_f[sample(nrow(non_f), 492),]
sampled_credit = rbind(fraud,non_f)
sampled_credit = sampled_credit[sample(nrow(sampled_credit)),]
table(sampled_credit$Class)

# Visualisation of new data distribution

ggplot(sampled_credit, aes(x = Class,  fill = Class)) +
  geom_bar() +
  theme(text = element_text(size=10)) +
  ggtitle("New Data Distribution")

# New correlation plot
# Variables have become more correlated
sampled_credit$Class <- as.numeric(sampled_credit$Class)
M <- cor(sampled_credit)
corrplot(M, method = 'circle')

# Distribution of amount with time
ggplot(credit, aes(x = Amount, fill = Class)) +
  geom_density(alpha = 0.33) +
  scale_x_continuous(limits = c(0, 250), breaks = seq(0, 250, 50)) +
  labs(title = "Distribution of Class with Amount",
       x = "Amount",
       y = "Density",
       col = "Class") +
  scale_fill_discrete(labels = c("Not Fraud, Fraud")) +
  theme_minimal() +
  theme(plot.background = element_rect("cornsilk2"),
        panel.background = element_rect("khaki2"),
        plot.title = element_text(face='bold', color = 'skyblue', 
                                  hjust = 0.5, size = 12))


sampled_credit$Class = as.factor(sampled_credit$Class)

# Split the dataset into the training and testing sets
intrain <- createDataPartition(y = sampled_credit$Class, p= 0.7, list = FALSE)
training <- sampled_credit[intrain,]
testing <- sampled_credit[-intrain,]

dim(training)
dim(testing)

# Use the trainControl() method to train our data on different algorithms
trainctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(Class~., data = training, method = "svmLinear", trControl = trainctrl, preProcess = c("center", "scale"), tuneLength = 10)
svm_Linear


# Test prediction
test_pred <- predict(svm_Linear, newdata = testing)
test_pred

# Use confusion matrix to predict the accuracy of our model
SVM_CF <- confusionMatrix(table(test_pred, testing$Class))
SVM_CF

# ROC of SVM
roc_SVM <- roc(as.numeric(test_pred), as.numeric(testing$Class))
plot(roc_SVM)
auc(roc_SVM)

# Use of KNN algorithm to build the model
# Finding the number of observation
NROW(training)

#Creating seperate dataframe for 'Class' feature which is our target.
train.credit <- sampled_credit[intrain,30]
test.credit <-sampled_credit[-intrain,30]

knn.27 <- knn(train=training, test=testing, cl=train.credit, k = 27)
knn.28 <- knn(train=training, test=testing, cl=train.credit, k= 28)

#Calculate the proportion of correct classification for k = 27, 28
ACC.27 <- 100 * sum(test.credit == knn.27)/NROW(test.credit)
ACC.28 <- 100 * sum(test.credit == knn.28)/NROW(test.credit)

ACC.27
ACC.28

# Check prediction against actual value in tabular form for k=27 and 28
table(knn.27 ,test.credit)
table(knn.28 ,test.credit)

# Use confusionmatrix with KNN to check the accuracy
KNN_CF <- confusionMatrix(table(knn.27 ,test.credit))
KNN_CF


#Optimization
i=1
k.optm=1
for (i in 1:28){ knn.mod <- knn(train=training, test=testing, cl=train.credit, k=i)
k.optm[i] <- 100 * sum(test.credit == knn.mod)/NROW(test.credit)
k=i
cat(k,'=',k.optm[i],'')}

# Graphically represent the accuracy
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

# ROC for KNN
knn_roc <- roc(as.numeric(knn.27),as.numeric(test.credit))
plot(knn_roc)
auc(knn_roc)

# Random Forest
randforest_model <- randomForest(Class~., data = training, 
                                 type = "classification", ntree = 100, mtry = 1)
randforest_model

# Testing
randforest_predict <- predict(randforest_model, testing, type = "class")
randforest_CF <- confusionMatrix(data = randforest_predict, testing$Class)
randforest_CF

# ROC for Random Forest
roc_randforest <- roc(testing$Class,
    predict(randforest_model, newdata = testing, type  = 'prob')[,2],
    plot=T)
roc_randforest
auc(roc_randforest)


# Four fold plot of all the models

fourfoldplot(SVM_CF$table, 
             main = paste('SVM(', round(SVM_CF$overall[1]*100), '%)', sep = ''))
             
fourfoldplot(KNN_CF$table, 
             main = paste('KNN(', round(KNN_CF$overall[1]*100), '%)', sep = ''))

fourfoldplot(randforest_CF$table,
             main = paste("RForest(", round(randforest_CF$overall[1]*100), "%)", sep = ""))














