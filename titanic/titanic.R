##########################
# Machine Learning HW #2
# 20111167 Taehoon Kim
##########################

library(randomForest)

classes = c("integer", "factor", "factor", "character", 
            "factor", "numeric", "integer", "integer", 
            "character", "numeric", "character", "factor")

train <- read.csv("data/train.csv",  colClasses=classes)
test <- read.csv("data/test.csv", stringsAsFactors=FALSE)

combined = rbind(train, test)

##########################
# Filling missing values
##########################

age_missing = mean(combined[!is.na(combined$Age),]$Age)
fare_missing = median(combined[!is.na(combined$Fare),]$Fare)

combined[is.na(combined$Age),]$Age = age_missing
combined[is.na(combined$Fare),]$Fare = fare_missing

table.embarked = table(combined[combined$Embarked!="",]$Embarked)
combined[combined$Embarked=="",]$Embarked = names(table.embarked)[which.max(table.embarked)]

###################
# Devide combined 
###################

train = combined[combined$PassengerId %in% train$PassengerId,]
test = combined[combined$PassengerId %in% test$PassengerId,]

formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked

forest = randomForest(formula, data=train)
test$Survived = predict(forest, newdata=test)

submit = test[c("PassengerId", "Survived")]
write.csv(submit, file="data/test_submit.csv", row.names=FALSE)

