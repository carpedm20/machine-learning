##########################
# Machine Learning HW #2
# 20111167 Taehoon Kim
##########################

library(randomForest)

train <- read.csv("data/train.csv", header=T, sep=",")
test <- read.csv("data/test.csv", header=T, sep=",")

train.processed <- train
test.processed <- test

##########################
# Filling missing values
##########################

age_missing = median(train.processed$Age, na.rm=T)
fare_missing = median(train.processed$Fare, na.rm=T)

train.processed$Age[is.na(train.processed$Age)] <- age_missing
test.processed$Age[is.na(test.processed$Age)] <- age_missing

test.processed$Fare[is.na(test.processed$Fare)] <- fare_missing

test.processed$Survived <- NA
combined <- rbind(train.processed, test.processed)

combined$Survived=as.factor(combined$Survived)
combined$Pclass=as.factor(combined$Pclass)
combined$Name=as.character(combined$Name)
combined$Ticket=as.character(combined$Ticket)
combined$Cabin=as.character(combined$Cabin)

table.embarked = table(combined[combined$Embarked!="",]$Embarked)
combined[combined$Embarked=="",]$Embarked = names(table.embarked)[which.max(table.embarked)]

###################
# Devide combined 
###################

train = combined[combined$PassengerId %in% train$PassengerId,]
test = combined[combined$PassengerId %in% test$PassengerId,]

formula = Survived ~ Pclass + Sex + Age + Fare + Embarked

forest = randomForest(formula, data=train, ntree=500)
test$Survived = predict(forest, newdata=test)

dict = c('Age'=0,'Embarked'=0,'Fare'=0,'Parch'=0,'Pclass'=0,'Sex'=0,'SibSp'=0)

for (i in 1:500) {
  tree = getTree(forest, i, labelVar=TRUE)
  label=tree[1,][,3]
  dict[label]=dict[label]+1
}

print(dict)

##########################
# Functions for plotting
##########################

#library("party")

#cf <- cforest(formula,
#              data = train,
#              controls = cforest_control(mtry=2, mincriterion=0)) 
#plot(cf)

#pt <- party:::prettytree(cf@ensemble[[1]], names(cf@data@get("input"))) 

#nt <- new("BinaryTree") 
#nt@tree <- pt 
#nt@data <- cf@data 
#nt@responses <- cf@responses 

#plot(nt) 

##########################
# Make a submission file 
##########################

print("Write file")
submit = test[c("PassengerId", "Survived")]
write.csv(submit, file="data/test_submit.csv", row.names=FALSE)

