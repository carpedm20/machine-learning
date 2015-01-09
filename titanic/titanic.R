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

###################
# Add new feature
###################

combined$FamilySize <- combined$SibSp + combined$Parch + 1

combined$Surname <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combined$FamilyID <- paste(as.character(combined$FamilySize), combined$Surname, sep="")
combined$FamilyID[combined$FamilySize <= 2] <- 'Small'

combined$FamilyID2 <- combined$FamilyID
combined$FamilyID2 <- as.character(combined$FamilyID2)
combined$FamilyID2[combined$FamilySize <= 3] <- 'Small'
combined$FamilyID2 <- factor(combined$FamilyID2)

table.embarked = table(combined[combined$Embarked!="",]$Embarked)
combined[combined$Embarked=="",]$Embarked = names(table.embarked)[which.max(table.embarked)]

###################
# Devide combined 
###################

train = combined[combined$PassengerId %in% train$PassengerId,]
test = combined[combined$PassengerId %in% test$PassengerId,]

formula = Survived ~ Pclass + Sex + Age + Fare + Embarked
#formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize + FamilyID2

forest = randomForest(formula, data=train, ntree=5000, maxnodes=8)
test$Survived = predict(forest, newdata=test)

dict = c('Age'=0,'Embarked'=0,'Fare'=0,'Parch'=0,'Pclass'=0,'Sex'=0,'SibSp'=0)

for (i in 1:5000) {
  tree = getTree(forest, i, labelVar=TRUE)
  label=tree[1,][,3]
  dict[label]=dict[label]+1
}

print(dict)
print(getTree(forest, 1, labelVar=TRUE))

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

