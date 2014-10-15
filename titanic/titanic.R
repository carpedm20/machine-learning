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

help(formula)
help(predict.randomForest)

##########################
# Functions for plotting
##########################

getConds<-function(tree){
  conds<-list()
  id.leafs<-which(tree$status==-1)
    j<-0
    for(i in id.leafs){
    j<-j+1
    prevConds<-prevCond(tree,i)
    conds[[j]]<-prevConds$cond
    while(prevConds$id>1){
      prevConds<-prevCond(tree,prevConds$id)
      conds[[j]]<-paste(conds[[j]]," & ",prevConds$cond)
      if(prevConds$id==1){
      conds[[j]]<-paste(conds[[j]]," => ",tree$prediction[i])
        break()
      }
    }

  }

  return(conds)
}

prevCond<-function(tree,i){
  if(i %in% tree$right_daughter){
    id<-which(tree$right_daughter==i)
    cond<-paste(tree$split_var[id],">",tree$split_point[id])
    }
    if(i %in% tree$left_daughter){
    id<-which(tree$left_daughter==i)
    cond<-paste(tree$split_var[id],"<",tree$split_point[id])
  }

  return(list(cond=cond,id=id))
}

collapse<-function(x){
  x<-sub(" ","_",x)

  return(x)
}

tree<-getTree(forest, k=1, labelVar=TRUE)
colnames(tree)<-sapply(colnames(tree),collapse)
rules<-getConds(tree)

#print(rules)

##########################
# Make a submission file 
##########################

submit = test[c("PassengerId", "Survived")]
write.csv(submit, file="data/test_submit.csv", row.names=FALSE)

