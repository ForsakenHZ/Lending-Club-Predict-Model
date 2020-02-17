
data=read.csv("Loan_data_part_I.csv")
data$loan_status=ifelse(data$loan_status=="Fail",1,0)
data$loan_status=factor(data$loan_status)
outcome=data$loan_status
data$home_ownership=factor(data$home_ownership)
data$purpose=factor(data$purpose)
data$sub_grade=factor(data$sub_grade)
data$verification_status=factor(data$verification_status)


library(MASS)
library(randomForest)
library(class)
library(caret)
set.seed(103)


#Q1
trctrl=trainControl(method="cv", number=10)
#####################
######LDA model######
#####################

fit=train(loan_status~., data=data, method="lda", preProcess =c("center","scale"), trControl = trctrl)
result_1=prop.table(confusionMatrix(fit)$table,2)

#####################
###Random Forest#####
#####################

fit=train(loan_status~., data=data, method="rf", ntree=100, preProcess =c("center","scale"), trControl = trctrl)
result_2=prop.table(confusionMatrix(fit)$table,2)

#####################
###50-Nearest########
#####################

process=preProcess(data, methods=c("center","scale"))
normdata=predict(process, data)
idx=createFolds(normdata$loan_status,k=10)
d=c()
for(i in seq(idx)){
  pred=knn(data.matrix(normdata[ -idx[[i]],-1 ]), data.matrix(normdata[ idx[[i]],-1 ]), data.matrix(normdata[ -idx[[i]], 1]), k=50)
  d=append(d,pred)
}
conf.table=table(d,normdata[unlist(idx),1])
result_3=prop.table(conf.table,2)  

#2a
###########################################################
##Try methods above with resampling########################
###########################################################
trctrl2=trainControl(method="cv", number=10, sampling = "down")
#####################
##LDA model##########
#####################

fit=train(loan_status~., data=data, method="lda", preProcess =c("center","scale"), trControl = trctrl2)
result_4=prop.table(confusionMatrix(fit)$table,2)

#####################
##Random Forest######
#####################

fit=train(loan_status~., data=data, method="rf", ntree=100, preProcess =c("center","scale"), trControl = trctrl2)
result_5=prop.table(confusionMatrix(fit,reference = data$loan_status)$table,2)

#####################
##50-Nearest#########
#####################

d=c()
for(i in seq(idx)){
  train.set=downSample(normdata[-idx[[i]], ],normdata[-idx[[i]], 1])[,-37]
  test.set=normdata[idx[[i]], ]
  pred=knn(data.matrix(train.set[, -1]), data.matrix(test.set[, -1]), data.matrix(train.set[, 1]), k=50)
  d=append(d,pred)
}
conf.table=table(d,normdata[unlist(idx), 1])
result_6=prop.table(conf.table, 2) 

###########################
#2b method 1#Don't run this code

trctrl=trainControl(method="cv", number=10)
set.seed(1)
fit=train(loan_status~int_rate+loan_amnt+installment+emp_length+home_ownership+annual_inc+num_accts_ever_120_pd+avg_cur_bal+term+revol_util, data=data, method="glm",family = "binomial",preProcess =c("center","scale"), trControl = trctrl, metric = "Accuracy")
confusionMatrix(fit)

###########################
#method 2#Don't run this code

ks <- 1:60
idx=createFolds(newdata$loan_status,k=10)
res=sapply(ks, function(k) {
  ##try out each version of k from 1 to 60
  res.k =sapply(seq_along(idx), function(i) {
    ##loop over each of the 10 cross-validation folds
    ##predict the held-out samples using k nearest neighbors
    pred = knn(data.matrix(newdata[ -idx[[i]], -1]),
               data.matrix(newdata[ idx[[i]], -1]),
               data.matrix(newdata[ -idx[[i]], 1]),k=k)
    ##the ratio of misclassified samples
    mean(data.matrix(newdata[idx[[i]], 1] )!= pred)
  })
  ##average over the 10 folds
  mean(res.k)
})

###################################
#3b
#logistic model
fit=train(loan_status~int_rate+loan_amnt+installment+emp_length+home_ownership+annual_inc+num_accts_ever_120_pd+avg_cur_bal+term+revol_util, data=data, method="glm",family="binomial",preProcess =c("center","scale"), trControl = trctrl2)
result_7=prop.table(confusionMatrix(fit)$table,2)
#randomforest
fit=train(loan_status~int_rate+loan_amnt+installment+emp_length+home_ownership+annual_inc+num_accts_ever_120_pd+avg_cur_bal+term+revol_util, data=data, method="rf", ntree=100, preProcess =c("center","scale"), trControl = trctrl2)
result_8=prop.table(confusionMatrix(fit)$table,2)
#50NN
d=c()
for(i in seq(idx)){
  train.set=downSample(normdata[-idx[[i]], c(1,2,3,4,6,7,8,13,14,19,30)],normdata[-idx[[i]],1])[,-12]
  test.set=normdata[idx[[i]], c(1,2,3,4,6,7,8,13,14,19,30)]
  pred=knn(data.matrix(train.set[,-1]), data.matrix(test.set[,-1]), data.matrix(train.set[, 1]), k=50)
  d=append(d,pred)
}
conf.table=table(d,normdata[unlist(idx),1])
result_9=prop.table(conf.table, 2) 

################################################################
#Q4

train_index = sample(1:nrow(normdata), 0.8 * nrow(normdata))
train_set=downSample(normdata[train_index, ],normdata[train_index, 1])[-37]
test_index = setdiff(1:nrow(normdata), train_index)
test_set=normdata[test_index,]

library(ROCR)
knn.fit=knn(data.matrix(train_set[ ,-1]), data.matrix(test_set[ ,-1]), data.matrix(train_set[ ,1]), k=50, prob=T)
prob=attr(knn.fit,"prob")
prob=2*ifelse(knn.fit=="0",1-prob, prob)-1
knn.predict=prediction(prob, test_set[,1])
knn.auc=as.numeric(ROCR::performance(knn.predict,"auc")@y.values)

lda.fit=lda(loan_status~.,data=train_set)
lda.pred=predict(lda.fit, test_set[,-1])
pred=ROCR::prediction(lda.pred$posterior[,2],test_set$loan_status)
lda.auc=as.numeric(ROCR::performance(pred,"auc")@y.values)

#################################################################
#Q6

library(glmnet)
pred=c()
test=c()
idx=createFolds(data$loan_status,k=10)
for (i in seq(idx)){
  train.set=downSample(data[-idx[[i]], ],data[-idx[[i]],1])[,-37]
  test.set=data[idx[[i]], ]
  model=glmnet(x=data.matrix(train.set[,-1]), y=data.matrix(train.set[, 1]),alpha=1,family="binomial",lambda=0.1)
  p=predict(model, data.matrix(test.set[,-1]), s=0.1, type="class")
  pred=append(pred,p)
  test=append(test,test.set$loan_status)
}
result_10=prop.table(table(pred,test),2)

##################################################################
#Q7

newdata=downSample(normdata,normdata[,1])[,-37]
counts=c()
mymodel=glmnet(x=data.matrix(newdata[,-1]), y=data.matrix(newdata[, 1]), alpha=1, family="binomial") 
for (i in c(1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10)){
  coeff=predict(mymodel,data.matrix(newdata[,-1]),s=i,type="coefficient")
  counti=sum(coeff!=0)-1
  counts=append(counts,counti)
}
  
predict(mymodel,data.matrix(newdata[,-1]), s=0.01225, type="coefficient")
# correct set of attributes:  term, int_rate, annual_inc, dti, fico_range_high, avg_cur_bal, earliest_cr_line, num_il_tl, num_op_rev_tl, home_ownership

##################################################################
#Q11
#Dataset(a)
data_a=read.csv("Loan_data_part_II_a.csv")
data_a$loan_status=ifelse(data_a$loan_status=="Fail",1,0)
data_a$loan_status=factor(data_a$loan_status)
data_a$home_ownership=factor(data_a$home_ownership)
data_a$purpose=factor(data_a$purpose)
data_a$sub_grade=factor(data_a$sub_grade)
data_a$verification_status=factor(data_a$verification_status)
process=preProcess(data_a, methods=c("center","scale"))
normdata_a=predict(process, data_a)
#######################
####Logistic model#####
#######################

library(pROC)
logistic.fit=glm(loan_status~., family = binomial(link="logit"), data=normdata)
log.pred=predict(logistic.fit, normdata_a[,-1])
roc(normdata_a$loan_status~log.pred)

#######################
####LDA model##########
#######################

lda.fit=lda(loan_status~.,data=normdata)
lda.pred=predict(lda.fit, normdata_a[ ,-1])
pred=prediction(lda.pred$posterior[,2],normdata_a$loan_status)
lda.auc=as.numeric(performance(pred,"auc")@y.values)


#######################
####KNN model##########
#######################

knn.pred=knn(data.matrix(normdata[,-1]), data.matrix(normdata_a[,-1]), data.matrix(normdata[,1]),k=50, prob=T)
prob=attr(knn.pred,"prob")
prob=2*ifelse(knn.pred=="0",1-prob, prob)-1
knn.predict=prediction(prob, normdata_a[,1])
knn.auc=as.numeric(ROCR::performance(knn.predict,"auc")@y.values)

#Dataset(b)
data_b=read.csv("Loan_data_part_II_b.csv")
data_b$loan_status=ifelse(data_b$loan_status=="Fail",1,0)
data_b$loan_status=factor(data_b$loan_status)
data_b$home_ownership=factor(data_b$home_ownership)
data_b$purpose=factor(data_b$purpose)
data_b$sub_grade=factor(data_b$sub_grade)
data_b$verification_status=factor(data_b$verification_status)
process=preProcess(data_b, methods=c("center","scale"))
normdata_b=predict(process, data_b)
#######################
####Logistic model#####
#######################

logistic.fit=glm(loan_status~., family = binomial(link="logit"), data=newdata)
log.pred=predict(logistic.fit, normdata_b[,-1])
roc(normdata_b$loan_status~log.pred, plot=T)

#######################
####LDA model##########
#######################

lda.fit=lda(loan_status~.,data=newdata)
lda.pred=predict(lda.fit, normdata_b[ ,-1])
pred=prediction(lda.pred$posterior[,2],normdata_b$loan_status)
lda.auc=as.numeric(performance(pred,"auc")@y.values)

#######################
####KNN model##########
#######################

knn.fit=knn(data.matrix(newdata[,-1]),data.matrix(normdata_b[,-1]),data.matrix(newdata[,1]),k=50,prob=T)
prob=attr(knn.fit,"prob")
prob=2*ifelse(knn.fit=="0",1-prob, prob)-1
knn.predict=prediction(prob, normdata_b[,1])
knn.auc=as.numeric(ROCR::performance(knn.predict,"auc")@y.values)
