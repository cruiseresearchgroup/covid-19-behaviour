#library(randomForest)
#library(e1071)
library(caret)
library(MLmetrics)
library(klaR)
data_path<-"D:/Other Work/Sam/CHI/Data/Processed Data"
setwd(data_path)
load("user_id.RData")
user_dem<-read.csv("users.csv",stringsAsFactors = FALSE)
#par(mfrow=c(3,3))
for (u in 1:length(user_id)) {
  
  load(paste(user_id[u],"_feature_set.RData"))
  if(u==1)
  {
    complete_feature_set<-na.omit(user_feature)
  }else
  {
    complete_feature_set<-rbind(complete_feature_set,na.omit(user_feature))
  }
  
  
}

male_id<-user_dem$code[user_dem$gender==1]
female_id<-user_dem$code[user_dem$gender==0]
n_employed_id<-user_dem$code[user_dem$employed=="False"]
employed_id<-user_dem$code[user_dem$employed=="True"]
group1<-user_dem$code[user_dem$age%in%c(18:24)]
group2<-user_dem$code[user_dem$age>24]


target_name<-"Q1"
label_column<-grep(target_name, colnames(complete_feature_set))#target variable
complete_feature_set1<-complete_feature_set[c(complete_feature_set[,label_column]!=3),]

excludecolumn<-c(1,35:41)
excludecolumn<-excludecolumn[!excludecolumn%in%label_column]
accuracy_result<-as.data.frame(matrix(nrow=5,ncol=4))
colnames(accuracy_result)<-c("Users","Acc_SVM","Acc_KNN","Acc_RF")
accuracy_result$Users<-c("All","Male","Female","Age 18-24","Age >25")

for (i  in 1:5) {
  
if(i==1)
{
  feature_set<-complete_feature_set1[,-excludecolumn]
  nm<-"1all"
}else if (i==2)
{
  feature_set<-complete_feature_set1[complete_feature_set1$user_id%in%male_id,-excludecolumn]
  nm<-"2male"
}else if (i==3)
{
  feature_set<-complete_feature_set1[complete_feature_set1$user_id%in%female_id,-excludecolumn]
  nm<-"3female"
}else if (i==4)
{
  feature_set<-complete_feature_set1[complete_feature_set1$user_id%in%group1,-excludecolumn]
  nm<-"4grp1"
}else if (i==5)
{
  feature_set<-complete_feature_set1[complete_feature_set1$user_id%in%group2,-excludecolumn]
  nm<-"5grp2"
}
#complete_feature_set$user_id%in%group4
colnames(feature_set)[ncol(feature_set)]<-"label"
colnames(feature_set)<-gsub(' ',"_",colnames(feature_set))
feature_set$label<-as.factor(feature_set$label)
par(mfrow=c(3,3))
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(7)
fit.svm <- train(label~., data=feature_set, method="svmRadial", metric=metric,trControl=control)
# kNN
set.seed(7)
fit.knn <- train(label~., data=feature_set, method="knn", metric=metric,trControl=control)
# Random Forest
set.seed(7)
#mtry <- sqrt(ncol(feature_set)-1)
#tunegrid <- expand.grid(.mtry=mtry)
fit.rf <- train(label~., data=feature_set, method="rf", metric=metric,trControl=control)
#png(file=paste(nm,"_Q3.png"))
#plot(varImp(fit.rf,scale = TRUE),top=20,xlab="Importance(%)")
#dev.off()

test<-varImp(fit.rf,scale = TRUE)
test<-test$importance
test$feature_name<-rownames(test)
test<-test[order(-test$Overall),]
test$group<-nm
if(i==1)
{
  var_imp<-test[1:20,]
}else
{
  var_imp<-rbind(var_imp,test[1:20,])
}

# collect resamples
results <- resamples(list(SVM=fit.svm, KNN=fit.knn, RF=fit.rf))
results$metrics<-metric
accuracy_result[i,(2:4)]<-unname(colMeans(results$values[,-c(1,3,5,7)],na.rm = TRUE))
save(accuracy_result,file=paste(target_name,"_acc.RData",sep=""))
write.csv(accuracy_result,file=paste(target_name,"_acc.csv",sep=""))
}
#scales <- list(x=list(relation="free"), y=list(relation="free"))
#bwplot(results, scales=scales)
save(var_imp,file=paste(target_name,"_var_imp.RData",sep=""))
write.csv(var_imp,file=paste(target_name,"_var_imp.csv",sep=""))



