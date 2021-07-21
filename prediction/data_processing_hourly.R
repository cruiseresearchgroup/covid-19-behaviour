library(reshape2)
library(plyr)
library(dplyr)
library(lubridate)
input_data_path<-"D:/Other Work/Sam/CHI/Data/Input Data"
output_data_path<-"D:/Other Work/Sam/CHI/Data/Processed Data"
setwd(input_data_path)
#app_activity<-read.csv("activity.csv",stringsAsFactors = FALSE)
#survey_data<-read.csv("anon_end_of_day.csv",stringsAsFactors = FALSE)
#app_category<-read.csv("categories.csv",stringsAsFactors = FALSE)
cluster_data<-read.csv("clusternames.csv",stringsAsFactors = FALSE)
cluster_features<-read.csv("features.csv",stringsAsFactors = FALSE)
user_id<-unique(cluster_features$code)

cluster_features$Date<-format(as.Date(cluster_features$from),"%Y-%m-%d",tz=Sys.timezone())
cluster_features$hour<-as.POSIXct(cluster_features$from,format="%Y-%m-%d %H:%M:%S",tz=Sys.timezone())
cluster_features$hour<-hour(cluster_features$hour)
cluster_features$stay_time<-as.numeric(difftime(as.POSIXct(cluster_features$to,format="%Y-%m-%d %H:%M:%S",tz=Sys.timezone()), as.POSIXct(cluster_features$from,format="%Y-%m-%d %H:%M:%S",tz=Sys.timezone()), units = "mins"))
colnames(cluster_data)[1]<-"label"
data_availability<-data.frame(user=user_id,date_count=0,stringsAsFactors = FALSE)

for(u in 1:length(user_id))
{
  per_user_data<-cluster_features[cluster_features$code==user_id[u],]
  feature_data_per_user<-as.data.frame(per_user_data%>%
                                         group_by(label,Date,hour)%>%
                                         summarize(stay_time=sum(stay_time),cluster_freq=length(label)))
  
  feature_data_per_user<-join(cluster_data,feature_data_per_user,by="label")
  
  feature_stay_time<-feature_data_per_user[,c("Date","hour","name","stay_time")]
  feature_stay_time<-as.data.frame(dcast(feature_stay_time,Date+hour~name,fill = 0))
  colnames(feature_stay_time)[3:18]<-paste("time_",colnames(feature_stay_time)[3:18],sep="")
  
  feature_cluster_freq<-feature_data_per_user[,c("Date","hour","name","cluster_freq")]
  feature_cluster_freq<-as.data.frame(dcast(feature_cluster_freq,Date+hour~name,fill = 0))
  colnames(feature_cluster_freq)[3:18]<-paste("freq_",colnames(feature_cluster_freq)[3:18],sep="")
  
  cluster_count<-feature_data_per_user[,c("Date","hour","name")]
  unique_cluster<-cluster_count%>%group_by(Date,hour)%>%summarise(cluster_no=length(name))
  
  user_feature<-as.data.frame(na.omit(join(feature_stay_time,feature_cluster_freq,by=c("Date","hour"))))
  user_feature<-join(user_feature,unique_cluster,by=c("Date","hour"))
  user_feature$user_id<-user_id[u]
  
  data_availability$date_count[data_availability$user==user_id[u]]<-nrow(user_feature)
  setwd(output_data_path)
  save(user_feature,file=paste(user_id[u],"_feature_set_hourly.RData"))
  
  
}

ggplot(data_availability,aes(x=user,y=date_count))+geom_bar(stat="identity", width=0.8,fill="darkblue")+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                                                                               panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x=element_text(angle = 90))+xlab("Use Id")+ylab("Day")


save(user_id,file="user_id.RData")
