library(lubridate)
library(reshape2)
library(plyr)
library(dplyr)
library(ggplot2)
input_data_path<-"D:/Other Work/Sam/CHI/Data/Input Data"
output_data_path<-"D:/Other Work/Sam/CHI/Data/Processed Data"
setwd(input_data_path)
#app_activity<-read.csv("activity.csv",stringsAsFactors = FALSE)
esm_data<-read.csv("surveys.csv",stringsAsFactors = FALSE)
load("user_id.RData")
esm_data$Dt<-format(as.POSIXct(esm_data$time/1000, origin = "1970-01-01",tz=Sys.timezone()),format="%Y-%m-%d %H:%M:%S")
esm_data$hour<-hour(esm_data$Dt)
esm_data$Date<-format(as.Date(esm_data$Dt),"%Y-%m-%d",tz=Sys.timezone() )
Response_data<-esm_data[,c("code","Date","hour","valence","arousal","interruption")]
col_num<-ncol(Response_data)

for(i in 4:col_num)
{
  Response_data[Response_data[,i]%in%c(1,2),i]<-1
  Response_data[Response_data[,i]%in%c(4,5),i]<-5
}

for (u in 1:length(user_id)) {
  
  setwd("D:/Other Work/Sam/CHI/Data/Processed Data")
  load(paste(user_id[u],"_feature_set_hourly.RData"))
  response_per_user<-Response_data[Response_data$code==user_id[u],-1]
  response_per_user$hour<-response_per_user$hour-1
  response_per_user$Date[response_per_user$hour==-1]<-format((as.Date(response_per_user$Date[response_per_user$hour==-1])-days(1)),"%Y-%m-%d",tz=Sys.timezone())
  response_per_user$hour[response_per_user$hour==-1]<-23
  user_feature<-join(user_feature,response_per_user,by=c("Date","hour"))
  save(user_feature,file=paste(user_id[u],"_feature_set_complete_hourly.RData"))
}

esm_count<-Response_data%>%
  group_by(code)%>%
  summarise(total_day=length(Date))

ggplot(esm_count,aes(x=code,y=total_day))+geom_bar(stat="identity", width=0.8,fill="darkblue")+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                                                                         panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x=element_text(angle = 90))+xlab("Use Id")+ylab("Day")
