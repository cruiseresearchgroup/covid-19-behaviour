library(ggplot2)
library(reshape2)
data_path<-"D:/Other Work/Sam/CHI/Data/Processed Data"
setwd(data_path)
load("valence_var_imp_hourly.RData")
colnames(var_imp)[1]<-"Importance"
var_imp$feature_name[var_imp$feature_name=="cluster_no"]="no of unique app category"

for(i in 1:2)
{
  if(i==1)
  {
    plot_data<-var_imp[var_imp$group%in%c("2male","3female"),]
  }else
  {
    plot_data<-var_imp[var_imp$group%in%c("4grp1","5grp2"),]
  }
  
ggplot(plot_data,aes(x=group,y=feature_name,fill=Importance)) + 
  geom_tile()+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                                     panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x=element_text(angle = 90,size = 14),axis.title.x=element_text(size = 14), axis.text.y=element_text(size = 14),legend.text = element_text(size = 14),legend.title = element_text(size = 14),axis.title.y = element_text(size = 14))+xlab(" ")+ylab("Features")+ theme(legend.position="top")+ scale_x_discrete(breaks=c("1all","2male","3female","4grp1","5grp2"),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       labels=c("All","Male","Female","Age 18-24","Age >=25"))+
  scale_fill_gradient(low = "aquamarine1", high = "blue4",na.value = 'white') 


ggsave(paste("user_group_",i,"_valence_var_imp.png",sep=""),width = 6, height=6)
}