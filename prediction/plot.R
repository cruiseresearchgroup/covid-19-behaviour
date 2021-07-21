library(ggplot2)
library(reshape2)
data_path<-"D:/Other Work/Sam/CHI/Data/Processed Data"
setwd(data_path)
load("Q2_acc.RData")
accuracy_result<-na.omit(accuracy_result)
accuracy_result$Users<-paste(1:5,accuracy_result$Users,sep="")
colnames(accuracy_result)[2:4]<-c("SVM","KNN","RF")
accuracy_result<-accuracy_result[-which(names(accuracy_result)=="KNN")]
label_ind<-c("All","Male","Female","Age 18-24","Age >=25")
break_ind<-c("1All","2Male","3Female","4Age 18-24","5Age >25")
for(i in seq(2,5,2))
{
acc_sub_result<-accuracy_result[i:(i+1),] 
plot_data<-melt(acc_sub_result,id.vars = "Users",measure.vars = c("SVM","RF"))
plot_data$variable<-as.character(plot_data$variable)
colnames(plot_data)<-c("User_Group","Classifier","Accuracy")
ggplot(plot_data,aes(x=User_Group,y=Accuracy,fill=Classifier)) + 
  geom_bar(position="dodge", stat="identity", width=0.5) +ylim(0,0.9)+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                                 panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x=element_text(angle = 90,size = 12),axis.title.x=element_text(size = 12), axis.text.y=element_text(size = 12),legend.text = element_text(size = 12),legend.title = element_text(size = 12),axis.title.y = element_text(size = 12))+ scale_x_discrete(breaks=break_ind[i:(i+1)],
                                                                                                                                                                                                         labels=label_ind[i:(i+1)])+ 
scale_fill_manual("Classifier", values = c("KNN" = "orangered3", "RF" = "seagreen4", "SVM" = "deepskyblue4"))+xlab(" ")+ theme(legend.position="top")


ggsave(paste("user_group_",i,"_Q2_acc.png"),width = 3.5, height=3.5)

}
