'''
z.test = function(a, mu, var){
  zeta = (mean(a) - mu) / (sqrt(var / length(a)))
  return(zeta)
}
rep<-function (x){
   x<-replace(x,x==0,mean(x))
}




rm(list = ls())
data<-read.csv("Final-dataset.csv")
library(dummies)
# m<-mean(data)
da1<-data[,-1]
da4<-data[,c(2,4:7)]
da4[,c(2:5)]<-scale(da4[,c(2:5)])
da3<-data[,-c(1,3,9,10,11,12,13,14,15,16:19,42,24)]
da2[,-1]<-scale(da3[,-1])

#da1[,25:43]<-sapply(data[,25:43],rep)
# da1[,3:42]<-scale(da1[,3:42])
#da1[,25:43]<-scale(da1[,25:43])
da4<-dummy.data.frame(data=da4)
d4<-dist(da4)
hus1<-hclust(d4)
plot(hus1)
cutree(hus1)
d2<-dummy.data.frame(data = da2)
d3<-dist(d2)
hus<-hclust(d3)
plot(hus)
cutree(hus,k=4)
###################
rm(list = ls())
data<-read.csv("Final-dataset.csv",header = TRUE)
#data2<-read.csv("Final-dataset.csv",header = FALSE)
#data3=data2[-1,]
library(dummies)
da4<-data[,c(1,4:7,8,16,17,20,21,22,26,29,31:35,40:42)]
da4[,c(2:6)]<-scale(da4[,c(2:6)])
da4<-dummy.data.frame(data=da4)
d4<-dist(da4)
hus1<-hclust(d4)
plot(hus1)
'''

setwd("~/Desktop/ml/competitions/open iit")
rm(list = ls())
data<-read.csv("dataset.csv")
library(dummies)
da4<-data[,c(29,30,31,36,39,22,40,27)]
da4<-as.data.frame(scale(da4))

#da4<-dummy.data.frame(data=da4)
d4<-dist(da4)
hus1<-hclust(d4)
plot(hus1)

######################RANDOM_FOREST###########

'''

rm(list = ls())
library(randomForest)
data<-read.csv("Final-dataset.csv",header = TRUE)
data<-data[,-c(1,3,39,31,36,30,29,22,40)]
library(dummies)
data<-dummy.data.frame(data)
data<-data[,-1]
data[,1]<-as.factor(data[,1])
library(caTools)
set.seed(123)
split <- sample.split(data, SplitRatio = 0.75)
datrain<-na.omit(subset(data,split==TRUE))
datest<-na.omit(subset(data,split==FALSE))

model_1=randomForest(City.TypeTown~.,data=datrain,ntree=7,
                     nodesize=15,maxnodes=6,importance=TRUE,
                     na.action = na.exclude)
pre1<-predict(model_1)
pre2<-predict(model_1,datest)
tab1<-table(pre1,datrain$City.TypeTown)
tab2<-table(pre2,datest$City.TypeTown)

accuracy1<-sum(diag(tab1))/sum(tab1)
accuracy2<-sum(diag(tab2))/sum(tab2)
varImpPlot(model_1)

'''

ana1<-da4[16:18,]
ana1[4,-1]=colMeans(da4[which(da4$City.Type=="City"),-1])

