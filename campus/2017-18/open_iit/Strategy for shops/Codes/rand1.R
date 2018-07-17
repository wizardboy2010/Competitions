setwd("~/Desktop/ml/competitions/open iit")
df=read.csv("tr1.csv", header =FALSE)
'''
library(randomForest)
model<-randomForest(V10~V2+V3+V4+V5+V6,data = df,importance=TRUE)

model$variable.importance
plot(model)
'''
summary(df[which(df$City.Type == 'City'),c(4,7,8)])
summary(df[which(df$City.Type != 'City'),c(4,7,8)])

#cor.test(as.matrix(df[,-c(1,2)]), method = "spearman",conf.level = 0.95)

#cor.test(~, method = "spearman",conf.level = 0.95)
library(hms)
t=tapply(df$V7,df$V2,mean)
m<-mean(df$V7)
df[,9]=unsplit(t,df$V2)
df[,10]=df$V7-m
df[,3:7]=scale(df[,3:7])
library(dummies)
d2<-dummy.data.frame(df[,2:7])
df2<-dist(d2)
model1<-hclust(df2)
#hcd<-as.dendrogram(model1)
#plot(cut(hcd, h = 2)$upper, main = "Upper tree of cut at h=75")
#plot(cut(hcd, h = 75)$lower[[2]], main = "Second branch of lower tree with cut at h=75")
plot(model1)
