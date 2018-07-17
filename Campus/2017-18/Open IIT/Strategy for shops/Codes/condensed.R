setwd("~/Desktop/ml/competitions/open iit")
df=read.csv("Combined trail.csv")

library(dummies)
library(cluster)

s = length(df)
j = s+1

#functions needed
rotate <- function(x) t(apply(x, 2, rev))

acc <- function(a,b){
  temp = table(a,b)
  tot = max(sum(diag(temp)), sum(diag(rotate(temp))))
  accuracy = tot/sum(temp)
  return(accuracy)
}

#making Data frames
e=list()            #list of df per SKU
l=list()            #list of scaled df per SKU
c=1
for(i in levels(df$Category.of.SKU)){
  e[[c]]=df[which(df$Category.of.SKU==i),]
  l[[c]]=dummy.data.frame(e[[c]],sep = ",")
  l[[c]]=scale(l[[c]])
  l[[c]]=as.data.frame(l[[c]])
  c=c+1
}

accuracy = list()
clust = list()

for(p in 1:19){
  for(i in 1:nrow(l[[p]])){
    if(e[[p]][i,5] == 'City'){
      e[[p]][i,j]=1
    }
    if(e[[p]][i,5] == 'Town'){
      e[[p]][i,j]=2
    }
  }
  temp_diff = c()
  for(i in 1:100){
    k=kmeans(l[[p]][,2:ncol(l[[p]])],centers=2,iter.max = 100)
    temp_diff = c(temp_diff, abs(diff(k$size)))
    if (min(temp_diff) == abs(diff(k$size))){
      temp_clust = k
      temp_acc = acc(e[[p]][,j],k$cluster)
    }
  }
  accuracy = append(accuracy, temp_acc)
  clust = append(clust, temp_clust)
}

total_accuracy = mean(unlist(accuracy))