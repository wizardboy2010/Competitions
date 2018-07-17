setwd("~/Desktop/ml/competitions/open iit")
library(dummies)
df=read.csv("Combined trail.csv")
#x = df$City.Type.x

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
  l[[c]]=dummy.data.frame(e[[c]][,c(5)],sep = ",")
  l[[c]]=scale(l[[c]])
  l[[c]]=as.data.frame(l[[c]])
  c=c+1
}
for(p in 1:19){
  for(i in 1:nrow(l[[p]])){
    if(e[[p]][i,5] == 'City'){
      e[[p]][i,j]=1
    }
    if(e[[p]][i,5] == 'Town'){
      e[[p]][i,j]=2
    }
  }
}

accuracy = list()
clust = list()

#cookware
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[1]][,2:ncol(l[[1]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[1]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#crockery
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[2]][,2:ncol(l[[2]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[2]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#electronics
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[3]][,2:ncol(l[[3]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[3]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#farm fresh
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[4]][,2:ncol(l[[4]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[4]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#fashion
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[5]][,2:ncol(l[[5]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[5]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#food
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[6]][,2:ncol(l[[6]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[6]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#shoes
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[7]][,2:ncol(l[[7]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[7]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#h.ess
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[8]][,2:ncol(l[[8]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[8]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#f.fash
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[9]][,2:ncol(l[[9]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[9]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)
#health
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[10]][,2:ncol(l[[10]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[10]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#trolley
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[11]][,2:ncol(l[[11]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[11]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Plastics
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[12]][,2:ncol(l[[12]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[12]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Processed Food
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[13]][,2:ncol(l[[13]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[13]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Sports
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[14]][,2:ncol(l[[14]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[14]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Staples
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[15]][,2:ncol(l[[15]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[15]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Stationery
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[16]][,2:ncol(l[[16]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[16]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Toys
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[17]][,2:ncol(l[[17]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[17]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#Utensils
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[18]][,2:ncol(l[[18]])],centers=2,iter.max = 100)
  temp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[18]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

#wellness
temp_diff = c()
for(i in 1:100){
  k=kmeans(l[[19]][,2:ncol(l[[19]])],centers=2,iter.max = 100)
  ttemp_diff = c(temp_diff, abs(diff(k$size)))
  if (min(temp_diff) == abs(diff(k$size))){
    temp_clust = list(k$cluster)
    temp_acc = acc(e[[19]][,j],k$cluster)
  }
}
accuracy = append(accuracy, temp_acc)
clust = append(clust, temp_clust)

total_accuracy = mean(unlist(accuracy))