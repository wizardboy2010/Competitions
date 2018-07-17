setwd("~/Desktop/ml/competitions/open iit")
library(dummies)
df=read.csv("Combined trail.csv")

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
  l[[c]]=dummy.data.frame(e[[c]][,1:s],sep = ",")
  l[[c]]=scale(l[[c]])
  l[[c]]=as.data.frame(l[[c]])
  c=c+1
}

accuracy = list()
clust = list()

#cookware
temp_diff = c()
for(i in 1:nrow(l[[1]])){
  if(e[[1]][i,4]<=mean(e[[1]]$Sale,na.rm = TRUE)){
    e[[1]][i,j]=1
  }
  if(e[[1]][i,4]>mean(e[[1]]$Sale,na.rm = TRUE)){
    e[[1]][i,j]=2
  }
}
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
for(i in 1:nrow(l[[2]])){
  if(e[[2]][i,4]<=mean(e[[2]]$Sale,na.rm = TRUE)){
    e[[2]][i,j]=1
  }
  if(e[[2]][i,4]>mean(e[[2]]$Sale,na.rm = TRUE)){
    e[[2]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[3]])){
  if(e[[3]][i,4]<=mean(e[[3]]$Sale,na.rm = TRUE)){
    e[[3]][i,j]=1
  }
  if(e[[3]][i,4]>mean(e[[3]]$Sale,na.rm = TRUE)){
    e[[3]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[4]])){
  if(e[[4]][i,4]<=mean(e[[4]]$Sale,na.rm = TRUE)){
    e[[4]][i,j]=1
  }
  if(e[[4]][i,4]>mean(e[[4]]$Sale,na.rm = TRUE)){
    e[[4]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[5]])){
  if(e[[5]][i,4]<=mean(e[[5]]$Sale,na.rm = TRUE)){
    e[[5]][i,j]=1
  }
  if(e[[5]][i,4]>mean(e[[5]]$Sale,na.rm = TRUE)){
    e[[5]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[6]])){
  if(e[[6]][i,4]<=mean(e[[6]]$Sale,na.rm = TRUE)){
    e[[6]][i,j]=1
  }
  if(e[[6]][i,4]>mean(e[[6]]$Sale,na.rm = TRUE)){
    e[[6]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[7]])){
  if(e[[7]][i,4]<=mean(e[[7]]$Sale,na.rm = TRUE)){
    e[[7]][i,j]=1
  }
  if(e[[7]][i,4]>mean(e[[7]]$Sale,na.rm = TRUE)){
    e[[7]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[8]])){
  if(e[[8]][i,4]<=mean(e[[8]]$Sale,na.rm = TRUE)){
    e[[8]][i,j]=1
  }
  if(e[[8]][i,4]>mean(e[[8]]$Sale,na.rm = TRUE)){
    e[[8]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[9]])){
  if(e[[9]][i,4]<=mean(e[[9]]$Sale,na.rm = TRUE)){
    e[[9]][i,j]=1
  }
  if(e[[9]][i,4]>mean(e[[9]]$Sale,na.rm = TRUE)){
    e[[9]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[10]])){
  if(e[[10]][i,4]<=mean(e[[10]]$Sale,na.rm = TRUE)){
    e[[10]][i,j]=1
  }
  if(e[[10]][i,4]>mean(e[[10]]$Sale,na.rm = TRUE)){
    e[[10]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[11]])){
  if(e[[11]][i,4]<=mean(e[[11]]$Sale,na.rm = TRUE)){
    e[[11]][i,j]=1
  }
  if(e[[11]][i,4]>mean(e[[11]]$Sale,na.rm = TRUE)){
    e[[11]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[12]])){
  if(e[[12]][i,4]<=mean(e[[12]]$Sale,na.rm = TRUE)){
    e[[12]][i,j]=1
  }
  if(e[[12]][i,4]>mean(e[[12]]$Sale,na.rm = TRUE)){
    e[[12]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[13]])){
  if(e[[13]][i,4]<=mean(e[[13]]$Sale,na.rm = TRUE)){
    e[[13]][i,j]=1
  }
  if(e[[13]][i,4]>mean(e[[13]]$Sale,na.rm = TRUE)){
    e[[13]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[14]])){
  if(e[[14]][i,4]<=mean(e[[14]]$Sale,na.rm = TRUE)){
    e[[14]][i,j]=1
  }
  if(e[[14]][i,4]>mean(e[[14]]$Sale,na.rm = TRUE)){
    e[[14]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[15]])){
  if(e[[15]][i,4]<=mean(e[[15]]$Sale,na.rm = TRUE)){
    e[[15]][i,j]=1
  }
  if(e[[15]][i,4]>mean(e[[15]]$Sale,na.rm = TRUE)){
    e[[15]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[16]])){
  if(e[[16]][i,4]<=mean(e[[16]]$Sale,na.rm = TRUE)){
    e[[16]][i,j]=1
  }
  if(e[[16]][i,4]>mean(e[[16]]$Sale,na.rm = TRUE)){
    e[[16]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[17]])){
  if(e[[17]][i,4]<=mean(e[[17]]$Sale,na.rm = TRUE)){
    e[[17]][i,j]=1
  }
  if(e[[17]][i,4]>mean(e[[17]]$Sale,na.rm = TRUE)){
    e[[17]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[18]])){
  if(e[[18]][i,4]<=mean(e[[18]]$Sale,na.rm = TRUE)){
    e[[18]][i,j]=1
  }
  if(e[[18]][i,4]>mean(e[[18]]$Sale,na.rm = TRUE)){
    e[[18]][i,j]=2
  } 
}
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
for(i in 1:nrow(l[[19]])){
  if(e[[19]][i,4]<=mean(e[[19]]$Sale,na.rm = TRUE)){
    e[[19]][i,j]=1
  }
  if(e[[19]][i,4]>mean(e[[19]]$Sale,na.rm = TRUE)){
    e[[19]][i,j]=2
  } 
}
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