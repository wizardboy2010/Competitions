setwd("~/Desktop/ml/competitions/open iit")
library(dummies)
df=read.csv("Total.csv")

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
  l[[c]]=dummy.data.frame(e[[c]][,1:9],sep = ",")
  l[[c]]=scale(l[[c]])
  l[[c]]=as.data.frame(l[[c]])
  c=c+1
}

accuracy = list()

#cookware
k=kmeans(l[[1]][,2:30],centers=2,iter.max = 100)
k$size
ncol(l[[1]])
for(i in 1:nrow(l[[1]])){
  if(e[[1]][i,4]<=mean(e[[1]]$Sale,na.rm = TRUE)){
    e[[1]][i,10]=1
  }
  if(e[[1]][i,4]>mean(e[[1]]$Sale,na.rm = TRUE)){
    e[[1]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[1]]$V10,k$cluster))

#crockery
k=kmeans(l[[2]][,2:ncol(l[[2]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[2]])){
  if(e[[2]][i,4]<=mean(e[[2]]$Sale,na.rm = TRUE)){
    e[[2]][i,10]=1
  }
  if(e[[2]][i,4]>mean(e[[2]]$Sale,na.rm = TRUE)){
    e[[2]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[2]]$V10,k$cluster))

#electronics
k=kmeans(l[[3]][,2:ncol(l[[3]])],centers=3,iter.max = 100)
k$size
ncol(l[[3]])
for(i in 1:nrow(l[[3]])){
  if(e[[3]][i,4]<=mean(e[[3]]$Sale,na.rm = TRUE)){
    e[[3]][i,10]=1
  }
  if(e[[3]][i,4]>mean(e[[3]]$Sale,na.rm = TRUE)){
    e[[3]][i,10]=2
  } 
}
table(e[[3]]$V10,k$cluster)
accuracy = append(accuracy, acc(e[[3]]$V10,k$cluster))

#farm fresh
k=kmeans(l[[4]][,2:ncol(l[[4]])],centers=2,iter.max = 100)
k$size
ncol(l[[4]])
for(i in 1:nrow(l[[4]])){
  if(e[[4]][i,4]<=mean(e[[4]]$Sale,na.rm = TRUE)){
    e[[4]][i,10]=1
  }
  if(e[[4]][i,4]>mean(e[[4]]$Sale,na.rm = TRUE)){
    e[[4]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[4]]$V10,k$cluster))

#fashion
k=kmeans(l[[5]][,2:ncol(l[[5]])],centers=2,iter.max = 100)
k$size
ncol(l[[5]])
for(i in 1:nrow(l[[5]])){
  if(e[[5]][i,4]<=mean(e[[5]]$Sale,na.rm = TRUE)){
    e[[5]][i,10]=1
  }
  if(e[[5]][i,4]>mean(e[[5]]$Sale,na.rm = TRUE)){
    e[[5]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[5]]$V10,k$cluster))

#food
k=kmeans(l[[6]][,2:ncol(l[[6]])],centers=2,iter.max = 100)
k$size
ncol(l[[6]])
for(i in 1:nrow(l[[6]])){
  if(e[[6]][i,4]<=mean(e[[6]]$Sale,na.rm = TRUE)){
    e[[6]][i,10]=1
  }
  if(e[[6]][i,4]>mean(e[[6]]$Sale,na.rm = TRUE)){
    e[[6]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[6]]$V10,k$cluster))

#shoes
k=kmeans(l[[7]][,2:ncol(l[[7]])],centers=2,iter.max = 100)
k$size
ncol(l[[7]])
for(i in 1:nrow(l[[7]])){
  if(e[[7]][i,4]<=mean(e[[7]]$Sale,na.rm = TRUE)){
    e[[7]][i,10]=1
  }
  if(e[[7]][i,4]>mean(e[[7]]$Sale,na.rm = TRUE)){
    e[[7]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[7]]$V10,k$cluster))

#h.ess
k=kmeans(l[[8]][,2:ncol(l[[8]])],centers=2,iter.max = 100)
k$size
ncol(l[[8]])
for(i in 1:nrow(l[[8]])){
  if(e[[8]][i,4]<=mean(e[[8]]$Sale,na.rm = TRUE)){
    e[[8]][i,10]=1
  }
  if(e[[8]][i,4]>mean(e[[8]]$Sale,na.rm = TRUE)){
    e[[8]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[8]]$V10,k$cluster))

#f.fash
k=kmeans(l[[9]][,2:ncol(l[[9]])],centers=2,iter.max = 100)
k$size
ncol(l[[9]])
for(i in 1:nrow(l[[9]])){
  if(e[[9]][i,4]<=mean(e[[9]]$Sale,na.rm = TRUE)){
    e[[9]][i,10]=1
  }
  if(e[[9]][i,4]>mean(e[[9]]$Sale,na.rm = TRUE)){
    e[[9]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[9]]$V10,k$cluster))

#health
k=kmeans(l[[10]][,2:ncol(l[[10]])],centers=2,iter.max = 100)
k$size
ncol(l[[10]])
for(i in 1:nrow(l[[10]])){
  if(e[[10]][i,4]<=mean(e[[10]]$Sale,na.rm = TRUE)){
    e[[10]][i,10]=1
  }
  if(e[[10]][i,4]>mean(e[[10]]$Sale,na.rm = TRUE)){
    e[[10]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[10]]$V10,k$cluster))

#trolley
k=kmeans(l[[11]][,2:ncol(l[[11]])],centers=2,iter.max = 100)
k$size
ncol(l[[11]])
for(i in 1:nrow(l[[11]])){
  if(e[[11]][i,4]<=mean(e[[11]]$Sale,na.rm = TRUE)){
    e[[11]][i,10]=1
  }
  if(e[[11]][i,4]>mean(e[[11]]$Sale,na.rm = TRUE)){
    e[[11]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[11]]$V10,k$cluster))

#Plastics
k=kmeans(l[[12]][,2:ncol(l[[12]])],centers=2,iter.max = 100)
k$size
ncol(l[[12]])
for(i in 1:nrow(l[[12]])){
  if(e[[12]][i,4]<=mean(e[[12]]$Sale,na.rm = TRUE)){
    e[[12]][i,10]=1
  }
  if(e[[12]][i,4]>mean(e[[12]]$Sale,na.rm = TRUE)){
    e[[12]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[12]]$V10,k$cluster))

#Processed Food
k=kmeans(l[[13]][,2:ncol(l[[13]])],centers=2,iter.max = 100)
k$size
ncol(l[[13]])
for(i in 1:nrow(l[[13]])){
  if(e[[13]][i,4]<=mean(e[[13]]$Sale,na.rm = TRUE)){
    e[[13]][i,10]=1
  }
  if(e[[13]][i,4]>mean(e[[13]]$Sale,na.rm = TRUE)){
    e[[13]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[13]]$V10,k$cluster))

#Sports
k=kmeans(l[[14]][,2:ncol(l[[14]])],centers=2,iter.max = 100)
k$size
ncol(l[[14]])
for(i in 1:nrow(l[[14]])){
  if(e[[14]][i,4]<=mean(e[[14]]$Sale,na.rm = TRUE)){
    e[[14]][i,10]=1
  }
  if(e[[14]][i,4]>mean(e[[14]]$Sale,na.rm = TRUE)){
    e[[14]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[14]]$V10,k$cluster))


#Staples
k=kmeans(l[[15]][,2:ncol(l[[15]])],centers=2,iter.max = 100)
k$size
ncol(l[[15]])
for(i in 1:nrow(l[[15]])){
  if(e[[15]][i,4]<=mean(e[[15]]$Sale,na.rm = TRUE)){
    e[[15]][i,10]=1
  }
  if(e[[15]][i,4]>mean(e[[15]]$Sale,na.rm = TRUE)){
    e[[15]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[2]]$V10,k$cluster))

#Stationery
k=kmeans(l[[16]][,2:ncol(l[[16]])],centers=2,iter.max = 100)
k$size
ncol(l[[16]])
for(i in 1:nrow(l[[16]])){
  if(e[[16]][i,4]<=mean(e[[16]]$Sale,na.rm = TRUE)){
    e[[16]][i,10]=1
  }
  if(e[[16]][i,4]>mean(e[[16]]$Sale,na.rm = TRUE)){
    e[[16]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[16]]$V10,k$cluster))

#Toys
k=kmeans(l[[17]][,2:ncol(l[[17]])],centers=2,iter.max = 100)
k$size
ncol(l[[17]])
for(i in 1:nrow(l[[17]])){
  if(e[[17]][i,4]<=mean(e[[17]]$Sale,na.rm = TRUE)){
    e[[17]][i,10]=1
  }
  if(e[[17]][i,4]>mean(e[[17]]$Sale,na.rm = TRUE)){
    e[[17]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[17]]$V10,k$cluster))

#Utensils
k=kmeans(l[[18]][,2:ncol(l[[18]])],centers=2,iter.max = 100)
k$size
ncol(l[[18]])
for(i in 1:nrow(l[[18]])){
  if(e[[18]][i,4]<=mean(e[[18]]$Sale,na.rm = TRUE)){
    e[[18]][i,10]=1
  }
  if(e[[18]][i,4]>mean(e[[18]]$Sale,na.rm = TRUE)){
    e[[18]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[18]]$V10,k$cluster))

#wellness
k=kmeans(l[[19]][,2:ncol(l[[19]])],centers=2,iter.max = 100)
k$size
ncol(l[[19]])
for(i in 1:nrow(l[[19]])){
  if(e[[19]][i,4]<=mean(e[[19]]$Sale,na.rm = TRUE)){
    e[[19]][i,10]=1
  }
  if(e[[19]][i,4]>mean(e[[19]]$Sale,na.rm = TRUE)){
    e[[19]][i,10]=2
  } 
}
accuracy = append(accuracy, acc(e[[19]]$V10,k$cluster))

mean(unlist(accuracy))