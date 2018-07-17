setwd("~/Desktop/ml/competitions/open iit")
df=read.csv("Total.csv")
df_normal=df[1:9]
df_normal$normal=scale(df_normal$Sale)
summary(df$Sale)
summary(df_normal$normal)
for(i in 1:23){
        if(f[i,4]<=mean(f$Sale,na.rm = TRUE)){
                f[i,10]=1
        }
        if(f[i,4]>mean(f$Sale,na.rm = TRUE)){
                f[i,10]=2
        } 
}
df$normal=as.factor(df$normal)
nlevels(df$normal)
summary(df$normal)
summary(df_normal$V11)
df_normal$V11=as.factor(df_normal$V11)
df_1=dummy.data.frame(df[,1:9],sep = ",")
set.seed(123)
for(i in 2:20){
        k=kmeans(f_1[,2:30],centers=i,iter.max = 100)
        print(k$size)
}
colSums(is.na(df))
class(k$cluster)
k$cluster=as.factor(k$cluster)
table(k$cluster,df$V10)
f=df[which(df$Category.of.SKU=="Shoes"),]
f_1=dummy.data.frame(f[,1:9],sep = ",")
f_1=scale(f_1)
f_1=as.data.frame(f_1)
k=kmeans(f_1[,2:31],centers=3,iter.max = 100)
for(i in 1:2){
        print(mean(f$Sale[f$clus==i]))
}
class(f$clus)
levels(df$Category.of.SKU)
e=list()
l=list()
c=1
for(i in levels(df$Category.of.SKU)){
        e[[c]]=df[which(df$Category.of.SKU==i),]
        l[[c]]=dummy.data.frame(e[[c]][,1:9],sep = ",")
        l[[c]]=scale(l[[c]])
        l[[c]]=as.data.frame(l[[c]])
        c=c+1
}
class(e[[1]])
e[[1]]$Category.of.SKU
length(e)
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
table(e[[1]]$V10,k$cluster)
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
table(e[[2]]$V10,k$cluster)
#electronics
k=kmeans(l[[3]][,2:ncol(l[[3]])],centers=3,iter.max = 100)
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
table(e[[2]]$V10,k$cluster)
#farm fresh
k=kmeans(l[[4]][,2:ncol(l[[4]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[4]])){
        if(e[[4]][i,4]<=mean(e[[4]]$Sale,na.rm = TRUE)){
                e[[4]][i,10]=1
        }
        if(e[[4]][i,4]>mean(e[[4]]$Sale,na.rm = TRUE)){
                e[[4]][i,10]=2
        } 
}
table(e[[4]]$V10,k$cluster)
#fashion
k=kmeans(l[[5]][,2:ncol(l[[5]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[5]])){
        if(e[[5]][i,4]<=mean(e[[5]]$Sale,na.rm = TRUE)){
                e[[5]][i,10]=1
        }
        if(e[[5]][i,4]>mean(e[[5]]$Sale,na.rm = TRUE)){
                e[[5]][i,10]=2
        } 
}
table(e[[5]]$V10,k$cluster)
#food
k=kmeans(l[[6]][,2:ncol(l[[6]])],centers=2,iter.max = 100)
k$size
#ncol(l[[2]])
for(i in 1:nrow(l[[6]])){
        if(e[[6]][i,4]<=mean(e[[6]]$Sale,na.rm = TRUE)){
                e[[6]][i,10]=1
        }
        if(e[[6]][i,4]>mean(e[[6]]$Sale,na.rm = TRUE)){
                e[[6]][i,10]=2
        } 
}
table(e[[6]]$V10,k$cluster)
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
table(e[[7]]$V10,k$cluster)
#h.ess
k=kmeans(l[[8]][,2:ncol(l[[8]])],centers=2,iter.max = 100)
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
table(e[[2]]$V10,k$cluster)
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
table(e[[9]]$V10,k$cluster)
#health
k=kmeans(l[[10]][,2:ncol(l[[10]])],centers=5,iter.max = 100)
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
table(e[[2]]$V10,k$cluster)
#trolley
k=kmeans(l[[11]][,2:ncol(l[[11]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[11]])){
        if(e[[11]][i,4]<=mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=1
        }
        if(e[[11]][i,4]>mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=2
        } 
}
table(e[[2]]$V10,k$cluster)

#Plastics
k=kmeans(l[[12]][,2:ncol(l[[12]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[12]])){
        if(e[[12]][i,4]<=mean(e[[12]]$Sale,na.rm = TRUE)){
                e[[12]][i,10]=1
        }
        if(e[[12]][i,4]>mean(e[[12]]$Sale,na.rm = TRUE)){
                e[[12]][i,10]=2
        } 
}
table(e[[12]]$V10,k$cluster)
#Processed Food
k=kmeans(l[[13]][,2:ncol(l[[13]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[13]])){
        if(e[[13]][i,4]<=mean(e[[13]]$Sale,na.rm = TRUE)){
                e[[13]][i,10]=1
        }
        if(e[[13]][i,4]>mean(e[[13]]$Sale,na.rm = TRUE)){
                e[[13]][i,10]=2
        } 
}
table(e[[13]]$V10,k$cluster)
#Sports
k=kmeans(l[[14]][,2:ncol(l[[14]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[14]])){
        if(e[[14]][i,4]<=mean(e[[14]]$Sale,na.rm = TRUE)){
                e[[14]][i,10]=1
        }
        if(e[[14]][i,4]>mean(e[[14]]$Sale,na.rm = TRUE)){
                e[[14]][i,10]=2
        } 
}
table(e[[14]]$V10,k$cluster)

#Staples
k=kmeans(l[[15]][,2:ncol(l[[15]])],centers=3,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[14]])){
        if(e[[11]][i,4]<=mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=1
        }
        if(e[[11]][i,4]>mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=2
        } 
}
table(e[[2]]$V10,k$cluster)

#Stationery
k=kmeans(l[[16]][,2:ncol(l[[16]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[16]])){
        if(e[[16]][i,4]<=mean(e[[16]]$Sale,na.rm = TRUE)){
                e[[16]][i,10]=1
        }
        if(e[[16]][i,4]>mean(e[[16]]$Sale,na.rm = TRUE)){
                e[[16]][i,10]=2
        } 
}
table(e[[16]]$V10,k$cluster)

#Toys
k=kmeans(l[[17]][,2:ncol(l[[17]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[17]])){
        if(e[[17]][i,4]<=mean(e[[17]]$Sale,na.rm = TRUE)){
                e[[17]][i,10]=1
        }
        if(e[[17]][i,4]>mean(e[[17]]$Sale,na.rm = TRUE)){
                e[[17]][i,10]=2
        } 
}
table(e[[17]]$V10,k$cluster)

#Utensils
k=kmeans(l[[18]][,2:ncol(l[[18]])],centers=2,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[18]])){
        if(e[[18]][i,4]<=mean(e[[18]]$Sale,na.rm = TRUE)){
                e[[18]][i,10]=1
        }
        if(e[[18]][i,4]>mean(e[[18]]$Sale,na.rm = TRUE)){
                e[[18]][i,10]=2
        } 
}
table(e[[18]]$V10,k$cluster)
#wellness
k=kmeans(l[[19]][,2:ncol(l[[19]])],centers=3,iter.max = 100)
k$size
ncol(l[[2]])
for(i in 1:nrow(l[[11]])){
        if(e[[11]][i,4]<=mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=1
        }
        if(e[[11]][i,4]>mean(e[[11]]$Sale,na.rm = TRUE)){
                e[[11]][i,10]=2
        } 
}
table(e[[2]]$V10,k$cluster)










