library("pgirmess", lib.loc="~/R/win-library/3.4")
library("stats", lib.loc="C:/Program Files/R/R-3.4.2/library")
library("agricolae", lib.loc="~/R/win-library/3.4")

##################################################################################################LOADING DATA
setwd("C:/Users/intel/Desktop/da gc")
fdata = read.csv("edited_main_data.csv")
que1 = subset.data.frame(x = fdata,select = c(Date,Fund.House,FundType,AUM,Net.Asset.Value,NAV_growth,AUM_growth))
freena = na.omit(que1)
freena$Net.Asset.Value = as.numeric(freena$Net.Asset.Value)
performance_metrics <- function(que1_nafree)
{
        #####################################################################  N A V
        print("Net Asset Value")
        print(kruskal.test(Net.Asset.Value ~ Fund.House , data = que1_nafree))
        print(kruskalmc(que1_nafree$Net.Asset.Value,que1_nafree$Fund.House, probs = 0.01))
        print(kruskal(que1_nafree$Net.Asset.Value,que1_nafree$Fund.House, console = TRUE,alpha = 0.01)) 
        #####################################################################  Nav_growth
        print("NAV Growth")
        data_nav_growth = subset.data.frame(que1_nafree,NAV_growth !=0 )
        print(kruskal.test(NAV_growth ~ Fund.House , data = data_nav_growth))
        print(kruskalmc(data_nav_growth$NAV_growth ,data_nav_growth$Fund.House, probs = 0.01))
        print(kruskal(data_nav_growth$NAV_growth,data_nav_growth$Fund.House, console = TRUE,alpha = 0.01) )
        
        ####################################################################  A U M
        print("AUM")
        data_aum = subset.data.frame(que1_nafree,AUM_growth !=1)
        print(kruskal.test(AUM ~ Fund.House , data = data_aum))
        print(kruskalmc(data_aum$AUM ,data_aum$Fund.House, probs = 0.001))
        print(kruskal(data_aum$AUM,data_aum$Fund.House, console = TRUE,alpha = 0.01))
        #################################################################### A U M_growth
        print("AUM Growth")
        data_aum_growth = subset.data.frame(data_aum,AUM_growth !=0)
        print(kruskal.test(AUM_growth ~ Fund.House , data = data_aum_growth))
        print(kruskalmc(data_aum_growth$AUM_growth ,data_aum_growth$Fund.House, probs = 0.01))
        print(kruskal(data_aum_growth$AUM_growth,data_aum_growth$Fund.House, console = TRUE,alpha = 0.01))
}

##########################################################################################################Comparing performance of past four years
performance_metrics(freena)

####################################################################Comparing performance of Fund Houses of past four years for respective Fund Types
fundtype = levels(freena$FundType)
datasets_fundtype = list()
for(i in 1:length(fundtype))
        {
        datasets_fundtype[[i]] = subset.data.frame(freena,FundType == fundtype[i] )
        }
for(i in 1:length(fundtype)){
        print(fundtype[i])
        performance_metrics(datasets_fundtype[[i]])
}
##################################################################################################comparing performance of Fund Houses for each year
class(freena$Date)
freena$Date= as.character(freena$Date)
freena$Date = substr(freena$Date,8,11)
freena$Date = as.factor(freena$Date)

year = levels(freena$Date)
datasets_year = list()
for(i in 1:length(year))
{
        datasets_year[[i]] = subset.data.frame(freena,Date == year[i] )
        
}
for(i in 1:length(year))
{
        print(year[i])
        print("/n")
        performance_metrics(datasets_year[[i]])
}


