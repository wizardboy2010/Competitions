import numpy as np
import pandas as pd
import tqdm as tqdm
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import os

############################ count number of timesa word appears in file
def numb(file, word):
    with open(file) as f:
        contents = f.read()
        count = contents.count(word)
    return count


############################ appends all lines containing a given word
def reqline(file, word):
    a = []
    with open(file) as f:
        for lines in f:
            if word in lines:
                a.append(lines[:-1])
    return a


files = os.listdir(os.getcwd()) ################## list of all 8 fundhouses datasets names


########################### FundType Names
a = np.array([])
for f in files:
    #print(f)
    a = np.append(a, reqline(f, 'Open Ended Schemes'))
funds = np.unique(a)


############################################## Number of schemes present in each fundhouse
for i in range(8):
	print(len(np.unique(pd.read_csv(files[i], delimiter=';', low_memory=False)['Scheme Code'])))


########################### list of all schemes present in each fundhouse
schemes = []
for f in files:
    d = pd.read_csv(f, delimiter=';', low_memory=False)
    schemes.append(np.unique(d['Scheme Name'][~d['Scheme Name'].isnull()]))


pkl.dump(schemes, open("schemes.pkl", "wb"))


c = 0       ############## total number of schemes
for i in schemes:
    c += len(i)


########## plotting and checking NAV values
_, a = plt.subplots(2)
d = pd.read_csv('Aditya Birla Sun Life Mutual Fund.txt', delimiter=';')
a[0].plot(d[d['Scheme Code'] == np.unique(d['Scheme Code'])[1]]['Net Asset Value'])
#plt.plot(d[d['Scheme_Code'] =='103188']['Repurchase_Price'])
#lt.plot(d[d['Scheme_Code'] =='103188']['Sale_Price'])
d = pd.read_csv('SBI Mutual Fund.txt', delimiter=';')
a[1].plot(d[d['Scheme Code'] == np.unique(d['Scheme Code'])[1]]['Net Asset Value'])


############################################## gives number of fundtypes in each fundhouse
for f in files:
    print(numb(f, 'Open Ended Schemes'))

#########################################################################################################################################
###### Data Extraction (FundType wise)

############## stores location at which new fundtype data occurs
ids = []
for i in files:
    l = []
    temp = pd.read_csv(i, delimiter = ';', low_memory=False)
    for p, q in enumerate(list(temp['Scheme Code'].values)):
        if q == i[:-4]:
            l.append(p)
    ids.append(l)

######## making data

t2 = []                           ################ Stores 12 arrays containing name of Fund House of each scheme in the order schemes 
fundtype = []                     ################ Stores 12 datasets of fundtypes(combining all 8 fund houses data)
for t in funds:
    funddata = []
    t1 = []
    for i, f in tqdm.tqdm(enumerate(files)):
        if numb(f, t) == 1:
            temp = reqline(f, 'Open Ended Schemes')[0].index(t)
            data = pd.read_csv(f, delimiter = ';', low_memory=False)
            if temp+1 != len(ids[i]):
                funddata.append(data[ids[i][temp]+1:ids[i][temp+1]-1])
                t1.append([f[:-4]]*len(range(ids[i][temp]+1,ids[i][temp+1]-1)))
            else:
                temp3 = len(data)
                funddata.append(data[ids[i][temp]+1:])
                t1.append([f[:-4]]*len(range(ids[i][temp]+1,temp3)))
    t2.append(t1)
    fundtype.append(funddata)

final = []
for i in range(12):
    final1 = np.array([])
    for j in t2[i]:
        final1 = np.append(final1, np.array(j))
    final.append(pd.DataFrame(final1, columns = ['Fund House']))

######################## making vanilla dataset

temp = []
for i in range(12):
    temp.append(pd.concat(fundtype[i]))       ######### makes a list of 12 dataframes with basic features(NAV, Repurchase_rate, etc..)
fundtype = temp
del temp

pkl.dump(fundtype, open('fundwise.pkl', 'wb'))   ######### save the data into pickle format

for i in range(12):
    fundtype[i].to_csv('fundtype'+'/'+funds[i][21:-2]+'.csv')       ######## also save as csv files with names of fundtypes

#########################################################################################################################################
########## Making Data for NAV analysis

for i in range(8):               ######## names of fundhouses
    files[i] = files[i][:-4]

data = []                        ######## list of given data
for i in tqdm.tqdm(files):
    d = pd.read_csv(i+'.txt', delimiter = ';', low_memory=False)
    a = pd.isnull(d['Scheme Name'])
    temp = []
    for j in a:
        temp.append(j)
    data.append(d.drop(d.index[temp]))

col = data[1].columns            ####### columns of the data

for i in range(8):               ####### remove all columns other than NAV(preffered over making new data due to less ram consumption)
    data[i] = data[i].drop(col[:2],1)
    data[i] = data[i].drop(col[3:],1)

for i in range(8):               ####### adding fundhouse names
    data[i]['house'] = files[i]

d = pd.concat(data)              ####### merge all 8 files to make a master NAV file
d.to_csv('nav/nav.csv')

#########################################################################################################################################
##################### ADDING NEW DATA

#...........................................................Method1

####### functions used
     
schemes = []   ############## each dataset wise list of schemes(inorder)
dates = []
nam = []
for i in range(12):
    dates.append(data[i]['Date'].values)
    schemes.append(data[i]['Scheme Name'].values)
    nam.append(data[i]['Fund House'].values)

ids = []        ############## stores locations which helps to cut the file(if all 12 files are mixed)
c = 0
for i in names:
    c += len(i)
    ids.append(c)

stot = []      ############# scheme names of all 12 files combined(in order)
dtot = []
name = []
for i in range(len(names)):
    stot = np.append(stot, schemes[i])
    dtot = np.append(dtot, dates[i])
    name = np.append(name, nam[i])

def sep(temp, ids):
    t = []
    c = 0
    for i in ids:
        t.append(temp[c:i])
        c = i
    return t

def create(given, val, names, data, ids, n):
    temp = []
    for i, p in tqdm.tqdm(enumerate(names)):
        c = 0
        for j in range(len(given)):
            if given[j] == p:
                temp.append(val[j])
                c = 1
        if c == 0:
            temp.append(np.nan)
    f = sep(temp, ids)
    for i in range(12):
        data[i][n] = f[i]
    return data

########## appending column named Scheme.Minimun.Amount from d into list of 12 datasets(data)
d = pd.read_csv('givenschemes.csv')
data = create(d['Scheme.NAV.Name'].values, d['Scheme.Minimum.Amount'].values, scheme, data, ids, 'Scheme.Minimum.Amount')
#### takes an average of 10-15 minutes for each file

#.............................................................Method2

##### For filling percapita
temp = {}                   ##### define a dictionary
temp['2013'] = 1520
temp['2014'] = 1560
temp['2015'] = 1600
temp['2016'] = 1670
temp['2017'] = np.nan

for i in tqdm.tqdm(range(12)):        ########## append by using the dictionary
    data[i]['percapita'] = [temp[val[-4:]] for val in data[i]['Date'].values]

###### adding GDP data from CSV
temp = {}
for i in range(len(d)):
    temp[d['date'][i]] = d['GDP'][i]
for i in tqdm.tqdm(range(12)):
    data[i]['GDP'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

#### In this way each feature is taking few seconds

##########################################################################################################################################
##### adding AUM data

f1 = os.listdir('aum')

f2 = []
for i in f1:
    f2.append(os.listdir('aum'+'/'+i))


def clean(d):
    a = pd.isnull(d[2])
    temp = []
    for j in a:
        temp.append(j)
    return temp


def create1(given, val, names, data, n):
    te = []
    for i in tqdm.tqdm(range(len(names))):
        c = 0
        for j in range(len(given)):
            if given[j] == names[i]:
                te.append(val[j])
                c = 1
        if c == 0:
            te.append(np.nan)
    data[n] = te
    return data


############## extracting the features from the quaterly files
def fn(f1, f2):
    t = []
    dat = []
    l = np.array([])
    for i in tqdm.tqdm(f2):
        d = pd.DataFrame(pd.read_html('aum'+'/'+f1+'/'+i)[0])
        x = d.iloc[1,0][58:-14]
        t.append(x[:3]+'-'+x[-4:])
        d = d.drop(d.index[clean(d)])
        d = d.drop(d.index[[0,-1]])
        l = np.append(l, d[1].values)
        dat.append([d[1].values, (d[2].astype(float)+d[3].astype(float)).values])
    return dat, t, np.unique(l)

d, t, l= fn(f1[0], f2[0])

m = []
for i in range(8):
    d, t, l= fn(f1[i], f2[i])
    temp = pd.DataFrame(l, columns = ['Scheme'])
    for j in [2013, 2014, 2015, 2016, 2017]:
        for k in ['Jan', 'Apr', 'Jul', 'Oct']:
            for i in range(20):
                if k+'-'+str(j) in t[i]:
                    for z in range(4):
                        if t[i][:3] == a[z][0]:
                            for y in a[z]:
                                temp = create1(d[i][0], d[i][1], l, temp, y+'-'+str(j))
    
    m.append(temp)

m1 = pd.concat(m)

sch = np.array([])
for i in schemes:
    sch = np.append(sch, i)

c = 0
adata = pd.DataFrame(columns = m1.columns)      ######## stores AUM data that needs to be appended
for i in sch:
    if i in list(m1['Scheme'].values):
        j = list(m1['Scheme'].values).index(i)
        adata.loc[c] = m1.iloc[j]
        c += 1

adata.to_csv('aum.csv')

####### adding the features to main list of data


###### Try1......................highly time complex...so rejected

def create2(scgiv, dagiv, val, sc, da, data, ids, n):
    temp = []
    for i in tqdm.tqdm(range(len(sc))):
        count = 0
        c = 0
        for j in range(len(scgiv)):
            if scgiv[j] == sc[i]:
                for k in range(len(da)):
                    for l in range(len(dagiv)):
                        if dagiv[l] in da[k]:
                            temp.append(val[j][l])
                            c = 1
                    if c == 0:
                        temp.append(np.nan)
        count += 1
        print(count)
    f = sep(temp, ids)
    for i in range(12):
        data[i][n] = f[i]
    return data

data = create2(adata['Scheme'].values, adata.columns[1:], adata.iloc[:,1:].values, stot, dtot, data, ids, 'AUM')

######Try2..................... Better compared to Try1

col = adata.columns
l = []
for i in tqdm.tqdm(range(len(adata))):
    a = pd.DataFrame(columns = [0,1])
    for j in range(60):
        if not pd.isnull(adata.iloc[i,j+1]):
            a.loc[j,0] = col[j+1]
            a.loc[j,1] = adata.iloc[i,j+1].values
    l.append(a.values)

temp = []
for i in schemes:
    temp.append(list(np.unique(i)))

c = []    
for i in tqdm.tqdm(range(12)):
    b = []
    for k in temp[i]:
        a = []
        for j in range(len(schemes[i])):
            if schemes[i][j] == k:
                a.append(dates[i][j])
        b.append(a)
    c.append(b)

giv = list(adata['Scheme'].values)

def comp(a1, a2):
    temp = [np.nan]*len(a1)
    for i in range(len(a1)):
        for j in range(len(a2)):
            if a2[j,0] in a1[i]:
                temp[i] = a2[j,1]
    return temp

fin = []
for i in tqdm.tqdm(range(12)):
    a = np.array([])
    for j in range(len(temp[i])):
        if not temp[i][j] in giv:
            a = np.append(a, [np.nan]*len(c[i][j]))
        else:
            p = giv.index(temp[i][j])
            a = np.append(a, comp(c[i][j], l[p]))
    fin.append(a)

for i in range(12):                   ######## adding features into main data
    tempdata[i]['AUM'] = fin[i]

######################################################################################
####### adding CPI data

d = pd.read_csv('cpi.csv')

d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['Date1'][i]+'-'+str(d['Date2'].values[i])

def create(given, val, names, data, ids, n):
    temp = []
    for i in tqdm.tqdm(range(len(names))):
        c = 0
        for j in range(len(given)):
            if given[j] in names[i]:
                temp.append(val[j])
                c = 1
        if c == 0:
            temp.append(np.nan)
    f = sep(temp, ids)
    for i in range(12):
        data[i][n] = f[i]
    return data

tempdata = create(d['date'].values, d['CPI_rural'].values, scheme, tempdata, ids, 'CPI_rural')
tempdata = create(d['date'].values, d['CPI_urban'].values, scheme, tempdata, ids, 'CPI_urban')
tempdata = create(d['date'].values, d['CPI_comb'].values, scheme, tempdata, ids, 'CPI_comb')
tempdata = create(d['date'].values, d['CPI_industrial_general'].values, scheme, tempdata, ids, 'CPI_industrial_general')
tempdata = create(d['date'].values, d['CPI_industrial_food'].values, scheme, tempdata, ids, 'CPI_industrial_food')
tempdata = create(d['date'].values, d['CPI_agri_labours'].values, scheme, tempdata, ids, 'CPI_agri_labours')

#####################################################################################
####### adding other features

funds = np.load('fundtypenames.npy')
for i in range(12):
    data[i]['FundType'] = funds[i]

temp = {}
temp['2013'] = 1520
temp['2014'] = 1560
temp['2015'] = 1600
temp['2016'] = 1670
temp['2017'] = np.nan

for i in tqdm.tqdm(range(12)):
    data[i]['percapita'] = [temp[val[-4:]] for val in data[i]['Date'].values]
    
temp = {}
temp['2013'] = 53.7757
temp['2014'] = 52.93609
temp['2015'] = 52.1299
temp['2016'] = 51.52284
temp['2017'] = np.nan

for i in tqdm.tqdm(range(12)):
    data[i]['dependency ratio'] = [temp[val[-4:]] for val in data[i]['Date'].values]
    
temp = {}
temp['2013'] = 10.90764
temp['2014'] = 6.6495
temp['2015'] = 4.906
temp['2016'] = 4.9414
temp['2017'] = np.nan

for i in tqdm.tqdm(range(12)):
    data[i]['inflation'] = [temp[val[-4:]] for val in data[i]['Date'].values]

###### GDP

d = pd.read_csv('Gdp.csv')
d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['Date1'][i]+'-'+str(d['Date2'].values[i])
temp = {}
for i in range(len(d)):
    temp[d['date'][i]] = d['GDP'][i]
for i in tqdm.tqdm(range(12)):
    data[i]['GDP'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date'] ]

####### Market value for Gold and Silver

f = os.listdir(os.getcwd())
d = pd.read_csv(f[-1])

d.iloc[1,2] = 'gold Mumbai 10 gm'
d.iloc[1,3] = 'gold london troy'
d.iloc[1,4] = 'gold 10gm'
d.iloc[1,5] = 'gold spread'
d.iloc[1,6] = 'silver Mumbai 10 gm'
d.iloc[1,7] = 'ny troy'
d.iloc[1,8] = 'silver per kg'
d.iloc[1,9] = 'silver spread'

d = d.drop(d.index[0])
col = d.iloc[0,:].values[2:]
d = d.drop(d.index[0])

d['date'] = np.nan
for i in range(len(d)):
    d.iloc[i,-1] = d.iloc[i,0]+'-'+d.iloc[i,1]
    #print(d.iloc[i,0][:3]+'-'+d.iloc[i,1])

for j in range(8):
    temp = {}
    for i in range(51):
        temp[d.iloc[i,-1]] = d.iloc[i,2+j]
    for i in tqdm.tqdm(range(12)):
        data[i][col[j]] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date'].values]

############# market

temp = {}
d = pd.read_csv('date1.csv')
num = len(d['Date'])

for t in range(num):
    temp[d['Date'][t]] = d['Open'][t]
print(temp)

for i in range(12):
    data[i]['Open'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Date'].values]

for t in range(num):
    temp[d['Date'][t]] = d['High'][t]
print(temp)

for i in range(12):
    data[i]['High'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Date'].values]

for t in range(num):
    temp[d['Date'][t]] = d['Low'][t]
print(temp)

for i in range(12):
    data[i]['Low'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Date'].values]

for t in range(num):
    temp[d['Date'][t]] = d['Close'][t]
print(temp)

for i in range(12):
    data[i]['Close'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Date'].values]

############### economic features

d = pd.read_csv('foreign.csv')

d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['date1'][i]+'-'+str(d['date2'].values[i])

d['date'] = d['date'].str.replace('january','Jan')
d['date'] = d['date'].str.replace('february','Feb')
d['date'] = d['date'].str.replace('march','Mar')
d['date'] = d['date'].str.replace('april','Apr')
d['date'] = d['date'].str.replace('may','May')
d['date'] = d['date'].str.replace('june','Jun')
d['date'] = d['date'].str.replace('july','Jul')
d['date'] = d['date'].str.replace('august','Aug')
d['date'] = d['date'].str.replace('september','Sep')
d['date'] = d['date'].str.replace('october','Oct')
d['date'] = d['date'].str.replace('november','Nov')
d['date'] = d['date'].str.replace('december','Dec')

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['Gross inflows/ Gross Investments'][i]
for i in range(12):
    data[i]['Gross inflows/ Gross Investments'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['Repatriation/Disinvestment'][i]
for i in range(12):
    data[i]['Repatriation/Disinvestment'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['Direct Investment to India'][i]
for i in range(12):
    data[i]['Direct Investment to India'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['FDI by India'][i]
for i in range(12):
    data[i]['FDI by India'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['Net Foreign Direct Investment'][i]
for i in range(12):
    data[i]['Net Foreign Direct Investment'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['Net Portfolio Investment'][i]
for i in range(12):
    data[i]['Net Portfolio Investment'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

temp = {}
for i in range(len(d['date'])):
    temp[d['date'][i]] = d['total'][i]
for i in range(12):
    data[i]['total'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date']]

################ sectorwise data

temp = {}
d = pd.read_csv('sectorwisetotal.csv')
num = len(d['Code'])

for t in range(num):
    temp[d['Code'][t]] = d['automotive'][t]
for i in range(12):
    data[i]['Automotive'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['banking'][t]
for i in range(12):
    data[i]['Banking'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['cement'][t]
for i in range(12):
    data[i]['Cement'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['chemicals'][t]
for i in range(12):
    data[i]['Chemicals'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['conglomerates'][t]
for i in range(12):
    data[i]['Conglomerates'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['durables'][t]
for i in range(12):
    data[i]['Durables'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['nondurables'][t]
for i in range(12):
    data[i]['Nondurables'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['capitalgood'][t]
for i in range(12):
    data[i]['Capitalgood'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['foodbeverage'][t]
for i in range(12):
    data[i]['Foodbeverage'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['infotech'][t]
for i in range(12):
    data[i]['Infotech'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['manufacturing'][t]
for i in range(12):
    data[i]['Manufacturing'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['mediaentertainment'][t]
for i in range(12):
    data[i]['Mediaentertainment'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['metals'][t]
for i in range(12):
    data[i]['Metals'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['miscellaneous'][t]
for i in range(12):
    data[i]['Miscellaneous'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['oilgas'][t]
for i in range(12):
    data[i]['Oilgas'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['pharmaceuticals'][t]
for i in range(12):
    data[i]['Pharmaceuticals'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['realestate'][t]
for i in range(12):
    data[i]['Realestate'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['services'][t]
for i in range(12):
    data[i]['Services'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['telecommunication'][t]
for i in range(12):
    data[i]['Telecommunication'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['tobacco'][t]
for i in range(12):
    data[i]['Tobacco'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]

for t in range(num):
    temp[d['Code'][t]] = d['utility'][t]
for i in range(12):
    data[i]['Utility'] = [temp[val] if val in temp.keys() else np.nan for val in data[i]['Scheme Code'].values]


#####################################################################################################################################
###### Finalised data
## data size is around 2-3Gb

pkl.dump(data[0], open('data0.pkl', 'wb'))
pkl.dump(data[1], open('data1.pkl', 'wb'))
pkl.dump(data[2], open('data2.pkl', 'wb'))
pkl.dump(data[3], open('data3.pkl', 'wb'))
pkl.dump(data[4], open('data4.pkl', 'wb'))
pkl.dump(data[5], open('data5.pkl', 'wb'))
pkl.dump(data[6], open('data6.pkl', 'wb'))
pkl.dump(data[7], open('data7.pkl', 'wb'))
pkl.dump(data[8], open('data8.pkl', 'wb'))
pkl.dump(data[9], open('data9.pkl', 'wb'))
pkl.dump(data[10], open('data10.pkl', 'wb'))
pkl.dump(data[11], open('data11.pkl', 'wb'))