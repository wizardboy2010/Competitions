
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import tqdm as tqdm
import os
import pickle as pkl


data = pkl.load(open('fundtype/fundwise.pkl', 'rb'))


def sep(temp, ids):
    t = []
    c = 0
    for i in ids:
        t.append(temp[c:i])
        c = i
    return t


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



schemes = []
dates = []
nam = []
for i in range(12):
    dates.append(data[i]['Date'].values)
    schemes.append(data[i]['Scheme Name'].values)
    nam.append(data[i]['Fund House'].values)

ids = []
c = 0
for i in dates:
    c += len(i)
    ids.append(c)

#stot = []
#dtot = []
name = []
for i in range(len(dates)):
    #stot = np.append(stot, schemes[i])
    #dtot = np.append(dtot, dates[i])
    name = np.append(name, nam[i])


tempdata = data


d = pd.read_csv('cpi.csv')


d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['Date1'][i]+'-'+str(d['Date2'].values[i])


d


tempdata = create(d['date'].values, d['CPI_rural'].values, scheme, tempdata, ids, 'CPI_rural')


data[0].iloc[2000]


#
tempdata = create(d['date'].values, d['CPI_urban'].values, scheme, tempdata, ids, 'CPI_urban')
tempdata = create(d['date'].values, d['CPI_comb'].values, scheme, tempdata, ids, 'CPI_comb')
tempdata = create(d['date'].values, d['CPI_industrial_general'].values, scheme, tempdata, ids, 'CPI_industrial_general')
tempdata = create(d['date'].values, d['CPI_industrial_food'].values, scheme, tempdata, ids, 'CPI_industrial_food')
tempdata = create(d['date'].values, d['CPI_agri_labours'].values, scheme, tempdata, ids, 'CPI_agri_labours')



pkl.dump(tempdata, open('fundtype/fundwise.pkl', 'wb'))


# ### GDP Data


d = pd.read_csv('Gdp.csv')
d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['Date1'][i]+'-'+str(d['Date2'].values[i])



col = d.columns[2:-1]
for i in col:
    tempdata = create(d['date'].values, d[i].values, scheme, tempdata, ids, i)


pkl.dump(tempdata, open('fundtype/fundwise.pkl', 'wb'))


# ### AUM Data

f1 = os.listdir('aum')
f1


f2 = []
for i in f1:
    f2.append(os.listdir('aum'+'/'+i))


f2[0][0]

df = pd.read_html('aum'+'/'+f1[0]+'/'+f2[0][1])
df = pd.DataFrame(df[0])
df.iloc[1,0][58:-14]


def clean(d):
    a = pd.isnull(d[2])
    temp = []
    for j in a:
        temp.append(j)
    return temp


d = []
t1 = []
t2 = []
t = []
for i in tqdm.tqdm(f2[0]):
    df = pd.DataFrame(pd.read_html('aum'+'/'+f1[0]+'/'+i)[0])
    t.append(df.iloc[1,0][58:-14])
    t1.append(df.iloc[1,0][58:-19])
    t2.append(df.iloc[1,0][-18:-14])
    d.append(df.drop(df.index[clean(df)]))


##### from CPI data
a = [['Jan', 'Feb', 'Mar'], ['Apr', 'May', 'Jun'], 
     ['Jul', 'Aug', 'Sep'], ['Oct', 'Nov', 'Dec']]
dates = []
for i in [2013, 2014, 2015, 2016, 2017]:
    for j in a:
        dates.append(j+'-'+str(i))


for i in range(20):
    t[i] = t[i][:3]+'-'+t[i][-4:]


for i in range(12):
    d[i] = d[i].drop(d[i].index[[0,-1]])


type(d[1][2].astype(float)+d[1][3].astype(float))



'Apr'+'-'+'2014' in t[0]



l = np.array([])
for i in range(12):
    l = np.append(l , d[i][1].values)
l = np.unique(l)



temp1 = pd.DataFrame(l, columns = ['a'])


for i in range(len(temp1)):
    for j in [2013, 2014, 2015, 2016, 2017]:
        for k in ['Jan', 'Apr', 'Jul', 'Oct']:
            for i in range(20):
                



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



for j in [2013, 2014, 2015, 2016, 2017]:
    for k in ['Jan', 'Apr', 'Jul', 'Oct']:
        for i in range(20):
            if k+'-'+str(j) in t[i]:
                temp1 = create1(d[i][1].values, d[i][2].values, l, temp1, t[i])


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
#pd.DataFrame(pd.read_html('aum'+'/'+f1[0]+'/'+f2[0][1])[0])


get_ipython().magic('time')
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

m1



s = pkl.load(open('schemes.pkl', 'rb'))



sch = np.array([])
for i in s:
    sch = np.append(sch, i)



c = 0
adata = pd.DataFrame(columns = m1.columns)
for i in sch:
    if i in list(m1['Scheme'].values):
        j = list(m1['Scheme'].values).index(i)
        adata.loc[c] = m1.iloc[j]
        c += 1



adata



house = pkl.load(open('housenames.pkl', 'rb'))



pkl.dump(adata, open('aumreq.pkl', 'wb'))



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


adata = pkl.load(open('aumreq.pkl', 'rb'))



tempdata = create2(adata['Scheme'].values, adata.columns[1:], adata.iloc[:,1:].values, stot, dtot, tempdata, ids, 'AUM')

c = 0
for i in range(len(dtot)):
    if i >0:
        if 'Jan-2013' in dtot[i] and 'Dec-2017' in dtot[i-1]:
            c += 1
print(c)

col = adata.columns
l = []
for i in tqdm.tqdm(range(len(adata))):
    a = pd.DataFrame(columns = [0,1])
    for j in range(60):
        if not pd.isnull(adata.iloc[i,j+1]):
            #a.append([col[j+1], adata.iloc[i,j+1]])
            a.loc[j,0] = col[j+1]
            a.loc[j,1] = adata.iloc[i,j+1].values
            #a = np.vstack((a, [col[j+1], adata.iloc[i,j+1]]))
    #a = np.concatenate(a, axis = 0)
    l.append(a.values)

adata.to_csv('aum.csv')
#######################################################

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
    
pkl.dump(c, open('datesofschemes.pkl', 'wb'))
##########################################################
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
        
###########################################################
    
for i in range(12):
    tempdata[i]['AUM'] = fin[i]
    
pkl.dump(tempdata, open('fundtype/fundwise.pkl', 'wb'))

##########################################################

d = pd.read_csv('sectorwisetotal.csv')
col = d.columns

t = create(d['Scheme.NAV.Name'].values, d['utility'].values, stot, tempdata, ids, 'utility')
pkl.dump(t, open('fundtype/fundwise.pkl', 'wb'))

for i in col[11:-1]:
    t = create(d['Scheme.NAV.Name'].values, d[i].values, stot, t, ids, i)
    
pkl.dump(t, open('fundtype/fundwise.pkl', 'wb'))

########################################################

abc = pd.DataFrame(columns = ['name', 'house'])
cde = 
for i in range(12):
    cde.append()
abc['name'] = stot
abc['house'] = name

req = []
for i in tqdm.tqdm(adata['Scheme']):
    p = list(abc['name'].values).index(i)
    req.append(abc.iloc[p,1])
    
adata['Fund House'] = req

############################################################################

funds = np.load('fundtypenames.npy')
for i in range(12):
    data[i]['FundType'] = funds[i]

maindata = pd.concat(data)
maindata.to_csv('maindata.csv')
################################################################################

d = pkl.load(open('fundtype/fundwise1.pkl', 'rb'))
col = d[1].columns[-6:]
for i in range(12):
    for j in col:
        data[i][j] = d[i][j].values
        
d = pkl.load(open('fundtype/data_.pkl', 'rb'))
col = d[1].columns[6:]
for i in range(12):
    for j in col:
        data[i][j] = d[i][j].values

pkl.dump(data, open('fundtype/fundwise.pkl', 'wb'))

#################################################################################

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
    

#################################################################################
    
d = pd.read_csv('Gdp.csv')
d['date'] = np.nan
for i in range(len(d)):
    d['date'][i] = d['Date1'][i]+'-'+str(d['Date2'].values[i])
temp = {}
for i in range(len(d)):
    temp[d['date'][i]] = d['GDP'][i]
for i in tqdm.tqdm(range(12)):
    data[i]['GDP'] = [temp[val[-8:]] if val[-8:] in temp.keys() else np.nan for val in data[i]['Date'] ]
    
#################################################################################
 
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
        
#################################################################################
        
d = pd.read_csv(f[11])
c = d.iloc[0,:].values
col = d.columns
d = d.drop(d.index[0])

a = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
l = []
for i in [2013, 2014, 2015, 2016, 2017]:
    for j in range(12):
        for k in range(31):
            l.append(str(k+1)+'-'+a[j]+'-'+str(i))
            
d['Date'] = np.nan
for i in range(len(d)):
    d.iloc[i,-1] = d.iloc[i,0]+'-'+d.iloc[i,1]+'-'+d.iloc[i,2]
    
p = pd.isnull(d[col[3]]).values
ids = []
for i in range(len(p)):
    if p[i]:
        ids.append(i)

d = d.drop(d.index[ids])    

temp = {}

for i in range(len(l)):
    for j in range(len(d)):
        if l[i] == d.iloc[j,-1]:
            flag = d.iloc[j, 3]
        temp[l[i]] = flagpkl.dump(data, open('fundtype/fundwise.pkl', 'wb'))
a = []   
for i in tqdm.tqdm(range(12)):
    a = [temp[val] for val in data[i]['Date'].values]

#################################################################################
    
for i in range(12):
    pkl.dump(data[i], open('fundwise'+str(i)+'.pkl', 'wb'))

#################################################################################
    
d = pkl.load(open('data0.pkl', 'rb'))
col = d.columns[6:]

for i in tqdm.tqdm(range(12)):
    d = pkl.load(open('data'+str(i)+'.pkl', 'rb'))
    col = d.columns[6:]
    for j in col:
        tempdata[i][j] = d[j].values
        
d = pkl.load(open('data0.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[0][i] = d[i].values
    
d = pkl.load(open('data1.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[1][i] = d[i].values
    
d = pkl.load(open('data2.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[2][i] = d[i].values
    
d = pkl.load(open('data3.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[3][i] = d[i].values
    
d = pkl.load(open('data4.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[4][i] = d[i].values
    
d = pkl.load(open('data5.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[5][i] = d[i].values
    
d = pkl.load(open('data6.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[6][i] = d[i].values
    
d = pkl.load(open('data7.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[7][i] = d[i].values
    
d = pkl.load(open('data8.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[8][i] = d[i].values
    
d = pkl.load(open('data9.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[9][i] = d[i].values
    
d = pkl.load(open('data10.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[10][i] = d[i].values
    
d = pkl.load(open('data11.pkl', 'rb'))
col = d.columns[6:]
for i in tqdm.tqdm(col):
    data[11][i] = d[i].values
    
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