# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:21:05 2019

@author: lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import datetime
import statsmodels.formula.api as sm
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from scipy.stats.stats import pearsonr

pd.set_option('display.max_columns', None)

data = pd.read_excel('Data_Train.xlsx')
data_test = pd.read_excel('Test_set.xlsx')

data.columns
data.head()
data.tail()
data.describe(include = 'all')
data.info()

data = data.drop_duplicates()
data_test = data_test.drop_duplicates()

data.isnull().sum() 
data_test.isnull().sum() 

# data = data.drop(data.loc[data['Route'].isnull()].index)


# data['Route'].corr(data['Price'])
data['Airline'].unique()
data['Airline'] = np.where(data['Airline']=='Vistara Premium economy', 'Vistara', data['Airline'])
data['Airline'] = np.where(data['Airline']=='Jet Airways Business', 'Jet Airways', data['Airline'])
data['Airline'] = np.where(data['Airline']=='Multiple carriers Premium economy', 'Multiple carriers', data['Airline'])

data_test['Airline'].unique()
data_test['Airline'] = np.where(data_test['Airline']=='Vistara Premium economy', 'Vistara', data_test['Airline'])
data_test['Airline'] = np.where(data_test['Airline']=='Jet Airways Business', 'Jet Airways', data_test['Airline'])
data_test['Airline'] = np.where(data_test['Airline']=='Multiple carriers Premium economy', 'Multiple carriers', data_test['Airline'])


data['Destination'].unique()
data['Destination'] = np.where(data['Destination']=='Delhi','New Delhi', data['Destination'])

data_test['Destination'].unique()
data_test['Destination'] = np.where(data_test['Destination']=='Delhi','New Delhi', data_test['Destination'])


#################### weekdays
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey']) 
data['day_of_week'] = data['Date_of_Journey'].dt.day_name()

data['Journey_Month'] = pd.to_datetime(data.Date_of_Journey, format='%d/%m/%Y').dt.month_name()
# Test Set
data_test['Date_of_Journey'] = pd.to_datetime(data_test['Date_of_Journey']) 
data_test['day_of_week'] = data_test['Date_of_Journey'].dt.day_name()

data_test['Journey_Month'] = pd.to_datetime(data_test.Date_of_Journey, format='%d/%m/%Y').dt.month_name()
# Compare the dates and delete the original date feature

#################DEPARTURE TIME
# Training Set
data['Departure_t'] = pd.to_datetime(data.Dep_Time, format='%H:%M')


##############grouping Dept_Time (departure)
   
a = data.assign(dept_session=pd.cut(data.Departure_t.dt.hour,[0,6,12,18,24],
                                labels=['Night','Morning','Afternoon','Evening']))

data['Departure_S'] = a['dept_session']

## test
data_test['Departure_t'] = pd.to_datetime(data_test.Dep_Time, format='%H:%M')
   
a = data_test.assign(dept_session=pd.cut(data_test.Departure_t.dt.hour,[0,6,12,18,24],
                                labels=['Night','Morning','Afternoon','Evening']))

data_test['Departure_S'] = a['dept_session']

data['Departure_S'].fillna("Night", inplace = True) 
data_test['Departure_S'].fillna("Night", inplace = True) 


################# Cleaning Duration
# spplitting duartion into two columns hr and min
# Training Set

duration = list(data['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  
 
for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
     
 
data['Duration_hours'] = dur_hours
data['Duration_minutes'] =dur_minutes

########################## test
duration = list(data_test['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  

for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
    
data_test['Duration_hours'] = dur_hours
data_test['Duration_minutes'] =dur_minutes


data.loc[:,'Duration_hours'] *= 60
data['Duration_Total_mins'] = data['Duration_hours']+data['Duration_minutes']

data_test.loc[:,'Duration_hours'] *= 60
data_test['Duration_Total_mins'] = data_test['Duration_hours']+data_test['Duration_minutes']



# Get names of indexes for which column Age has value 30
indexNames = data[data.Duration_Total_mins < 60].index
# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)

##Test
indexNames = data_test[data_test.Duration_Total_mins < 60].index
# Delete these row indexes from dataFrame
data_test.drop(indexNames , inplace=True)
#################################)'Dep_Time'(##drop columns##########################################
data.drop(labels = ['Arrival_Time','Dep_Time' ,'Date_of_Journey','Duration','Departure_t','Duration_hours','Duration_minutes'], axis=1, inplace = True)
data_test.drop(labels = ['Arrival_Time','Dep_Time' ,'Date_of_Journey','Duration','Departure_t','Duration_hours','Duration_minutes'], axis=1, inplace = True)

####### putting price at last
data = data[['Airline', 'Source', 'Destination', 'Route', 'Total_Stops',
       'Additional_Info', 'day_of_week', 'Journey_Month',
       'Departure_S','Duration_Total_mins','Price']]


# =============================================================================
# ####VIZUALIZATION
# data['Price'].describe()
# sns.boxplot(data['Price'])
# 
# data = data[data.Price < 40000]
# 
# print("Skew: %.2f" %data['Price'].skew())
# print("Kurtosis: %.2f" %data['Price'].kurt())
# 
# sns.distplot(data['Price'], fit = None)
# =============================================================================


###CREATE DUMMY VAR

cat_vars = ['Airline', 'Source', 'Destination', 'Route', 'Total_Stops',
       'Additional_Info', 'day_of_week', 'Journey_Month', 'Departure_S' ]
for var in cat_vars:
    catList = 'var'+'_'+var
    catList = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(catList)
    data = data1
    
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
# =============================================================================
# data_final.columns.values
# =============================================================================
cat_vars2 = ['Airline', 'Source', 'Destination', 'Route', 'Total_Stops',
       'Additional_Info', 'day_of_week', 'Journey_Month', 'Departure_S' ]
for var in cat_vars2:
    catList = 'var'+'_'+var
    catList = pd.get_dummies(data_test[var], prefix=var)
    data1 = data_test.join(catList)
    data_test = data1
    

data_vars_test = data_test.columns.values.tolist()
to_keep = [i for i in data_vars_test if i not in cat_vars2]

data_final_test=data_test[to_keep]

#############SCALE FEATUREs
data_final2 = data_final.copy()
data_final_test2 = data_final_test.copy()

sc = StandardScaler()
data_final2 = sc.fit_transform(data_final2)
data_final_test2 = sc.transform(data_final_test2)

#############  PCA

pca = PCA()
df_pca = pca.fit_transform(df_scaled)

num_components=len(pca.explained_variance_ratio_)

ind = np.arange(num_components)
vals = pca.explained_variance_ratio_
 
plt.figure(figsize=(20, 10))
ax = plt.subplot(111)
cumvals = np.cumsum(vals)
ax.plot(ind, vals)
ax.plot(ind, cumvals)


ax.xaxis.set_tick_params(width=0,gridOn=True)
ax.yaxis.set_tick_params(width=2, length=12,gridOn=True)
 
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
plt.title('Explained Variance Per Principal Component')

## Addition based on Requires Changes suggestion:

n_components = min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.85)[0]+1)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1],True)
ax2 = ax.twinx()
ax.plot(pca.explained_variance_ratio_, label='Variance',)
ax2.plot(np.cumsum(pca.explained_variance_ratio_), label='Cumulative Variance',color = 'red');
ax.set_title('n_components needed for >85% explained variance: {}'.format(n_components));
ax.axvline(n_components, linestyle='dashed', color='black')
ax2.axhline(np.cumsum(pca.explained_variance_ratio_)[n_components], linestyle='dashed', color='black')
fig.legend(loc=(0.6,0.2));

pca = PCA(n_components=69)
df_pca = pca.fit_transform(df_scaled)
print(pca.explained_variance_ratio_.sum())
df_pca = pd.DataFrame(df_pca)





























data_final = data_final[['Duration_Total_mins', 'Airline_Air Asia',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Multiple carriers',
       'Airline_SpiceJet', 'Airline_Trujet', 'Airline_Vistara',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi',
       'Source_Kolkata', 'Source_Mumbai', 'Destination_Banglore',
       'Destination_Cochin', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi',
       'Route_BLR → AMD → DEL', 'Route_BLR → BBI → DEL',
       'Route_BLR → BDQ → DEL', 'Route_BLR → BOM → AMD → DEL',
       'Route_BLR → BOM → BHO → DEL', 'Route_BLR → BOM → DEL',
       'Route_BLR → BOM → IDR → DEL', 'Route_BLR → BOM → IDR → GWL → DEL',
       'Route_BLR → BOM → IXC → DEL', 'Route_BLR → BOM → JDH → DEL',
       'Route_BLR → BOM → NAG → DEL', 'Route_BLR → BOM → UDR → DEL',
       'Route_BLR → CCU → BBI → DEL', 'Route_BLR → CCU → BBI → HYD → DEL',
       'Route_BLR → CCU → BBI → HYD → VGA → DEL', 'Route_BLR → CCU → DEL',
       'Route_BLR → CCU → GAU → DEL', 'Route_BLR → COK → DEL',
       'Route_BLR → DEL', 'Route_BLR → GAU → DEL',
       'Route_BLR → GOI → DEL', 'Route_BLR → HBX → BOM → AMD → DEL',
       'Route_BLR → HBX → BOM → BHO → DEL',
       'Route_BLR → HBX → BOM → NAG → DEL', 'Route_BLR → HYD → DEL',
       'Route_BLR → HYD → VGA → DEL', 'Route_BLR → IDR → DEL',
       'Route_BLR → LKO → DEL', 'Route_BLR → MAA → DEL',
       'Route_BLR → NAG → DEL', 'Route_BLR → PNQ → DEL',
       'Route_BLR → STV → DEL', 'Route_BLR → TRV → COK → DEL',
       'Route_BLR → VGA → DEL', 'Route_BLR → VGA → HYD → DEL',
       'Route_BLR → VGA → VTZ → DEL', 'Route_BOM → AMD → ISK → HYD',
       'Route_BOM → BBI → HYD', 'Route_BOM → BDQ → DEL → HYD',
       'Route_BOM → BHO → DEL → HYD', 'Route_BOM → BLR → CCU → BBI → HYD',
       'Route_BOM → BLR → HYD', 'Route_BOM → CCU → HYD',
       'Route_BOM → COK → MAA → HYD', 'Route_BOM → DED → DEL → HYD',
       'Route_BOM → DEL → HYD', 'Route_BOM → GOI → HYD',
       'Route_BOM → GOI → PNQ → HYD', 'Route_BOM → HYD',
       'Route_BOM → IDR → DEL → HYD', 'Route_BOM → JAI → DEL → HYD',
       'Route_BOM → JDH → DEL → HYD', 'Route_BOM → JDH → JAI → DEL → HYD',
       'Route_BOM → JLR → HYD', 'Route_BOM → MAA → HYD',
       'Route_BOM → NDC → HYD', 'Route_BOM → RPR → VTZ → HYD',
       'Route_BOM → UDR → DEL → HYD', 'Route_BOM → VNS → DEL → HYD',
       'Route_CCU → AMD → BLR', 'Route_CCU → BBI → BLR',
       'Route_CCU → BBI → BOM → BLR', 'Route_CCU → BBI → HYD → BLR',
       'Route_CCU → BBI → IXR → DEL → BLR', 'Route_CCU → BLR',
       'Route_CCU → BOM → AMD → BLR', 'Route_CCU → BOM → BLR',
       'Route_CCU → BOM → COK → BLR', 'Route_CCU → BOM → GOI → BLR',
       'Route_CCU → BOM → HBX → BLR', 'Route_CCU → BOM → PNQ → BLR',
       'Route_CCU → BOM → TRV → BLR', 'Route_CCU → DEL → AMD → BLR',
       'Route_CCU → DEL → BLR', 'Route_CCU → DEL → COK → BLR',
       'Route_CCU → DEL → COK → TRV → BLR', 'Route_CCU → DEL → VGA → BLR',
       'Route_CCU → GAU → BLR', 'Route_CCU → GAU → DEL → BLR',
       'Route_CCU → GAU → IMF → DEL → BLR', 'Route_CCU → HYD → BLR',
       'Route_CCU → IXA → BLR', 'Route_CCU → IXB → BLR',
       'Route_CCU → IXB → DEL → BLR', 'Route_CCU → IXR → BBI → BLR',
       'Route_CCU → IXR → DEL → BLR', 'Route_CCU → IXZ → MAA → BLR',
       'Route_CCU → JAI → BOM → BLR', 'Route_CCU → JAI → DEL → BLR',
       'Route_CCU → KNU → BLR', 'Route_CCU → MAA → BLR',
       'Route_CCU → NAG → BLR', 'Route_CCU → PAT → BLR',
       'Route_CCU → PNQ → BLR', 'Route_CCU → RPR → HYD → BLR',
       'Route_CCU → VNS → DEL → BLR', 'Route_CCU → VTZ → BLR',
       'Route_DEL → AMD → BOM → COK', 'Route_DEL → AMD → COK',
       'Route_DEL → ATQ → BOM → COK', 'Route_DEL → BBI → COK',
       'Route_DEL → BDQ → BOM → COK', 'Route_DEL → BHO → BOM → COK',
       'Route_DEL → BLR → COK', 'Route_DEL → BOM → COK',
       'Route_DEL → CCU → BOM → COK', 'Route_DEL → COK',
       'Route_DEL → DED → BOM → COK', 'Route_DEL → GOI → BOM → COK',
       'Route_DEL → GWL → IDR → BOM → COK', 'Route_DEL → HYD → BOM → COK',
       'Route_DEL → HYD → COK', 'Route_DEL → HYD → MAA → COK',
       'Route_DEL → IDR → BOM → COK', 'Route_DEL → IXC → BOM → COK',
       'Route_DEL → IXU → BOM → COK', 'Route_DEL → JAI → BOM → COK',
       'Route_DEL → JDH → BOM → COK', 'Route_DEL → LKO → BOM → COK',
       'Route_DEL → LKO → COK', 'Route_DEL → MAA → BOM → COK',
       'Route_DEL → MAA → COK', 'Route_DEL → NAG → BOM → COK',
       'Route_DEL → PNQ → COK', 'Route_DEL → RPR → NAG → BOM → COK',
       'Route_DEL → TRV → COK', 'Route_DEL → UDR → BOM → COK',
       'Route_MAA → CCU', 'Total_Stops_1 stop', 'Total_Stops_2 stops',
       'Total_Stops_3 stops', 'Total_Stops_4 stops',
       'Total_Stops_non-stop', 'Additional_Info_1 Long layover',
       'Additional_Info_1 Short layover',
       'Additional_Info_2 Long layover', 'Additional_Info_Business class',
       'Additional_Info_Change airports',
       'Additional_Info_In-flight meal not included',
       'Additional_Info_No Info',
       'Additional_Info_No check-in baggage included',
       'Additional_Info_No info', 'Additional_Info_Red-eye flight',
       'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
       'day_of_week_Sunday', 'day_of_week_Thursday',
       'day_of_week_Tuesday', 'day_of_week_Wednesday',
       'Journey_Month_April', 'Journey_Month_December',
       'Journey_Month_January', 'Journey_Month_June',
       'Journey_Month_March', 'Journey_Month_May',
       'Journey_Month_September', 'Departure_S_Night',
       'Departure_S_Morning', 'Departure_S_Afternoon',
       'Departure_S_Evening','Price']]

corrmat = data_final.corr()
fig, ax = plt.subplots(figsize=(20,18))
sns.heatmap(corrmat, vmax=0.8, square=True)
######################################################

corr = data_final.corr()
corr.sort_values(["Price"], ascending = False, inplace = True)
corr_Price = corr.Price
print(corr_Price)

################## WORK from HEre
X = data_final.iloc[:, :-1]
y = data_final.iloc[:,181].to_frame()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)









































