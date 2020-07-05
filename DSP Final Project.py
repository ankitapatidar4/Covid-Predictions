#!/usr/bin/env python
# coding: utf-8

# # Analysis of Cases, as of May 1, 2020

# In[1]:


from IPython.display import Image


# ![title](coronavirus.png)

# ### Import the required libraries and load the data used in analysis

# * Pandas - for dataset handeling
# * Numpy - to perform operations on array
# * Matplotlib - for visualization 
# * pycountry_convert - Library for getting continent (name) to from their country names
# * ARIMA, Prophet - Prediction Models

# In[2]:


#import required libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker 
import seaborn as sns
import pycountry
import pycountry_convert as pc
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
from datetime import datetime, timedelta,date
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
from fbprophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


#load the data
case_time = pd.read_csv("cases_time_1.csv",parse_dates=['Last_Update'])
df_confirmed = pd.read_csv('time_series_covid19_confirmed_global_1.csv')
df_deaths = pd.read_csv('time_series_covid19_deaths_global_1.csv')
covid=pd.read_csv("covid_19_data.csv")


# ### Preprocessing of data analyzing the cases around the world

# In[4]:


print("Size/Shape of the dataset: ",covid.shape)
print("Checking for null values:\n",covid.isnull().sum())
print("Checking Data-type of each column:\n",covid.dtypes)


# In[5]:


#Dropping column as SNo is of no use, and "Province/State" contains too many missing values
covid.drop(["SNo"],1,inplace=True)


# In[6]:


#Converting "Observation Date" into Datetime format
covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])


# In[7]:


#Grouping different types of cases as per the date
datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()


# In[8]:


print("Basic Information")
print("Totol number of countries with Disease Spread: ",len(covid["Country/Region"].unique()))
print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])
print("Total number of Recovered Cases around the World: ",datewise["Recovered"].iloc[-1])
print("Total number of Deaths Cases around the World: ",datewise["Deaths"].iloc[-1])
print("Total number of Active Cases around the World: ",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))
print("Total number of Closed Cases around the World: ",datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1])
print("Approximate number of Confirmed Cases per Day around the World: ",np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Recovered Cases per Day around the World: ",np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Death Cases per Day around the World: ",np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Confirmed Cases per hour around the World: ",np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24)))
print("Approximate number of Recovered Cases per hour around the World: ",np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24)))
print("Approximate number of Death Cases per hour around the World: ",np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24)))
print("Number of Confirmed Cases in last 24 hours: ",datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2])
print("Number of Recovered Cases in last 24 hours: ",datewise["Recovered"].iloc[-1]-datewise["Recovered"].iloc[-2])
print("Number of Death Cases in last 24 hours: ",datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2])


# Active Cases = Number of Confirmed Cases - Number of Recovered Cases - Number of Death Cases

# In[9]:



plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
plt.title("Distribution Plot for Active Cases Cases over Date")
plt.xticks(rotation=90)


# Increase in number of Active Cases is probably an indication of Recovered case or Death case number is dropping in comparison to number of Confirmed Cases drastically. Will look for the conclusive evidence for the same
# 

# In[10]:


plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Recovered"]+datewise["Deaths"])
plt.title("Distribution Plot for Closed Cases Cases over Date")
plt.xticks(rotation=90)


# Closed Cases = Number of Recovered Cases + Number of Death Cases
# 
# Increase in number of Closed classes imply either more patients are getting recovered from the disease or more pepole are dying because of COVID-19
# 

# In[11]:



datewise["WeekOfYear"]=datewise.index.weekofyear

week_num=[]
weekwise_confirmed=[]
weekwise_recovered=[]
weekwise_deaths=[]
w=1
for i in list(datewise["WeekOfYear"].unique()):
    weekwise_confirmed.append(datewise[datewise["WeekOfYear"]==i]["Confirmed"].iloc[-1])
    weekwise_recovered.append(datewise[datewise["WeekOfYear"]==i]["Recovered"].iloc[-1])
    weekwise_deaths.append(datewise[datewise["WeekOfYear"]==i]["Deaths"].iloc[-1])
    week_num.append(w)
    w=w+1

plt.figure(figsize=(8,5))
plt.plot(week_num,weekwise_confirmed,linewidth=3)
plt.plot(week_num,weekwise_recovered,linewidth=3)
plt.plot(week_num,weekwise_deaths,linewidth=3)
plt.ylabel("Number of Cases")
plt.xlabel("Week Number")
plt.title("Weekly progress of Different Types of Cases")
plt.xlabel


# In[12]:


print("Average increase in number of Confirmed Cases every day: ",np.round(datewise["Confirmed"].diff().fillna(0).mean()))
print("Average increase in number of Recovered Cases every day: ",np.round(datewise["Recovered"].diff().fillna(0).mean()))
print("Average increase in number of Deaths Cases every day: ",np.round(datewise["Deaths"].diff().fillna(0).mean()))

plt.figure(figsize=(15,6))
plt.plot(datewise["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",linewidth=3)
plt.plot(datewise["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",linewidth=3)
plt.plot(datewise["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)
plt.xlabel("Timestamp")
plt.ylabel("Daily Increment")
plt.title("Daily increase in different Types of Cases Worldwide")
plt.xticks(rotation=90)
plt.legend()


# # Predictions using Machine Learning Models
# 
# ### Linear Regression Model for Confirmed Cases Prediction Worldwide

# In[13]:


datewise["Days Since"]=datewise.index-datewise.index[0]
datewise["Days Since"]=datewise["Days Since"].dt.days

train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
model_scores=[]


# In[14]:


lin_reg=LinearRegression(normalize=True)

lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))


# In[15]:


prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
plt.plot(datewise["Confirmed"],label="Actual Confirmed Cases")
plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()


# The Linear Regression Model is absolutely falling apart. As it is clearly visible that the trend of Confirmed Cases in absolutely not Linear.

# In[16]:


#Polynomial Regression for Prediction of Confirmed Cases

train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]

poly = PolynomialFeatures(degree = 2) 

train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]

linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)

prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)


# In[17]:


comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,predictions_poly, linestyle='--',label="Best Fit for Polynomial Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Polynomial Regression Prediction")
plt.xticks(rotation=90)
plt.legend()


# In[18]:


new_prediction_poly=[]
for i in range(1,18):
    new_date_poly=poly.fit_transform(np.array(datewise["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])


# In[19]:


#Support Vector Machine ModelRegressor for Prediction of Confirmed Cases
    
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]    

#Intializing SVR Model
svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)

#Fitting model on the training data
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))


# In[20]:


plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Support Vector Machine Regressor Prediction")
plt.xticks(rotation=90)
plt.legend()

new_date=[]
new_prediction_lr=[]
new_prediction_svm=[]
for i in range(1,18):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])

pd.set_option('display.float_format', lambda x: '%.6f' % x)
model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_poly,new_prediction_svm),columns=["Dates","Linear Regression Prediction","Polynonmial Regression Prediction","SVM Prediction"])
model_predictions.head()


# The Polynomial Regression has low RMSE value than Linear Regression and Support Vector Machine in predicting the number of cases worldwide.

# ## Preprocessing data for Continent-wise Analysis

# In[21]:


#drop the columns not used in our analysis
case_time = case_time.drop(['Recovered','Active','Delta_Confirmed', 'Delta_Recovered',
       'Incident_Rate', 'People_Tested', 'People_Hospitalized',
       'Province_State', 'FIPS', 'UID', 'iso3', 'Report_Date_String'],axis =1)

#renaming the column names
df_confirmed = df_confirmed.rename(columns={"Province/State":"State","Country/Region": "Country"})

df_deaths = df_deaths.rename(columns={"Province/State":"State","Country/Region": "Country"})


# The name of countries in the dataset is not in standard format as per pycountry library, so the names needs to be modified.

# In[22]:


#replacing the names of the country
case_time["Country_Region"].replace({'US': 'USA','Korea, South':'South Korea',
                'Taiwan*':'Taiwan','Congo (Kinshasa)':'Democratic Republic of the Congo',
                "Cote d'Ivoire":"Côte d'Ivoire",'Reunion':'Réunion',
                'Congo (Brazzaville)':'Republic of the Congo','Bahamas, The':'Bahamas',
                'Gambia, The':'Gambia'}, inplace=True)

df_confirmed["Country"].replace({'US': 'USA','Korea, South':'South Korea',
                'Taiwan*':'Taiwan','Congo (Kinshasa)':'Democratic Republic of the Congo',
                "Cote d'Ivoire":"Côte d'Ivoire",'Reunion':'Réunion',
                'Congo (Brazzaville)':'Republic of the Congo','Bahamas, The':'Bahamas',
                'Gambia, The':'Gambia'}, inplace=True)

df_deaths["Country"].replace({'US': 'USA','Korea, South':'South Korea',
                'Taiwan*':'Taiwan','Congo (Kinshasa)':'Democratic Republic of the Congo',
                "Cote d'Ivoire":"Côte d'Ivoire",'Reunion':'Réunion',
                'Congo (Brazzaville)':'Republic of the Congo','Bahamas, The':'Bahamas',
                'Gambia, The':'Gambia'}, inplace=True)


# In[23]:


# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}


# In[24]:


# Defininng Function for getting continent code for country.
def country_to_continent_code(country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
    except :
        return 'na'


# In[25]:


#insert continent column in all the three datasets 
case_time.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in case_time["Country_Region"].values])
case_time = case_time[case_time["continent"] != "Others"]

df_confirmed.insert(2,"continent", [continents[country_to_continent_code(country)] for country in df_confirmed["Country"].values])

df_deaths.insert(2,"continent", [continents[country_to_continent_code(country)] for country in df_confirmed["Country"].values])


# ### Defining Functions

# * plot_params()
# * visualize_covid_cases()

# In[26]:


#replace nan values with blanks
df_confirmed = df_confirmed.replace(np.nan, '', regex=True)
df_deaths = df_deaths.replace(np.nan, '', regex=True)


# In[27]:


#functions for vizualizations
def plot_params(ax,axis_label= None, plt_title = None,label_size=15, axis_fsize = 15, title_fsize = 20, scale = 'linear' ):
    # Tick-Parameters
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', width=1,labelsize=label_size)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3, color='0.8')
    
    # Grid
    plt.grid(lw = 1, ls = '-', c = "0.7", which = 'major')
    plt.grid(lw = 1, ls = '-', c = "0.9", which = 'minor')

    # Plot Title
    plt.title( plt_title,{'fontsize':title_fsize})
    
    # Yaxis sacle
    plt.yscale(scale)
    plt.minorticks_on()
    # Plot Axes Labels
    xl = plt.xlabel(axis_label[0],fontsize = axis_fsize)
    yl = plt.ylabel(axis_label[1],fontsize = axis_fsize)


# In[28]:


#continent-wise trend analysis function
def contitent_wise_covid_cases(confirmed, deaths, continent=None , country = None , state = None, period = None, figure = None, scale = "linear"):
    x = 0
    if figure == None:
        f = plt.figure(figsize=(10,10))
        # Sub plot
        ax = f.add_subplot(111)
    else :
        f = figure[0]
        # Sub plot
        ax = f.add_subplot(figure[1],figure[2],figure[3])
    ax.set_axisbelow(True)
    plt.tight_layout(pad=10, w_pad=5, h_pad=5)
    
    stats = [confirmed, deaths]
    label = ["Confirmed", "Deaths"]
    
    if continent != None:
        params = ["continent",continent]

    color = ["maroon","green"]
    marker_style = dict(linewidth=3,marker='o',markersize=4, markerfacecolor='#ffffff')
    for i,stat in enumerate(stats):
        cases = np.sum(np.asarray(stat[stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        date = np.arange(1,cases.shape[0]+1)[x:]
        plt.plot(date,cases,label = label[i]+" (Total : "+str(cases[-1])+")",color=color[i],**marker_style)
        plt.fill_between(date,cases,color=color[i],alpha=0.3)
        Total_confirmed =  np.sum(np.asarray(stats[0][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        
    # Plot Axes Labels
        axis_label = ["Days ("+df_confirmed.columns[5]+" - "+df_confirmed.columns[-1]+")","No of Cases"]
    
    # Plot Parameters
    plot_params(ax,axis_label,scale = scale)
    
    # Title for each subplots
    plt.title("COVID-19 Cases for "+params[1] ,{'fontsize':25}) 
    
    # Legend Location
    l = plt.legend(loc= "best",fontsize = 15)
    
    if figure == None:
        plt.show()


# ### Exploring the data Continent-wise

# In[29]:


df_continents= df_confirmed.groupby(["continent"]).sum()      
continents = df_continents.sort_values(df_continents.columns[-1],ascending = False).index


# In[30]:


cols =2
rows = int(np.ceil(continents.shape[0]/cols))
f = plt.figure(figsize=(20,9*rows))
for i,continent in enumerate(continents):
    contitent_wise_covid_cases(df_confirmed,df_deaths,continent = continent,figure = [f,rows,cols, i+1])

plt.show()


# From above plots, it can be seen that:-
# * The number of confirmed cases are increasing exponentially over time in all the continents. 
# * There is a increasing trend in number of deaths due to covid in all the continents
# 

# In[31]:


case_time.head()


# ### Exploring the data for top 10 countries having most confirmed cases

# In[32]:


top = case_time[case_time['Last_Update'] == case_time['Last_Update'].max()]
top_cases_country = top.groupby(by = 'Country_Region')['Confirmed'].sum().sort_values(ascending = False).head(10).reset_index()
top_cases_country


# In[33]:


plt.figure(figsize= (10,8))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Total cases",fontsize = 14)
plt.ylabel('Country',fontsize = 14)
plt.title("Top 10 countries having most confirmed cases" , fontsize = 18)
ax = sns.barplot(x = top_cases_country.Confirmed, y = top_cases_country.Country_Region)
for i, (value, name) in enumerate(zip(top_cases_country.Confirmed,top_cases_country.Country_Region)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=8, ha='left',  va='center')
ax.set(xlabel='Total cases', ylabel='Country')


# In[34]:


case_nums_country = df_confirmed.groupby("Country").sum().drop(['Lat','Long'],axis =1).apply(lambda x: x[x > 0].count(), axis =0)
d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in case_nums_country.index]


# ### Exploring the data for top 10 countries having most confirmed cases on daily basis

# In[35]:


thoudand = 1000
prediction_days =10
temp = df_confirmed.groupby('Country').sum().diff(axis=1).sort_values(df_deaths.columns[-1],ascending =False).head(10).replace(np.nan,0)
threshold = 0
f = plt.figure(figsize=(20,12))
ax = f.add_subplot(111)
for i,country in enumerate(temp.index):
    t = temp.loc[temp.index== country].values[0]
    t = t[t>=threshold]
    
    date = np.arange(0,len(t[:]))
    plt.plot(date,t/thoudand,'-o',label = country,linewidth =2, markevery=[-1])
case_nums_country = df_confirmed.groupby("Country").sum().drop(['Lat','Long'],axis =1).apply(lambda x: x[x > 0].count(), axis =0)


nextdays = [(datetime.strptime(d[-1],'%d %b')+timedelta(days=i)).strftime("%d %b") for i in range(1,prediction_days+1)]
total =d+nextdays

# X-axis
plt.xticks(list(np.arange(0,len(total),int(len(total)/5))),total[:-1:int(len(total)/5)]+[total[-1]])

# Tick-Parameters
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(which='both', width=1,labelsize=14)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3, color='0.8')

# Grid
plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')
plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')

# Plot Title
plt.title("COVID-19 Daily Confirmed Cases in Different Countries",{'fontsize':24})

# Axis Lable
plt.xlabel("Date",fontsize =18)
plt.ylabel("Number of Daily Confirmed Cases (Thousand)",fontsize =18)

# plt.yscale("log")
plt.legend(fontsize=18) 
#plt.savefig(out+"daily confirmed cases countrywise.png")
plt.show()


# We see that the number of covid confirmed cases are increasing tremendously for US than any other country daily.

# ## Analysis of US Cases

# ## Time Series Analysis

# In[36]:


#Creating the dataframe for cases in USA
series_us = case_time[case_time['Country_Region'] == "USA"]
#Creating dataframe for time series implementation
series_us = series_us[["Last_Update","Confirmed"]]


# In[37]:


#setting Last_Update as index
series_us = series_us.set_index("Last_Update")
series_us.info()


# In[38]:


# line plot of dataset
plt.figure(figsize=(12,8))
plt.plot(series_us)


# As per above plot,there is exponential increase in the number of cases.

# In[39]:


plt.figure(figsize=(10,8))
autocorrelation_plot(series_us)


# The number of cases had positive correlation with initial lags and then there is a negative correlation after 25 lags.

# In[40]:


plot_acf(series_us)
pyplot.show()


# From above plot it can be seen that the correlation had decreasing gradually.

# ### Predictions using ARIMA model

# The parameters of the ARIMA model are defined as follows:
# 
# * p: The number of lag observations included in the model.
# * d: The number of times that the raw observations are differenced.
# * q: The size of the moving average window.

# In[41]:


model2 = ARIMA(series_us, order=(5,1,0))
model_us = model2.fit(disp=0)


# In[42]:


print(model_us.summary())


# In[43]:


# plot residual errors
residuals = pd.DataFrame(model_us.resid)
plt.plot(residuals)
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# This suggests that the mean and variance is not stationary and will require differencing to make it stationary

# In[44]:


X = series_us.values
size = int(len(X) * 0.70)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[45]:


mse_us_confirmed = mean_squared_error(test, predictions)
print('ARIMA Model MSE: %.3f' % mse_us_confirmed)
rmse_us_confirmed = np.sqrt(mse_us_confirmed)
print('ARIMA Model RMSE: %.3f' % rmse_us_confirmed)


# In[46]:


# plot
plt.figure(figsize=(8,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# The blue line represents the actual data points while red line shows the prediction values. The predictions looks to be pretty good though the root mean square value is quite on the higher side. This can be attributed to the huge variation in the cases.

# ### Predictions using Facebook Prophet

# In[47]:


df_us = series_us.copy()
plt.figure(figsize=(10,6))
plt.plot(df_us)
plt.legend(['Confirmed'])
df_us.head()


# Preparing the data explicitly as per facebook prophet model i.e creating the columns 'ds and 'y'.

# In[48]:


df_us = df_us.reset_index()
df_us.columns = ['ds','y']
df_us.head()


# In[49]:


us_prophet = Prophet(daily_seasonality=True)
us_prophet.fit(df_us)


# In[50]:


future_us = us_prophet.make_future_dataframe(periods=5)
forecast_us = us_prophet.predict(future_us)
forecast_us.tail().T


# In[51]:


forecast_us[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[52]:


us_prophet.plot(forecast_us);
us_prophet.plot_components(forecast_us);


# In[53]:


from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics


# In[54]:


cross_validation_results = cross_validation(us_prophet, initial='95 days', period='5 days', horizon='5 days')
print(cross_validation_results)


# In[55]:


performance_metrics_results = performance_metrics(cross_validation_results)
print(performance_metrics_results)


# cross validation to measure forecast error using historical data. 

# In[56]:


from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(cross_validation_results, metric='mape')


# ARIMA model seems to have low RMSE value than Facebook Prophet model and hence, it performs better the available data.

# ### US Death Cases Prediction using Time Series

# #### ARIMA Model

# In[57]:


#Creating the dataframe for cases in USA
series_us_deaths = case_time[case_time['Country_Region'] == "USA"]
#Creating dataframe for time series implementation
series_us_deaths = series_us_deaths[["Last_Update","Deaths"]]


# In[58]:


#setting Last_Update as index
series_us_deaths = series_us_deaths.set_index("Last_Update")
series_us_deaths.info()


# In[59]:


print(series_us_deaths.head(5))


# In[60]:


plt.figure(figsize=(10,8))
plt.plot(series_us_deaths)


# In[61]:


autocorrelation_plot(series_us_deaths)


# In[ ]:





# In[62]:


X = series_us_deaths.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,2,0))
	model_fit = model.fit(disp=5)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[63]:


mse_us_death = mean_squared_error(test, predictions)
print('ARIMA model MSE: %.3f' % mse_us_death)


# In[64]:


rmse_us_death = np.sqrt(mse_us_death)
print('ARIMA model RMSE: %.3f' % rmse_us_death)


# In[65]:


plt.figure(figsize=(8,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# #### Facebook Prophet Model

# In[66]:


df_us_deaths = series_us_deaths.copy()
plt.figure(figsize=(10,6))
plt.plot(df_us_deaths)
plt.legend(['Deaths'])


# In[67]:


df_us_deaths = df_us_deaths.reset_index()
df_us_deaths.columns = ['ds','y']
df_us_deaths.head()


# In[68]:


prophet_us_deaths = Prophet(daily_seasonality=True)
prophet_us_deaths.fit(df_us_deaths)


# In[69]:


future_us_deaths = prophet_us_deaths.make_future_dataframe(periods=5)
forecast_us_deaths = prophet_us_deaths.predict(future_us_deaths)
forecast_us_deaths.tail().T


# In[70]:


forecast_us_deaths[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[71]:


prophet_us_deaths.plot(forecast_us_deaths);
prophet_us_deaths.plot_components(forecast_us_deaths);


# In[72]:


cross_validation_results_deaths = cross_validation(prophet_us_deaths, initial='90 days', period='10 days', horizon='10 days')
print(cross_validation_results_deaths)


# In[73]:


performance_metrics_results_deaths = performance_metrics(cross_validation_results_deaths)
print(performance_metrics_results_deaths)


# From the above two model,the facebook prophet model performs better in predicting the death cases in US.
# Comparing the root mean square value of two models
# 
# * Facbook prophet model has RMSE of 89.  
# * ARIMA has RMSE of 520.

# In[ ]:




