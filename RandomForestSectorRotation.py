#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
import seaborn
from scipy import stats
#import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from empyrical import max_drawdown
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[3]:


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# # Data Input

# In[4]:


hs300 = pd.read_excel("沪深300.xlsx", sheet_name = "沪深300")
hs300 = hs300.set_index("Date")
hs300


# ### Put the information of sector index (dataframe) into a dictionary

# In[5]:


work_area = ["沪深300能源","沪深300材料","沪深300工业","沪深300可选","沪深300消费","沪深300医药","沪深300金融","沪深300信息","沪深300电信","沪深300公用"]
dict_hs_work = {}
for work in work_area:
    dict_hs_work[work] = pd.read_excel("沪深300行业指数ENG.xlsx", sheet_name = work)
    dict_hs_work[work] = dict_hs_work[work].set_index('Date')
    dict_hs_work[work] = dict_hs_work[work].fillna(dict_hs_work[work].mean()) 
    dict_hs_work[work]["Tomr_Trend"] = dict_hs_work[work]["open"].shift(-1) > dict_hs_work[work]["open"]
    dict_hs_work[work].iloc[dict_hs_work[work].shape[0]-1,dict_hs_work[work].shape[1]-1] = np.nan
    dict_hs_work[work]["5_day_Trend"] = dict_hs_work[work]["open"].shift(-5) > dict_hs_work[work]["open"]
    dict_hs_work[work].iloc[-5:,dict_hs_work[work].shape[1]-1] = np.nan
li_hs_work = [dict_hs_work[work] for work in work_area]
df_hs_work = pd.concat(li_hs_work, axis = 1, keys = work_area)
df_hs_work.columns.names = ['Work','Daily info']
df_hs_work


# ### Divide the origin dictionary into 3 dictionary

# ### 3 dictionary:
# #### About the factor information for each sector (Explanatory Variables)
# #### About the trend in next 5 days for each sector, upward or downward (Explained Variables)
# #### About the trading information for each sector (like close price, open price)

# In[6]:


dict_hs300work_factor = {}
dict_hs300work_trend5 = {}
dict_hs300work_trade_info = {}
for work in work_area:
    dict_hs300work_factor[work] = dict_hs_work[work].iloc[ : ,5:-2]
    dict_hs300work_trend5[work] = dict_hs_work[work].iloc[ : ,-1]
    dict_hs300work_trade_info[work] = dict_hs_work[work].iloc[ : ,0:5]


# In[7]:


for work in work_area:
    df_extreme_factor = dict_hs300work_factor[work].quantile([0.025, 0.975])
    dict_hs300work_factor[work] = dict_hs300work_factor[work].clip(df_extreme_factor.iloc[0], df_extreme_factor.iloc[1],axis = 1)
    dict_hs300work_factor[work] = (dict_hs300work_factor[work]-dict_hs300work_factor[work].mean())/dict_hs300work_factor[work].std()


# In[8]:


df_hs_work.loc["2015-05-11":"2020-05-07"].shape[0]


# In[9]:


train_prop = np.round(df_hs_work.loc["2015-05-11":"2020-05-07"].shape[0] / df_hs_work.shape[0],1)
train_prop


# #### Using open price to get the return and put each sectors' return series into a dataframe 

# In[246]:


dict_Return_Work = {}
for work in work_area:
    dict_Return_Work[work] = dict_hs300work_trade_info[work]["open"] / dict_hs300work_trade_info[work]["open"].iloc[0] - 1
dict_Return_Work ["Date"] = hs300.index
df_Return_Work = pd.DataFrame(dict_Return_Work).set_index("Date")
df_Return_Work


# In[247]:


df_Return_Work.plot(figsize = (12,8))


# #### For Cross Validation

# In[11]:


dict_factor_train = {}
dict_trend5_train = {}
dict_factor_test = {}
dict_trend5_test ={}

for work in work_area:
    x_train, x_test, y_train, y_test = train_test_split(dict_hs300work_factor[work].iloc[:-5],dict_hs300work_trend5[work].dropna(),train_size = train_prop, random_state = 1)
    dict_factor_train[work] = x_train
    dict_factor_test[work] = x_test
    dict_trend5_train[work] = y_train
    dict_trend5_test[work] = y_test


# ### The following Function returns the accruacy measurement of random forest model with given unique minimum sample leaf but different minimum sample splits

# In[12]:


def para_select_RF2(dict_hs300work_factor, dict_hs300work_trend5,dict_factor_train,dict_trend5_train, msl):
    dict_score_min_ss_work = {}
    min_sample_split_candi = range(6, 55, 6)
    for work in work_area:
        dict_accur = {}
        oob = []
        CV =[]
        accur_train = []
        accur_test = []
        auc_li = []
        for mss in min_sample_split_candi:
            RF = RandomForestClassifier(n_estimators = 100, max_features= int(np.ceil(math.sqrt(dict_factor_train[work].shape[1]))), max_depth = None, min_samples_split = mss, bootstrap=True, oob_score = True, min_samples_leaf = msl)
            oob.append(RF.oob_score)
            CV.append(cross_val_score(RF, dict_factor_train[work], dict_trend5_train[work]).mean())
            RF.fit(dict_hs300work_factor[work].loc["2015-05-11":"2020-05-07"], dict_hs300work_trend5[work].loc["2015-05-11":"2020-05-07"])
            trend5_train_pred = RF.predict(dict_hs300work_factor[work].loc["2015-05-11":"2020-05-07"])
            trend5_test_pred = RF.predict(dict_hs300work_factor[work].loc["2020-05-08":"2021-04-27"])
            accur_train.append(accuracy_score(dict_hs300work_trend5[work].loc["2015-05-11":"2020-05-07"],trend5_train_pred))
            accur_test.append(accuracy_score(dict_hs300work_trend5[work].loc["2020-05-08":"2021-04-27"],trend5_test_pred))
        
            trend5_preprob = RF.predict_proba(dict_hs300work_factor[work].loc["2020-05-08":"2021-04-27"])[:,1]
            fpr_Nb, tpr_Nb, _ = roc_curve(dict_hs300work_trend5[work].loc["2020-05-08":"2021-04-27"], trend5_preprob)
            accval = auc(fpr_Nb, tpr_Nb)
            auc_li.append(accval)
        dict_accur["OOB"] = oob
        dict_accur["Cross Validation Score"] = CV
        dict_accur["Accuracy Train"] = accur_train
        dict_accur["Accuracy Test"] = accur_test
        dict_accur["AUC"] = auc_li
        dict_accur["min sample split"] = min_sample_split_candi
        dict_score_min_ss_work[work] = pd.DataFrame(dict_accur).set_index("min sample split")
    return(dict_score_min_ss_work) 


# ### Through the iteration given different minimum sample leaf, then we can find the dictionary about the accruacy of random forest model with different minimum sample leaf

# ### This dictionary's element is also the dictionary about the accruacy of random forest model with different minimum sample split corresponding to the unique minimum sample leaf (keys of dictionary)

# In[13]:


dict_para_select_msl_2 = {}
min_sample_leaf_candi = range(1,16,2)
for msl in min_sample_leaf_candi:
    dict_para_select_msl_2[msl] = para_select_RF2(dict_hs300work_factor, dict_hs300work_trend5,dict_factor_train,dict_trend5_train, msl)


# #### Generate the 4 dictionaries about accuracy measurement correpoding to each sectors' random forest and  all possible parameters combinations

# In[14]:


dict_CV_work2 = {}
dict_Accur_Train_work2 = {}
dict_Accur_Test_work2 = {}
dict_Accur_AUC_work2 = {}

for work in work_area:
    Li_work_accu2 = [dict_para_select_msl_2[msl][work] for msl in min_sample_leaf_candi]
    df_work_accu2 = pd.concat(Li_work_accu2, axis = 1, keys = min_sample_leaf_candi)
    df_work_accu2.columns.names = ["min sample leaf","Accuracy measure"]
    dict_CV_work2[work] = df_work_accu2.xs("Cross Validation Score", level = "Accuracy measure", axis = 1)
    dict_Accur_Train_work2[work] = df_work_accu2.xs("Accuracy Train", level = "Accuracy measure", axis = 1)
    dict_Accur_Test_work2[work] = df_work_accu2.xs("Accuracy Test", level = "Accuracy measure", axis = 1)
    dict_Accur_AUC_work2[work] = df_work_accu2.xs("AUC", level = "Accuracy measure", axis = 1)


# In[15]:


dict_work_model_para2 = {}


# ## The following 10 heatmap shows all kinds of parameters combination and corresponding accuracy

# # Notice that this is the random forest model, if you re-run this code, the results of best parameter selection may become different as what we analyze in report

# In[16]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300能源"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300能源"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300能源"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300能源"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[36]:


dict_work_model_para2["沪深300能源"] = [18,1]


# In[18]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300材料"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300材料"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300材料"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300材料"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[37]:


dict_work_model_para2["沪深300材料"] = [30,13]


# In[20]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300工业"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300工业"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300工业"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300工业"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[38]:


dict_work_model_para2["沪深300工业"] = [36,5]


# In[22]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300可选"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300可选"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300可选"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300可选"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[39]:


dict_work_model_para2["沪深300可选"] = [48,13]


# In[24]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300消费"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300消费"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300消费"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300消费"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[40]:


dict_work_model_para2["沪深300消费"] = [48,13]


# In[26]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300医药"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300医药"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300医药"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300医药"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[41]:


dict_work_model_para2["沪深300医药"] = [54,15]


# In[28]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300金融"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300金融"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300金融"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300金融"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[42]:


dict_work_model_para2["沪深300金融"] = [36,13]


# In[30]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300信息"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300信息"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300信息"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300信息"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[43]:


dict_work_model_para2["沪深300信息"] = [36,5]


# In[32]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300电信"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300电信"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300电信"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300电信"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[44]:


dict_work_model_para2["沪深300电信"] = [30,15]


# In[34]:


fig, axes = plt.subplots(figsize = (10,8), nrows = 2, ncols = 2)
seaborn.heatmap(dict_CV_work2["沪深300公用"], annot = True, cmap = "Blues_r", ax = axes[0][0])
axes[0][0].set_title("Cross Validation Score")
seaborn.heatmap(dict_Accur_Train_work2["沪深300公用"], annot = True, cmap = "Greens_r", ax = axes[0][1])
axes[0][1].set_title("Accuracy Train")
seaborn.heatmap(dict_Accur_Test_work2["沪深300公用"], annot = True, cmap = "Oranges_r", ax = axes[1][0])
axes[1][0].set_title("Accuracy Test")
seaborn.heatmap(dict_Accur_AUC_work2["沪深300公用"], annot = True, cmap = "Greys_r", ax = axes[1][1])
axes[1][1].set_title("AUC")
plt.tight_layout()


# In[45]:


dict_work_model_para2["沪深300公用"] = [18,5]


# ### Record the parameters selection for random forest model of each sectors into a dictionary

# In[46]:


dict_work_model_para2


# ### Fitting the random forest model for each sectors using the parameters we select

# In[47]:


dict_RF_2 = {}
for work in work_area:
    RF = RandomForestClassifier(n_estimators = 100, max_features= int(np.ceil(math.sqrt(dict_factor_train[work].shape[1]))), max_depth = None, min_samples_split = dict_work_model_para2[work][0], bootstrap=True, oob_score = True, min_samples_leaf = dict_work_model_para2[work][1], random_state = 1)
    RF.fit(dict_hs300work_factor[work].loc["2015-05-11":"2020-05-07"], dict_hs300work_trend5[work].loc["2015-05-11":"2020-05-07"])
    dict_RF_2[work] = RF


# ## This is the back testing platform ONLY for Timing Strategy
# 
# ### This is because we conmbine the strategy and back test trading

# In[457]:


class Timing_Strategy:
    def __init__(self, name,df_hist_trade_info, df_hist_factor, df_hist_trend5, init_value, RF):
        self.name = name
        self.df_hist_trade_info = df_hist_trade_info
        self.df_hist_factor = df_hist_factor
        self.df_hist_trend5 = df_hist_trend5
        self.trading_days = df_hist_trade_info.shape[0]
        self.init_value = init_value
        self.Position = init_value * np.ones(self.trading_days)
        self.Return = np.zeros(self.trading_days)
        self.Based_Return = np.zeros(self.trading_days)
        self.share = np.zeros(self.trading_days) 
        self.hold = False
        self.RF = RF
        self.charge = 0
        self.long_times = 0
        self.short_times = 0
        self.Daily_Return = []
        self.Daily_Based_Return = []
        
    def Back_testing(self):
        for date_index in range(0,self.trading_days):
            if date_index != 0:
                self.Position[date_index] = self.Position[date_index-1] 
            if date_index % 5 == 0:
                trend5_forecast = (self.RF).predict(np.array(self.df_hist_factor.iloc[date_index]).reshape(1,41))[0]
                if self.hold == False:
                    if trend5_forecast == 1:
                        self.share[date_index] = self.Position[date_index] / self.df_hist_trade_info["open"].iloc[date_index]
                        self.charge += 0.0007 * self.Position[date_index]
                        self.Position[date_index] = (1-0.0007) * self.Position[date_index]
                        self.long_times += 1
                        self.hold = True
                else:
                    if trend5_forecast == 0:
                        self.share[date_index] = 0
                        self.short_times += 1
                        self.hold = False
                    else:
                        self.share[date_index] = self.share[date_index - 1]
            else:
                if date_index != 0:
                    self.share[date_index] = self.share[date_index - 1]
                if self.hold == True:
                    self.Position[date_index] = self.share[date_index] * self.df_hist_trade_info["open"].iloc[date_index]
        
        self.Return = self.Position / self.init_value - 1
        self.Based_Return = self.df_hist_trade_info["open"] / self.df_hist_trade_info["open"].iloc[0] - 1 
        
        for date_index in range(1,self.trading_days):
            self.Daily_Return.append(self.Position[date_index]/self.Position[date_index-1] - 1)
            self.Daily_Based_Return.append(self.df_hist_trade_info["open"].iloc[date_index]/self.df_hist_trade_info["open"].iloc[date_index-1] - 1)
            
    def Back_testing_plot(self):
        dict_compare_return = {}
        dict_compare_return["Random Forest Timing"] = self.Return
        dict_compare_return[self.name] = self.Based_Return
        df_compare_return = pd.DataFrame(dict_compare_return)
        df_compare_return.plot.area(title = self.name, figsize = (12,8), alpha = 0.4, stacked = False,ylabel = "Cumulative Return")
        
    def ex_ret(self):
        self.ex_ret = self.Return[self.trading_days-1] - self.Based_Return[self.trading_days-1]
        return np.round(self.ex_ret,3)
    
    def tim_maxdrawdown(self):
        return np.round(max_drawdown(np.array(self.Daily_Return)),3)
    
    def base_maxdrawdown(self):
        return np.round(max_drawdown(np.array(self.Daily_Based_Return)),3)
    
    
    def reduced_drawdown(self):
        return self.tim_maxdrawdown() - self.base_maxdrawdown()    
        
    def __del__(self):
        print("Timing Strategy for ", self.name," is Over")
    


# ### Do the back-testing for each sectors' Timing strategy

# In[458]:


dict_strategy_object2 = {}
for work in work_area:
    dict_strategy_object2[work] = Timing_Strategy(work,dict_hs300work_trade_info[work].loc["2020-05-08":"2021-05-07"], dict_hs300work_factor[work].loc["2020-05-08":"2021-05-07"], dict_hs300work_trend5[work].loc["2020-05-08":"2021-05-07"], 1000000, dict_RF_2[work])
    dict_strategy_object2[work].Back_testing()


# ### The following 10 area line plot is the back-testing for Timing strategies corresponding to 10 sectors

# In[459]:


dict_strategy_object2["沪深300能源"].Back_testing_plot()


# In[460]:


dict_strategy_object2["沪深300材料"].Back_testing_plot()


# In[461]:


dict_strategy_object2["沪深300工业"].Back_testing_plot()


# In[462]:


dict_strategy_object2["沪深300可选"].Back_testing_plot()


# In[463]:


dict_strategy_object2["沪深300消费"].Back_testing_plot()


# In[464]:


dict_strategy_object2["沪深300医药"].Back_testing_plot()


# In[465]:


dict_strategy_object2["沪深300金融"].Back_testing_plot()


# In[466]:


dict_strategy_object2["沪深300信息"].Back_testing_plot()


# In[467]:


dict_strategy_object2["沪深300电信"].Back_testing_plot()


# In[468]:


dict_strategy_object2["沪深300公用"].Back_testing_plot()


# ### This is to draw the table of back-tesing result

# In[382]:


dict_back_test_info = {}
Long_times = []
Short_times = []
Charge = []
excess_return = []
tim_drawdown = []
base_drawdown = []
reduce_drawdown =[]
for work in work_area:
    Long_times.append(dict_strategy_object2[work].long_times)
    Short_times.append(dict_strategy_object2[work].short_times)
    Charge.append(np.round(dict_strategy_object2[work].charge,3))
    excess_return.append(dict_strategy_object2[work].ex_ret())
    tim_drawdown.append(dict_strategy_object2[work].tim_maxdrawdown())
    base_drawdown.append(dict_strategy_object2[work].base_maxdrawdown())
    reduce_drawdown.append(dict_strategy_object2[work].reduced_drawdown())
dict_back_test_info["Long Times"] = Long_times
dict_back_test_info["Short Times"] = Short_times
dict_back_test_info["Total Charge"] = Charge
dict_back_test_info["Excess Return of Timing Strategy"] = excess_return
dict_back_test_info["Max Drawdown of Timing Strategy"] = tim_drawdown
dict_back_test_info["Max Drawdown of Base Portfolio"]d = base_drawdown
dict_back_test_info["Reduced Drawdown"] = reduce_drawdown
dict_back_test_info["Industry"] = work_area
df_back_test_info = pd.DataFrame(dict_back_test_info).set_index("Industry")
df_back_test_info


# In[442]:


df_back_test_info["Excess Return of Timing Strategy"].mean()


# In[443]:


df_back_test_info["Reduced Drawdown"].mean()


# In[455]:


dict_position ={}
for work in work_area:
    dict_position[work] = dict_strategy_object2[work].Position
dict_position["Date"] = hs300.loc["2020-05-08":"2021-05-07"].index
df_position = pd.DataFrame(dict_position).set_index("Date")
df_position.iloc[-1]


# In[456]:


df_position.iloc[-1].mean()


# ## This is the back-testing platform ONLY for sector rotation strategy

# In[469]:


class Sector_Rotation:
    def __init__(self, df_hs300,dict_hs300work_trade_info, dict_hs300work_factor, dict_hs300work_trend5, init_value, dict_RF, work_area):
        self.df_hs300 = df_hs300
        self.dict_hs300work_trade_info = dict_hs300work_trade_info
        self.dict_hs300work_factor = dict_hs300work_factor
        self.dict_hs300work_trend5 = dict_hs300work_trend5
        self.trading_days = dict_hs300work_trade_info[work_area[0]].shape[0]
        self.Position = init_value * np.ones(self.trading_days)
        self.Return = np.zeros(self.trading_days)
        self.Based_Return = self.df_hs300["open"] / self.df_hs300["open"].iloc[0] - 1
        self.share = list(np.zeros(self.trading_days).reshape(self.trading_days,1)) 
        self.hold = list(np.zeros(self.trading_days).reshape(self.trading_days,1)) 
        self.whether_hold = False
        self.dict_RF = dict_RF
        self.charge = 0
        self.long_times = 0
        self.short_times = 0
        self.Daily_Return = []
        self.Daily_Based_Return = []
        self.work_area = work_area
        self.init_value = init_value
        self.probability_record = {"1st Work":[],"1st Work prob":[], "2nd Work":[],"2nd Work prob":[], "mean prob":[], "Date":[]}
        
    def Back_testing(self):
        for date_index in range(0,self.trading_days):
            if date_index != 0:
                self.Position[date_index] = self.Position[date_index-1] 
            if date_index % 5 == 0:
                dict_Prob_up = {}
                li_Prob_up = []
                for work in self.work_area:
                    dict_Prob_up[work] = self.dict_RF[work].predict_proba(np.array(dict_hs300work_factor[work].iloc[date_index]).reshape(1,41))[0,1]    
                    li_Prob_up.append(dict_Prob_up[work])
                Prob_up_sort = sorted(dict_Prob_up.items(), key = lambda acd:acd[1], reverse=True)
                self.probability_record["mean prob"].append(np.array(li_Prob_up).mean())
                work1 = Prob_up_sort[0][0]
                work2 = Prob_up_sort[1][0]
                work1_prob = Prob_up_sort[0][1]
                work2_prob = Prob_up_sort[1][1]
                self.probability_record["1st Work"].append(work1)
                self.probability_record["1st Work prob"].append(work1_prob)
                self.probability_record["2nd Work"].append(work2)
                self.probability_record["2nd Work prob"].append(work2_prob)
                self.probability_record["Date"].append(self.df_hs300.index[date_index])
                
                if work1_prob >= 0.5 and work2_prob < 0.5:
                    if work1 not in self.hold[date_index-1] or date_index == 0:
                        self.hold[date_index] = [work1]
                        self.share[date_index] = [self.Position[date_index]/dict_hs300work_trade_info[work1]["open"].iloc[date_index]]
                        self.charge += 0.0007 * self.Position[date_index]
                        self.Position[date_index] = (1-0.0007) * self.Position[date_index]
                        self.long_times += 1
                        if date_index != 0:
                            if self.share[date_index-1] != 0 and len(self.share[date_index-1]) == 1:
                                self.short_times += 1
                            elif len(self.share[date_index-1]) == 2:
                                self.short_times += 2
                        self.whether_hold = True
                    else:
                        self.hold[date_index] = [work1]
                        if len(self.hold[date_index]) == 2:
                            share_work1 = self.Position[date_index]/dict_hs300work_trade_info[work1]["open"].iloc[date_index]
                            self.share[date_index] = [share_work1]
                            if self.hold[date_index-1][0] == work1:
                                share_work1_yest = self.share[date_index-1][0] 
                            else:
                                share_work1_yest = self.share[date_index-1][1]
                            pos_other_part = self.Position[date_index] - share_work1_yest * dict_hs300work_trade_info[work1]["open"].iloc[date_index]
                            self.charge += 0.0007 * pos_other_part
                            self.Position[date_index] -= 0.0007 * pos_other_part
                            self.long_times += 1
                            self.short_times += 1   
                            self.whether_hold = True
                        else:
                            self.share[date_index] = self.share[date_index-1]
                            
                elif work2_prob >= 0.5:
                    if date_index == 0 or (work1 not in self.hold[date_index-1] and work2 not in self.hold[date_index-1]):
                        self.hold[date_index] = [work1, work2]
                        Total_prob = work1_prob + work2_prob
                        self.share[date_index] = [work1_prob/Total_prob * self.Position[date_index]/dict_hs300work_trade_info[work1]["open"].iloc[date_index] ,work2_prob/Total_prob * self.Position[date_index]/dict_hs300work_trade_info[work2]["open"].iloc[date_index]]
                        self.charge += 0.0007 * self.Position[date_index]
                        self.Position[date_index] = (1-0.0007) * self.Position[date_index]
                        self.long_times += 2
                        if date_index != 0:
                            if self.share[date_index-1] != 0 and len(self.share[date_index-1]) == 1:
                                self.short_times += 1
                            elif len(self.share[date_index-1]) == 2:
                                self.short_times += 2   
                        self.whether_hold = True
                    
                    elif work1 not in self.hold[date_index-1] and work2 in self.hold[date_index-1]:
                        self.hold[date_index] = [work1, work2]
                        Total_prob = work1_prob + work2_prob
                        if len(self.hold[date_index-1]) == 2:
                            if self.hold[date_index-1][0] == work2:
                                share_work2 = self.share[date_index-1][0] 
                            else:
                                share_work2 = self.share[date_index-1][1]
                            pos_work1 = self.Position[date_index] - share_work2 * dict_hs300work_trade_info[work2]["open"].iloc[date_index]
                            self.share[date_index] = [pos_work1/dict_hs300work_trade_info[work1]["open"].iloc[date_index],share_work2]  
                            self.charge += 0.0007 * pos_work1
                            self.Position[date_index] -= 0.0007 * pos_work1 
                        else:
                            self.share[date_index] = [work1_prob/Total_prob * self.Position[date_index]/dict_hs300work_trade_info[work1]["open"].iloc[date_index] ,work2_prob/Total_prob * self.Position[date_index]/dict_hs300work_trade_info[work2]["open"].iloc[date_index]]
                            self.charge += 0.0007 * work1_prob/Total_prob * self.Position[date_index]
                            self.Position[date_index] -= 0.0007 * work1_prob/Total_prob * self.Position[date_index]
                        self.long_times += 1
                        self.short_times += 1
                        self.whether_hold = True  
                        
                    elif work1 in self.hold[date_index-1] and work2 not in self.hold[date_index-1]:
                        if len(self.hold[date_index-1]) == 2:
                            self.hold[date_index] = [work1, work2]
                            Total_prob = work1_prob + work2_prob
                            if self.hold[date_index-1][0] == work1:
                                share_work1 = self.share[date_index-1][0]
                            else:
                                share_work1 = self.share[date_index-1][1]
                            pos_work2 = self.Position[date_index] - share_work1 * dict_hs300work_trade_info[work1]["open"].iloc[date_index]
                            self.share[date_index] = [share_work1,pos_work2/dict_hs300work_trade_info[work2]["open"].iloc[date_index]]  
                            self.charge += 0.0007 * pos_work2
                            self.Position[date_index] -= 0.0007 * pos_work2
                            self.long_times += 1
                            self.short_times += 1
                            self.whether_hold = True
                        else:
                            self.hold[date_index] = self.hold[date_index-1]
                            self.share[date_index] = self.share[date_index-1]
                    
                    else:
                        self.hold[date_index] = self.hold[date_index-1]
                        self.share[date_index] = self.share[date_index-1]
                        
                else:
                    if date_index != 0:
                        if self.share[date_index-1] != 0 and len(self.share[date_index-1]) == 1:
                            self.short_times += 1
                        elif len(self.share[date_index-1]) == 2 :
                            self.short_times += 2
                    self.whether_hold = False
            else:
                self.hold[date_index] = self.hold[date_index-1]
                self.share[date_index] = self.share[date_index-1]
                if self.whether_hold == True:
                    if len(self.hold[date_index]) == 1:
                        work1 = self.hold[date_index][0]
                        share_work1 = self.share[date_index][0]
                        self.Position[date_index] = share_work1 * dict_hs300work_trade_info[work1]["open"].iloc[date_index]
                    elif len(self.hold[date_index]) == 2:
                        work1 = self.hold[date_index][0]
                        share_work1 = self.share[date_index][0]
                        work2 = self.hold[date_index][1]
                        share_work2 = self.share[date_index][1]
                        self.Position[date_index] = share_work1 * dict_hs300work_trade_info[work1]["open"].iloc[date_index] + share_work2 * dict_hs300work_trade_info[work2]["open"].iloc[date_index]
        self.Return = self.Position / self.init_value - 1
        for date_index in range(1,self.trading_days):
            self.Daily_Return.append(self.Position[date_index]/self.Position[date_index-1] - 1)
            self.Daily_Based_Return.append(self.df_hs300["open"].iloc[date_index]/self.df_hs300["open"].iloc[date_index-1] - 1)
                        
    
    def Back_testing_plot(self):
        dict_compare_return = {}
        dict_compare_return["Sector Rotation"] = self.Return
        dict_compare_return["沪深300"] = self.Based_Return
        df_compare_return = pd.DataFrame(dict_compare_return)
        df_compare_return.plot.area(title ="Sector Rotation" , figsize = (12,8), alpha = 0.4, stacked = False, ylabel = "Cumulative Return")
    
    def __del__(self):
        print("Industry rotation for is Over")
   
    def ex_ret(self):
        self.ex_ret = self.Return[self.trading_days-1] - self.Based_Return[self.trading_days-1]
        return np.round(self.ex_ret,3)
    
    def tim_maxdrawdown(self):
        return np.round(max_drawdown(np.array(self.Daily_Return)),3)
        
    def SharpeRatio(self):
        return self.Annualized_Return()/self.Annualized_Volatility()
        
#-------------
#def victory_ratio(self):
#        return (self.Daily_Return>0).mean()# pricesell是array或list
#--------------
    
    def Annualized_Return(self):
        return (1+(self.Position[self.trading_days-1]-self.Position[0])/self.Position[0])**(365/365)-1
    
    def Annualized_Volatility(self):
        DailyReturns=np.array(self.Daily_Return)
        return np.std(DailyReturns)*np.sqrt(self.trading_days)# self.trading_days为一年期的交易日
    
    def beta(self):
        cov_mat = np.cov(np.array(self.Daily_Based_Return),np.array(self.Daily_Return))/np.var(np.array(self.Daily_Based_Return))
        return cov_mat[0,1]
    
    def alpha(self):
        Rf=0
        Rm=np.array(self.Daily_Based_Return).mean()
        R=np.array(self.Daily_Return).mean()
        return ((R-Rf)-self.beta()*(Rm-Rf))
    
    def Trading_detail(self):
        dict_detail_info = {}
        dict_detail_info["Cumulaitve Return"] = self.Return
        work1_record = []
        work1_share = []
        work1_price = []
        work2_record = []
        work2_share = []
        work2_price = []
        for date_index in range(0,self.trading_days):
            if len(self.hold[date_index]) == 2:
                work1_record.append(self.hold[date_index][0])
                work1_share.append(self.share[date_index][0])
                work1_price.append(self.dict_hs300work_trade_info[self.hold[date_index][0]]["open"].iloc[date_index])
                work2_record.append(self.hold[date_index][1])
                work2_share.append(self.share[date_index][1])
                work2_price.append(self.dict_hs300work_trade_info[self.hold[date_index][1]]["open"].iloc[date_index])
            elif len(self.hold[date_index]) == 1 and self.hold[date_index] != 0:
                work1_record.append(self.hold[date_index][0])
                work1_share.append(self.share[date_index][0])
                work1_price.append(self.dict_hs300work_trade_info[self.hold[date_index][0]]["open"].iloc[date_index])
                work2_record.append(0)
                work2_share.append(0)
                work2_price.append(0)
            else:
                work1_record.append(0)
                work1_share.append(0)
                work1_price.append(0)
                work2_record.append(0)
                work2_share.append(0)
                work2_price.append(0)
        dict_detail_info["work1 Record"] = work1_record
        dict_detail_info["work1 Share Record"] = work1_share
        dict_detail_info["work1 Price"] = work1_price
        dict_detail_info["work2 Record"] = work2_record
        dict_detail_info["work2 Share Record"] = work2_share
        dict_detail_info["work2 Price"] = work2_price       
        
        df_detail_info = pd.DataFrame(dict_detail_info)
        df_detail_info.index = self.df_hs300.index
        return(df_detail_info)
    
    def Probability_Record(self):
        df_Probability_Record = pd.DataFrame(self.probability_record).set_index("Date")
        return(df_Probability_Record)


# In[364]:


dict_hs300work_trade_info_Test = {}
dict_hs300work_factor_Test = {}
dict_hs300work_trend5_Test = {}
for work in work_area:
    dict_hs300work_trade_info_Test[work] = dict_hs300work_trade_info[work].loc["2020-05-08":"2021-05-07"]
    dict_hs300work_factor_Test[work] = dict_hs300work_factor[work].loc["2020-05-08":"2021-05-07"]
    dict_hs300work_trend5_Test[work] = dict_hs300work_trend5[work].loc["2020-05-08":"2021-05-07"]


# ### Generate the sector strategy object

# In[470]:


strategy_ob_SR = Sector_Rotation(hs300.loc["2020-05-08":"2021-05-07"],dict_hs300work_trade_info_Test, dict_hs300work_factor_Test, dict_hs300work_trend5_Test, 1000000,dict_RF_2,work_area)


# ### Do the back testing of sector strategy

# In[471]:


strategy_ob_SR.Back_testing()


# ### Plot the back testing result of sector strategy

# In[472]:


strategy_ob_SR.Back_testing_plot()


# ### Make a table about the result of back testing of sector rotation strategy

# In[473]:


SR_back_test_info = {}
SR_back_test_info["Long Times"] = strategy_ob_SR.long_times
SR_back_test_info["Short Times"] = strategy_ob_SR.short_times
SR_back_test_info["Total Charge"] = np.round(strategy_ob_SR.charge,3)
SR_back_test_info["Excess Return"] = strategy_ob_SR.ex_ret()
SR_back_test_info["Max Drawdown"] = strategy_ob_SR.tim_maxdrawdown()
SR_back_test_info["Annualized Return"] = strategy_ob_SR.Annualized_Return()
SR_back_test_info["Annualized Volatility"] = strategy_ob_SR.Annualized_Volatility()
SR_back_test_info["Sharpe Ratio"] =strategy_ob_SR.SharpeRatio()
SR_back_test_info["Beta"] =strategy_ob_SR.beta()
SR_back_test_info["Alpha"] =strategy_ob_SR.alpha()


# In[331]:


df_SR_back_test_info = pd.DataFrame(pd.Series(SR_back_test_info))
df_SR_back_test_info


# ### Detail about upward probability estimation

# In[332]:


df_Probability_Record = strategy_ob_SR.Probability_Record()
df_Probability_Record


# ### Trading Detail

# In[413]:


df_Position_detail = strategy_ob_SR.Trading_detail().loc[df_Probability_Record.index]
df_Position_detail["work1 weight"] = df_Position_detail["work1 Share Record"]*df_Position_detail["work1 Price"]/(df_Position_detail["work1 Share Record"]*df_Position_detail["work1 Price"] + df_Position_detail["work2 Share Record"]*df_Position_detail["work2 Price"])
df_Position_detail["work2 weight"] = df_Position_detail["work2 Share Record"]*df_Position_detail["work2 Price"]/(df_Position_detail["work1 Share Record"]*df_Position_detail["work1 Price"] + df_Position_detail["work2 Share Record"]*df_Position_detail["work2 Price"])
df_Position_detail


# In[414]:


dict_Return_Work_20_21 = {}
for work in work_area:
    dict_Return_Work_20_21[work] = dict_hs300work_trade_info[work]["open"].loc["2020-05-08":"2021-05-08"] / dict_hs300work_trade_info[work]["open"].loc["2020-05-08"]- 1
dict_Return_Work_20_21["Date"] = hs300.loc["2020-05-08":"2021-05-08"].index
df_Return_Work_20_21 = pd.DataFrame(dict_Return_Work_20_21).set_index("Date")


# ### cumulative return of CSI 300 sector index

# In[474]:


df_Return_Work_20_21.plot(figsize = (12,8), ylabel = "Cumulative Return")


# In[420]:


dict_Daily_Return_Work_20_21 = {}
for work in work_area:
    dict_Daily_Return_Work_20_21[work] = dict_hs300work_trade_info[work]["open"].loc[df_Probability_Record.index].pct_change()
dict_Daily_Return_Work_20_21["Date"] = df_Probability_Record.index
df_Daily_Return_Work_20_21 = pd.DataFrame(dict_Daily_Return_Work_20_21).set_index("Date")


# ### weekly return of CSI sector index

# In[475]:


df_Daily_Return_Work_20_21.plot(figsize = (15,9), ylabel="Daily Return")


# ## This is the accuracy of timing strategy for each sector

# In[451]:


dict_accuracy_RF = {}
for work in work_area:
    trend5_test_pred_ = dict_RF_2[work].predict(dict_hs300work_factor[work].loc[df_Probability_Record.index])
    dict_accuracy_RF[work] = accuracy_score(dict_hs300work_trend5[work].loc[df_Probability_Record.index].dropna(),trend5_test_pred_[:-1])
df_accracy_RF = pd.DataFrame(pd.Series(dict_accuracy_RF))
df_accracy_RF


# In[452]:


df_accracy_RF.mean()


# ### Plot the estimation of upward probability in each trading date

# In[336]:


marker_work = ["o", "D", "x", "^", ">", "+", "_", "s", "d", "3"]
dict_work_mark = dict(zip(work_area,marker_work))


# In[478]:


fig, axes = plt.subplots(1,figsize = (12,8))
axes.plot(df_Probability_Record.index,df_Probability_Record["1st Work prob"], label ="1st Work prob", color = "gold")
axes.plot(df_Probability_Record.index,df_Probability_Record["2nd Work prob"], label ="2nd Work prob", color = "limegreen") 
axes.plot(df_Probability_Record.index,df_Probability_Record["mean prob"], label = "mean up probability", color = "navy" )
axes.plot(df_Probability_Record.index,0.5 * np.ones(df_Probability_Record.shape[0]), label = "0.5", color = "k")
for work in work_area:
    x = df_Probability_Record.loc[df_Probability_Record["1st Work"] == work].index
    y = df_Probability_Record.loc[df_Probability_Record["1st Work"] == work]["1st Work prob"]
    axes.scatter(x,y, color = "gold", label = "1st Work: " + work, marker = dict_work_mark[work], s = 100)
for work in work_area:
    x = df_Probability_Record.loc[df_Probability_Record["2nd Work"] == work].index
    y = df_Probability_Record.loc[df_Probability_Record["2nd Work"] == work]["2nd Work prob"]
    axes.scatter(x,y, color = "limegreen", label = "2nd Work: " + work, marker = dict_work_mark[work], s = 100)
plt.ylabel("Probability")
plt.legend()


# ### Plot the detail about weight position in each trading date

# In[480]:


df_Position_detail[["work1 weight","work2 weight"]].plot.area(alpha = 0.4, color = ["gold","limegreen"], figsize = (12,6),stacked = False)
for work in work_area:
    x = df_Position_detail.loc[df_Position_detail["work1 Record"] == work].index
    y = df_Position_detail.loc[df_Position_detail["work1 Record"] == work]["work1 weight"]
    plt.scatter(x, y, color = "gold", label = "1st Work: " + work, marker = dict_work_mark[work], s = 100 )
for work in work_area:
    x = df_Position_detail.loc[df_Position_detail["work2 Record"] == work].index
    y = df_Position_detail.loc[df_Position_detail["work2 Record"] == work]["work2 weight"]
    plt.scatter(x, y, color = "limegreen", label = "2nd Work: " + work, marker = dict_work_mark[work], s = 100 )
plt.ylabel("Weight")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




