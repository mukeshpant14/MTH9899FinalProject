import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler

def ts_normalize(df, columns):
    result = df.copy()
    sc = StandardScaler()
    #scale variables along time in the same security
    for col in columns:
        result[col] = sc.fit_transform(np.array(result[col]).reshape(-1,1))  
    return result


def ts_fillna(df, column, default_value=0):
    result = df.copy()
    #backfill missing data, try the previous value, if not existing use default 
    na_rows = result[column].isna().sum()
    if na_rows > 0:
        result.vol.fillna(method='ffill', inplace=True)    
        result.vol.fillna(default_value, inplace=True)          
    return result

def get_weight_ffd(d, thres, lim):
  w, k = [1.], 1
  ctr = 0
  while True:
    w_ = -w[-1] / k * (d - k + 1)
    if abs(w_) < thres:
      break
    w.append(w_)
    k += 1
    ctr += 1
    if ctr == lim - 1:
      break
  w = np.array(w[::-1]).reshape(-1, 1)
  return w

def frac_diff_ffd(x, d, thres=1e-3):
  w = get_weight_ffd(d, thres, len(x))
  width = len(w) - 1
  output = []
  output.extend([0] * width)
  for i in range(width, len(x)):
    output.append(np.dot(w.T, x[i - width:i + 1])[0])
  return np.array(output)

def add_vol_ffd(series, fd):
  series['vol_ffd'] = frac_diff_ffd(series.vol, fd)    


def find_min_ffd(df1,confidence=0.05):
    for d in np.linspace(0,1,21):
        diff = frac_diff_ffd(df1,d)
        adf = adfuller(diff, maxlag=1, regression='c', autolag=None)
        pval = adf[1]
        if pval <= 0.05:
            return d
    return 1

def plot_min_ffd(df1):
    out = pd.DataFrame(columns=['adfStat','pVal','lags','nobs','95% conf','corr'])
    best = 0
    for d in np.linspace(0,1,21):
        df2 = frac_diff_ffd(df1,d)
        corr = np.corrcoef(df1, df2)[0,1]
        df2 = adfuller(df2, maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]
        if df2[1] < 0.05 and best == 0:
            best = d
            print(f"Best d: {best}")
    ax = out[['adfStat','corr']].plot(secondary_y='adfStat')
    ax.set_xlabel('Differentiation factor')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.axvline(best, linestyle='dotted')
    plt.show()
    return


df = pd.read_csv('dat_final.csv')
df.set_index('Date')
df.drop('Time',axis=1,inplace=True)
df.vol.replace(0,np.nan,inplace=True)
sec_ids = df['sec_id'].unique()
series = [df[df['sec_id']== i] for i in sec_ids]
series = [ x.set_index('Date') for x in series] #sets the index to the Date
empty_thres = 0.25
empty_secs = set()
for i in range(len(series)):
    s = series[i]
    total = s.shape[0]
    zeros = len(s[s.fut_ret==0.0])
    if zeros>total*empty_thres:
        empty_secs.add(s.sec_id[0])
discarded = len(empty_secs)
print(f"Discarding {discarded} securities with incomplete data (threshold={empty_thres})")

clean_series = []
for s in series:
    if s.sec_id[0] not in empty_secs:
        clean_series.append(s)
  
median = df['vol'].median()
input_vars = ['vol','X1','X2','X3','X4','X5','X6','X7']
clean_series = [ ts_fillna(x, 'vol', median) for x in clean_series]
clean_series = [ ts_normalize(x, input_vars) for x in clean_series] 

vol = clean_series[150]['vol']
plot_min_ffd(vol)
#count non-stationary values
count = 0
for i in clean_series:
    vol = i.vol
    adf = adfuller(vol)
    if adf[1] > 0.05:
        count += 1
print(count)

mins = []
for i in clean_series:
    vol = i.vol
    d = find_min_ffd(vol)
    mins.append(d)     
print(mins)

plt.hist(mins, bins=50,cumulative=True)
plt.xlabel('Miimum fractional differentiating factor')
plt.ylabel('Number of securities with stationary vol after differentiation')
plt.show()


[add_vol_ffd(x, 0.7) for x in clean_series]
#count non-stationary values
count = 0
for i in clean_series:
    vol = i.vol_ffd
    adf = adfuller(vol)
    if adf[1] > 0.05:
        count += 1
print(count)

    
#  clean_series = [ ts_normalize(x, input_vars) for x in clean_series] 
  
  #make vol a stationary variable using fractional differencing
  #the differencing factor is decided as the value that makes more than 95% 
  #of the samples in the dataset stationary
#  [add_vol_ffd(x, 0.7) for x in clean_series]  
  
#  org_data = pd.concat(clean_series)
#  org_data = ts_normalize(org_data, input_vars)
    