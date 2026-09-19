from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

# basic setting
alpha_1=0
alpha_2=30
N_max=25      
NN=range(1,1+N_max)
PCPA_w0var=pd.Series([0]*N_max)

pbar=trange(N_max)
#模拟
for k in range(len(NN)):
    N=NN[k]
    prec=10**4
    c=np.sort(np.random.normal(loc=10,scale=3,size=N))
    for i in range(999):
        c+=np.sort(np.random.normal(loc=10,scale=3,size=N))
    c=c/1000
    
    #模拟tau
    tau=np.linspace(0,1-1/prec,prec)
    IPA_p=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    IPA_x=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    IPA_DU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    IPA_SU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    PCPA_p=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    PCPA_x=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    PCPA_DU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    PCPA_SU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
    PCPA_w=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))

    #IPM模拟
    for i in range(prec):
        IPA_p[i]=np.maximum(np.sqrt(alpha_2*c/(1-tau[i])),c/(1-tau[i]))
        IPA_x[i]=np.maximum(alpha_2/IPA_p[i]-1,0)
        IPA_DU[i]=alpha_1+alpha_2*np.log(1+IPA_x[i])-IPA_x[i]*IPA_p[i]
        IPA_SU[i]=(1-tau[i])*IPA_x[i]*IPA_p[i]-IPA_x[i]*c
    IPA_EU=tau*np.sum(IPA_p*IPA_x,axis=0)/N

    #PCPM模拟
    def create_equations(N):
        def b0_equations(vars, tau, c, alpha_2):
            eqs = []
            for i in range(N):
                b=list(vars)
                eq=sum(vars)-np.sqrt((sum(vars)-b[i]+(1-tau)/c[i])*(sum(vars)-b[i]+1/alpha_2))
                eqs.append(eq)
            return eqs
        return b0_equations
    equations = create_equations(N)
    initial_guess = [1]*N
    for i in range(prec):
        solution = pd.Series(fsolve(equations, initial_guess, args=(tau[i], c, alpha_2)))
        PCPA_p[i]=np.maximum(1/solution,c/(1-tau[i]))
        PCPA_w[i]=(1/PCPA_p[i])/np.sum(1/PCPA_p[i])*N
        PCPA_x[i]=np.maximum(alpha_2/PCPA_p[i]-1,0)
        PCPA_DU[i]=alpha_1+alpha_2*np.log(1+PCPA_x[i])-PCPA_x[i]*PCPA_p[i]
        PCPA_SU[i]=((1-tau[i])*PCPA_x[i]*PCPA_p[i]-PCPA_x[i]*c)*PCPA_w[i]
    PCPA_EU=tau/N*(PCPA_w*PCPA_p*PCPA_x).sum(axis=0)

    #寻找均衡竞争力方差
    PCPA_i0=PCPA_EU.idxmax()
    PCPA_w0var[k]=np.var(PCPA_w.iloc[:,PCPA_i0])
    
    #进度更替
    pbar.update(1)

#进度条结束
pbar.close()

# 定义拟合函数，kln(x)的形式
def fit_func(x, k):
    return k * np.log(x)

# 使用curve_fit进行拟合
params, covariance = curve_fit(fit_func,NN,PCPA_w0var)

# 提取拟合参数
k_fit= params

# 使用拟合参数生成拟合数据
x_fit = np.linspace(min(NN), max(NN), prec)
y_fit = fit_func(x_fit, k_fit)

#绘制图2
plt.figure(figsize=(12,8),dpi=1200)
plt.scatter(NN,PCPA_w0var,color='r')
plt.plot(x_fit,y_fit,'k--')
plt.legend(['original data','fit: var=kln(N)'])
plt.xlabel('the number of data factor suppliers')
plt.ylabel('market competitiveness variance of data factor suppliers in PCPA')