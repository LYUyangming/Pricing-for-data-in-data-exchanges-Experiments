"""
不同质量水平参数下的最优价格
"""
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# basic setting
alpha_1=0
N=5
c=np.sort(np.random.normal(loc=500,scale=150,size=N))
for i in range(499):
    c+=np.sort(np.random.normal(loc=500,scale=150,size=N))
c=c/500
prec=10**3
alpha2_big=np.linspace(100,10000,prec)
alpha2_small=np.linspace(750,1250,prec)
IPA_p0=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
PCPA_p0=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))

pbar=trange(2*prec)
#全局模拟
for k in range(len(alpha2_big)):
    alpha_2=alpha2_big[k]
    
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

    #寻找数据要素最优价格
    IPA_i0=IPA_EU.idxmax()
    PCPA_i0=PCPA_EU.idxmax()
    IPA_p0[k]=IPA_p.iloc[:,IPA_i0]
    PCPA_p0[k]=PCPA_p.iloc[:,PCPA_i0]
    
    #进度更替
    pbar.update(1)

#p0-alpha_2(全局)图像绘制
fig,axs=plt.subplots(1,2,figsize=(14,7),dpi=1200,sharex=False,sharey=False)
axs[0].plot(alpha2_big,np.mean(IPA_p0,axis=0),alpha2_big,np.mean(PCPA_p0,axis=0))
axs[0].legend(['IPA_average_p*','PCPA_average_p*'])
axs[0].set_xlabel('data quality parameter: alpha_2')
axs[0].set_ylabel('average optimal pricing of data suppliers')

#局部模拟
for k in range(len(alpha2_small)):
    alpha_2=alpha2_small[k]
    
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

    #寻找数据要素最优价格
    IPA_i0=IPA_EU.idxmax()
    PCPA_i0=PCPA_EU.idxmax()
    IPA_p0[k]=IPA_p.iloc[:,IPA_i0]
    PCPA_p0[k]=PCPA_p.iloc[:,PCPA_i0]
    
    #进度更替
    pbar.update(1)

#进度条结束
pbar.close()

#p0-alpha_2(局部)图像绘制
axs[1].plot(alpha2_small,np.mean(IPA_p0,axis=0),alpha2_small,np.mean(PCPA_p0,axis=0))
axs[1].legend(['IPA_average_p*','PCPA_average_p*'])
axs[1].set_xlabel('data quality parameter: alpha_2')
axs[1].set_ylabel('average optimal pricing of data suppliers')
plt.tight_layout()