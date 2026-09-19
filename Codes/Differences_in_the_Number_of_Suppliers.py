"""
不同的供给者数量模拟
"""
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# basic setting
alpha_1=0
alpha_2=1000
N_max=40      
NN=range(1,1+N_max)
IPA_tau0=pd.Series([0]*N_max)
PCPA_tau0=pd.Series([0]*N_max)
IPA_w0var=pd.Series([0]*N_max)
PCPA_w0var=pd.Series([0]*N_max)
IPA_average_p0=pd.Series([0]*N_max)
PCPA_average_p0=pd.Series([0]*N_max)
IPA_average_x0=pd.Series([0]*N_max)
PCPA_average_x0=pd.Series([0]*N_max)

pbar=trange(N_max)

for k in range(len(NN)):
    N=NN[k]
    prec=10**4
    c=np.sort(np.random.normal(loc=500,scale=150,size=N))
    for i in range(499):
        c+=np.sort(np.random.normal(loc=500,scale=150,size=N))
    c=c/500
    
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

    #寻找数据交易所最优价格
    IPA_i0=IPA_EU.idxmax()
    PCPA_i0=PCPA_EU.idxmax()
    IPA_tau0[k]=tau[IPA_i0]
    PCPA_tau0[k]=tau[PCPA_i0]
    
    #寻找均衡竞争力方差
    PCPA_w0var[k]=np.var(PCPA_w.iloc[:,PCPA_i0])
    
    #寻找数据要素最优价格均值
    IPA_average_p0[k]=np.mean(IPA_p.iloc[:,IPA_i0])
    PCPA_average_p0[k]=np.mean(PCPA_p.iloc[:,PCPA_i0])
    
    #寻找数据要素最优购买量均值
    IPA_average_x0[k]=np.mean(IPA_x.iloc[:,IPA_i0])
    PCPA_average_x0[k]=np.mean(PCPA_x.iloc[:,PCPA_i0])
    
    #进度更替
    pbar.update(1)

#进度条结束
pbar.close()

#均衡-N图像绘制
fig,axs=plt.subplots(2,2,figsize=(14,9),dpi=1200,sharex=False,sharey=False)
axs[0,0].plot(NN,IPA_tau0,NN,PCPA_tau0)
axs[0,0].legend(['IPA_tau*','PCPA_tau*'])
axs[0,0].set_xlabel('number of data suppliers')
axs[0,0].set_ylabel('optimal pricing of the data exchange')
axs[0,1].plot(NN,IPA_average_p0,NN,PCPA_average_p0)
axs[0,1].legend(['IPA_average_p*','PCPA_average_p*'])
axs[0,1].set_xlabel('number of data suppliers')
axs[0,1].set_ylabel('average optimal pricing of data suppliers')
axs[1,0].plot(NN,IPA_w0var,NN,PCPA_w0var)
axs[1,0].legend(['IPA_var_w*','PCPA_var_w*'])
axs[1,0].set_xlabel('number of data suppliers')
axs[1,0].set_ylabel('the variance of equilibrium market competitiveness')
axs[1,1].plot(NN,IPA_average_x0,NN,PCPA_average_x0)
axs[1,1].legend(['IPA_average_x*','PCPA_average_x*'])
axs[1,1].set_xlabel('number of data suppliers')
axs[1,1].set_ylabel('average optimal puchase amount of data demanders')
plt.tight_layout()