"""
模拟一定范围的tau，寻找最优的tau和市场参与者效用关于tau的拐点
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# basic setting
alpha_1=0
alpha_2=30
N=5
c=np.sort(np.random.normal(loc=10,scale=3,size=N))
for i in range(499):
    c+=np.sort(np.random.normal(loc=10,scale=3,size=N))
c=c/500

#模拟tau
prec=10**4
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

#EU-tau图像绘制
plt.figure(figsize=(9,6),dpi=1200)
plt.plot(tau,IPA_EU,tau,PCPA_EU)
plt.legend(['IPA_EU','PCPA_EU'])
plt.xlabel('pricing of the data exchange')
plt.ylabel('the utility of the data exchange')

#寻找最优的tau
IPA_tau0=tau[IPA_EU.idxmax()]
PCPA_tau0=tau[PCPA_EU.idxmax()]
print('The optimal pricing for the data exchange in the IPA is',IPA_tau0,sep='\n')
print('The optimal pricing for the data exchange in the PCPA is',PCPA_tau0,sep='\n')