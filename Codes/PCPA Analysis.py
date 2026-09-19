"""
PCPA引入竞争后的优缺点
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# basic setting
alpha_1=0
alpha_2=1000
N=5
c=np.sort(np.random.normal(loc=500,scale=150,size=N))
for i in range(499):
    c+=np.sort(np.random.normal(loc=500,scale=150,size=N))
c=c/500

#模拟tau
prec=10**4
tau=np.linspace(0,1-1/prec,prec)
PCPA_p=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
PCPA_x=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
PCPA_DU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
PCPA_SU=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))
PCPA_w=pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N,prec))

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

#w-tau图像绘制
plt.figure(figsize=(9,6),dpi=1200)
for i in range(N):
    plt.plot(tau,PCPA_w.iloc[i,:])
plt.plot([0,1-1/prec],[1,1])
plt.legend(['supplier 1','supplier 2','supplier 3','supplier 4','supplier 5','datum line'])
plt.xlabel('pricing of the data exchange')
plt.ylabel('market competitiveness of suppliers')