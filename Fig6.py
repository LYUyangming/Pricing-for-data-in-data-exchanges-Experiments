from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

# basic setting
prec0 = 10**2
prec = 10**3
alpha_1 = 0
alpha2 = np.linspace(20, 45, prec0)
Q = np.linspace(0, 2, prec0)
N = 10
c = np.sort(np.random.normal(loc=10, scale=3, size=N))
for i in range(999):
    c += np.sort(np.random.normal(loc=10, scale=3, size=N))
c = c/1000
SWI = pd.DataFrame(
    np.array([[0]*(len(alpha2)*len(Q))]).reshape(len(Q), len(alpha2)))
SWC = pd.DataFrame(
    np.array([[0]*(len(alpha2)*len(Q))]).reshape(len(Q), len(alpha2)))

pbar = trange(len(alpha2)*len(Q))

for s in range(len(Q)):
    for t in range(len(alpha2)):
        alpha_2 = alpha2[t]
        q = Q[s]

        

        # 模拟tau
        tau = np.linspace(0, 1-1/prec, prec)
        IPA_p = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        IPA_x = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        IPA_DU = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        IPA_SU = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        PCPA_p = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        PCPA_x = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        PCPA_DU = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        PCPA_SU = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))
        PCPA_w = pd.DataFrame(np.array([[0]*(N*prec)]).reshape(N, prec))

        # IPM模拟
        for i in range(prec):
            IPA_p[i] = np.maximum(np.sqrt(alpha_2*c/(1-tau[i])), c/(1-tau[i]))
            IPA_x[i] = np.maximum(alpha_2/IPA_p[i]-1, 0)
            IPA_DU[i] = alpha_1+alpha_2*np.log(1+IPA_x[i])-IPA_x[i]*IPA_p[i]
            IPA_SU[i] = (1-tau[i])*IPA_x[i]*IPA_p[i]-IPA_x[i]*c
        IPA_EU = tau*np.sum(IPA_p*IPA_x, axis=0)/N

        # PCPM模拟
        def create_equations(N):
            def b0_equations(vars, tau, c, alpha_2):
                eqs = []
                for i in range(N):
                    b = list(vars)
                    eq = sum(vars)-np.sqrt((sum(vars) -
                                            b[i]+(1-tau)/c[i])*(sum(vars)-b[i]+1/alpha_2))
                    eqs.append(eq)
                return eqs
            return b0_equations
        equations = create_equations(N)
        initial_guess = [1]*N
        for i in range(prec):
            solution = pd.Series(
                fsolve(equations, initial_guess, args=(tau[i], c, alpha_2)))
            PCPA_p[i] = np.maximum(1/solution, c/(1-tau[i]))
            PCPA_w[i] = (1/PCPA_p[i])/np.sum(1/PCPA_p[i])*N
            PCPA_x[i] = np.maximum(alpha_2/PCPA_p[i]-1, 0)
            PCPA_DU[i] = alpha_1+alpha_2 * \
                np.log(1+PCPA_x[i])-PCPA_x[i]*PCPA_p[i]
            PCPA_SU[i] = ((1-tau[i])*PCPA_x[i]*PCPA_p[i]-PCPA_x[i]*c)*PCPA_w[i]
        PCPA_EU = tau/N*(PCPA_w*PCPA_p*PCPA_x).sum(axis=0)

        IPA_tau0_index = IPA_EU.idxmax()
        IPA_i0_index = (IPA_DU.iloc[:, IPA_tau0_index]).idxmax()
        PCPA_tau0_index = PCPA_EU.idxmax()
        PCPA_i0_index = (PCPA_DU.iloc[:, IPA_tau0_index]).idxmax()

        SWI.iloc[s, t] = alpha_1+alpha_2*np.log(1+IPA_x.iloc[IPA_i0_index, IPA_tau0_index]) - \
            q*c[IPA_i0_index]*IPA_x.iloc[IPA_i0_index, IPA_tau0_index]
        SWC.iloc[s, t] = alpha_1+alpha_2*np.log(1+PCPA_x.iloc[PCPA_i0_index, PCPA_tau0_index]) - \
            q*c[PCPA_i0_index]*PCPA_x.iloc[PCPA_i0_index, PCPA_tau0_index]

        # 进度更替
        pbar.update(1)

# 进度条结束
pbar.close()

# 图6
# fig=plt.figure(figsize=(12,9),dpi=300)
#ax=fig.add_subplot(111, projection='3d')
# ax.plot_surface(Q,alpha2,SWI,cmap='rainbow')
# ax.plot_surface(Q,alpha2,SWC,color='r',alpha=0.3)
plt.figure(figsize=(12, 9), dpi=1200)
level = list(range(math.ceil(min(np.min(SWI), np.min(SWC))),math.floor(max(np.max(SWI), np.max(SWC))), 4))
contourI = plt.contour(alpha2, Q, SWI, levels=level,linestyles='--', cmap='cool')
plt.clabel(contourI, inline=True, fontsize=12, colors='b')
contourC = plt.contour(alpha2, Q, SWC, levels=level, cmap='gist_heat_r')
plt.clabel(contourC, inline=True, fontsize=12, colors='r')

plt.plot([],[],'--b')
plt.plot([],[],'r')
plt.legend(['isoline of SW in IPA','isoline of SW in PCPA'])
plt.xlabel('the value of the parameter alpha2')
plt.ylabel('the value of the social welfare priority parameter Q')
plt.savefig('C:/Users/dell/Desktop/数据要素市场最优定价模型/数值模拟及其结果/结果/图6.png')