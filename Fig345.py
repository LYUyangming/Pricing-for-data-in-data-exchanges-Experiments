from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# basic setting
prec=10**3
alpha_1=0
alpha2=np.linspace(20,45,prec)
N_max=25      
NN=range(5,1+N_max,5)
DUI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
DUC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
XI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
XC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
SUI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
SUC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
PI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
PC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
EUI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
EUC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
TAUI=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))
TAUC=pd.DataFrame(np.array([[0]*(len(alpha2)*len(NN))]).reshape(len(NN),len(alpha2)))

pbar=trange(len(alpha2)*len(NN))

for s in range(len(NN)):
    N=NN[s]
    
    c=np.sort(np.random.normal(loc=10,scale=3,size=N))
    for i in range(999):
        c+=np.sort(np.random.normal(loc=10,scale=3,size=N))
    c=c/1000
    for t in range(len(alpha2)):
        
        alpha_2=alpha2[t]
        
        
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
        
        IPA_tau0_index=IPA_EU.idxmax()
        IPA_i0_index=(IPA_DU.iloc[:,IPA_tau0_index]).idxmax()
        PCPA_tau0_index=PCPA_EU.idxmax()
        PCPA_i0_index=(PCPA_DU.iloc[:,IPA_tau0_index]).idxmax()
        
        DUI.iloc[s,t]=IPA_DU.iloc[IPA_i0_index,IPA_tau0_index]
        DUC.iloc[s,t]=PCPA_DU.iloc[PCPA_i0_index,PCPA_tau0_index]
        XI.iloc[s,t]=IPA_x.iloc[IPA_i0_index,IPA_tau0_index]
        XC.iloc[s,t]=PCPA_x.iloc[PCPA_i0_index,PCPA_tau0_index]
        SUI.iloc[s,t]=IPA_SU.iloc[IPA_i0_index,IPA_tau0_index]
        SUC.iloc[s,t]=PCPA_SU.iloc[PCPA_i0_index,PCPA_tau0_index]
        PI.iloc[s,t]=IPA_p.iloc[IPA_i0_index,IPA_tau0_index]
        PC.iloc[s,t]=PCPA_p.iloc[PCPA_i0_index,PCPA_tau0_index]
        EUI.iloc[s,t]=IPA_EU[IPA_tau0_index]
        EUC.iloc[s,t]=PCPA_EU[PCPA_tau0_index]
        TAUI.iloc[s,t]=tau[IPA_tau0_index]
        TAUC.iloc[s,t]=tau[PCPA_tau0_index]
        
        
        #进度更替
        pbar.update(1)

#进度条结束
pbar.close()

#图3
X,Y=np.meshgrid(alpha2,NN)
# 初始化
canvas=plt.figure(figsize=(12,9),dpi=300)#创建画布
axs1=canvas.add_subplot(121, projection='3d')#添加三维子图
axs2=canvas.add_subplot(122, projection='3d')#添加三维子图
axs1.plot([5],[20],[0],'b')
axs1.plot([5],[20],[0],'r')
axs1.legend(['IPA','PCPA'])
axs2.plot([5],[20],[0],'b')
axs2.plot([5],[20],[0],'r')
axs2.legend(['IPA','PCPA'])

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(DUI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(DUI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(DUI)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(DUI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]+1.2, X[i, k], np.array(DUI)[i, k]-1, f'{np.array(DUI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(DUI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(DUC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(DUC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(DUC)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(DUC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]-1.2, X[i, k], np.array(DUC)[i, k]+1, f'{np.array(DUC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(DUC)[:, k], linestyle='--', color='r') 

axs1.set_xticks([5,10,15,20,25])
axs1.set_xlabel('the number of data factor suppliers')
axs1.set_ylabel('the value of the parameter alpha2')
axs1.set_zlabel('demander utility')

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(XI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(XI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(XI)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(XI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]+1.2, X[i, k], np.array(XI)[i, k]-1, f'{np.array(XI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(XI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(XC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(XC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(XC)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(XC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]-1.2, X[i, k], np.array(XC)[i, k]+1, f'{np.array(XC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(XC)[:, k], linestyle='--', color='r') 

axs2.set_xticks([5,10,15,20,25])
axs2.set_xlabel('the number of data factor suppliers')
axs2.set_ylabel('the value of the parameter alpha2')
axs2.set_zlabel('volumn of transactions')

plt.savefig('C:/Users/dell/Desktop/数据要素市场最优定价模型/数值模拟及其结果/结果/图3.png')

#图4
X,Y=np.meshgrid(alpha2,NN)
# 初始化
canvas=plt.figure(figsize=(12,9),dpi=300)#创建画布
axs1=canvas.add_subplot(121, projection='3d')#添加三维子图
axs2=canvas.add_subplot(122, projection='3d')#添加三维子图
axs1.plot([5],[20],[0],'b')
axs1.plot([5],[20],[0],'r')
axs1.legend(['IPA','PCPA'])
axs2.plot([5],[20],[0],'b')
axs2.plot([5],[20],[0],'r')
axs2.legend(['IPA','PCPA'])

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(SUI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(SUI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(SUI)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(SUI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]+1.2, X[i, k], np.array(SUI)[i, k]-1, f'{np.array(SUI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(SUI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(SUC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(SUC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(SUC)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(SUC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]-1.2, X[i, k], np.array(SUC)[i, k]+1, f'{np.array(SUC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(SUC)[:, k], linestyle='--', color='r') 

axs1.set_xticks([5,10,15,20,25])
axs1.set_xlabel('the number of data factor suppliers')
axs1.set_ylabel('the value of the parameter alpha2')
axs1.set_zlabel('supplier utility')

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(PI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(PI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(PI)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(PI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]+1.2, X[i, k], np.array(PI)[i, k]-1, f'{np.array(PI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(PI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(PC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(PC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(PC)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(PC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]-1.2, X[i, k], np.array(PC)[i, k]+1, f'{np.array(PC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(PC)[:, k], linestyle='--', color='r') 

axs2.set_xticks([5,10,15,20,25])
axs2.set_xlabel('the number of data factor suppliers')
axs2.set_ylabel('the value of the parameter alpha2')
axs2.set_zlabel('unit price')
plt.savefig('C:/Users/dell/Desktop/数据要素市场最优定价模型/数值模拟及其结果/结果/图4.png')

#图5
X,Y=np.meshgrid(alpha2,NN)
# 初始化
canvas=plt.figure(figsize=(12,9),dpi=300)#创建画布
axs1=canvas.add_subplot(121, projection='3d')#添加三维子图
axs2=canvas.add_subplot(122, projection='3d')#添加三维子图
axs1.plot([5],[20],[0],'b')
axs1.plot([5],[20],[0],'r')
axs1.legend(['IPA','PCPA'])
axs2.plot([5],[20],[0],'b')
axs2.plot([5],[20],[0],'r')
axs2.legend(['IPA','PCPA'])

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(EUI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(EUI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(EUI)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(EUI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]+1.2, X[i, k], np.array(EUI)[i, k]-1, f'{np.array(EUI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(EUI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs1.plot(Y[i],X[i],np.array(EUC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs1.plot(Y[i], X[i], np.zeros_like(np.array(EUC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(EUC)[i, j]])
    axs1.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs1.scatter(Y[i, k], X[i, k], np.array(EUC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs1.text(Y[i, k]-1.2, X[i, k], np.array(EUC)[i, k]+1, f'{np.array(EUC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs1.plot(Y[:, k], X[:, k], np.array(EUC)[:, k], linestyle='--', color='r') 

axs1.set_xticks([5,10,15,20,25])
axs1.set_xlabel('the number of data factor suppliers')
axs1.set_ylabel('the value of the parameter alpha2')
axs1.set_zlabel('exchange utility')

# 绘制折线面
for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(TAUI*PI)[i],color='b',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(TAUI*PI)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(TAUI*PI)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='b', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(TAUI*PI)[i, k],color='b')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]+1.2, X[i, k], np.array(TAUI*PI)[i, k]-1, f'{np.array(TAUI*PI)[i, k]:.2f}',
                 #color='b', ha='center', size=7) 
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(TAUI*PI)[:, k], linestyle='--', color='b') 

for i in range(len(NN)):  # 遍历
    # z值线，即实际数据。
    axs2.plot(Y[i],X[i],np.array(TAUC*PC)[i],color='r',alpha=0.3)
    # 0值线（z=0），与“地面”连接。
    axs2.plot(Y[i], X[i], np.zeros_like(np.array(TAUC*PC)[i]), color='gray', alpha=0.5) 
    # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
    polygon = [
        [Y[i, 0], X[i, 0], 0],    # 左下
        [Y[i, -1], X[i, -1], 0],  # 右下
    ]
    for j in range(len(alpha2)-1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
        polygon.append([Y[i, j], X[i, j], np.array(TAUC*PC)[i, j]])
    axs2.add_collection3d(Poly3DCollection([polygon], color='r', alpha=0.3))
    for k in [0,round((prec-1)*2/5),prec-1]:
        axs2.scatter(Y[i, k], X[i, k], np.array(TAUC*PC)[i, k],color='r')
    # 标注数字（z值）
    #for k in [0,round((prec-1)/3),round((prec-1)*2/3),prec-1]:
        #axs2.text(Y[i, k]-1.2, X[i, k], np.array(TAUC*PC)[i, k]+1, f'{np.array(TAUC*PC)[i, k]:.2f}',
                 #color='r', ha='center', size=7)         
# 用虚线将需要标注的时间（y）连起来
for k in [0,round((prec-1)*2/5),prec-1]:
    axs2.plot(Y[:, k], X[:, k], np.array(TAUC*PC)[:, k], linestyle='--', color='r') 

axs2.set_xticks([5,10,15,20,25])
axs2.set_xlabel('the number of data factor suppliers')
axs2.set_ylabel('the value of the parameter alpha2')
axs2.set_zlabel('unit revenue of the exchange')
plt.savefig('C:/Users/dell/Desktop/数据要素市场最优定价模型/数值模拟及其结果/结果/图5.png')



