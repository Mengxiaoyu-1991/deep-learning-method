'''
1.plot the line
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import rc
random.seed(2019)
plt.style.use('classic')
mpl.rc('axes',edgecolor='k')

#setting the size
rc('font',size=14)
rc('font',family='Times New Roman')
rc('axes',labelsize=14)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('font', weight='normal')

plt.close('all')
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x1,y1,'-',color='r',markersize=6,linewidth=1.5, markeredgecolor='r',label= 'Exponential') #plot
plt.xlabel('Z coordinates', fontname = 'Times New Roman')
plt.ylabel('Thermal conductivity k(z)', fontname = 'Times New Roman')


ax.set_ylim(0, 45)  #limit the domain of y_axis
ax.set_xticks(np.arange(0, 1.1, 0.1)) #define the scale in x_axis

plt.grid(linestyle='dotted',color='k')
ax.legend(loc='upperleft', fancybox=True, shadow=True,  ncol=1,  prop={'size': 17}) 
plt.savefig('conductivity of irregular.pdf',dpi=600,bbox_inches='tight')
plt.show()













'''
2.plot the bar
'''
import pandas as pd
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
plt.style.use('classic')
#mpl.rcParams['axes.facecolor'] = 'w' # Or any suitable colour...
mpl.rc('axes',edgecolor='k')

#rc('text',usetex=True)
# format the figure
rc('font',size=14)
rc('font',family='Times New Roman')
rc('axes',labelsize=14)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('font', weight='normal')


fig,ax = plt.subplots()
x=np.arange(len(layer2))
height = 0.2 #adjust the distance of the bar
# plotting
plt.barh(x-1.5*height,layer2,height,color='r',label = "Hidden layer=2") #plot bar in horizontal direction
plt.xlabel('Error (*$10^{ -5}$)', fontname = 'Times New Roman', size = 14)
plt.ylabel('Sampling', fontname = 'Times New Roman', size = 14)
#ax.set_title("error",fontsize=14,fontweight='bold')
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", left="on", right="off", labelleft="on")
tick_spacing =1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.yticks(x,function)
font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 14,
         }

plt.legend(loc='center right',prop=font1)
plt.savefig("compare-sampling.pdf",bbox_inches = 'tight')
plt.show()















'''
 3.plot bar
 (1) Use color to show value 
 (2) add colar bar
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import rc
import pandas as pd

random.seed(2019)
# Using seaborn's style
plt.style.use('classic')
#mpl.rcParams['axes.facecolor'] = 'w' # Or any suitable colour...
mpl.rc('axes',edgecolor='k')

#rc('text',usetex=True)
rc('font',size=14)
rc('font',family='Times New Roman')
rc('axes',labelsize=14)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('font', weight='normal')

plt.close('all')

cmap = plt.get_cmap('YlOrRd')

pos = np.arange(len(Point)) 
fig = plt.figure()   
norm_data=Error.flatten()    
norm_data= (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())
colors1 = plt.cm.coolwarm(norm_data)
plot = plt.scatter(Error.flatten(), Error.flatten(), c = 1e-5*Error.flatten(), cmap = 'coolwarm')
plt.clf()
plt.colorbar(plot)
plt.barh(pos,Error.flatten(),color=colors1)

plt.yticks(pos, point1)
plt.xlabel('Error (*$10^{ -5}$)', fontname = 'Times New Roman', size = 14)
plt.ylabel('Points', fontname = 'Times New Roman', size = 14)
plt.savefig('compare-point_N_b.pdf',dpi=600,bbox_inches='tight')
plt.show()




















'''
4.plot cloud figure
'''
import meshio
MeshData=meshio.read('cubeunit.msh') #read data
Grid=MeshData.points
tetra=MeshData.cells['tetra']
# save data to vtu
cells={"tetra":tetra}
field={"T_pred":u_pred,
   "T_exact":u_exact,
   "T_err":u_pred_err,
   "flux_pred":flux_pred,
   "flux_err":flux_pred_err,
}
#
meshio.write_points_cells(
Filename,
Grid,
cells,
field,
)














'''
5.plot the plate
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
X1= pd.read_excel('jayalm_2m_1.xlsx') #read data
X1=X1.as_matrix()
Node_per_layer=X1[:,0]
Num_layer=X1[:,1]
Error_u=X1[:,2]

#find the maximum and minimum value, to define the size of plate
max=np.max(Error_u)
min=np.min(Error_u)
X_max=np.max(Node_per_layer)+1
X_min=np.min(Node_per_layer)-1
Y_max=np.max(Num_layer)+1
Y_min=np.min(Num_layer)-1
fig, ax = plt.subplots()
vmax=np.log10(max)
vmin=np.log10(min)
norm = colors.Normalize(vmax, vmin)

#plot point in plate
cm = plt.cm.get_cmap('RdYlBu')
sc = ax.scatter(Num_layer,Node_per_layer, s=75, c=np.log10(Error_u), norm=norm, alpha=1, cmap=cm)
ax.set_yticks(np.arange(X_min, X_max, 2))
ax.set_xticks(np.arange(Y_min, Y_max, 2))

#plot the color bar
cbar = plt.colorbar(sc)
cbar_range = np.arange(vmin-0.1, vmax+0.1, 0.75)
cbar.set_ticks(cbar_range)
labels = np.round(np.power(10, cbar_range), 6)
cbar.set_ticklabels(labels)
cbar.set_label('Error_u',size=15) 


ax.set_xlabel(r'Width', fontsize=16)
ax.set_ylabel(r'Depth', fontsize=16)
ax.set_title('Error',size=16)
ax.grid(True)

#plot the best position and label
min = np.argmin(Error_u)
ax.scatter( Num_layer[min], Node_per_layer[min] ,color='', marker='s', edgecolors='#AD03DE', s=200)
plt.text(Num_layer[min]+2, Node_per_layer[min]-0,[int(Num_layer[min]), int(Node_per_layer[min])], ha='center', va='center', fontsize=16)

fig.tight_layout()
plt.savefig("layer_lm2_m1.pdf",bbox_inches = 'tight')
#plt.savefig("Error_parameter_random.pdf",bbox_inches = 'tight')
#plt.savefig("Error_parameter_boyer.pdf",bbox_inches = 'tight')
plt.show()





















'''
6.Plot line and expand a defined area
'''

#import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib import rc
#from matplotlib.ticker import MaxNLocator
import pandas as pd
#import xlrd
import math
from matplotlib.ticker import FormatStrFormatter
random.seed(2019)
# Using seaborn's style
plt.style.use('classic')
#mpl.rcParams['axes.facecolor'] = 'w' # Or any suitable colour...
mpl.rc('axes',edgecolor='k')

#rc('text',usetex=True)

rc('font',size=20)
rc('font',family='Times New Roman')
rc('axes',labelsize=20)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc('font', weight='normal')

plt.close('all')

cmap = plt.get_cmap('winter')

# Change to the directory which contains the current script
#dirFile = os.path.dirname(os.path.join('\\\\in.uni-weimar.de\\fs\\home\\b\\m\\xima1978\\desktop',
                          #'3Dplottrail.py'))
#dirFile=os.path.dirname(os.path.abspath(__file__))
#dirFile="/C:/Users/45287/Desktop/Master Thesis/my_tensorflow/tanh_excel/"
    
def generate_colormap(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up, refcheck=False)
    arr = arr.reshape(7,N_up//7).T.reshape(-1)
    ret = cmap(arr)
    n = ret[:,3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
    ret[n//2:,3] *= np.arange(1,0.1,-0.9/b)
#     print(ret)
    return ret

if __name__ == "__main__":
    
    # ...............DEM relative error of the maximum deflection error.............
    # maximum defle
    X1= pd.read_excel('lm=1M=1_data.xlsx')
    X1=X1.as_matrix()
    X2= pd.read_excel('lm=1M=2_data.xlsx')
    X2=X2.as_matrix()
    X3= pd.read_excel('lm=1M=3_data.xlsx')
    X3=X3.as_matrix()
    #X4= pd.read_excel('lm=1.9 beta=1.3 M=2_data.xlsx')
    #X4=X4.as_matrix()
    x1=X1[:,1:2] 
    f_x1=X1[:,3:4] 
    g1=X1[:,5:6]
    
    x2=X2[:,1:2] 
    f_x2=X2[:,3:4] 
    g2=X2[:,5:6]
    
    x3=X3[:,1:2] 
    f_x3=X3[:,3:4] 
    g3=X3[:,5:6]
    
    #x4=X4[:,1:2] 
    #f_x4=X4[:,3:4] 
    #g4=X4[:,5:6]       
    clmap=generate_colormap(12)
    sampling = random.choices(clmap, k=6)
    clmap=['#0000FF','#A52A2A','#00FFFF','#FFD700','#008000','#FF0000']
    
    fig=plt.figure()
    #plt.figure(figsize=(16,20),dpi=200)
    left, bottom, width, height = 0, 0, 1, 1
    p1 = fig.add_axes([left, bottom, width, height])
    
    #p1 = plt.subplot(121,aspect=100/15)
    left, bottom, width, height = 0.3, 0.5, 0.25, 0.3
    p2 = fig.add_axes([left, bottom, width, height])  #plot the expand area
    #p2 = plt.subplot(121,aspect=0.5/0.2)
    p1.plot(x1,f_x1,'-',color='r',markersize=6,linewidth=1.5, markeredgecolor=clmap[0],label=r'$\lambda=1,M=1$')
    p1.plot(x2,f_x2,'-',color='y',markersize=6,linewidth=1.5, markeredgecolor=clmap[1],label= r'$\lambda=1,M=2$') 
    p1.plot(x3,f_x3,'-',color='c',markersize=6,linewidth=1.5,markeredgecolor=clmap[2],label= r'$\lambda=1,M=3$')               
    p2.plot(x1,f_x1,'-',color='r',markersize=6,linewidth=1.5, markeredgecolor=clmap[0])
    p2.plot(x2,f_x2,'-',color='y',markersize=6,linewidth=1.5, markeredgecolor=clmap[1]) 
    p2.plot(x3,f_x3,'-',color='c',markersize=6,linewidth=1.5,markeredgecolor=clmap[2])  
    #ax.plot(x4,g4,'-',color='r',markersize=6,linewidth=1.5,markeredgecolor=clmap[3],label= r'$\lambda=1.9,M=2$,$\beta=4/3$')
    
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    p1.axis([0.0,10,0,1.01])
    p1.set_xlabel('$\eta$',  fontname = 'Times New Roman', size=20)
    p1.set_ylabel('$df/d\eta$',  fontname = 'Times New Roman',size=20) 
    p1.grid(linestyle='dotted',color='k')
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 20,
             }
    
    p1.legend(loc='lower right',prop=font1)
    #frame = legend.get_frame()
    #frame.set_edgecolor('k')
    #plt.ylim(0,0.25)
    p2.axis([1.35,1.6,0.94,0.97])  #plot the expand square
    p2.set_xticks([])
    p2.set_yticks([])
    #p2.set_ylabel('$\eta$',  fontname = 'Times New Roman', size=14)
    #p2.set_xlabel('$g(\eta)$',  fontname = 'Times New Roman',size=14)
    #p2.grid(linestyle='dotted',color='k')
    
    #p2.legend(loc='lower right',prop=font1)
    #plot the expand square
    tx0 = 1.345
    tx1 = 1.605
    ty0 = 0.925
    ty1 = 0.985
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    p1.plot(sx,sy,"purple", linewidth=1.5)
    x = [[1.345, 3], [1.605, 5.4]]
    y = [[0.925, 0.57], [0.985, 0.801]]  # define the position of line
    for i in range(len(x)):
    
        p1.plot(x[i], y[i], color='k', linewidth=1.5)


 
    plt.savefig('fx1.pdf',bbox_inches='tight')
    plt.show()

