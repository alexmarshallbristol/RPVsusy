
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import glob 
from scipy.interpolate import griddata
from matplotlib import colors
plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 100
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})


files = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/NEW/grid_data_rpv_2_1_1_2000_*_0.npy')


data_total = np.empty((0,15))

for file in files:

	current = np.expand_dims(np.load(file),0)

	data_total = np.concatenate((data_total, current),axis=0)

# data = [theMass,theCouplings[0],0,sum_weight_vis,theCouplings[1], sum_weight_all, sum_weight_no_cuts,br_to_vis,np.sum(weights_times_BR_b0),NUM_B0,np.sum(weights_times_BR_bplus),NUM_BPLUS,np.sum(weights_times_BR_b0_no_cuts),np.sum(weights_times_BR_bplus_no_cuts),total_weight] #will want to divide this sum by number of events simulated

acceptance = np.divide(data_total[:,5],100)
masses = data_total[:,0]
couplings = data_total[:,1]*1E-6

where_0 = np.where(data_total[:,5]==0)

acceptance = np.delete(acceptance, where_0)
masses = np.delete(masses, where_0)
couplings = np.delete(couplings, where_0)


# print(data_total[np.where(acceptance<0.01)])
# quit()

plt.scatter(masses, couplings, c=acceptance, marker='s', s=250, norm=LogNorm())
plt.yscale('log')
plt.ylim(1E-9,1E-5)
plt.xlim(0,2)
plt.title('Reconstructed weight/total weight')
plt.colorbar()
# plt.ylabel(r'\textbf{$\frac{\lambda^\prime_{122}}{m^2_{\bar{f}}}=\frac{\lambda^\prime_{112}}{m^2_{\bar{f}}}$ in GeV$^{-2}$}', fontsize=12)
# plt.xlabel(r'\textbf{$m_{\widetilde{\chi}^0_1}$ in MeV}', fontsize=12, labelpad=12)
plt.ylabel('Coupling', fontsize=12)
plt.xlabel('Mass', fontsize=12, labelpad=12)
plt.savefig('acceptance_divide_by_100',bbox_inches='tight')

quit()

















data = data_total



num_mesons = 1.37632E17
events_in_5_years = num_mesons*data[:,2]*data[:,5]*data[:,7]/100
# events_in_5_years_all = num_mesons*data_all[:,2]*data_all[:,5]/1000 

data[:,1] = data[:,1]/(1000*1000)
data[:,4] = data[:,4]/(1000*1000)

# data_all[:,1] = data_all[:,1]/(1000*1000)
# data_all[:,4] = data_all[:,4]/(1000*1000)

# Add points if kinematic limits are not well represented in contour plots
# for i in range(0, 50):
# 	rangea = 9.5 - 5
# 	exp = float(i/50.)*rangea
# 	data = np.append(data, [[1.97, 10**(-5 - exp),0,0,10**(-5 - exp),0,0,0]], axis=0)
# 	events_in_5_years = np.append(events_in_5_years, 0)


# for i in range(0, 50):
# 	rangea = 9.5 - 5
# 	exp = float(i/50.)*rangea
# 	data_all = np.append(data_all, [[1.97, 10**(-5 - exp),0,0,10**(-5 - exp),0,0]], axis=0)
# 	events_in_5_years_all = np.append(events_in_5_years_all, 0)



xi,yi = np.meshgrid(np.unique(data[:,0]),np.unique(data[:,1]))

grid_x = data[:,0]
grid_y = data[:,1]

zi = griddata((grid_x,grid_y),events_in_5_years,(xi,yi),method='nearest')


# xi_all,yi_all = np.meshgrid(np.unique(data_all[:,0]),np.unique(data_all[:,1]))

# grid_x_all = data_all[:,0]
# grid_y_all = data_all[:,1]






# zi_all = griddata((grid_x_all,grid_y_all),events_in_5_years_all,(xi_all,yi_all),method='nearest')


fig = plt.figure(figsize=(7,6))

ax1 = fig.add_subplot(1,1,1)

# cmap = colors.ListedColormap(['r','#F6D7DA'])#bristol
# bounds = [0, 2.3, 1E16]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# plt.contourf(xi_all,yi_all,zi_all, [2.3,1E16], cmap=cmap,norm=norm)

# cmap = colors.ListedColormap(['k','k'])
# bounds = [0, 2.3, 1E16]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# plt.contour(xi_all,yi_all,zi_all, [2.3,1E16], cmap=cmap,linewidths=1, linestyles='dashed')


cmap = colors.ListedColormap(['r','#ECB1B6','#DB7B84','#BC515B'])#bristol
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contourf(xi,yi,zi, [2.3, 3E3, 3E6, 1E16], cmap=cmap,norm=norm)


cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi,yi,zi, [3E6, 1E16], cmap=cmap,linewidths=1)

cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi,yi,zi, [3E3,1E16], cmap=cmap,linewidths=1)

cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi,yi,zi, [2.3,1E16], cmap=cmap,linewidths=2)


cmap = plt.get_cmap('viridis')
cmap.set_under(color='white')  


# plt.scatter(data_all[:,0], data_all[:,1], c=events_in_5_years_all, norm=LogNorm(), cmap=cmap)
plt.yscale('log')
plt.ylim(1E-5,0.0000000005)
plt.gca().invert_yaxis()
plt.xlim(0,2)
plt.xticks([0,0.5,1,1.5,2],[0,500,1000,1500,2000])
plt.grid(linestyle=':')

sfermion_mass = 1000
value = (0.03*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,2],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(1,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 1 TeV', horizontalalignment='center', verticalalignment='center', fontsize='10')

sfermion_mass = 250
value = (0.03*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,2],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(1,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 250 GeV', horizontalalignment='center', verticalalignment='center', fontsize='10')


sfermion_mass = 5000
value = (0.03*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,2],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(1,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 5 TeV', horizontalalignment='center', verticalalignment='center', fontsize='10')


plt.legend(loc='upper left', bbox_to_anchor=(0, 0.99), fontsize=10,framealpha=0,borderpad=0.1,labelspacing=0.1,handletextpad=0,borderaxespad=0,columnspacing=0,markerfirst=True)

plt.ylabel(r'\textbf{$\frac{\lambda^\prime_{122}}{m^2_{\bar{f}}}=\frac{\lambda^\prime_{112}}{m^2_{\bar{f}}}$ in GeV$^{-2}$}', fontsize=18)
plt.xlabel(r'\textbf{$m_{\widetilde{\chi}^0_1}$ in MeV}', fontsize=18, labelpad=15)

# plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('sensitivity.png')


