# from numpy import linspace, meshgrid
# from matplotlib.mlab import griddata

# import numpy as np
# from scipy.interpolate import griddata

# xi = np.linspace(4, 8, 10)
# yi = np.linspace(1, 4, 10)
# zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
# plt.contour(xi, yi, zi)
# quit()





# quit()

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import glob 
from scipy.interpolate import griddata
from matplotlib import colors
# # from matplotlib.mlab import griddata
plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 100
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})



files = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/visgrid_data_rpv_elec_2_1_2000*')

data = np.empty((0,8))
i = -1
for file in files:
	# try:
		i+=1
		print(file)
		jobid = file[83:-6]


		# bins = 50

		# y_float = jobid/bins

		# y_row = int(np.floor(y_float))

		# x_column = int(jobid - (y_row*bins))

		# # Stop mass of 1 TeV
		# mass2_for_test = 1000*1000

		# lambda_112_over_m2 = 10**((np.float(y_row)/50.)*4.5 - 9.5)

		# lambda_121_over_m2 = lambda_112_over_m2

		# lambda_112 = lambda_112_over_m2*mass2_for_test
		# lambda_121 = lambda_121_over_m2*mass2_for_test

		# theCouplings = [lambda_112, lambda_121, 1000]

		# theMass = (np.float(x_column)/50.)*plot1massmax*u.GeV


		current = np.load(file)
		print(current)
		quit()	
		
		data = np.append(data, [current], axis=0)
	# except:
		print('o')


# np.save('plot_data/data_2_1',data)
quit()
data = np.load('plot_data/data_2_1.npy')

data_all = np.load('plot_data/data_all_2_1.npy')

# num_mesons = 1.4E17
num_mesons = 1.37632E17
events_in_5_years = num_mesons*data[:,2]*data[:,5]*data[:,7]/1000 
events_in_5_years_all = num_mesons*data_all[:,2]*data_all[:,5]/1000 

data[:,1] = data[:,1]/(1000*1000)
data[:,4] = data[:,4]/(1000*1000)

data_all[:,1] = data_all[:,1]/(1000*1000)
data_all[:,4] = data_all[:,4]/(1000*1000)

# Add points if kinematic limits are not well represented in contour plots
for i in range(0, 50):
	rangea = 9.5 - 5
	exp = float(i/50.)*rangea
	data = np.append(data, [[1.97, 10**(-5 - exp),0,0,10**(-5 - exp),0,0,0]], axis=0)
	events_in_5_years = np.append(events_in_5_years, 0)


for i in range(0, 50):
	rangea = 9.5 - 5
	exp = float(i/50.)*rangea
	data_all = np.append(data_all, [[1.97, 10**(-5 - exp),0,0,10**(-5 - exp),0,0]], axis=0)
	events_in_5_years_all = np.append(events_in_5_years_all, 0)



xi,yi = np.meshgrid(np.unique(data[:,0]),np.unique(data[:,1]))

grid_x = data[:,0]
grid_y = data[:,1]

# for i in range(0, 50):
# 	rangea = 2
# 	exp = float(i/50.)*rangea
# 	grid_x = np.append(grid_x, exp)
# 	grid_y = np.append(grid_y, 1E-5)

# print(np.shape(grid_y), np.shape(xi))
# quit()
zi = griddata((grid_x,grid_y),events_in_5_years,(xi,yi),method='nearest')


xi_all,yi_all = np.meshgrid(np.unique(data_all[:,0]),np.unique(data_all[:,1]))

grid_x_all = data_all[:,0]
grid_y_all = data_all[:,1]
# for i in range(0, 50):
# 	rangea = 2
# 	exp = float(i/50.)*rangea
# 	grid_x = np.append(grid_x, exp)
# 	grid_y = np.append(grid_y, 1E-5)

# print(np.shape(grid_y), np.shape(xi))
# quit()
zi_all = griddata((grid_x_all,grid_y_all),events_in_5_years_all,(xi_all,yi_all),method='nearest')



fig = plt.figure(figsize=(7,6))



ax1 = fig.add_subplot(1,1,1)

# cmap = colors.ListedColormap(['r','#E1E1FD'])
# cmap = colors.ListedColormap(['r','#E6D9F8'])#purple
cmap = colors.ListedColormap(['r','#F6D7DA'])#bristol
bounds = [0, 2.3, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contourf(xi_all,yi_all,zi_all, [2.3,1E16], cmap=cmap,norm=norm)

cmap = colors.ListedColormap(['k','k'])
bounds = [0, 2.3, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi_all,yi_all,zi_all, [2.3,1E16], cmap=cmap,linewidths=1, linestyles='dashed')

# cmap = colors.ListedColormap(['r','#8888FF','#6D6DCF','#5A5AAB'])#ship 2015 colours
# cmap = colors.ListedColormap(['r','#A0A0FE','#7879FD','#5355FE'])#mathusla colours
# cmap = colors.ListedColormap(['r','#D2B8F6','#BC92F6','#A76EF7'])#purple2
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



points_ship = np.loadtxt('overlay_data/overlay_ship_bench2_plot1.txt', delimiter=', ')
plt.plot(points_ship[:,0], points_ship[:,1], '-',c='r',label='SHiP 2015',linewidth=1)
points_math = np.loadtxt('overlay_data/overlay_mathusla_bench2_plot1.txt', delimiter=', ')
plt.plot(points_math[:,0], points_math[:,1], '-',c='b',label='MATHUSLA',linewidth=1)


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
plt.savefig('plots/plot1_bench1_compare.png')


