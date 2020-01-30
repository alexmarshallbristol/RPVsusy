# from numpy import linspace, meshgrid
# from matplotlib.mlab import griddata

# import numpy as np
# from scipy.interpolate import griddata

# xi = np.linspace(4, 8, 10)
# yi = np.linspace(1, 4, 10)
# zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
# plt.contour(xi, yi, zi)
# quit()







# import ROOT 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import glob 
# import argparse
# import rpvsusy
from scipy.interpolate import griddata
from matplotlib import colors
import os.path, time

# # from matplotlib.mlab import griddata
plt.rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 100
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 15})




# # print(np.shape(glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/4visgrid_data_rpv_5_1_5500_*_0.npy')))
# # print(np.shape(glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/4visgrid_data_rpv_5_1_5500_*_1.npy')))
# # quit()


# data_all = np.empty((0,14))
# for i in range(0, 2500):
# 	# print(i)
# 	bool_0 = False
# 	bool_1 = False
# 	try:
# 		current_0 = np.load('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/4visgrid_data_rpv_5_1_5500_%d_0.npy'%i)
# 		if np.shape(current_0)[0] == 7:
# 			current_0 = np.append(current_0, [0,0,0,0,0,0,0])
# 		bool_0 = True
# 		# print(current_0[8],current_0[10])
# 	except:
# 		print('no 0')
# 	try:
# 		current_1 = np.load('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/4visgrid_data_rpv_5_1_5500_%d_1.npy'%i)
# 		if np.shape(current_1)[0] == 7:
# 			current_1 = np.append(current_1, [0,0,0,0,0,0,0])
# 		bool_1 = True
# 		# print(current_1[8],current_1[10])
# 	except:
# 		print('no 1')

# 	if bool_0 == True and bool_1 == True:
# 		current_0[9] = current_1[9]
# 		current_0[12] = current_1[12]
# 		# if i == 710:
# 		# 	current_0[9] = 10000000
# 		data_all = np.append(data_all, [current_0], axis=0)
# 	elif bool_0 == True:
# 		# print(i, current_0[0])
# 		# quit()
# 		if current_0[0]>3.4:
# 		# current_0[9] = current_1[9]
# 		# current_0[12] = current_1[12]
# 		# if i == 710:
# 		# 	current_0[9] = 10000000
# 			data_all = np.append(data_all, [current_0], axis=0)
# 	# elif bool_0 == True and bool_1 == False:
# 	# 	data_all = np.append(data_all, [current_0], axis=0)
# 	# elif bool_0 == False and bool_1 == True:
# 	# 	data_all = np.append(data_all, [current_1], axis=0)
# 	else:
# 		continue

# print(np.shape(data_all))
# # quit()

# print(np.shape(data_all))
# np.save('plot_data/plot1_bench5_data',data_all)
# quit()
data_all = np.load('plot_data/plot1_bench5_data.npy')
# quit()

data_all_all = np.load('plot_data/plot1_bench5_data_all_all.npy')
'''
import rpvsusy_new as rpvsusy
# import rpvsusy


for i in range(0,np.shape(data_all)[0]):
		# print(data_all[i][0],data_all[i][1], data_all[i][4])
		try:
			b = rpvsusy.RPVSUSY(data_all[i][0],[data_all[i][1], data_all[i][4]],1e3,5, 3,False,0)
			# print(b.findProdBranchingRatio('B+ -> N tau+'))
			if data_all[i][7] > 0:
				data_all[i][7] = data_all[i][7]/b.findProdBranchingRatio('B0 -> N nu_tau')
		except:
			print(data_all[i][7])
		try:
			b = rpvsusy.RPVSUSY(data_all[i][0],[data_all[i][1], data_all[i][4]],1e3,5, 3,False,1)
			# print(b.findProdBranchingRatio('B+ -> N tau+'))
			if data_all[i][9] > 0:
				data_all[i][9] = data_all[i][9]/b.findProdBranchingRatio('B+ -> N tau+')
		except:
			print(data_all[i][9])
'''


# 		quit()

# quit()





# files_all_1 = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/2allgrid_data_rpv_5_1_5500*_0.*')
# # files_all = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/2allgrid_data_rpv_5_1_5500_614.npy')
# # quit()
# # files_all = glob.glob('allgrid_data_rpv_5_1_5500*')
# # quit()
# # files = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/visgrid_data_rpv_elec_4_1_5500*')
# # # [theMass,theCouplings[0],prod_branching_ratio,sum_weight_vis,theCouplings[1], sum_weight_all, sum_weight_no_cuts, br_to_vis]
# # data = np.empty((0,8))
# data_all = np.empty((0,13))
# for i in range(0, np.shape(files_all_1)[0]):
# 	try:
# 		current_1 = np.load(files_all_1[i])
# 		if np.shape(current_1)[0] == 7:
# 			current_1 = np.append(current_1, [0,0,0,0,0,0])
# 		data_all = np.append(data_all, [current_1], axis=0)
# 	except:
# 		print('o')
# # quit()


# data_all = np.empty((0,13))
# i = -1
# for file in files_all_0:
# 	try:
# 		i+=1
# 		current = np.load(file)
# 		# print(current)
# 		# if np.shape(current)[0] == 11:
# 		# 	print(file,current)
# 		# if current[7] != 0:
# 		# 	print(file,current)
# 		# 	quit()

# 		if np.shape(current)[0] == 7:
# 			current = np.append(current, [0,0,0,0,0,0])
# 		# if current[10] > 0:
# 		# 	print(current[8], current[10], file) 
# 		# 	print("Last modified: %s" % os.path.getmtime(file))
# 		if os.path.getmtime(file) < 1566299316.97:
# 			print(file, time.ctime(os.path.getmtime(file)))
# 			os.remove(file)
# 			quit()
# 		if current[9] < 0:
# 			print(file)
# 		# if current[5] > 0:
# 			# print(current)
# 		data_all = np.append(data_all, [current], axis=0)
# 	except:
# 		print('o')
# # quit()
number_of_b0_mesons_SHiP = 4.54784E13
number_of_bplus_mesons_SHiP = 4.53696E13#*((0.418/0.417)*(0.83/0.80))

num_from_b0 = 500
num_from_bplus = 500

print(np.sum(num_from_b0), np.sum(num_from_bplus))

# quit()
# np.save('num_mesons_614',[num_from_b0, num_from_bplus])
# quit()
# print(np.shape(num_from_b0))
events_in_5_years_all_b0 = np.zeros(np.shape(data_all)[0])
events_in_5_years_all_bplus = np.zeros(np.shape(data_all)[0])
events_in_5_years = np.zeros(np.shape(data_all)[0])

print(np.shape(data_all), 'data_all')

print(np.shape(events_in_5_years_all_b0))

for i in range(0, np.shape(data_all)[0]):
	# print(data_all[i])

	# if num_from_b0[i] > 0:
		events_in_5_years_all_b0[i] = number_of_b0_mesons_SHiP*data_all[i][7]*data_all[i][13]/(500)
	# else:
	# 	events_in_5_years_all_b0[i] = 0
	# if num_from_bplus[i] > 0:
		# print('here')
		events_in_5_years_all_bplus[i] = number_of_bplus_mesons_SHiP*data_all[i][9]*data_all[i][13]/500
	# else:
	# 	events_in_5_years_all_bplus[i] = 0

		events_in_5_years[i] = events_in_5_years_all_bplus[i] + events_in_5_years_all_b0[i]

print(np.sum(events_in_5_years))





events_in_5_years_all_b0 = np.zeros(np.shape(data_all_all)[0])
events_in_5_years_all_bplus = np.zeros(np.shape(data_all_all)[0])
events_in_5_years_all = np.zeros(np.shape(data_all_all)[0])

# print(np.shape(data_all), 'data_all')

print(np.shape(events_in_5_years_all_b0))

for i in range(0, np.shape(data_all_all)[0]):
	# print(data_all[i])

	# if num_from_b0[i] > 0:
		events_in_5_years_all_b0[i] = number_of_b0_mesons_SHiP*data_all_all[i][7]/(500)
	# else:
	# 	events_in_5_years_all_b0[i] = 0
	# if num_from_bplus[i] > 0:
		# print('here')
		events_in_5_years_all_bplus[i] = number_of_bplus_mesons_SHiP*data_all_all[i][9]/500
	# else:
	# 	events_in_5_years_all_bplus[i] = 0

		events_in_5_years_all[i] = events_in_5_years_all_bplus[i] + events_in_5_years_all_b0[i]

print(np.sum(events_in_5_years_all))
# quit()
'''
files_all_0 = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/2allgrid_data_rpv_5_1_5500*_0.*')
files_all_1 = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/2allgrid_data_rpv_5_1_5500*_1.*')
# files_all = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/2allgrid_data_rpv_5_1_5500_614.npy')
# quit()
# files_all = glob.glob('allgrid_data_rpv_5_1_5500*')
print(np.shape(files_all_0),np.shape(files_all_1))
# quit()
# files = glob.glob('/eos/experiment/ship/user/amarshal/RPV_output/check_elec/visgrid_data_rpv_elec_4_1_5500*')
# # [theMass,theCouplings[0],prod_branching_ratio,sum_weight_vis,theCouplings[1], sum_weight_all, sum_weight_no_cuts, br_to_vis]
# data = np.empty((0,8))
data_all = np.empty((0,13))
for i in range(0, np.shape(files_all_0)[0]):
	try:
		print(files_all_0[i], files_all_1[i])
		current_0 = np.load(files_all_0[i])
		current_1 = np.load(files_all_1[i])
		if np.shape(current_0)[0] == 7:
			current_0 = np.append(current_0, [0,0,0,0,0,0])
		if np.shape(current_1)[0] == 7:
			current_1 = np.append(current_1, [0,0,0,0,0,0])
		current_0[9] = current_1[9]
		data_all = np.append(data_all, [current_0], axis=0)
	except:
		print('o')
# quit()


# data_all = np.empty((0,13))
# i = -1
# for file in files_all_0:
# 	try:
# 		i+=1
# 		current = np.load(file)
# 		# print(current)
# 		# if np.shape(current)[0] == 11:
# 		# 	print(file,current)
# 		# if current[7] != 0:
# 		# 	print(file,current)
# 		# 	quit()

# 		if np.shape(current)[0] == 7:
# 			current = np.append(current, [0,0,0,0,0,0])
# 		# if current[10] > 0:
# 		# 	print(current[8], current[10], file) 
# 		# 	print("Last modified: %s" % os.path.getmtime(file))
# 		if os.path.getmtime(file) < 1566299316.97:
# 			print(file, time.ctime(os.path.getmtime(file)))
# 			os.remove(file)
# 			quit()
# 		if current[9] < 0:
# 			print(file)
# 		# if current[5] > 0:
# 			# print(current)
# 		data_all = np.append(data_all, [current], axis=0)
# 	except:
# 		print('o')
# # quit()
number_of_b0_mesons_SHiP = 4.54784E13
number_of_bplus_mesons_SHiP = 4.53696E13

num_from_b0 = 500
num_from_bplus = 500

print(np.sum(num_from_b0), np.sum(num_from_bplus))

# quit()
# np.save('num_mesons_614',[num_from_b0, num_from_bplus])
# quit()
# print(np.shape(num_from_b0))
events_in_5_years_all_b0 = np.zeros(np.shape(data_all)[0])
events_in_5_years_all_bplus = np.zeros(np.shape(data_all)[0])
events_in_5_years_all = np.zeros(np.shape(data_all)[0])

print(np.shape(data_all), 'data_all')

print(np.shape(events_in_5_years_all_b0))

for i in range(0, np.shape(data_all)[0]):
	# print(data_all[i])

	# if num_from_b0[i] > 0:
		events_in_5_years_all_b0[i] = number_of_b0_mesons_SHiP*data_all[i][7]/(500)
	# else:
	# 	events_in_5_years_all_b0[i] = 0
	# if num_from_bplus[i] > 0:
		# print('here')
		events_in_5_years_all_bplus[i] = number_of_bplus_mesons_SHiP*data_all[i][9]/500
	# else:
	# 	events_in_5_years_all_bplus[i] = 0

		events_in_5_years_all[i] = events_in_5_years_all_bplus[i] + events_in_5_years_all_b0[i]

# print(np.sum(events_in_5_years_all_bplus), np.sum(events_in_5_years_all_b0),'events')
# quit()
	# events_in_5_years_all[i] = events_in_5_years_all_b0[i]

# print(events_in_5_years_all[np.where(events_in_5_years_all!=0)])
'''

# for i in range(0, np.shape(num_from_b0)[0]):
# 	# print(data_all[i])

# 	if num_from_b0[i] > 0:
# 		events_in_5_years_all_b0[i] = number_of_b0_mesons_SHiP*data_all[i][10]/num_from_b0[i]
# 	else:
# 		events_in_5_years_all_b0[i] = 0
# 	if num_from_bplus[i] > 0:
# 		# print('here')
# 		events_in_5_years_all_bplus[i] = number_of_bplus_mesons_SHiP*data_all[i][11]/num_from_bplus[i]
# 	else:
# 		events_in_5_years_all_bplus[i] = 0

# 	events_in_5_years_all[i] = events_in_5_years_all_bplus[i] + events_in_5_years_all_b0[i]
# 	# events_in_5_years_all[i] = events_in_5_years_all_b0[i]

# print(events_in_5_years_all[np.where(events_in_5_years_all!=0)])
'''
'''


# quit()



# if num_from_b0 > 0:
# 	events_in_5_years_all_b0 = number_of_b0_mesons_SHiP*data_all[:,7]/num_from_b0
# else:
# 	events_in_5_years_all_b0 = 0
# if num_from_bplus > 0:
# 	events_in_5_years_all_bplus = number_of_bplus_mesons_SHiP*data_all[:,9]/num_from_bplus
# else:
# 	events_in_5_years_all_bplus = 0

# events_in_5_years_all = events_in_5_years_all_b0 + events_in_5_years_all_bplus



# data[:,1] = data[:,1]/(1000*1000)
# data[:,4] = data[:,4]/(1000*1000)

data_all[:,1] = data_all[:,1]/(1000*1000)
data_all[:,4] = data_all[:,4]/(1000*1000)

data_all_all[:,1] = data_all_all[:,1]/(1000*1000)
data_all_all[:,4] = data_all_all[:,4]/(1000*1000)

# for i in range(0, 50):
# 	rangea = 9.5 - 5
# 	exp = float(i/50.)*rangea
# 	# data = np.append(data, [[5.28, 10**(-5 - exp),0,0,10**(-5 - exp),0,0,0]], axis=0)
# 	# events_in_5_years = np.append(events_in_5_years, 0)
# 	data_all = np.append(data_all, [[5.28, 10**(-5 - exp),0,0,10**(-5 - exp),0,0]], axis=0)
# 	events_in_5_years_all = np.append(events_in_5_years_all, 0)

# Add points if kinematic limits are not well represented in contour plots
# for i in range(0, 50):
# 	rangea = 9.5 - 5
# 	exp = float(i/50.)*rangea
# 	data = np.append(data, [[1.97, 10**(-5 - exp),0,0,10**(-5 - exp),0,0,0]], axis=0)
# 	events_in_5_years = np.append(events_in_5_years, 0)

xi_all,yi_all = np.meshgrid(np.unique(data_all[:,0]),np.unique(data_all[:,1]))

grid_x_all = data_all[:,0]
grid_y_all = data_all[:,1]

zi_all = griddata((grid_x_all,grid_y_all),events_in_5_years,(xi_all,yi_all),method='nearest')




xi_all_all,yi_all_all = np.meshgrid(np.unique(data_all_all[:,0]),np.unique(data_all_all[:,1]))

grid_x_all_all = data_all_all[:,0]
grid_y_all_all = data_all_all[:,1]

zi_all_all = griddata((grid_x_all_all,grid_y_all_all),events_in_5_years_all,(xi_all_all,yi_all_all),method='nearest')

# zi_b0 = griddata((grid_x_all,grid_y_all),events_in_5_years_all_b0,(xi_all,yi_all),method='nearest')
# zi_bplus = griddata((grid_x_all,grid_y_all),events_in_5_years_all_bplus,(xi_all,yi_all),method='nearest')



# xi,yi = np.meshgrid(np.unique(data[:,0]),np.unique(data[:,1]))

# grid_x = data[:,0]
# grid_y = data[:,1]

# zi = griddata((grid_x,grid_y),events_in_5_years,(xi,yi),method='nearest')





fig = plt.figure(figsize=(7,6))

# cmap = colors.ListedColormap(['r','#E1E1FD'])
# cmap = colors.ListedColormap(['r','#E6D9F8'])#purple







cmap = plt.get_cmap('viridis')
cmap.set_under(color='white')  



# points_ship_vis = np.loadtxt('overlay_data/bench4_vis_ship_2015.txt', delimiter=', ')
# points_ship_all = np.loadtxt('overlay_data/bench4_all_ship_2015.txt', delimiter=', ')

# points_math_vis = np.loadtxt('overlay_data/bench4_vis_math_2015.txt', delimiter=', ')
# points_math_all = np.loadtxt('overlay_data/bench4_all_math_2015.txt', delimiter=', ')

# print(points_math_all,np.shape(points_math_all))

# print(np.shape(points_ship_vis), np.shape(points_ship_all))
# plt.plot(points_ship_vis[:,0]/1000, points_ship_vis[:,1],'-',c='r',label='SHiP 2015',linewidth=1)
# plt.plot(points_ship_all[:,0]/1000, points_ship_all[:,1], '--',c='r',linewidth=1)

# plt.plot(points_math_vis[:,0], points_math_vis[:,1],'-',c='b',label='MATHUSLA',linewidth=1)
# plt.plot(points_math_all[:,0], points_math_all[:,1], '--',c='b',linewidth=1)

# plt.legend(loc='upper left', bbox_to_anchor=(0, 0.99), fontsize=10,framealpha=0,borderpad=0.1,labelspacing=0.1,handletextpad=0,borderaxespad=0,columnspacing=0,markerfirst=True)

cmap = colors.ListedColormap(['r','#F6D7DA'])#bristol
bounds = [0, 2.3, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contourf(xi_all_all,yi_all_all,zi_all_all, [2.3,1E16], cmap=cmap,norm=norm)

cmap = colors.ListedColormap(['k','k'])
bounds = [0, 2.3, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi_all_all,yi_all_all,zi_all_all, [2.3,1E16], cmap=cmap,linewidths=1, linestyles='dashed')






# cmap = colors.ListedColormap(['r','#A0A0FE','#7879FD','#5355FE'])
cmap = colors.ListedColormap(['r','#ECB1B6','#DB7B84','#BC515B'])#purple2
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contourf(xi_all,yi_all,zi_all, [2.3, 3E3, 3E6, 1E16], cmap=cmap,norm=norm)

cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi_all,yi_all,zi_all, [3E6, 1E16], cmap=cmap,linewidths=1)

cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi_all,yi_all,zi_all, [3E3,1E16], cmap=cmap,linewidths=1)

cmap = colors.ListedColormap(['k','k','k','k'])
bounds = [0, 2.3, 3E3, 3E6, 1E16]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.contour(xi_all,yi_all,zi_all, [2.3,1E16], cmap=cmap,linewidths=2)

print(np.amin(events_in_5_years_all), np.amax(events_in_5_years_all))
# plt.scatter(data_all[:,0], data_all[:,1], c=events_in_5_years_all, norm=LogNorm(), cmap=cmap)

# plt.scatter(data_all[:,0], data_all[:,1], c=events_in_5_years_all, norm=LogNorm(), cmap=cmap,vmin=2.3)

# plt.scatter(data_all[:,0], data_all[:,1], c=events_in_5_years_all, norm=LogNorm(), cmap=cmap,vmax=2.3)
# plt.colorbar()
plt.yscale('log')
plt.ylim(0.0000000005,1E-5)
#plt.ylim(1E-9,1E-8)
plt.xlim(0,5.5)
# plt.colorbar()
plt.xticks([0,1,2,3,4,5],[0,1000,2000,3000,4000,5000])
# plt.gca().invert_yaxis()
plt.grid(linestyle=':')


sfermion_mass = 1000
value = (0.06*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,5.5],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(5.5/2,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 1 TeV', horizontalalignment='center', verticalalignment='center', fontsize='10')

sfermion_mass = 250
value = (0.06*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,5.5],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(5.5/2,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 250 GeV', horizontalalignment='center', verticalalignment='center', fontsize='10')


sfermion_mass = 5000
value = (0.06*sfermion_mass/100)/sfermion_mass**2
plt.axhline(value, color='k')
log_value = np.log10(value)
log_value_upper = log_value + 0.15 
log_value_text = log_value - 0.09
plt.fill_between([0,5.5],[value,value],[10**log_value_upper,10**log_value_upper], facecolor="none", hatch="/", edgecolor="k", linewidth=0.0)
plt.text(5.5/2,10**log_value_text,'Present RPV limit for $m_{\\tilde{f}} =$ 5 TeV', horizontalalignment='center', verticalalignment='center', fontsize='10')

plt.ylabel(r'\textbf{$\frac{\lambda^\prime_{312}}{m^2_{\bar{f}}}=\frac{\lambda^\prime_{313}}{m^2_{\bar{f}}}$ in GeV$^{-2}$}', fontsize=18)
plt.xlabel(r'\textbf{$m_{\widetilde{\chi}^0_1}$ in MeV}', fontsize=18, labelpad=15)

# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)

plt.savefig('plots/plot1_bench5.png')

# modern (0.0, 18554854345.689705)
# 2015 (0.0, 18938265095.54097)

