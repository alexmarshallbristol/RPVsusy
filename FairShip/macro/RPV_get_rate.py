
import ROOT 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import glob 
import argparse
import rpvsusy
from rootpyPickler import Unpickler
import shipunit as u
from ShipGeoConfig import ConfigRegistry
from decorators import *
import shipRoot_conf
shipRoot_conf.configure()
from array import array

# Pass in jobid - index of current point in 50x50 grid.
parser = argparse.ArgumentParser()
parser.add_argument('-jobid', action='store', dest='jobid', type=int,
					help='jobid')
parser.add_argument('-benchmark', action='store', dest='benchmark', type=int,
					help='benchmark')
parser.add_argument('-plot', action='store', dest='plot', type=int,
					help='plot')
parser.add_argument('-plot2mass', action='store', dest='plot2mass', type=float,
					help='plot2mass GeV',default=0.6)
parser.add_argument('-plot1massmax', action='store', dest='plot1massmax', type=float,
					help='plot1massmax GeV',default=2.0)
parser.add_argument('-leptongeneration', action='store', dest='leptongeneration', type=int,
					help='leptongeneration',default=2.0)
parser.add_argument('-visible', action='store', dest='visible', type=int,
					help='visible',default=0)
parser.add_argument('-bzerobplus', action='store', dest='bzerobplus', type=int,
					help='bzerobplus',default=0)
results = parser.parse_args()


jobid = int(results.jobid)
benchmark = int(results.benchmark)
plot = int(results.plot)
bzerobplus = int(results.bzerobplus)


visible = int(results.visible)
if visible == 1:
  visible = True
else:
  visible = False


print('benchmark:',benchmark,'plot:',plot)
plot2mass = results.plot2mass
plot1massmax = results.plot1massmax
leptongeneration = results.leptongeneration

# Compute mass and couplings - exact same procedure as in the RPV run_simScript.py code.
if plot == 1: 
	bins = 15

	y_float = jobid/bins

	y_row = int(np.floor(y_float))

	x_column = int(jobid - (y_row*bins))

	# Stop mass of 1 TeV
	mass2_for_test = 1000*1000

	lambda_112_over_m2 = 10**((np.float(y_row)/15.)*4.5 - 9.5)

	lambda_121_over_m2 = lambda_112_over_m2

	lambda_112 = lambda_112_over_m2*mass2_for_test
	lambda_121 = lambda_121_over_m2*mass2_for_test

	theCouplings = [lambda_112, lambda_121, 1000]

	theMass = (np.float(x_column)/15.)*plot1massmax*u.GeV

	print(' ')
	print('theCouplings',theCouplings)
	print('theMass',theMass)
	print(' ')
elif plot == 2: 
	bins = 50

	y_float = jobid/bins

	y_row = int(np.floor(y_float))

	x_column = int(jobid - (y_row*bins))

	# Stop mass of 1 TeV
	mass2_for_test = 1000*1000

	lambda_112_over_m2 = 10**((np.float(y_row)/50.)*7. - 12.)
	lambda_121_over_m2 = 10**((np.float(x_column)/50.)*7. - 12.)

	lambda_112 = lambda_112_over_m2*mass2_for_test
	lambda_121 = lambda_121_over_m2*mass2_for_test

	theCouplings = [lambda_112, lambda_121, 1000]

	theMass = plot2mass*u.GeV

	print(' ')
	print('theCouplings',theCouplings)
	print('theMass',theMass)
	print(' ')

print('got indexes')


# Start of script is all copied from ShipAna.py
try:
	PDG = ROOT.TDatabasePDG.Instance()

	eosship = ROOT.gSystem.Getenv("EOSSHIP")

	geoFile = 'geofile_full.conical.Pythia8-TGeant4.root'
	fgeo = ROOT.TFile(geoFile)

	# new geofile, load Shipgeo dictionary written by run_simScript.py
	upkl    = Unpickler(fgeo)
	ShipGeo = upkl.load('ShipGeo')
	ecalGeoFile = ShipGeo.ecal.File
	dy = ShipGeo.Yheight/u.m

	# -----Create geometry----------------------------------------------
	import shipDet_conf
	run = ROOT.FairRunSim()
	run.SetName("TGeant4")  # Transport engine
	run.SetOutputFile("dummy")  # Output file
	run.SetUserConfig("g4Config_basic.C") # geant4 transport not used, only needed for the mag field
	rtdb = run.GetRuntimeDb()
	# -----Create geometry----------------------------------------------
	modules = shipDet_conf.configure(run,ShipGeo)
	run.Init()
	import geomGeant4
	if hasattr(ShipGeo.Bfield,"fieldMap"):
			fieldMaker = geomGeant4.addVMCFields(ShipGeo, '', True)

	sGeo   = ROOT.gGeoManager
	geoMat =  ROOT.genfit.TGeoMaterialInterface()
	ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
	bfield = ROOT.genfit.FairShipFields()
	fM = ROOT.genfit.FieldManager.getInstance()
	fM.init(bfield)


	def dist2InnerWall(X,Y,Z):
	  dist = 0
	 # return distance to inner wall perpendicular to z-axis, if outside decayVolume return 0.
	  node = sGeo.FindNode(X,Y,Z)
	  if ShipGeo.tankDesign < 5:
	     if not 'cave' in node.GetName(): return dist  # TP 
	  else:
	     if not 'DecayVacuum' in node.GetName(): return dist
	  start = array('d',[X,Y,Z])
	  nsteps = 8
	  dalpha = 2*ROOT.TMath.Pi()/nsteps
	  rsq = X**2+Y**2
	  minDistance = 100 *u.m
	  for n in range(nsteps):
	    alpha = n * dalpha
	    sdir  = array('d',[ROOT.TMath.Sin(alpha),ROOT.TMath.Cos(alpha),0.])
	    node = sGeo.InitTrack(start, sdir)
	    nxt = sGeo.FindNextBoundary()
	    if ShipGeo.tankDesign < 5 and nxt.GetName().find('I')<0: return 0    
	    distance = sGeo.GetStep()
	    if distance < minDistance  : minDistance = distance
	  return minDistance

	def isInFiducial(X,Y,Z):
				# print('open isInFiducial')
				# print(X,Y,Z)
				if Z > ShipGeo.TrackStation1.z : return False
				# print(X,Y,Z)
				if Z < ShipGeo.vetoStation.z+100.*u.cm : return False
				# print(X,Y,Z, dist2InnerWall(X,Y,Z))
				# typical x,y Vx resolution for exclusive HNL decays 0.3cm,0.15cm (gaussian width)
				if dist2InnerWall(X,Y,Z)<5*u.cm: return False
				# print(X,Y,Z)
				return True 
	#

	def myVertex(t1,t2,PosDir):
		# closest distance between two tracks
					# d = |pq . u x v|/|u x v|
				# print(' ')
				# print('myvertex',PosDir)
				# print(np.shape(PosDir))
				# print(PosDir[0])
				# print(PosDir[0][0])
				a = ROOT.TVector3(PosDir[0][0](0) ,PosDir[0][0](1), PosDir[0][0](2))
				u = ROOT.TVector3(PosDir[0][1](0),PosDir[0][1](1),PosDir[0][1](2))
				c = ROOT.TVector3(PosDir[1][0](0) ,PosDir[1][0](1), PosDir[1][0](2))
				v = ROOT.TVector3(PosDir[1][1](0),PosDir[1][1](1),PosDir[1][1](2))
				pq = a-c
				uCrossv = u.Cross(v)
				dist  = pq.Dot(uCrossv)/(uCrossv.Mag()+1E-8)
				# u.a - u.c + s*|u|**2 - u.v*t    = 0
				# v.a - v.c + s*v.u    - t*|v|**2 = 0
				E = u.Dot(a) - u.Dot(c) 
				F = v.Dot(a) - v.Dot(c) 
				A,B = u.Mag2(), -u.Dot(v) 
				C,D = u.Dot(v), -v.Mag2()
				t = -(C*E-A*F)/(B*C-A*D)
				X = c.x()+v.x()*t
				Y = c.y()+v.y()*t
				Z = c.z()+v.z()*t
				return X,Y,Z,abs(dist)

	def  RedoVertexing(t1,t2):    
						PosDir = [] 
						for tr in [t1,t2]:
							xx  = tr
							# help(xx)
							# PosDir[tr] = [xx.getPos(),xx.getDir()]
							# print(xx.getPos()[0])
							PosDir.append([xx.getPos(),xx.getDir()])
						# print(PosDir)
						# PosDir[0] = [t1.getPos(),t1.getDir()]
						# PosDir[1] = [t2.getPos(),t2.getDir()]
						# print('here',PosDir)
						xv,yv,zv,doca = myVertex(t1,t2,PosDir)
	# as we have learned, need iterative procedure
						dz = 99999.
						reps,states,newPosDir = {},{},{}
						newPosDir = []
						parallelToZ = ROOT.TVector3(0., 0., 1.)
						rc = True 
						step = 0
						while dz > 0.1:
							zBefore = zv
							newPos = ROOT.TVector3(xv,yv,zv)
						# make a new rep for track 1,2
							for tr in [t1,t2]:     
								xx = tr
								reps[tr]   = ROOT.genfit.RKTrackRep(xx.getPDG())
								states[tr] = ROOT.genfit.StateOnPlane(reps[tr])
								reps[tr].setPosMom(states[tr],xx.getPos(),xx.getMom())
								try:
									reps[tr].extrapolateToPoint(states[tr], newPos, False)
								except:
									# print 'SHiPAna: extrapolation did not work'
									rc = False  
									break
								# help(reps[tr].getPos(states[tr]))
								# print(reps[tr].getPos(states[tr]))
								# print(reps[tr].getPos(states[tr])[0])
								# print(float(reps[tr].getPos(states[tr])[0]))

								# newPosDir[tr] = [reps[tr].getPos(states[tr]),reps[tr].getDir(states[tr])]

								# newPosDir.append([float(reps[tr].getPos(states[tr])[0]),float(reps[tr].getPos(states[tr])[1]),float(reps[tr].getPos(states[tr])[2]),float(reps[tr].getDir(states[tr])[0]),float(reps[tr].getDir(states[tr])[1]),float(reps[tr].getDir(states[tr])[2])])
								newPosDir.append([reps[tr].getPos(states[tr]),reps[tr].getDir(states[tr])])
								# print('newposdir',newPosDir)
							if not rc: break
							xv,yv,zv,doca = myVertex(t1,t2,newPosDir)
							dz = abs(zBefore-zv)
							step+=1
							if step > 10:  
										# print 'abort iteration, too many steps, pos=',xv,yv,zv,' doca=',doca,'z before and dz',zBefore,dz
										rc = False
										break 
						if not rc: return xv,yv,zv,doca # extrapolation failed, makes no sense to continue
				
						return xv,yv,zv,doca

	def ImpactParameter2(point,tPos,tMom):
			t = 0
			# if hasattr(tMom,'P'): P = tMom.P()
			# else:                 P = tMom.Mag()
			P = math.sqrt(tMom[0]**2 + tMom[1]**2 + tMom[2]**2)
			for i in range(3):   t += tMom[i]/P*(point(i)-tPos[i]) 
			dist = 0
			for i in range(3):   dist += (point[i]-tPos[i]-t*tMom[i]/P)**2
			dist = ROOT.TMath.Sqrt(dist)
			return dist

	##############################

	# Calculate prod_branching_ratio using Kostas' rpvsusy.py script
	b = rpvsusy.RPVSUSY(theMass,[theCouplings[0], theCouplings[1]],1e3,benchmark, leptongeneration,visible,bzerobplus)

	if leptongeneration == 1:
		if benchmark == 1: meson_decay = 'D+ -> N e+'
		if benchmark == 2: meson_decay = 'D_s+ -> N e+'
		if benchmark == 3: meson_decay = 'B0 -> N nu_e'
		if benchmark == 4: meson_decay = 'B0 -> N nu_e'
		prod_branching_ratio = b.findProdBranchingRatio(meson_decay)

		br_to_vis = 0
		
		if benchmark == 4:
			br_to_vis += b.findDecayBranchingRatio('N -> D+ e-')
			br_to_vis += b.findDecayBranchingRatio('N -> D*+ e-')
		else:
			br_to_vis += b.findDecayBranchingRatio('N -> K+ e-')
			br_to_vis += b.findDecayBranchingRatio('N -> K*+ e-')

	elif leptongeneration == 2:
		if benchmark == 1: meson_decay = 'D+ -> N mu+'
		if benchmark == 2: meson_decay = 'D_s+ -> N mu+'
		if benchmark == 3: meson_decay = 'B0 -> N nu_mu'
		if benchmark == 4: meson_decay = 'B0 -> N nu_mu'
		prod_branching_ratio = b.findProdBranchingRatio(meson_decay)

		br_to_vis = 0

		if benchmark == 4:
			br_to_vis += b.findDecayBranchingRatio('N -> D+ mu-')
			br_to_vis += b.findDecayBranchingRatio('N -> D*+ mu-')
		else:
			br_to_vis += b.findDecayBranchingRatio('N -> K+ mu-')
			br_to_vis += b.findDecayBranchingRatio('N -> K*+ mu-')

	if leptongeneration == 3:
		if benchmark == 5: 
			meson_decay = 'B0 -> N nu_tau'

			meson_decay = 'B+ -> N tau+'


	# 521 is B+
	# 511 is B0


	# Open rec root file containing output of simulation.
	f = ROOT.TFile.Open('ship.conical.Pythia8-TGeant4_rec.root')

	tree = f.Get("cbmsim")

	N = tree.GetEntries()

	# Initialise array to contain list of weights of signal events. 
	weights_all = np.empty(0)
	weights_vis = np.empty(0)
	weights_no_cuts = np.empty(0)
	weights_bench5_b0 = np.empty((0,2))
	weights_bench5_bplus = np.empty((0,2))
	weights_bench5_b0_no_cuts = np.empty((0,2))
	weights_bench5_bplus_no_cuts = np.empty((0,2))
	weights_bench5 = np.empty((0,2))
	i = -1

	if benchmark == 5:
		# num_mesons = np.load('num_mesons.npy') # old method - before separate simulations were run for b0 and bplus

		# if bzerobplus == 0
		NUM_B0 = 0
		NUM_BPLUS = 0


	print('starting event loop')

	total_weight = 0

	for event in tree:

		if bzerobplus == 0 and benchmark == 5:
			NUM_B0 += 1
		if bzerobplus == 1 and benchmark == 5:
			NUM_BPLUS += 1

		i += 1
		j = 0

		list_of_fitted_states = []

		list_pdg_and_mothers = np.empty((0,2))

		for iter_i, e in enumerate(event.MCTrack):
			list_pdg_and_mothers = np.append(list_pdg_and_mothers, [[e.GetPdgCode(),e.GetMotherId()]],axis=0)
			if iter_i == 1:
				event_weight = e.GetWeight()
		total_weight += event_weight

		for e in event.FitTracks:

			fittedState = e.getFittedState()
			list_of_fitted_states = np.append(list_of_fitted_states, fittedState)
			fitStatus   = e.getFitStatus()
			nmeas = fitStatus.getNdf()
			chi2 = fitStatus.getChi2()
			rchi2 = chi2/nmeas

			P = fittedState.getMomMag()

			# Only record track if pass initial cuts. 
			if rchi2 > 0 and rchi2 < 25 and nmeas > 0 and P > 1:
				j+=1

		print(i,j)

		# If more than 1 fitted tracks in an event... 
		if j > 1:

			# Calculate momentum of RPVn by combining momentum of all tracks...
			momentum_vectors = np.empty((0, 3))
			for track_i in range(0, j):
				momentum_vectors = np.append(momentum_vectors, [[list_of_fitted_states[track_i].getMom()[0],list_of_fitted_states[track_i].getMom()[1],list_of_fitted_states[track_i].getMom()[2]]], axis = 0)
			RPVnMom = np.zeros(3)
			for track_i in range(0, j):
				RPVnMom[0] += momentum_vectors[track_i, 0]
				RPVnMom[1] += momentum_vectors[track_i, 1]
				RPVnMom[2] += momentum_vectors[track_i, 2]

			# Calculate position of decay vertex and the doca of tracks.
			# Do this for all combinations of tracks and pick the reconstruction with lowest DOCA 
			xv_list = yv_list = zv_list = doca_list = np.empty(0)
			for track_i in range(0, j):
				for track_j in range(track_i+1, j):
					xv_c,yv_c,zv_c,doca_c = RedoVertexing(list_of_fitted_states[track_i],list_of_fitted_states[track_j])

					xv_list = np.append(xv_list, xv_c)
					yv_list = np.append(yv_list, yv_c)
					zv_list = np.append(zv_list, zv_c)
					doca_list = np.append(doca_list, doca_c)
			where = np.where(doca_list==np.amin(doca_list))
			xv = xv_list[where]
			yv = yv_list[where]
			zv = zv_list[where]
			doca = doca_list[where]

			# Is reconstructed vertex in the fiducial volume?
			# print(xv, yv, zv)
			fid = isInFiducial(xv,yv,zv)

			# Does the reconstructed RPVn point back to the target?
			tr = ROOT.TVector3(0,0,ShipGeo.target.z0)
			RPVpos = [xv,yv,zv]
			dist = ImpactParameter2(tr,RPVpos,RPVnMom)

			# Get the weight of the event...
			weight = event.MCTrack[1].GetWeight()

			weights_no_cuts = np.append(weights_no_cuts, weight)

			if benchmark == 5:

				which_meson = abs(list_pdg_and_mothers[np.where(list_pdg_and_mothers[:,1]==0)][:,0])

				if which_meson[0] == 511:#B0

					weights_bench5_b0_no_cuts = np.append(weights_bench5_b0_no_cuts, [[weight, b.findProdBranchingRatio('B0 -> N nu_tau')]], axis=0)
				elif which_meson[0] == 521:#B+
					weights_bench5_bplus_no_cuts = np.append(weights_bench5_bplus_no_cuts, [[weight, b.findProdBranchingRatio('B+ -> N tau+')]], axis=0)




			# If the event passes all cuts, save the weight...

			decay_products = abs(list_pdg_and_mothers[np.where(list_pdg_and_mothers[:,1]==2)][:,0])
			print(decay_products)
			if doca < 1 and fid == True and dist < 250:
				weights_all = np.append(weights_all, weight)

				if benchmark == 5:
					which_meson = abs(list_pdg_and_mothers[np.where(list_pdg_and_mothers[:,1]==0)][:,0])
					decay_products = abs(list_pdg_and_mothers[np.where(list_pdg_and_mothers[:,1]==2)][:,0])

					if which_meson[0] == 511:#B0
						weights_bench5_b0 = np.append(weights_bench5_b0, [[weight, b.findProdBranchingRatio('B0 -> N nu_tau')]], axis=0)
						weights_bench5 = np.append(weights_bench5, [[weight, b.findProdBranchingRatio('B0 -> N nu_tau')]], axis=0) #wrong
					elif which_meson[0] == 521:#B+
						weights_bench5_bplus = np.append(weights_bench5_bplus, [[weight, b.findProdBranchingRatio('B+ -> N tau+')]], axis=0)
						weights_bench5 = np.append(weights_bench5, [[weight, b.findProdBranchingRatio('B+ -> N tau+')]], axis=0) #wrong
					print('passed')
				else:
					decay_products = abs(list_pdg_and_mothers[np.where(list_pdg_and_mothers[:,1]==2)][:,0])

					if benchmark == 2 or benchmark == 1:
						if (321 in decay_products or 323 in decay_products) and 13 in decay_products: 
							weights_vis = np.append(weights_vis, weight)
					print('passed')


	print('out of event loop')



	if benchmark == 5:
		weights_times_BR = weights_bench5[:,0] * weights_bench5[:,1]

		if np.shape(weights_bench5_b0)[0] > 0:
			weights_times_BR_b0 = weights_bench5_b0[:,0] * weights_bench5_b0[:,1]
		else:
			weights_times_BR_b0 = np.zeros(1)
			
		if np.shape(weights_bench5_bplus)[0] > 0:
			weights_times_BR_bplus = weights_bench5_bplus[:,0] * weights_bench5_bplus[:,1]
		else:
			weights_times_BR_bplus = np.zeros(1)


		if np.shape(weights_bench5_b0_no_cuts)[0] > 0:
			weights_times_BR_b0_no_cuts = weights_bench5_b0_no_cuts[:,0] * weights_bench5_b0_no_cuts[:,1]
		else:
			weights_times_BR_b0_no_cuts = np.zeros(1)
			
		if np.shape(weights_bench5_bplus_no_cuts)[0] > 0:
			weights_times_BR_bplus_no_cuts = weights_bench5_bplus_no_cuts[:,0] * weights_bench5_bplus_no_cuts[:,1]
		else:
			weights_times_BR_bplus_no_cuts = np.zeros(1)

		sum_weight_vis = np.sum(weights_vis)
		sum_weight_all = np.sum(weights_times_BR) # this one
		sum_weight_no_cuts = np.sum(weights_no_cuts)


		br_to_vis = 0
		br_to_vis += b.findDecayBranchingRatio('N -> K+ tau-')
		br_to_vis += b.findDecayBranchingRatio('N -> K*+ tau-')

		data = [theMass,theCouplings[0],0,sum_weight_vis,theCouplings[1], sum_weight_all, sum_weight_no_cuts,br_to_vis,np.sum(weights_times_BR_b0),NUM_B0,np.sum(weights_times_BR_bplus),NUM_BPLUS,np.sum(weights_times_BR_b0_no_cuts),np.sum(weights_times_BR_bplus_no_cuts),total_weight] #will want to divide this sum by number of events simulated

		print(NUM_B0, NUM_BPLUS)

	else:
		sum_weight_vis = np.sum(weights_vis)
		sum_weight_all = np.sum(weights_all)
		sum_weight_no_cuts = np.sum(weights_no_cuts)

		data = [theMass,theCouplings[0],prod_branching_ratio,sum_weight_vis,theCouplings[1], sum_weight_all, sum_weight_no_cuts,br_to_vis,0,0,0,0,0,0,total_weight] #will want to divide this sum by number of events simulated


	print('blurgh',data)


	print('np sum weights only',np.sum(weights_bench5[:,0]))

	print(' ')
	print('Neutralino Mass GeV:',theMass)
	print('Production Couplings:',theCouplings[0])
	print('Decay Couplings:',theCouplings[1])
	print('sFermion mass (degenerate):',theCouplings[2])
	print('Production BR','NOT SUPPLIED FOR BENCH5')
	print('Sum of weights visible:',sum_weight_vis)
	print('Sum of weights all - is really sum(weights*prod_BR):',sum_weight_all)
	print('Sum of weights no cuts:',sum_weight_no_cuts)
	print('Sum of weights - total_weight:',total_weight)
	print(int(N),'events simulated')
	print(' ')

except:
	# If script fails - point not kinematically possible etc.. 
	# Save sum_weight as 0
	print('Something went wrong...')
	data = [theMass, theCouplings[0], 0, 0, theCouplings[1],0,0,0,0,0,0,0,0,0,0]
	print(data)



# Need to organise output nicely...
print(benchmark)
print(visible)
print(plot)
print(plot1massmax*1000)
print(jobid)
print(bzerobplus)

if visible == True:
	visible = 1
else:
	visible = 0


np.save('grid_data_rpv_%d_%d_%d_%d_%d_%d'%((benchmark),(visible),(plot),((plot1massmax*1000)),(jobid),(bzerobplus)),data)







