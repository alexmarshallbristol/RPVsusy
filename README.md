# Sensitivity of SHiP to R-Parity Violating Supersymmetry

Sensitivity study of the SHiP experiment (CERN) to R-Parity Violating Supersymmetry. 

Five benchmark scenarios are analysed based on those in this [paper](https://arxiv.org/abs/1511.07436). These benchmarks have become the standard with which to compare similar experiments, see this [paper](https://arxiv.org/abs/1810.03617) also.

This analysis relies on the FairShiP software, avaliable [here](https://github.com/ShipSoft/FairShip). This repository contains the modified code for this purpose. 

## Getting Started

Download the project code
```
git clone https://github.com/alexmarshallbristol/RPVsusy
```
source SHiP setup script from CVMFS (accessible on LXPLUS)
```
source /cvmfs/ship.cern.ch/SHiP-2018/latest/setUp.sh
```
build the software using alibuild
```
aliBuild build FairShip --default fairship --always-prefer-system --config-dir $SHIPDIST
```

Then do once, in the install directory
```
alienv printenv FairShip/latest >> config.sh
chmod u+x config.sh
```

For running quick test jobs, load the enviroment with
```
source /cvmfs/ship.cern.ch/SHiP-2018/latest/setUp.sh
alienv enter FairShip/latest
```

For batch scripts the process is different. Load the environment with
```
#!/bin/bash
source /cvmfs/ship.cern.ch/SHiP-2018/latest/setUp.sh
SHIPBUILD_mymaster=/afs/cern.ch/user/a/amarshal/RPV_new
export ALIBUILD_WORK_DIR=$SHIPBUILD_mymaster/sw #for alienv
source $SHIPBUILD_mymaster/config.sh
```
the submit script may or may not need the line
```
getenv = true
```


## Changes from master FairShip

### Addition of the file [RPV_get_rate.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/macro/RPV_get_rate.py).

This file runs through reconstructed events, checks which events pass background selection cuts. It is a tailored version of [ShipAna.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/macro/ShipAna.py).

### Editing of [RPV_get_rate.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/macro/run_simScript.py).

Options are added which are specific to the RPV susy analysis. A loop is added to define values for mass and couplings based on jobid values from condor.

### Correction of and additions to [rpvsusy.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/python/rpvsusy.py)

Changed some constants, corrected some indexing and added if statements which provide more control for lepton generation and visible only/all states runs.

### Correction of and additions to [pythia8_conf.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/python/pythia8_conf.py)

Added if statements which provide more control for lepton generation and visible only/all states runs.

### Correction and addition of .dat files.

Files are synthesised by iterating over [rpvsusy.py](https://github.com/alexmarshallbristol/RPVsusy/blob/master/FairShip/python/rpvsusy.py). New electron files are synthesised. 



## Steps to run a standard job







