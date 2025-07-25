# PyAFSpectra

## Introduction
PyAFSpectra is an application for the analysis of single molecule force spectrosopy data from Acoustic force spectrosopy.

The code for the following analysis are were written:
- On the ramp (OTR) force calibration - Calibate an individual bead during a force ramp as close to as bond rupture.
- OTR global force calibration  - 


This code was developed as a part of the publication:
 
LINK TO THE PAPER

If you have any suggestions, comments or experience any issues. 
Please write to us @ yogesh.saravanan@inserm.fr ou claire.valotteau@inserm.fr ou felix.rico@inserm.fr


## Basic usage
### To run from source
- Clone the repository
```
git clone  https://github.com/DyNaMo-INSERM/PyAFSpectra.git
cd ./PyAFSpectra
```
- Create an environment with python 3.9
```
conda create -n yourenvname python=3.9 
conda activate yourenvname
```

- Install the dependencies from requirements.txt
```
pip install -r requirements.txt
```

- run main.py
```
python main.py
```

### Navigating the GUI 
This GUI has keyboard functionality, to have a more user friendly approch to analyse the data.
- Define the rupture time of the trace by manually dragging T_rup and double clicking to record it.
- Dragable OTR fitting window across the trace to find the right window to fit the PSD.(ideally close to T_rup)
- Press S to save the results locally in the computer at any point of analysis.
- Press D to go to the next bead/trace.( Automatically save the current T_rup and the OTR window position to the results dataframe)
- Press A to go back to the previous trace. 
- Press Space to skip the analysis of the current trace.


## Acknowledgements

The authors would like to acknowledge all the funding sources that supported this work. 
This project received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement number 772257 to FR) and the Marie Curie Sklodowska action (MSCA-IF) (grant agreement no. 895819 to CV), from ITMO Cancer of Aviesan on funds Cancer 2021 (ATIP Avenir to CV) and from the Agence Nationale de la Recherche (ANR AAPG2023 - PRC – XXL to MM and FR). The AFS setup was acquired thanks to the grant Projet exploratoire region PACA 2017 – AcouLeuco cofunded by the ANR (ANR-15-CE11-0007-01 to FR). We also acknowledge Inserm, CNRS, and amU for regular support.


