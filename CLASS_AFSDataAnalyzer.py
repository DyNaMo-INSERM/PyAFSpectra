#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:01:29 2024

@author: yogehs
"""

import matplotlib.pyplot as plt

from decimal import Decimal
from scipy import signal

from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d

from scipy.stats import norm
import numpy as np
import pandas as pd
import datetime
import pygame as pyg
import chime 

import pyqtgraph as pg
pg.setConfigOption('background', '#f0f0f0')
pg.setConfigOption('foreground', 'black')
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton


from pyqtgraph.Qt import QtWidgets, QtCore
import os
import warnings
warnings.filterwarnings("ignore")

from nptdms import TdmsFile

from collections import defaultdict
 
from utils import *
from pprint import pprint

#some important parameters
bool_out_xyz = False;    CleanOutlier_med =False ;

SaveGraph =True;CloseAfterSave = True;
save_alpha_fit = True

first_z  =True ; saveZ = False ;power_spec_bool = True
out_Z =False


# To map the keyboard keys
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)
class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)
    
def DENn(f,fc,fs,n): return 1 + (np.abs(f+n*fs)/fc)**2  
def SINCn(f,te,fs,n): return np.sinc(te*np.abs(f+n*fs))**2 
 

class AFSDataAnalyzer():
    """ 
    Troubleshoot AFS data.
    
    Takes index and log as inputs and has methods to prepare and analyze traces.
    """
    def __init__(self, log_afs, save_path, start_index=0, N_psd_build=1200, gui_bool=True, anal_label=None):
        """
        Initialize the AFSDataAnalyzer class.

        Parameters:
        log_afs (DataFrame): Log of AFS data.
        save_path (str): Path to save the analysis results.
        start_index (int): Starting index for analysis.
        N_psd_build (int): Number of PSD builds.
        gui_bool (bool): Flag to enable/disable GUI.
        anal_label (str): Label for analysis.
        """
        super().__init__()
        self.joysticks = {}

        self.app = pg.mkQApp("PyAFSpectra") 
        self.mw = QtWidgets.QMainWindow()
        self.log_trouble_sess = pd.DataFrame(columns=log_afs.columns)
        self.save_path = save_path
        self.save_folder = os.path.dirname(save_path)

        self.gui_bool = gui_bool
        #anal params
    
        self.i_file = start_index
        self.log_afs_df = log_afs
        self.log_afs_df['B_ref'] = self.log_afs_df['B_ref'].astype("string")
        self.log_afs_df['radius_nm'] = self.log_afs_df['radius_nm'].astype("string")

        self.num_of_beads = len(log_afs)
        
        self.N_psd_build= N_psd_build
        
        self.n_sec_otf = 0
        self.otf_end_T_thresh =2
    
        self.ht_correction_factor = 1.33
        #importatnt parameters for the D_avg_ global method of calibrating the forces OTF.
        self.d_dt_min = 0.8 ;self.d_dt_max = 1.2
        self.fmin_Hz=0.1;self.fmax_Hz = 0  # frequency range for spectrum fit (Hz)
        self.save_anal_label = None
        if anal_label is not None:
            self.save_anal_label = f"{self.save_folder}/{anal_label}"
            os.makedirs(self.save_anal_label,exist_ok=True)

        
    def init_trace_params(self):
        """
        Initialize trace parameters.
        """
        #trace params
        self.file_label = 'test'
        self.kBT_pN_nm = 0
        self.fs = 0
        self.te = 0
        self.f_c_threshold_global = 0
        self.f_c_avg_otr = 0
        self.xx = [];self.yy = []
        self.zz = [];self.T_s = []
        self.L_t = [];self.P0 = []
        self.non_zero_pos = 0
        self.ramp_loc = []
        
        self.rupture_pos=0;self.rupture_time =0
        self.P_rup = 0
        self.load_time = 0
        self.rup_ramp_loc =np.array([0,0])
        self.symmetry_factor = 0
        self.rms_X_Y = 0
        #note these are index of the OTR window
        self.OTR_B = 0;self.OTR_A = 0
        self.bool_D_G = False
        
        self.D_global_arr = [False,0.0,0.0]
        self.D_G_found = self.D_global_arr[0]
        self.D_X_array=[];self.D_Y_array=[]
        
        #final rupture force dfs data 
        self.L_t_rup = 0

        self.LDR = 0
        self.rup_force = 0
        self.calib_force_arr =  np.nan*np.ones(2)
        self.calib_K_arr =  np.nan*np.ones(2)

        self.calib_D_arr = np.nan*np.ones(2)
        self.D_t = 0
        self.pow_rate_rup_ramp = 0
        self.avg_pow_OTR = 0
             
    def launch_gui(self):
        """
        Launch the GUI for the application.
        """
        self.create_app()
        self.init_bead()
    def init_bead(self):
        """
        Initialize bead parameters and load bead data.
        """
        self.trace_plt.setYRange(-400,400)
        self.trace_plt.clear();self.trace_plt.addItem(self.rup_pos)
        self.trace_plt.addItem(self.otr_win)
        if 'bool_troubleshoot' in self.log_afs_df:
            self.load_bead_post_gui()
        else:
            self.load_bead()
        [tx,ty] = self.rup_ramp_loc

        self.trace_plt.setYRange(-400,400)
        self.trace_plt.enableAutoRange(axis='x')
        self.trace_plt.setAutoVisible(x=True)
        self.file_label_btn.setText(self.file_label)
        self.trace_plt.plot(self.T_s[:],self.L_t[:])
        self.rup_pos.setValue(self.rupture_time)
        self.otr_win.setRegion([self.T_s[self.OTR_A],self.T_s[self.OTR_B]])

    def load_bead(self):
        """
        Load bead data and prepare trace.
        """
        self.init_trace_params()
        self.is_bass_trace = False

        print("load the data")
        
        self.ori_path = self.log_afs_df['File path'][self.i_file]
        self.tdms_file_name = self.log_afs_df['TDMS file name '][self.i_file]

        
        self.file_label = str(self.i_file)+"_"+self.tdms_file_name
        #print(self.log_afs_df['radius_nm'][self.i_file])
        if self.log_afs_df['radius_nm'][self.i_file].isdigit():
            self.bead_radius = int(self.log_afs_df['radius_nm'][self.i_file])
        else:
            #this was the default radius in our case 
            self.bead_radius = 3101/2.0
        self.prep_trace()

        self.rupture_pos = len(self.xx)-1
        self.OTR_A= 0;self.OTR_B= self.N_psd_build

    def load_bead_post_gui(self):
        """
        Load bead data after GUI initialization.
        """
        self.init_trace_params()
        self.is_bass_trace = False

        print("load the data")
        
        self.ori_path = self.log_afs_df.iloc[self.i_file]['File path']
        self.tdms_file_name = self.log_afs_df.iloc[self.i_file]['TDMS file name ']

        self.rupture_pos = int(self.log_afs_df.iloc[self.i_file]['rupture_pos_pt_idx'])
        self.rupture_time = float(self.log_afs_df.iloc[self.i_file]['rupture_pos_s'])
        
        self.fs = float(self.log_afs_df.iloc[self.i_file]['frame_rate_hz'])
        self.fmax_Hz=self.fs ;self.f_c_threshold_global = self.fs/6 ;self.n_sec_otf = self.N_psd_build/self.fs 
        #self.fmin_Hz = self.fs/self.N_psd_build

        
        self.te = float(self.log_afs_df.iloc[self.i_file]['exposure_time_s'])
        
        
        self.file_label = str(self.i_file)+"_"+self.tdms_file_name
        #print(self.log_afs_df['radius_nm'][self.i_file])
        if pd.isna(self.log_afs_df.iloc[self.i_file]['radius_nm']):
            self.bead_radius = 3101/2.0
        else:
            self.bead_radius = float(self.log_afs_df.iloc[self.i_file]['radius_nm'])
            
        bool_troubleshoot = bool(self.log_afs_df.iloc[self.i_file]['bool_troubleshoot'])
        if bool_troubleshoot:
            
            self.kBT_pN_nm= float(self.log_afs_df.iloc[self.i_file]['k_Bt_nm'])
            
            #otr index
            self.OTR_A= int(self.log_afs_df.iloc[self.i_file]['OTR_A_pt_idx'])
            self.OTR_B= int(self.log_afs_df.iloc[self.i_file]['OTR_B_pt_idx'])
            print(self.OTR_A)
            #self.bool_D_G =bool(self.log_afs_df.iloc[self.i_file]['bool_D_G'])
            #self.D_G_found = bool(self.log_afs_df.iloc[self.i_file]['D_G_found'])
            if self.bool_D_G and self.D_G_found :
                dg_x = float(self.log_afs_df.iloc[self.i_file]['DG_X'])
                dg_y =float(self.log_afs_df.iloc[self.i_file]['DG_Y'])
                self.D_global_arr = [True,dg_x,dg_y]

            self.prep_trace_post_gui()

    def prep_trace_post_gui(self):
        """
        Prepare trace data after GUI initialization.
        """
        CleanOutlier_med =False 
        did_it_unbind =  ~np.isnan(self.rupture_pos)

        b, bref = bead_extract_from_df(self.log_afs_df,self.i_file)
        self.bref = bref
        self.file_label+= "_"+ str(b[0])
        
        
    
    
        fname = self.tdms_file_name+".tdms"
        
        
    
        #In this code block i shall try to obtain the step values for each data file
        #the fs and the exposure time are also read from the meta data 
        
        tdms_file = TdmsFile.open(f"{self.ori_path}/{self.tdms_file_name}.tdms") 
        tdms_groups = tdms_file.groups()   
        tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
        (TE0,NE0,DE0) = events(tdms_event)
        (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[0]) # test bead
        
      
        T0, self.fs = timet(tdms_track, False)
        self.fmax_Hz=self.fs ;self.f_c_threshold_global = self.fs/6 ;self.n_sec_otf = self.N_psd_build/self.fs 
        #self.fmin_Hz = self.fs/self.N_psd_build
        
        
        self.te = find_te_tdms_index(fname,self.ori_path)
    
    
        #extracting the step and ramps values for this TDMS file.
        step_pos = str(self.log_afs_df.iloc[self.i_file]['Step_position'])
        
        step_pos = np.array([int(e) if e.isdigit() else e for e in step_pos.split('_')])
        
        P0 = BuildP(T0, TE0, NE0, DE0)
        value, location, lengths = island_props(P0)
    
    
    
        step_values = [] ;self.step_loc = []
        for i in step_pos:
            step_values.append(value[i])
            self.step_loc += [(location[i],location[i+1]-1)]

        self.ramp_loc = ramp_extraction(P0,T_s)
    
    

 
        #maybe will be updated over time and functionality. 
        pprint = True
        RangeNZlavg=[400]       # size of moving average for smoothing z before anchor point determination
        NZrefavg=5000
        mode='reflect' #'nearest'  # mode for rolling average   # edge modes = ['full', 'same', 'valid', 'nearst']
        # see details on https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        AnchorPointStateList=['custom'] # correction with AnchorPoint as determined by anchor point
    
    
        plot_fig_name = fname[-11:-5] #+"_R_"+str(self.bead_radius) + "_bead_"+str(b[0]) 
        os.chdir(self.ori_path)

        print('Loading data', fname)
       
        print('======================BEAD ', b[0] ,' ===========================')
        tdms_groups = tdms_file.groups()        
        tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
        (TE0,NE0,DE0) = events(tdms_event)
        P0 = BuildP(T0, TE0, NE0, DE0)


        dmax=1000000.; imax=(len(tdms_track)-1)//5; dX=1500; dY=1500    
        #print('Nb of traj=', imax)
        refname=fname[:-5]+'_Bead'+str(b[0])+'_'; print(refname)
        (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[0]) # test bead



        zero_force_pos = zero_force_pos_func(P0)
        non_0_force = np.argwhere(P0).squeeze()
        
        #remove the nan values 
        i0 = np.isnan(X0) + np.isnan(Y0) + np.isnan(Z0) 
        mX0 = np.median(X0[~i0]);  X0[i0] = mX0
        mY0 = np.median(Y0[~i0]);  Y0[i0] = mY0
        mZ0 = np.median(Z0[~i0]);  Z0[i0] = mZ0
        #remove and substitute the median values 
        if CleanOutlier_med:
            mm=0.2; print('Clean outliers nan or +/-', mm)
            if pprint: print('median X0=', mX0, 'median Y0=', mY0, 'median Z0=', mZ0 )
            j0 = (X0>(1+mm)*mX0) + (X0<(1-mm)*mX0) + (Y0>(1+mm)*mY0) + (Y0<(1-mm)*mY0) + (Z0>(1+mm)*mZ0) + (Z0<(1-mm)*mZ0)
            #print(len(j0))
            X0[j0] = mX0; Y0[j0] = mY0; Z0[j0] = mZ0; 
        #    m0 = np.median(X0[~i0]);  X0[i0] = m0*(1+.01*np.random.rand(len(X0[i0]))); X0[ (X0>(1+mm)*m0) | (X0<(1-mm)*m0) ] = m0
            m0 = np.median(MinLUT0[~i0]);  MinLUT0[i0] = m0 ; MinLUT0[ (MinLUT0>(1+mm)*m0) | (MinLUT0<(1-mm)*m0) ] = m0
            m0 = np.median(SSIM0[~i0]);  SSIM0[i0] = m0 ; SSIM0[ (SSIM0>(1+mm)*m0) | (SSIM0<(1-mm)*m0) ] = m0

        Z0[:non_0_force[0]],count = qt_anchor_Z(Z0[:non_0_force[0]],0.02)       
        #this is to replace all values of the Z_bf dist using the 2endth quantile

        X_bf = X0[:non_0_force[0]] ; T_bf = T_s[:non_0_force[0]] ; Y_bf = Y0[:non_0_force[0]]
        Z_bf = Z0[:non_0_force[0]] 


        #step 0: trimming X-Y traces of outliers
        if bool_out_xyz:
                
            temp_out_Xpos = (X0 > (X0.mean() - 3*X0.std())) * (X0 < (X0.mean() + 3*X0.std()))
            temp_out_Ypos = (Y0 > (Y0.mean() - 3*Y0.std())) * (Y0 < (Y0.mean() + 3*Y0.std()))
            temp_out_Zpos = (Z_bf > (Z_bf.mean() - 3*Z_bf.std())) * (Z_bf < (Z_bf.mean() + 3*Z_bf.std()))
    
            for i in range(len(Y0)):
                if ~temp_out_Ypos[i]:
                    Y0[i] = np.median(Y0)
    
            for i in range(len(X0)):
                if ~temp_out_Xpos[i]:
                    X0[i] = np.median(X0)


        #Z_bf  = np.array([x for x in Z_bf if (x > Z_bf.mean() - 2 * Z_bf.std())])

        #step 1 determine anchor point before force, mean of X,Y & Z traces

        #x_anchor_pt = X_bf.mean() ;y_anchor_pt = Y_bf.mean()
        #z_anchor_pt = Z_bf.min()


        #step 2 : anchor pt correction to X,Y , Z

        #X0 -= x_anchor_pt ; Y0 -= y_anchor_pt; Z0 -= z_anchor_pt



        #step 3 : average the traje out for reference beads and cleaning the outliers 

        print("Calculating pool of reference bead(s)", bref )
        if len(bref)>1:

            (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])
            for i in range(1,len(bref)):

                (Xr0_,Yr0_,Zr0_,MinLUTr0_,SSIMr0_,Temp_t)=track(tdms_track,bref[i])
                Xr0=Xr0+Xr0_; Yr0=Yr0+Yr0_; Zr0=Zr0+Zr0_
            Xr0=Xr0/len(bref); Yr0=Yr0/len(bref); Zr0=Zr0/len(bref)
            #Zr0 = Zr0[temp_out_Z_pos]
        elif len(bref)==1:
            (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])   # reference bead

        i0 = np.isnan(Xr0) + np.isnan(Yr0) + np.isnan(Zr0) 
        mXr0 = np.median(Xr0[~i0]);  Xr0[i0] = mXr0
        mYr0 = np.median(Yr0[~i0]);  Yr0[i0] = mYr0
        mZr0 = np.median(Zr0[~i0]);  Zr0[i0] = mZr0
    
        if CleanOutlier_med:
            mm=0.5; print('Clean outliers nan or +/-', mm)
            if pprint: print('median X0=', mXr0, 'median Y0=', mYr0, 'median Z0=', mZr0 )
            j0 = (Xr0>(1+mm)*mXr0) + (Xr0<(1-mm)*mXr0) + (Yr0>(1+mm)*mYr0) + (Yr0<(1-mm)*mYr0) + (Zr0>(1+mm)*mZr0) + (Zr0<(1-mm)*mZr0)
            print(len(j0))
            Xr0[j0] = mXr0; Yr0[j0] = mYr0; Zr0[j0] = mZr0; 
        #    m0 = np.median(X0[~i0]);  X0[i0] = m0*(1+.01*np.random.rand(len(X0[i0]))); X0[ (X0>(1+mm)*m0) | (X0<(1-mm)*m0) ] = m0
            m0 = np.median(MinLUT0[~i0]);  MinLUT0[i0] = m0 ; MinLUT0[ (MinLUT0>(1+mm)*m0) | (MinLUT0<(1-mm)*m0) ] = m0
            m0 = np.median(SSIM0[~i0]);  SSIM0[i0] = m0 ; SSIM0[ (SSIM0>(1+mm)*m0) | (SSIM0<(1-mm)*m0) ] = m0

        #step 4: substract out the anchor point(2) from the reference beads


        #Xr0 -= Xr0[:non_0_force[0]].mean(); Yr0 -= Yr0[:non_0_force[0]].mean()
        #Zr0 -= Zr0[:non_0_force[0]].min()

        #step 5: Rolling average for the reference. 

        Xr0,Xr0avg= Rollavg(Xr0, NZrefavg, mode); Yr0,Yr0avg=Rollavg(Yr0, NZrefavg, mode) 
        Zr0, Zr0avg = Rollavg(Zr0, NZrefavg, mode) 

        #step 6: drift correction :Subt the rolling avg refrence from actual trace
        xx = X0 - Xr0avg + Xr0avg[0];yy = Y0- Yr0avg + Yr0avg[0]
        Z01 = Z0 - Zr0avg + Zr0avg[0] 


        #step 7: trimming the Z traces. 

        if out_Z:
            temp_out_Zpos = (Z_bf > (Z_bf.mean() - 3*Z_bf.std())) * (Z_bf < (Z_bf.mean() + 3*Z_bf.std()))

            Z01_med = np.median(Z01);Z01_m = np.mean(Z01) ;Z01_s = np.std(Z01)
            temp_lout_Zpos= (Z_bf > (Z_bf.mean() -  3* Z_bf.std()))
            temp_hout_Zpos = (Z01 > (Z01_m )) * (Z01 < (Z01_m + 3*Z01_s))

            for i in range(len(Z_bf)):
                if ~temp_lout_Zpos[i] :
                    Z01[i] = Z01_med
            for i in range(len(Z01)):
                if ~temp_hout_Zpos[i]:
                    Z01[i] = Z01_med

        #step 8:final anchor pont sub for Z
        z_anchor_pt = Z01[:non_0_force[0]].min()
        zz = Z01-Z01[:non_0_force[0]].min()
     
        #step 9 : determining the rupture force through the Z trace
        #need not do it as this file as it already.
        
        #Compute the lenghts distribution for the whole range now 
        anchor_pt_arr = [(np.mean(xx[:non_0_force[0]]),np.mean(yy[:non_0_force[0]]))]
        for a in anchor_pt_arr :
            xx -= a[0];yy-=a[1]
            if did_it_unbind : 
                non_0_force = pre_rup_func(non_0_force,self.rupture_pos-int(self.fs*self.otf_end_T_thresh))
                zero_force_pos = pre_rup_func(zero_force_pos,self.rupture_pos)
                self.non_zero_pos = non_0_force[0]
                if self.rupture_pos == len(zz)-1:
                    zz = zz - zz[zero_force_pos].min()
                else:
                    zz = zz - zz[zero_force_pos].min()
                    
            #accounting for the height correction  
            zz = zz*self.ht_correction_factor

                
            #print(zero_force_pos)
            sigma_x = np.quantile(X_bf- np.mean(X_bf),0.975) - np.quantile(X_bf- np.mean(X_bf),0.025)
            sigma_y = np.quantile(Y_bf- np.mean(Y_bf),0.975) - np.quantile(Y_bf- np.mean(Y_bf),0.025)

            L_t_planar = np.sqrt((xx[zero_force_pos])**2 + (yy[zero_force_pos] )**2)
            self.rms_X_Y = np.sqrt(np.average((xx[:self.non_zero_pos])**2 + (yy[:self.non_zero_pos] )**2))
            max_L_t_plane = np.quantile(L_t_planar,0.95)

            L_t_bforce = (np.sqrt((xx[zero_force_pos] )**2 + (yy[zero_force_pos] )**2+
                                 (zz[zero_force_pos] + self.bead_radius )**2 ) - self.bead_radius)

            L_t_aforce = (np.sqrt((xx[non_0_force])**2 + (yy[non_0_force])**2+
                                 (zz[non_0_force] + self.bead_radius )**2 ) - self.bead_radius)
            L_t = (np.sqrt((xx )**2 + (yy )**2+
                                 (zz + self.bead_radius )**2 ) - self.bead_radius)
            
            
            
            L_T_bforce_avg = L_t_bforce.mean();L_T_aforce_avg = L_t_aforce.mean()
            AnchorCov=np.cov(xx[:non_0_force[0]],yy[:non_0_force[0]])
            w, v = np.linalg.eig(AnchorCov); self.symmetry_factor = np.sqrt(np.amax(w)/np.amin(w))

            
            max_L_t_bforce = np.quantile(L_t_bforce,0.95);max_L_t_aforce = np.quantile(L_t_aforce,0.95)
            min_L_t_bforce_qt = np.quantile(L_t_bforce,0.05);min_L_t_aforce_qt = np.quantile(L_t_aforce,0.05)
            min_L_t_bforce = L_t_bforce.min();min_L_t_aforce =L_t_aforce.min()
            min_L_t_aforce_rup_ramp,qt_L_t_aforce_rup_ramp = np.nan ,np.nan

            Z_rup_avg = np.nan
                
        for j in range(len(self.ramp_loc)):
            (tx,ty) = self.ramp_loc[j]
            if self.rupture_pos >tx and self.rupture_pos<ty :
                self.rup_ramp_loc[0] = tx;self.rup_ramp_loc[1] = ty 
                #self.pow_rate_rup_ramp=(P0[self.rupture_pos] - P0[tx])/(T_s[self.rupture_pos]-T_s[tx])

        self.xx = xx;self.yy = yy
        self.zz = zz;self.T_s = T_s
        self.L_t = L_t;self.P0 = P0


    def prep_trace(self):
        """
        Prepare trace data.
        """
        CleanOutlier_med =False 
        did_it_unbind =  ~np.isnan(self.rupture_pos)

        b, bref = bead_extract_from_df(self.log_afs_df,self.i_file)
        self.bref = bref
        self.file_label+= "_"+ str(b[0])
        
        step_pos = str(self.log_afs_df['Step_position'][self.i_file])
        
        step_pos = np.array([int(e) if e.isdigit() else e for e in step_pos.split('_')])
    
    
        fname = self.tdms_file_name+".tdms"
        
        
        range_anchorpoint_min=(0,100); range_spectrum_min=(0, 100); Select_range_spectrum=False
    
        #In this code block i shall try to obtain the step values for each data file
        #the fs and the exposure time are also read from the meta data 
        
        #tdms_file = TdmsFile.open(f"{self.ori_path}/{fname}") 
        tdms_file = TdmsFile.read(f"{self.ori_path}/{fname}") 

        tdms_groups = tdms_file.groups()   

        tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
        (TE0,NE0,DE0) = events(tdms_event)
        (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[0]) # test bead
        
      
        T0, self.fs = timet(tdms_track, False)
        self.fmax_Hz=self.fs ;self.f_c_threshold_global = self.fs/6 ;self.n_sec_otf = self.N_psd_build/self.fs 
        self.fmin_Hz = self.fs/self.N_psd_build
        
        
        self.te = find_te_tdms_index(fname,self.ori_path)
    
    
        #extracting the step and ramps values for this TDMS file. 
        P0 = BuildP(T0, TE0, NE0, DE0)
        value, location, lengths = island_props(P0)
    
    
    
        step_values = [] ;self.step_loc = []
        for i in step_pos:
            step_values.append(value[i])
            self.step_loc += [(location[i],location[i+1]-1)]

        self.ramp_loc = ramp_extraction(P0,T_s)
    
    
        #extracting and setting hte average temperature of the chamber dyanmicsally 
    
        temperature = tdms_temp['Actual Temp'][:]
        Temperature_C = np.nanmean(temperature); #global kBT_pN_nm
        if Temperature_C==0:
            Temperature_C = 25
            
        self.kBT_pN_nm= 1.38e-23*(Temperature_C+273)*1.e12*1.e9
    
        
 
        #maybe will be updated over time and functionality. 
        loaddata = True 
        pprint = True
        RangeNZlavg=[400]       # size of moving average for smoothing z before anchor point determination
        NZrefavg=5000
        mode='reflect' #'nearest'  # mode for rolling average   # edge modes = ['full', 'same', 'valid', 'nearst']
        # see details on https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        AnchorPointStateList=['custom'] # correction with AnchorPoint as determined by anchor point    
    
        plot_fig_name = fname[-11:-5] #+"_R_"+str(self.bead_radius) + "_bead_"+str(b[0]) 
        os.chdir(self.ori_path)

        print('Loading data', fname)
       
        print('======================BEAD ', b[0] ,' ===========================')
        tdms_groups = tdms_file.groups()        
        tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
        (TE0,NE0,DE0) = events(tdms_event)
        P0 = BuildP(T0, TE0, NE0, DE0)


        dmax=1000000.; imax=(len(tdms_track)-1)//5; dX=1500; dY=1500    
        #print('Nb of traj=', imax)
        refname=fname[:-5]+'_Bead'+str(b[0])+'_'; print(refname)
        (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[0]) # test bead
        #redudunnt code as the previous code block stores this value T

        #T0, fs = time(tdms_track, False)
        #P0 = BuildP(T0, TE0, NE0, DE0)

        zero_force_pos = zero_force_pos_func(P0)
        non_0_force = np.argwhere(P0).squeeze()
        #remove the nan values 
        i0 = np.isnan(X0) + np.isnan(Y0) + np.isnan(Z0) 
        mX0 = np.median(X0[~i0]);  X0[i0] = mX0
        mY0 = np.median(Y0[~i0]);  Y0[i0] = mY0
        mZ0 = np.median(Z0[~i0]);  Z0[i0] = mZ0
        #remove and substitute the median values 
        if CleanOutlier_med:
            mm=0.2; print('Clean outliers nan or +/-', mm)
            if pprint: print('median X0=', mX0, 'median Y0=', mY0, 'median Z0=', mZ0 )
            j0 = (X0>(1+mm)*mX0) + (X0<(1-mm)*mX0) + (Y0>(1+mm)*mY0) + (Y0<(1-mm)*mY0) + (Z0>(1+mm)*mZ0) + (Z0<(1-mm)*mZ0)
            #print(len(j0))
            X0[j0] = mX0; Y0[j0] = mY0; Z0[j0] = mZ0; 
        #    m0 = np.median(X0[~i0]);  X0[i0] = m0*(1+.01*np.random.rand(len(X0[i0]))); X0[ (X0>(1+mm)*m0) | (X0<(1-mm)*m0) ] = m0
            m0 = np.median(MinLUT0[~i0]);  MinLUT0[i0] = m0 ; MinLUT0[ (MinLUT0>(1+mm)*m0) | (MinLUT0<(1-mm)*m0) ] = m0
            m0 = np.median(SSIM0[~i0]);  SSIM0[i0] = m0 ; SSIM0[ (SSIM0>(1+mm)*m0) | (SSIM0<(1-mm)*m0) ] = m0
        Z0[:non_0_force[0]],count = qt_anchor_Z(Z0[:non_0_force[0]],0.02)       
        #this is to replace all values of the Z_bf dist using the 2endth quantile

        X_bf = X0[:non_0_force[0]] ; T_bf = T_s[:non_0_force[0]] ; Y_bf = Y0[:non_0_force[0]]
        Z_bf = Z0[:non_0_force[0]] 


        #step 0: trimming X-Y traces of outliers
        if bool_out_xyz:
                
            temp_out_Xpos = (X0 > (X0.mean() - 3*X0.std())) * (X0 < (X0.mean() + 3*X0.std()))
            temp_out_Ypos = (Y0 > (Y0.mean() - 3*Y0.std())) * (Y0 < (Y0.mean() + 3*Y0.std()))
            temp_out_Zpos = (Z_bf > (Z_bf.mean() - 3*Z_bf.std())) * (Z_bf < (Z_bf.mean() + 3*Z_bf.std()))
    
            for i in range(len(Y0)):
                if ~temp_out_Ypos[i]:
                    Y0[i] = np.median(Y0)
    
            for i in range(len(X0)):
                if ~temp_out_Xpos[i]:
                    X0[i] = np.median(X0)


        #Z_bf  = np.array([x for x in Z_bf if (x > Z_bf.mean() - 2 * Z_bf.std())])

        #step 1 determine anchor point before force, mean of X,Y & Z traces

        #x_anchor_pt = X_bf.mean() ;y_anchor_pt = Y_bf.mean()
        #z_anchor_pt = Z_bf.min()


        #step 2 : anchor pt correction to X,Y , Z

        #X0 -= x_anchor_pt ; Y0 -= y_anchor_pt; Z0 -= z_anchor_pt



        #step 3 : average the traje out for reference beads and cleaning the outliers 

        print("Calculating pool of reference bead(s)", bref )
        if len(bref)>1:

            (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])
            for i in range(1,len(bref)):

                (Xr0_,Yr0_,Zr0_,MinLUTr0_,SSIMr0_,Temp_t)=track(tdms_track,bref[i])
                Xr0=Xr0+Xr0_; Yr0=Yr0+Yr0_; Zr0=Zr0+Zr0_
            Xr0=Xr0/len(bref); Yr0=Yr0/len(bref); Zr0=Zr0/len(bref)
            #Zr0 = Zr0[temp_out_Z_pos]
        elif len(bref)==1:
            (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])   # reference bead

        i0 = np.isnan(Xr0) + np.isnan(Yr0) + np.isnan(Zr0) 
        mXr0 = np.median(Xr0[~i0]);  Xr0[i0] = mXr0
        mYr0 = np.median(Yr0[~i0]);  Yr0[i0] = mYr0
        mZr0 = np.median(Zr0[~i0]);  Zr0[i0] = mZr0
    
        if CleanOutlier_med:
            mm=0.5; print('Clean outliers nan or +/-', mm)
            if pprint: print('median X0=', mXr0, 'median Y0=', mYr0, 'median Z0=', mZr0 )
            j0 = (Xr0>(1+mm)*mXr0) + (Xr0<(1-mm)*mXr0) + (Yr0>(1+mm)*mYr0) + (Yr0<(1-mm)*mYr0) + (Zr0>(1+mm)*mZr0) + (Zr0<(1-mm)*mZr0)
            print(len(j0))
            Xr0[j0] = mXr0; Yr0[j0] = mYr0; Zr0[j0] = mZr0; 
        #    m0 = np.median(X0[~i0]);  X0[i0] = m0*(1+.01*np.random.rand(len(X0[i0]))); X0[ (X0>(1+mm)*m0) | (X0<(1-mm)*m0) ] = m0
            m0 = np.median(MinLUT0[~i0]);  MinLUT0[i0] = m0 ; MinLUT0[ (MinLUT0>(1+mm)*m0) | (MinLUT0<(1-mm)*m0) ] = m0
            m0 = np.median(SSIM0[~i0]);  SSIM0[i0] = m0 ; SSIM0[ (SSIM0>(1+mm)*m0) | (SSIM0<(1-mm)*m0) ] = m0

        #step 4: substract out the anchor point(2) from the reference beads


        #Xr0 -= Xr0[:non_0_force[0]].mean(); Yr0 -= Yr0[:non_0_force[0]].mean()
        #Zr0 -= Zr0[:non_0_force[0]].min()

        #step 5: Rolling average for the reference. 

        Xr0,Xr0avg= Rollavg(Xr0, NZrefavg, mode); Yr0,Yr0avg=Rollavg(Yr0, NZrefavg, mode) 
        Zr0, Zr0avg = Rollavg(Zr0, NZrefavg, mode) 

        #step 6: drift correction :Subt the rolling avg refrence from actual trace
        xx = X0 - Xr0avg + Xr0avg[0];yy = Y0- Yr0avg + Yr0avg[0]
        Z01 = Z0 - Zr0avg + Zr0avg[0] 


        #step 7: trimming the Z traces. 

        if out_Z:
            temp_out_Zpos = (Z_bf > (Z_bf.mean() - 3*Z_bf.std())) * (Z_bf < (Z_bf.mean() + 3*Z_bf.std()))

            Z01_med = np.median(Z01);Z01_m = np.mean(Z01) ;Z01_s = np.std(Z01)
            temp_lout_Zpos= (Z_bf > (Z_bf.mean() -  3* Z_bf.std()))
            temp_hout_Zpos = (Z01 > (Z01_m )) * (Z01 < (Z01_m + 3*Z01_s))

            for i in range(len(Z_bf)):
                if ~temp_lout_Zpos[i] :
                    Z01[i] = Z01_med
            for i in range(len(Z01)):
                if ~temp_hout_Zpos[i]:
                    Z01[i] = Z01_med

        #step 8:final anchor pont sub for Z
        z_anchor_pt = Z01[:non_0_force[0]].min()
        zz = Z01-Z01[:non_0_force[0]].min()
        self.non_zero_pos = non_0_force[0]

        #step 9 : determining the rupture force through the Z trace
        #need not do it as this file as it already.
        
        #Compute the lenghts distribution for the whole range now 
        anchor_pt_arr = [(np.mean(xx[:non_0_force[0]]),np.mean(yy[:non_0_force[0]]))]
        for a in anchor_pt_arr :
            xx -= a[0];yy-=a[1]
           
            #accounting for the height correction  
            zz = zz*self.ht_correction_factor

                
            
            self.rms_X_Y = np.sqrt(np.average((xx[:self.non_zero_pos])**2 + (yy[:self.non_zero_pos] )**2))

           
            L_t = (np.sqrt((xx )**2 + (yy )**2+
                                 (zz + self.bead_radius )**2 ) - self.bead_radius)
            
            
            
                

        self.xx = xx;self.yy = yy
        self.zz = zz;self.T_s = T_s
        self.L_t = L_t;self.P0 = P0


    def create_app(self):
        """
        Create the GUI application.
        """
        #initial setup 
        self.mw.resize(1000,800)
        #self.view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        self.view = KeyPressWindow(show=True)
        self.mw.setCentralWidget(self.view)
        #self.view.sigKeyPress.connect(lambda event: print(keys_mapping[event.key()]))
        self.view.sigKeyPress.connect(lambda event: self.key_pressed_YS(event))
        self.mw.show()

        self.mw.setWindowTitle('Lumicks analysis sucks')
        #creating buttons  
        self.prev_bead_btn = QtWidgets.QPushButton("Prev bead")
        self.next_bead_btn = QtWidgets.QPushButton("Next bead")
        self.next_bead_btn.setShortcut
        self.t_rup_redo_btn = QtWidgets.QPushButton("Redo T rup")
        self.save_trup_btn = QtWidgets.QPushButton("Save T_rup")
        self.PSD_plt_btn = QtWidgets.QPushButton("Plot PSD")
        self.save_otr_btn = QtWidgets.QPushButton("Save OTR")
        self.save_sess_btn = QtWidgets.QPushButton("Save Sess")
        
        self.bool_ignore_X =     QtWidgets.QCheckBox("Ignore X")
        self.bool_ignore_Y =     QtWidgets.QCheckBox("Ignore Y")

        self.file_label_btn = pg.LabelItem("Name of file")
        self.X_PSD_label_itm = pg.LabelItem("X PSD results")
        self.Y_PSD_label_itm = pg.LabelItem("Y PSD results")
        self.trace_label_itm = pg.LabelItem("Trace deets")

        #connecting the buttons
                
        self.t_rup_redo_btn.clicked.connect(self.onButtonClicked_t_rup_redo)
        self.PSD_plt_btn.clicked.connect(self.onButtonClicked_PSD_plt_b)
        self.prev_bead_btn.clicked.connect(self.btn_prev_file)
        self.next_bead_btn.clicked.connect(self.btn_next_file)

        self.save_sess_btn.clicked.connect(self.save_sess_log)

        ## create fview
        self.b_pf = self.view.addItem(self.add_button(self.prev_bead_btn));self.fn_label = self.view.addItem(self.file_label_btn)
        self.b_nf = self.view.addItem(self.add_button(self.next_bead_btn))#;self.b_s_sess = self.view.addItem(self.add_button(self.save_sess_btn))

        self.view.nextRow()
        self.trace_plt = self.view.addPlot();self.trace_plt.setYRange(-400,400)
        self.trace_plt.setAspectLocked(lock=True,ratio= 1.40)
        self.PSD_plt = self.view.addPlot();self.PSD_plt.setAspectLocked()#;PSD_plt.enableAutoRange()
 
        
        self.view.nextRow()
                
        self.b_red_rup  = self.view.addItem(self.add_button(self.t_rup_redo_btn));self.b_save_rup = self.view.addItem(self.add_button(self.save_trup_btn))
        #self.b_psd_plt = self.view.addItem(self.add_button(self.PSD_plt_btn));
        self.b_s_sess = self.view.addItem(self.add_button(self.save_sess_btn))
        
        self.view.nextRow()
        self.view.addItem(self.X_PSD_label_itm);self.view.addItem(self.Y_PSD_label_itm);self.view.addItem(self.trace_label_itm);


        #adding rup pos line and otr window
        self.rup_pos = pg.InfiniteLine(movable=True, angle=90, pen=(0, 0, 200) , hoverPen=(0,200,0), label='T_rup ={value:0.2f}s', 
                               labelOpts={'color': (200,0,0), 'movable': True, 'fill': (0, 0, 200, 100)})
        
        self.rup_pos.sigClicked.connect(self.save_t_rup)
        

        #OTR window added
        
        self.otr_win = pg.LinearRegionItem(values=[0,0+self.n_sec_otf])
        
        #this is done to stop the lines from moving and changing the windowsize
        for line in self.otr_win.lines:
            line.setMovable(False)
        self.otr_label = pg.InfLineLabel(self.otr_win.lines[0], f"OTR_A_{self.otr_win.getRegion()} ",color='black', position=0.95, rotateAxis=(0,-1), anchor=(1, 1))

        self.otr_win.sigRegionChanged.connect(self.onButtonClicked_PSD_plt_b)
        
    def move_rup_pos(self, ax, V_x=1):
        """
        Move the rupture position.

        Parameters:
        ax (int): Axis value.
        V_x (int): Value to move the rupture position.
        """
        cur_rup_pos = self.rup_pos.value()
        new_pos  = cur_rup_pos +ax *V_x
        self.rup_pos.setPos(new_pos)
    def move_otr(self, ax, V_x=0.1):
        """
        Move the OTR window.

        Parameters:
        ax (int): Axis value.
        V_x (float): Value to move the OTR window.
        """
        cur_rup_pos = self.otr_win.getRegion()
        new_pos  = cur_rup_pos[0] +ax *V_x
        self.otr_win.setRegion([new_pos,new_pos+self.n_sec_otf])

    def add_button(self, btn):
        """
        Add a button to the GUI.

        Parameters:
        btn (QPushButton): Button to add.
        """
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(btn)
    
        return(proxy)
        
    
    def btn_skip_file(self):
        """
        Skip the current file and move to the next.
        """
        chime.info()
        self.i_file +=1

        if self.i_file==self.num_of_beads:
            self.file_label_btn.setText("C'est fini")
        else:
            self.onButtonClicked_t_rup_redo()
            if self.is_bass_trace:
                self.load_bead_gui_bas()
            else:
                self.load_bead_gui()        
    
    def btn_next_file(self):
        """
        Move to the next file.
        """
        self.save_sess_log()

        self.i_file +=1

        if self.i_file==self.num_of_beads:
            self.file_label_btn.setText("C'est fini")
        else:
            self.onButtonClicked_t_rup_redo()
            if self.is_bass_trace:
                self.load_bead_gui_bas()
            else:
                self.load_bead_gui()               
    def btn_prev_file(self):
        """
        Move to the previous file.
        """
        self.i_file -=1
        if self.i_file<0:
            self.file_label_btn.setText("Index 0, no more file post")
        else:
            self.onButtonClicked_t_rup_redo()
            self.pop_sess_log()
            if self.is_bass_trace:
                self.load_bead_gui_bas()
            else:
                self.load_bead_gui()               
    def pop_sess_log(self):
        """
        Pop the session log.
        """
        self.log_trouble_sess = self.log_trouble_sess.loc[:len(self.log_trouble_sess)-2,:]
        self.log_trouble_sess.to_excel(self.save_path,sheet_name='troubleshooted')

    def update_OTR_WIN(self, otr_b):
        """
        Update the OTR window.

        Parameters:
        otr_b (float): OTR window value.
        """
        self.OTR_B = int(otr_b*self.fs); self.OTR_A =  int(self.fs*(otr_b-self.n_sec_otf))
    
    def round_avg_win(self, arr):
        """
        Round the average window.

        Parameters:
        arr (array): Array of values.
        """
        return np.round(np.average(arr[self.OTR_A:self.OTR_B]),3)
    def rms_over_wins(self, win_size_s=2):
        """
        Calculate RMS over windows.

        Parameters:
        win_size_s (int): Window size in seconds.
        """
        N_pts_win =int(win_size_s*self.fs)
        sym_fac_array = [];rms_arr = []
        x_bf = self.xx[:self.non_zero_pos]
        y_bf = self.yy[:self.non_zero_pos]

        for i in range(N_pts_win,len(x_bf),N_pts_win):
            AnchorCov=np.cov(x_bf[:i],y_bf[:i],rowvar = False)
            rms = np.sqrt(np.average(x_bf[:i]**2+y_bf[:i]**2))
            w, v = np.linalg.eig(AnchorCov); 
            rms_arr.append(rms)
            sym_fac_array.append(  np.sqrt(np.amax(w)/np.amin(w)))
        return sym_fac_array,rms_arr 
    def cal_rupture_angle(self):
        """
        Calculate the rupture angle.
        
        Computes the angle at a particular point and returns the angle in degrees.
        """
        #computes the angle at a particular point, 
        #returns the angle in degrees  
        temp = np.sqrt(self.xx[self.OTR_A:self.OTR_B]**2+self.yy[self.OTR_A:self.OTR_B]**2)
        angle  = (temp/(self.L_t[self.OTR_A:self.OTR_B]+self.bead_radius))

        angle = np.rad2deg(np.arcsin(angle))
        self.rup_angle = np.average(angle)
        return(angle)
        
    def onButtonClicked_PSD_plt_b(self):
        """
        Handle the event when the PSD plot button is clicked.
        """
        self.PSD_plt.clear()
   
        #print("updates")
        rgn = self.otr_win.getRegion()
        self.update_OTR_WIN(rgn[1])

        
        axis_arr =['R','R']

        self.avg_pow_OTR  = np.round(np.average(self.P0[self.OTR_A:self.OTR_B]),3)
        temp_pow_rate_rup_ramp=(self.P0[self.OTR_B]- self.P0[self.OTR_A])/(self.T_s[self.OTR_B]- self.T_s[self.OTR_A])
        self.pow_rate_rup_ramp = temp_pow_rate_rup_ramp
        Z_rup_avg = np.round(np.mean(self.zz[self.OTR_A:self.OTR_B]),3)
        trace_deet_label = 'L_t : '+ str(self.L_t_rup)+'_sym : '+f'{self.symmetry_factor:.2f}'+"_P_avg : "+str(self.avg_pow_OTR)+f"_dp_dt_{temp_pow_rate_rup_ramp:.6f}"
        #pow_per_sec = (self.P0[self.OTR_B] - self.P0[self.OTR_A])/(self.T_s[self.OTR_B]-self.T_s[self.OTR_A])
        #self.otr_label.setText("Avg_P = "+ str(avg_pow))
        self.trace_label_itm.setText(trace_deet_label,color = 'black')
        trace_arr = [self.xx,self.yy]
        fc_l_arr = np.zeros(2)
        for i in range(2):
        
            trace = trace_arr[i]
            temp_label = "temp_label"
            temp_out_spec_l= self.PSD_spectrum_plotting(trace[self.OTR_A:self.OTR_B],  
                                                        Z_rup_avg,i,temp_label) 
            fc_l_arr[i] = temp_out_spec_l[4]
            self.calib_K_arr[i] = temp_out_spec_l[0]
            
            self.calib_force_arr[i] = 0.5*temp_out_spec_l[0]*(self.L_t_rup+self.bead_radius)
            self.calib_D_arr[i] = temp_out_spec_l[1]
            
        self.f_c_avg_otr = np.average(fc_l_arr)
        if np.average(fc_l_arr)>=self.f_c_threshold_global:
            print(self.f_c_threshold_global,np.average(fc_l_arr))
            self.bool_D_G = True

    def find_rup_force(self):
        """
        Find the rupture force.
        """
        self.bool_troubleshoot  = True
        print("find the rup force ")
        
        trace_arr = [self.xx,self.yy]
        fc_l_arr = np.zeros(2)
        Z_rup_avg = np.round(np.mean(self.zz[self.OTR_A:self.OTR_B]),3)
        self.L_t_rup = np.round(np.nanmean(self.L_t[self.rupture_pos-self.N_psd_build :self.rupture_pos ]),3)

        self.avg_pow_OTR  = np.round(np.average(self.P0[self.OTR_A:self.OTR_B]),3)
        self.pow_rate_rup_ramp = (self.P0[self.OTR_B]- self.P0[self.OTR_A])/(self.T_s[self.OTR_B]- self.T_s[self.OTR_A])
        self.P_rup = self.P0[ self.rupture_pos]

        if self.pow_rate_rup_ramp ==0: 
            if self.avg_pow_OTR>0:
                self.pow_rate_rup_ramp= (self.P0[self.rupture_pos]- self.P0[self.OTR_A])/(self.T_s[self.rupture_pos]- self.T_s[self.OTR_A])
                self.load_time = self.P_rup/self.pow_rate_rup_ramp

            else:
                self.bool_troubleshoot  =False
                self.load_time = 0
        else:
            self.load_time = self.P_rup/self.pow_rate_rup_ramp

        #self.pow_rate_rup_ramp = 
        for i in range(2):
        
            trace = trace_arr[i]
            temp_label = "temp_label"

            temp_out_spec_l= self.PSD_spectrum_plotting(trace[self.OTR_A:self.OTR_B], 
                                                        Z_rup_avg,i,temp_label,display_psd=True) 
            fc_l_arr[i] = temp_out_spec_l[4]
            self.calib_K_arr[i] = temp_out_spec_l[0]

            self.calib_force_arr[i] = temp_out_spec_l[0]*(self.L_t_rup+self.bead_radius)
            self.calib_D_arr[i] = temp_out_spec_l[1]
            self.D_t = self.kBT_pN_nm/temp_out_spec_l[5]
            self.f_c_avg_otr = np.average(fc_l_arr)
            
            
        if np.average(fc_l_arr)>=self.f_c_threshold_global:
            self.bool_D_G = True
        else:
            self.bool_D_G = False


        plt.show()

        if self.bool_D_G and not(self.D_G_found):
            self.find_global()

        if self.bool_D_G:
            if self.D_G_found:
                for i in range(2):
                    trace = trace_arr[i]
                    temp_label = "glbalfit"
                    
    
                    temp_out_spec_l= self.spectrum_D_g(trace[self.OTR_A:self.OTR_B], 
                                                                Z_rup_avg,i,temp_label) 
                    fc_l_arr[i] = temp_out_spec_l[4]
                    self.calib_K_arr[i] = temp_out_spec_l[0]
    
                    self.calib_force_arr[i] = temp_out_spec_l[0]*(self.L_t_rup+self.bead_radius)
                    
                    self.calib_D_arr[i] = temp_out_spec_l[1]
                    print(temp_out_spec_l[5])
                    self.D_t = self.kBT_pN_nm/temp_out_spec_l[5]

            else:
                self.bool_troubleshoot  =False
        self.cal_rupture_angle()
 
        self.rup_force =np.nanmean(self.calib_force_arr)
        self.LDR = (self.rup_force/self.avg_pow_OTR)*self.pow_rate_rup_ramp
        self.rup_force =self.LDR *self.load_time

    def find_global(self):
        """
        Find the global D value.
        """
        print("finding D_g agaim")
        if self.bool_D_G:
            self.search_DG()
            if self.D_global_arr[0]== False:
                print('find D_G in all ramps')

                for rl in self.ramp_loc:
                    self.search_DG(rl)
        self.D_G_found = self.D_global_arr[0]        
    def save_t_rup(self):
        """
        Save the rupture time.
        """
        #print("save_ trup" , self.rup_pos.value())
        self.rup_pos.setPen((0,200,0))
        self.rup_pos.label.setColor((0,200,0))
    
        self.rup_pos.movable = False
        self.rupture_time = self.rup_pos.value()
        self.rupture_pos = int(self.rupture_time*self.fs)
        self.P_rup = self.P0[ self.rupture_pos]
        for j in range(len(self.ramp_loc)):
            (tx,ty) = self.ramp_loc[j]
            if self.rupture_pos >tx and self.rupture_pos<ty :
                self.rup_ramp_loc[0] = tx;self.rup_ramp_loc[1] = ty 

        self.L_t_rup = np.round(np.nanmean(self.L_t[self.rupture_pos-self.N_psd_build :self.rupture_pos ]),3)
        self.update_OTR_WIN(self.rupture_time)
        self.otr_win.setRegion([self.rupture_time-self.n_sec_otf,self.rupture_time])
        
    def onButtonClicked_t_rup_redo(self):
        """
        Handle the event when the redo rupture time button is clicked.
        """
        self.rup_pos.setPen((0,0,200))
        self.rup_pos.label.setColor((200,0,0))
        self.rup_pos.movable = True
    def save_sess_log_post_clean(self, do_you_want_an_excel=False):
        """
        Save the session log after cleaning.

        Parameters:
        do_you_want_an_excel (bool): Flag to save as Excel.
        """
        N_trouble_sess = len(self.log_trouble_sess)
        #print(N_trouble_sess,self.log_afs_df.iloc[self.i_file])

        self.log_trouble_sess.loc[N_trouble_sess] = self.log_afs_df.iloc[self.i_file]
        
        self.log_trouble_sess.loc[N_trouble_sess,'bool_troubleshoot'] = self.bool_troubleshoot
        self.log_trouble_sess.loc[N_trouble_sess,'k_Bt_nm'] = self.kBT_pN_nm

        self.log_trouble_sess.loc[N_trouble_sess,'rupture_pos_s'] = self.rupture_time
        self.log_trouble_sess.loc[N_trouble_sess,'rupture_pos_pt_idx'] = self.rupture_pos
        
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_A_s'] = self.T_s[self.OTR_A]
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_B_s'] = self.T_s[self.OTR_B]
        
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_A_pt_idx'] = self.OTR_A
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_B_pt_idx'] = self.OTR_B
        self.log_trouble_sess.loc[N_trouble_sess,'f_c_otr_avg'] = self.f_c_avg_otr
        self.log_trouble_sess.loc[N_trouble_sess,'fs'] = self.fs
        self.log_trouble_sess.loc[N_trouble_sess,'f_c_threshold_global'] = self.f_c_threshold_global

        self.log_trouble_sess.loc[N_trouble_sess,'bool_D_G'] = self.bool_D_G
        
        
        #if self.bool_D_G:self.find_global()
        
        self.find_rup_force()
        self.log_trouble_sess.loc[N_trouble_sess,'D_G_found'] = self.D_global_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'DG_X'] = self.D_global_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'DG_Y'] = self.D_global_arr[2]

        self.log_trouble_sess.loc[N_trouble_sess,'L_t_rup'] = self.L_t_rup

        self.log_trouble_sess.loc[N_trouble_sess,'symmetry'] = self.symmetry_factor
        self.log_trouble_sess.loc[N_trouble_sess,'rms_X_Y'] = self.rms_X_Y
        self.log_trouble_sess.loc[N_trouble_sess,'rup_angle_deg'] = self.rup_angle


        self.log_trouble_sess.loc[N_trouble_sess,'K_X_otr'] = self.calib_K_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'K_Y_otr'] = self.calib_K_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'bead_radius'] = self.bead_radius
        self.log_trouble_sess.loc[N_trouble_sess,'power_per_sec'] = self.pow_rate_rup_ramp
        self.log_trouble_sess.loc[N_trouble_sess,'avg_pow_otr'] = self.avg_pow_OTR

        self.log_trouble_sess.loc[N_trouble_sess,'D_X_otr'] = self.calib_D_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'D_Y_otr'] = self.calib_D_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'force_X'] = self.calib_force_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'force_Y'] = self.calib_force_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'D_T'] = self.D_t
        self.log_trouble_sess.loc[N_trouble_sess,'rup_force'] = self.rup_force
        self.log_trouble_sess.loc[N_trouble_sess,'LDR'] = self.LDR
        self.log_trouble_sess.loc[N_trouble_sess,'correctioin_factor'] = self.ht_correction_factor
        self.log_trouble_sess.loc[N_trouble_sess,'load_time'] = self.load_time

        print("save success,  " , self.file_label)
        if do_you_want_an_excel:self.log_trouble_sess.to_excel(self.save_path,sheet_name='troubleshooted')        
    def save_sess_log(self):
        """
        Save the session log.
        """
        N_trouble_sess = len(self.log_trouble_sess)
        #print(N_trouble_sess,self.log_afs_df.iloc[self.i_file])

        self.log_trouble_sess.loc[N_trouble_sess] = self.log_afs_df.loc[self.i_file]
        
        self.log_trouble_sess.loc[N_trouble_sess,'bool_troubleshoot'] = True
        self.log_trouble_sess.loc[N_trouble_sess,'k_Bt_nm'] = self.kBT_pN_nm

        self.log_trouble_sess.loc[N_trouble_sess,'rupture_pos_s'] = self.rupture_time
        self.log_trouble_sess.loc[N_trouble_sess,'rupture_pos_pt_idx'] = self.rupture_pos
        
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_A_s'] = self.T_s[self.OTR_A]
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_B_s'] = self.T_s[self.OTR_B]
        
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_A_pt_idx'] = self.OTR_A
        self.log_trouble_sess.loc[N_trouble_sess,'OTR_B_pt_idx'] = self.OTR_B
        self.log_trouble_sess.loc[N_trouble_sess,'f_c_otr_avg'] = self.f_c_avg_otr
        self.log_trouble_sess.loc[N_trouble_sess,'fs'] = self.fs
        self.log_trouble_sess.loc[N_trouble_sess,'f_c_threshold_global'] = self.f_c_threshold_global

        self.log_trouble_sess.loc[N_trouble_sess,'bool_D_G'] = self.bool_D_G
        
        
        #if self.bool_D_G:self.find_global()
        
        self.find_rup_force()
        self.log_trouble_sess.loc[N_trouble_sess,'D_G_found'] = self.D_global_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'DG_X'] = self.D_global_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'DG_Y'] = self.D_global_arr[2]

        self.log_trouble_sess.loc[N_trouble_sess,'L_t_rup'] = self.L_t_rup

        self.log_trouble_sess.loc[N_trouble_sess,'symmetry'] = self.symmetry_factor
        self.log_trouble_sess.loc[N_trouble_sess,'rms_X_Y'] = self.rms_X_Y
        self.log_trouble_sess.loc[N_trouble_sess,'rup_angle_deg'] = self.rup_angle


        self.log_trouble_sess.loc[N_trouble_sess,'K_X_otr'] = self.calib_K_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'K_Y_otr'] = self.calib_K_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'bead_radius'] = self.bead_radius
        self.log_trouble_sess.loc[N_trouble_sess,'power_per_sec'] = self.pow_rate_rup_ramp
        self.log_trouble_sess.loc[N_trouble_sess,'avg_pow_otr'] = self.avg_pow_OTR

        self.log_trouble_sess.loc[N_trouble_sess,'D_X_otr'] = self.calib_D_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'D_Y_otr'] = self.calib_D_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'force_X'] = self.calib_force_arr[0]
        self.log_trouble_sess.loc[N_trouble_sess,'force_Y'] = self.calib_force_arr[1]
        self.log_trouble_sess.loc[N_trouble_sess,'D_T'] = self.D_t
        self.log_trouble_sess.loc[N_trouble_sess,'rup_force'] = self.rup_force
        self.log_trouble_sess.loc[N_trouble_sess,'LDR'] = self.LDR
        self.log_trouble_sess.loc[N_trouble_sess,'correctioin_factor'] = self.ht_correction_factor
        self.log_trouble_sess.loc[N_trouble_sess,'load_time'] = self.load_time

        chime.success()


        print("save success,  " , self.file_label)
        self.log_trouble_sess.to_excel(self.save_path,sheet_name='troubleshooted')
    def FitSpectrumGenNew(self, f, k, D):
        """
        Generate the spectrum fit.

        Parameters:
        f (array): Frequency array.
        k (float): Spring constant.
        D (float): Diffusion coefficient.
        """
        #te - exp time in seconds 
        #te = 0.0029#; fs = 59.82279832617854
        #print(te,"check the te in s read from meta data and fs : ",fs)
        fc = D*k/(2*np.pi*self.kBT_pN_nm) 
        PREF = 2*self.kBT_pN_nm**2/(D*k*k) 
        S=0 
        for n in [0, 1,2]: 
            #print(n)
        #for n in [0, 1]: 
    
            S+= (SINCn(f,self.te,self.fs,n)/DENn(f,fc,self.fs,n)) 
        return PREF*S   # with correction 
    
    def FitSpectrumGenDaldrop(self, f, *p):
        """
        Generate the spectrum fit using Daldrop method.

        Parameters:
        f (array): Frequency array.
        *p (tuple): Parameters for the fit.
        """
        return self.FitSpectrumGenNew(f, p[0], p[1])
    def FitSpectrum_plot_global(self, x, *p):
        """
        Plot the global spectrum fit.

        Parameters:
        x (array): X values.
        *p (tuple): Parameters for the fit.
        """
        return p[1]/(2*np.pi**2)/( x**2 + (p[0]*p[1]/(2*np.pi*self.kBT_pN_nm))**2 )   # Sitter 2015

    
    def PSD_spectrum_plotting(self, xx, p1Zo, i, label='lolol', ramp_time=None, build_time=None, plot_ax=None, display_psd=False):
        """
        Plot the PSD spectrum.

        Parameters:
        xx (array): Trace array.
        p1Zo (float): Z position.
        i (int): Index.
        label (str): Label for the plot.
        ramp_time (float): Ramp time.
        build_time (float): Build time.
        plot_ax (Axes): Plot axis.
        display_psd (bool): Flag to display PSD.
        """
        display = True ; SaveGraph = True
        axis_arr = ['X','Y'];color_arr= [(255,0,0),(0,0,255)]
        friction0 = 6*np.pi*1.e-9*self.bead_radius       # units pN.s/nm
        
        friction = friction0 / ( 1 - (9/16)*(self.bead_radius/(p1Zo+self.bead_radius)) + (1/8)*(self.bead_radius/(p1Zo+self.bead_radius))**3 )

        f, Pxx_spec = signal.periodogram(xx, self.fs, scaling='density') 
    
        Pxxt_spec= Pxx_spec[(f>self.fmin_Hz)&(f<self.fmax_Hz)]; ft=f[(f>self.fmin_Hz)&(f<self.fmax_Hz)]
        nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
        for m in range(nbins-1):
            u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
            Pbins[m]=np.mean(u)
            dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
            
            
        while True:
            try:
                pEq, pcovEq=curve_fit(self.FitSpectrumGenDaldrop, ft, Pxxt_spec, p0=[1.e-3, 1.e5])
                eEq=np.sqrt(np.diag(pcovEq))
                break
            except RuntimeError:
                print("No convergence"); pEq=[np.nan, np.nan]; eEq=[np.nan, np.nan]; break 
            FitPxxt_spec=self.FitSpectrumGenNew(ft, pEq[0], pEq[1])

        FitPxxt_spec=self.FitSpectrumGenNew(ft, pEq[0], pEq[1])
        pEq[0] = np.abs(pEq[0])
        fc = pEq[0]*pEq[1]/(2*np.pi*self.kBT_pN_nm)
        
        D_Dtheo = pEq[1]* (friction/self.kBT_pN_nm)
        o_p_label= 'f_c: '+f'{fc:.2f}'+ '_k(e-4):'+f'{pEq[0]*10**4:.2f}'+'_D_Dt: '+f'{D_Dtheo:.2f}'
        if self.gui_bool:
            if i==0:
                self.X_PSD_label_itm.setText(o_p_label,color =color_arr[i] )
            else:
                self.Y_PSD_label_itm.setText(o_p_label,color =color_arr[i])  
            EB_itm = pg.ErrorBarItem()
            EB_itm.setData(x = fbins, y = Pbins, top = Pbins+ dPbins,bottom = Pbins- dPbins)
            #PSD_plt.addItem(EB_itm)
            self.PSD_plt.plot(fbins, Pbins,pen=color_arr[i], name = axis_arr[i])
        
            self.PSD_plt.plot(ft, FitPxxt_spec, pen='black', alpha=0.8)
            self.PSD_plt.setLogMode(x=True, y=True)
                    
            self.PSD_plt.addLegend(frame=False,QSizeF = 10)
            self.PSD_plt.setYRange(1, 5);self.PSD_plt.setXRange(-2, 3)  
        elif display_psd:
            color_arr_plt = ['blue','red']
            if plot_ax ==None:
                plt.figure(self.file_label, figsize=(6,6), dpi=100); ax = plt.gca()
                plt.title(self.file_label)
            else:
                ax =plot_ax
            ax.errorbar(fbins, Pbins, dPbins, marker='.', c=color_arr_plt[i], label = f"{axis_arr[i]}_{fc}", alpha=0.2)
            ax.set_ylim(1e-1, 1e5); ax.set_xscale('log'); ax.set_xlim(1e-2, 1e3); ax.set_yscale('log')    
            ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+self.file_label);    #     ax.set_ylabel('spectrum [nm²] '+label); 
            ax.plot(ft, FitPxxt_spec, c='black', alpha=0.8)
            plt.legend()
            
            '''
            if ramp_time!=None:
                ax.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r');
                a = '%.2E' % Decimal(ramp_time[1])
                ax.text(1e0, 1e4, 'Power per second  = '+a,c='r');

            if build_time!=None:ax.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')
            '''
        
        avg_pow =  np.round(np.average(self.P0[self.OTR_A:self.OTR_B]),3)
        
        #param_label =pg.LabelItem(str(avg_pow),parent=self.PSD_plt,size = '10pt',bold = False)
        #param_label.anchor(itemPos=(0.4, 0.1), parentPos=(0.4, 0.1))
        #PSD_plt.setRange(xRange=[-2,3])

        #PSD_plt.set_ylabel('PSD [nm²/Hz] '+label)    
        if ramp_time!=None:
            #axbis.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r')
            a = '%.2E' % Decimal(ramp_time[1])
            axbis.text(1e0, 1e4, 'Power per second  = '+a,c='r')
    
        if build_time!=None:axbis.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')
            
        
        #print('Spectrum'+label, ' k (pN/nm)=',"%.5f" %  (pEq[0]),' D (µm²/s)=',"%.3f" %  (pEq[1]*1.e-6), 'D/Dtheo=', pEq[1]*friction/kBT_pN_nm)
        temp_arr_round = [pEq[0], pEq[1], eEq[0], eEq[1], fc, friction ,D_Dtheo]
        
        return(temp_arr_round)
    
    def spectrum_D_g(self, xx, p1Zo, i, label='lolol', plot_ax=None):
        """
        Plot the global spectrum D.

        Parameters:
        xx (array): X values.
        p1Zo (float): Z position.
        i (int): Index.
        label (str): Label for the plot.
        plot_ax (Axes): Plot axis.
        """
        friction0 = 6*np.pi*1.e-9*self.bead_radius 
        bead_radius =self.bead_radius
        # units pN.s/nm
        D_global=self.D_global_arr[i+1]
        axis_arr = ['X','Y']
        
        friction = friction0 / ( 1 - (9/16)*(self.bead_radius/(p1Zo+self.bead_radius)) + (1/8)*(self.bead_radius/(p1Zo+self.bead_radius))**3 )


        def FitSpectrum_global(x, k1,D_g=D_global): 
            return D_g/(2*np.pi**2)/( x**2 + (k1*D_g/(2*np.pi*self.kBT_pN_nm))**2 )   # Sitter 2015
            
        D_Dtheo = D_global*friction/self.kBT_pN_nm ; dl = 0

        f, Pxx_spec = signal.periodogram(xx, self.fs, scaling='density') 
        #print("GLobal spec F_max ",fmax_Hz,f)
        Pxxt_spec= Pxx_spec[(f>self.fmin_Hz)&(f<self.fmax_Hz)]; ft=f[(f>self.fmin_Hz)&(f<self.fmax_Hz)]

        #print(ft)
        nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
        
        for m in range(nbins-1):
            u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
            Pbins[m]=np.mean(u)
            dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
            

        while True:
            try:
                pEq, pcovEq=curve_fit(FitSpectrum_global, ft, Pxxt_spec, p0=[1.e-2])
                #print("pEq",pEq)
                eEq=np.sqrt(np.diag(pcovEq))
                break
            except RuntimeError:
                print("No convergence"); pEq=[np.nan, np.nan]; eEq=[np.nan, np.nan]; break 
        
        pEq = np.abs(pEq[0])
        fc = pEq*D_global/(2*np.pi*self.kBT_pN_nm)
        ft_global = np.append(ft,np.array([1e4,5e4]))    #just extending the limits of the predicted curve

        FitPxxt_spec=self.FitSpectrum_plot_global(ft_global, pEq, D_global)
        if self.gui_bool==False:
            color_arr_plt = ['blue','red']
            if plot_ax ==None:
                plt.figure(self.file_label, figsize=(6,6), dpi=100); ax = plt.gca()
                plt.title(self.file_label)
            else:
                ax =plot_ax
            ax.errorbar(fbins, Pbins, dPbins, marker='.', c=color_arr_plt[i], label = f"{axis_arr[i]}_{fc}", alpha=0.2)
            ax.set_ylim(1e-1, 1e5); ax.set_xscale('log'); ax.set_xlim(1e-2, 1e3); ax.set_yscale('log')    
            ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+self.file_label);    #     ax.set_ylabel('spectrum [nm²] '+label); 
            ax.plot(ft_global, FitPxxt_spec, c='black', alpha=0.8)
            plt.legend()
                

        temp_arr_round = [pEq, D_global, eEq[0], dl,fc, friction ,D_Dtheo]
        return(temp_arr_round)
    def search_DG(self, ramp_loc=None):
        """
        Search for the global D value.

        Parameters:
        ramp_loc (tuple): Ramp location.
        """
        d_dt_max,d_dt_min = 1.2 ,0.8
     
        if ramp_loc==None:
            [tx,ty] = self.rup_ramp_loc

            print("Global D search initiaited ")

            start_time = tx+int(0* self.fs) ; end_time =  self.rupture_pos - int( 0* self.fs)
        else:
            print("Global D search all ramps ")

            if type(self.D_X_array)==np.ndarray:
                print("change to list")
                self.D_X_array = self.D_X_array.tolist()
                self.D_Y_array = self.D_Y_array.tolist()
                
                
            start_time = ramp_loc[0];end_time=ramp_loc[1]
            
        
        pos_a = start_time ; 
        axis_arr = ['X','Y','Z'] 
        trace_arr = [self.xx,self.yy]
        while pos_a < end_time and pos_a<self.rupture_pos and self.D_global_arr[0] == False:
            pos_b = pos_a +self.N_psd_build

            D_arr = np.zeros(2); D_Dt_arr = np.zeros(2);fc_l_arr = np.zeros(2)

            avg_power_1 = (self.P0[pos_a]+self.P0[pos_b])/2.0
            temp_force = 0;Z_rup_avg = np.round(np.mean(self.zz[pos_a:pos_b]),3)
            L_trup_avg = np.round(np.mean(self.L_t[pos_a:pos_b]),3)
            for i in range(2):

                trace = trace_arr[i]
                temp_label = "_PowerSpectrum_alias_avg_P"+str(avg_power_1)+'_' + str(self.n_sec_otf) +"_secs_" +axis_arr[i]+'_D_g_otf_Fc_thresh_'+str(self.f_c_threshold_global)
                plot_fig_glob = "_global_alias_felix_otf_"+axis_arr[i]+"_P_"+str(avg_power_1)
                temp_out_spec_l= self.PSD_spectrum_plotting(trace[pos_a:pos_b], 
                                                            Z_rup_avg,i,temp_label) 
    
                [k_l, D_arr[i], dk_l, dD_l, fc_l_arr[i], friction_l,D_Dt_arr[i]]= temp_out_spec_l
                

            #print(D_arr,D_Dt_arr)
            if fc_l_arr[0] < self.f_c_threshold_global and not(self.D_global_arr[0]):
                if (d_dt_min <= D_Dt_arr[0] <=d_dt_max) and len(self.D_X_array)<101:
                    print(D_arr[0],D_Dt_arr[0],self.D_t,D_arr[0]/D_Dt_arr[0])
                    self.D_X_array.append(D_arr[0])

                
            if fc_l_arr[1] < self.f_c_threshold_global and not(self.D_global_arr[0]):
       
                if (d_dt_min <= D_Dt_arr[1] <=d_dt_max)and len(self.D_Y_array)<101:
                    self.D_Y_array.append(D_arr[1])

            elif self.f_c_threshold_global < fc_l_arr[1]and self.f_c_threshold_global < fc_l_arr[0] and not(self.D_global_arr[0]):
                self.D_X_array = np.array(self.D_X_array);self.D_Y_array = np.array(self.D_Y_array)

                self.D_global_arr = [True , np.nanmean(self.D_X_array),np.nanmean(self.D_Y_array)]

            pos_a += int( 0.1*self.N_psd_build)
        self.D_X_array = np.array(self.D_X_array)
        self.D_Y_array = np.array(self.D_Y_array)
        
        if len(self.D_X_array)>=1 and len(self.D_Y_array) >=1 :
            self.D_global_arr = [True , np.nanmean(self.D_X_array),np.nanmean(self.D_Y_array)]
            #print("Global D alias found ")
            
        elif len(self.D_X_array)>=1 :
            print("Global D only X  found ")
            self.D_global_arr = [True , np.nanmean(self.D_X_array),np.nanmean(self.D_X_array)]
        elif len(self.D_Y_array)>=1 :
            print("Global D only Y  found ")
            self.D_global_arr = [True , np.nanmean(self.D_Y_array),np.nanmean(self.D_Y_array)]

        else :
            self.D_global_arr = [False , 0,0]
            print("NOT FOUNDGlobal D ")
    def controller_cont(self):
        """
        Handle controller events.
        """
    
        for event in pyg.event.get():

            if event.type == pyg.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pyg.joystick.Joystick(event.device_index)
                self.joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connencted")
            if event.type == pyg.JOYDEVICEREMOVED:
                del self.joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")
        for controller in self.joysticks.values():
                
            if controller.get_button(11): #A
                self.btn_next_file()
            if controller.get_button(12): #B
                self.btn_prev_file()

            if controller.get_button(13): #X
                self.save_sess_btn()

            if controller.get_button(14): #Y
                self.move_otr()
            if controller.get_axis(0): #Y
                self.move_rup_pos()

    def find_temp(self):
        """
        Find the temperature from the TDMS file.
        """
        tdms_file = TdmsFile.read_metadata(f"{self.ori_path}/{self.tdms_file_name}.tdms")  
        for group in tdms_file.groups():
            te = float(group.properties['Camera.Exposure time (ms)'])
            break
        self.te = te* 0.001 # to transform the data into s
        
        tdms_file = TdmsFile.open(f"{self.ori_path}/{self.tdms_file_name}.tdms") 
        tdms_groups = tdms_file.groups()   
        tdms_temp = tdms_groups[1]
        
        temperature = tdms_temp['Actual Temp'][:]

        Temperature_C = np.nanmean(temperature); #global kBT_pN_nm
        
        temp_kBT_pN_nm= 1.38e-23*(Temperature_C+273)*1.e12*1.e9
        return Temperature_C
    def key_pressed_YS(self, ev):
        """
        Handle key press events.

        Parameters:
        ev (QEvent): Key event.
        """
        
      
        if keys_mapping[ev.key()]=='D':
            self.btn_next_file()
        elif keys_mapping[ev.key()]=='A':
            self.btn_prev_file()
        elif keys_mapping[ev.key()]=='S':
            self.save_sess_log()
        elif keys_mapping[ev.key()]=='Space':
            self.btn_skip_file()