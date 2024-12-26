#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:04:12 2024

@author: yogehs
"""
from scipy import signal

from scipy.optimize import curve_fit
#import tweezepy as tp
import numpy as np
from nptdms import TdmsFile #from nptdms import tdms  # pip install nptdms
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

from scipy.stats import norm
import csv 
import os
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d
import matplotlib
from decimal import Decimal
from math import log10, floor

import gc
import chime 
save_out_root_path  = "/Users/yogehs/Documents/MS_thesis_all_files /analysis_files/rupture_force_data_september_22/output_anal_10"

#%% tdmfile input 
def cal_rupture_angle(x,y,L,R):
    #computes the angle at a particular point, 
    #returns the angle in degrees  
    temp = np.sqrt(x**2+y**2)
    angle  = (temp/(L+R))
    angle = np.rad2deg(np.arcsin(angle))
    return(angle)
def find_te_tdms_index(fname,ori_path):
#this function read hte metadata and return the te in s
    if os.getcwd() != (ori_path) : os.chdir(ori_path)
    tdms_file = TdmsFile.read_metadata(fname)  
    for group in tdms_file.groups():
        te = float(group.properties['Camera.Exposure time (ms)'])
        break
    te *= 0.001 # to transform the data into s
    
    return (te)


def round_sig(x, sig=4):
    if x==0:return x
    else:return round(x, sig-int(floor(log10(abs(x))))-1)
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

#this function returns the coordinates of the anchor point via the contour estimation 
#method, i could use this to solve the lenght distribution. 

def anchor_point_contour_cal(X1,Y1,n2= 30 ,levels = 10):

    #X1 = xx ;Y1 = yy
    ax = plt.gca()
    print(len(X1[::n2]))
    plt.figure(1,figsize=(6,6),dpi = 100)
    sns.kdeplot(x=X1[::n2], y=Y1[::n2], color='k', levels=levels, alpha=0.3, ax=ax, linestyles="--") #    sns.heatmap(X1[::30], Y1[::30], ax=ax)
    contourdata=[]
    #plt.show()
    for i,c in enumerate(ax.collections):     
        if isinstance(c, matplotlib.collections.LineCollection):
            if len(c.get_segments())>0:
                v = c.get_segments()[0]; cv=centeroidnp(v)                  
                print("contour:", i, " nb of points", len(v), "centroid=", cv)
                ax.plot(list(zip(*v))[0], list(zip(*v))[1], color='k', alpha=0.3, label="Outer Contour", linewidth=2.0)
                ax.scatter(cv[0], cv[1], color='k', alpha=0.3, label="Contour center")
                contourdata.append((i, len(v), cv))
                break
    ax.scatter(np.mean(X1), np.mean(Y1), color='r', alpha=0.3, label='center+label1')
    #plt.xlim(-500,500);plt.ylim(-500,500)
    print(contourdata)
    return contourdata[0][2]
    #print("this is the value",contourdata[0])
    #print("np.mean(X1), np.mean(Y1)",np.mean(X1), np.mean(Y1))


def track(d0, i):    # retrieves coordinates / data of bead i
    X = d0['ROI{:04d}'.format(i)+' X (nm)'][:]
    Y = d0['ROI{:04d}'.format(i)+' Y (nm)'][:]
    Z = d0['ROI{:04d}'.format(i)+' Z (nm)'][:]
    T = d0['Time (ms)'][:]
    T_s=(1/1000.0)*T

    MinLUT = d0['ROI{:04d}'.format(i)+' MinLUT'][:]
    SSIM = d0['ROI{:04d}'.format(i)+' SSIM'][:]
    return (X,Y,Z, MinLUT, SSIM,T_s)

def events(d2):  # retrieve events from tdms file Time/Name/Data
    T=d2['Time (ms)'][:]
    N=d2['Name'][:]
    D=d2['Data'][:]
    return (T,N,D)

def MakeGraphXYall(d0, imax, n, refrms, name, outname,  OutFormat, SaveGraph, corr=False, Xr0avg=None, Yr0avg=None, pprint=False):   
# Displays all XY traj with number and SD on the same graph
    if corr: name=name+"_Corr"
    figname=name+'_XYall'
    plt.figure(figname, figsize=(6,6), dpi=100); ax = plt.gca()
    xmin=-1000; xmax=350000; ymin=-1000; ymax=250000
    for i in range(1,imax+1):
   #     (X,Y,Z,MinLUT,SSIM255)=track(d0, i)
        X = d0['ROI{:04d}'.format(i)+' X (nm)'][:]
        Y = d0['ROI{:04d}'.format(i)+' Y (nm)'][:]
        if corr: X=X-Xr0avg; Y=Y-Yr0avg
        xx=X[::n]; yy=Y[::n]
        ds=np.sqrt((X-X.mean())**2+(Y-Y.mean())**2)
        dsmean=ds.mean()
        if np.isnan(dsmean): lab=str(i)+'/'+'Nan'
        else: lab=str(i)+'/'+str(int(dsmean))
        if dsmean<refrms: lab=lab+'ref'
        if pprint: print('plot xy traj:'+lab)
        ax.scatter(xx, yy, marker='.', alpha=0.5, label=str(i))
        plt.text(xx[-1], yy[-1], lab, fontsize=6)
    ax.axis([xmin, xmax, ymin, ymax])
    if corr: ax.axis([xmin-np.mean(Xr0avg), xmax-np.mean(Xr0avg), ymin-np.mean(Yr0avg), ymax-np.mean(Yr0avg)])
    print(outname,figname,OutFormat)
    if SaveGraph: plt.savefig(outname+figname+OutFormat, transparent=True)


def BuildP(T, TE, NE, DE):      # builds power time trace form event times
    I=(NE=='SigGen.Power (%)')
    TP=TE[I]; DP=DE[I]
    print('Build Power trace:', len(TP), 'events') #   print(TP)
    P=np.zeros(len(T)); j1=0
    for i in range(1,len(TP)):
        j=int(np.where(T==TP[i])[0][0])
        P[j1:j]=DP[i-1]; j1=j
    return P


def timet(d0, Graph):    # retrieves time trace from tdms file and measures time step
    T = d0['Time (ms)'][:]
    print('Total points:', len(T))
    dT=np.gradient(T)[(np.gradient(T)>0) & (np.gradient(T)<1000)]
    print('Total points gradient:', len(dT))
    avfreq = 1000/np.mean(dT)
    print('dTrawhisto (frame time ms)', np.amin(dT), np.amax(dT), ' average frequency (Hz)', "%.3f" % avfreq)   
    if Graph:
        figname='dTrawhisto'
        plt.figure(figname, figsize=(6,6), dpi=100); ax = plt.gca()
        ax=sns.distplot( dT, kde=False, bins=np.linspace(0.,20.,num=200) )
        ax.set_xlabel("Time step (ms)");  ax.set_ylabel("Number"); ax.set_yscale('log')
        plt.ylim(0.1, 1.e7)
    return T, avfreq

def Rollavg(Z, N, mode):      # rolling average on N time steps
 #   Zavg=np.convolve(Z, np.ones((N,))/N, mode=mode)
    Zavg = uniform_filter1d(Z, size=N, mode=mode)
 #   n=int(N/2.); Zavg[range(n)]=Zavg[n+1]; Zavg[range(-1,-n-1,-1)]= Zavg[-n]
 #   if mode=='valid': Z=Z[0: len(Z)-N+1];# Z=Z[0: len(Z)-N+1]
    return Z, Zavg



def data_dict(data,header):
    temp_dict = {}
    for i in range(len(data)):
        temp_dict[header[i]] = data[i]
    return(temp_dict)

def create_csv_file_header(file ,hc ):
    #csv_file = tdms_file_name[:8] +".csv"

    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hc)

        csv_dict = data_dict(hc,hc)
        writer.writerow(csv_dict)
#%% RUp imporatnta fUnctions 

def rupture_probe_func(xx,std_level = 4):
    count = 0
    NZrefavg = 5000
    xx,roll_x  = Rollavg(xx, NZrefavg, mode)
    roll_std_x = np.sqrt((xx-roll_x)**2)
    rupture_level =roll_std_x.mean()- std_level *np.std(roll_std_x)
    temp_pos = np.nan 
    for i in range(len(xx)):
        if xx[i] < rupture_level:
            temp_pos = i ;break
        
    return(temp_pos)
def rupture_probe_func1(xx,std_level = 4):
    count = 0
    xx,roll_x  = Rollavg(xx, NZrefavg, mode)
    roll_std_x = np.sqrt((xx-roll_x)**2)
    rupture_level =roll_std_x.mean()- std_level *np.std(roll_std_x)
    temp_pos = np.nan 
    for i in range(len(xx)):
        if xx[i] > rupture_level:
            print(xx[i],rupture_level)
            temp_pos = i ;break
        
    return(temp_pos,rupture_level)



def island_props(v):
    # Get one-off shifted slices and then compare element-wise, to give
    # us a mask of start and start positions for each island.
    # Also, get the corresponding indices.
    mask = np.concatenate(( [True], v[1:] != v[:-1], [True] ))
    loc0 = np.flatnonzero(mask)

    # Get the start locations
    loc = loc0[:-1]

    # The values would be input array indexe by the start locations.
    # The lengths woul be the differentiation between start and stop indices.
    return v[loc], loc, np.diff(loc0)
def zero_force_pos_func(P0):
    value, location, lengths = island_props(P0)
    zero_force_pos = np.empty([0],dtype=int)
    for i in np.where(value==0)[0]:
        temp = np.arange(location[i],location[i]+lengths[i],1)
        zero_force_pos = np.concatenate((zero_force_pos,temp))
    return(zero_force_pos)

def pre_rup_func(pos_arr,rupture_pos):
    
    temp = []
    for t in pos_arr:
        if t<rupture_pos:
            temp.append(t)
    temp = np.array(temp)
    return(temp)
    #return(pre_rupture_force_array)
#power spectrum fuctions


def bead_extract_from_df(log_afs_df,i):
    bref =log_afs_df.iloc[i]['B_ref']
    
    if isinstance(bref, str):
        bref= np.array([int(e) if e.isdigit() else e for e in bref.split('_')])
    else:
        bref= np.array([bref])
        print( "\n check ,reference beads array : " , bref)
    

    b = log_afs_df.iloc[i]['B']

    if isinstance(b, str):
        b= np.array([int(e) if e.isdigit() else e for e in b.split(',')])
    else:
        b= np.array([b])
        print( "\n check ,beads array : " , b)
    return(b,bref)
def compute_rup_force_trad(b,P0,T_s,ramp_loc,pcov,alpha,rupture_pos,temp_label = " "):
        
    rupture_force = np.nan;loading_rate= np.nan;load_time = np.nan;pow_per_sec = np.nan
    
    for j in range(len(ramp_loc)):
        (tx,ty) = ramp_loc[j]
        
        if rupture_pos >tx and rupture_pos<ty:
            pow_per_sec = ((P0[ty] - P0[tx]) / (T_s[ty] - T_s[tx]));loading_rate = pow_per_sec * alpha
            loading_rate = round_sig(loading_rate,6)
            load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
            rupture_force_ld = loading_rate * load_time
            rupture_force = round_sig(alpha * P0[rupture_pos],6)
                
        

    csv_file = "trad_anal_10_master_rup_force_spec.csv"
    alpha = round_sig(alpha,6);p_error = round_sig( np.sqrt(np.diag(pcov))[0],4)
    #load_time = round_sig(load_time,4);loading_rate = round_sig(loading_rate,4);
    #rupture_force = round_sig(rupture_force,4)


    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_csv_rup)
        csv_dict = data_dict([temp_label,'step', b ,alpha, p_error,pow_per_sec,load_time,loading_rate,rupture_force],header_csv_rup)
        writer.writerow(csv_dict)


def faxen_correction_glob(D_Dt_glob,L_t_new,axis,bead_radius):
    
    friction0 = 6*np.pi*1.e-9*bead_radius       # units pN.s/nm
    friction_new = friction0 / ( 1 - (9/16)*(bead_radius/(L_t_new+bead_radius)) + (1/8)*(bead_radius/(L_t_new+bead_radius))**3 )
    
    #friction = friction0 / ( 1 - (9/8)*(bead_radius/(L_t_new+bead_radius)) + (1/2)*(bead_radius/(L_t_new+bead_radius))**3 )
    
    D_t_new = kBT_pN_nm/friction_new
    D_new = D_t_new *D_Dt_glob
    #print(D_Dt_glob,D_t_new)
    #print("faxen corrected for D_glob")
    return(D_new)


#%% Spectrums

def Spectrum_l(xx, axis, p1Zo, fs, label, display,plot_fig_name, save_path = save_out_root_path,ramp_time=None,build_time = None,exportname=None, axtable=None):
    display = True ; SaveGraph = True
    friction0 = 6*np.pi*1.e-9*bead_radius       # units pN.s/nm
    
    if axis=='XY' or axis=='R':
        friction = friction0 / ( 1 - (9/16)*(bead_radius/(p1Zo+bead_radius)) + (1/8)*(bead_radius/(p1Zo+bead_radius))**3 )
    elif axis=='Z':
        friction = friction0 / ( 1 - (9/8)*(bead_radius/(p1Zo+bead_radius)) + (1/2)*(bead_radius/(p1Zo+bead_radius))**3 )
 
    if axis=='XY' or axis=='Z':
        #print('Spectrum Model Schäffer')
        def FitSpectrum(x, *p): return p[1]/(np.pi**2)/( x**2 + (p[0]*p[1]/(2*np.pi*kBT_pN_nm))**2 )   # schaffer 2015
    if axis=='R':
        #print('Spectrum Model Sitter')
        def FitSpectrum(x, *p): return p[1]/(2*np.pi**2)/( x**2 + (p[0]*p[1]/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015

    f, Pxx_spec = signal.periodogram(xx, fs, scaling='density') 
    #print("normal spec fmax",fmax_Hz)
    Pxxt_spec= Pxx_spec[(f>fmin_Hz)&(f<fmax_Hz)]; ft=f[(f>fmin_Hz)&(f<fmax_Hz)]
    nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
    
    for m in range(nbins-1):
        u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
        Pbins[m]=np.mean(u)
        dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
        
    if display:        
        #print("FUCK")
        figname = label
        plt.figure(figname+'bis', figsize=(6,6), dpi=100); ax = plt.gca()
        if axtable!=None:
            wax=[ax, axtable]
        else:
            wax=[ax]   
        for axbis in wax:
            axbis.errorbar(fbins, Pbins, dPbins, marker='.', c='b', alpha=0.2)
            axbis.set_ylim(1e-1, 1e5); axbis.set_xscale('log'); axbis.set_xlim(1e-2, 1e3); axbis.set_yscale('log')    
            axbis.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+label)    #     ax.set_ylabel('spectrum [nm²] '+label); 
            if ramp_time!=None:
                #axbis.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r')
                a = '%.2E' % Decimal(ramp_time[1])
                axbis.text(1e0, 1e4, 'Power per second  = '+a,c='r')

            if build_time!=None:axbis.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')

            if axbis==axtable: axbis.axes.get_xaxis().set_visible(False); axbis.axes.get_yaxis().set_visible(False)
    
    while True:
        try:
            pEq, pcovEq=curve_fit(FitSpectrum, ft, Pxxt_spec, p0=[1.e-3, 1.e5])
            eEq=np.sqrt(np.diag(pcovEq))
            break
        except RuntimeError:
            print("No convergence"); pEq=[0, 0]; eEq=[0, 0]; break 
            
            
    pEq[0] = np.abs(pEq[0])
    fc = pEq[0]*pEq[1]/(2*np.pi*kBT_pN_nm)
    
    #print('friction / friction_bulk =', friction/friction0, ' Corner frequency fc=', fc)
    FitPxxt_spec=FitSpectrum(ft, pEq[0], pEq[1])
    if display:
        for axbis in wax: axbis.plot(ft, FitPxxt_spec, c='r', alpha=0.8)
        if SaveGraph: 
            #print(SaveGraph,os.getcwd() ,os.getcwd() != (save_path))
            
            if os.getcwd() != (save_path) : os.chdir(save_path)
            plt.savefig(plot_fig_name+".png")
        if CloseAfterSave: plt.close()
    D_Dtheo = pEq[1]* (friction/kBT_pN_nm)
    
    #print('Spectrum'+label, ' k (pN/nm)=',"%.5f" %  (pEq[0]),' D (µm²/s)=',"%.3f" %  (pEq[1]*1.e-6), 'D/Dtheo=', pEq[1]*friction/kBT_pN_nm)
    temp_arr_round = [pEq[0], pEq[1], eEq[0], eEq[1], fc, friction ,D_Dtheo]
    '''
    #this is code to round the values; I am ignoring for now
    for i in range(len(temp_arr_round)):
        if ~np.isnan(temp_arr_round[i]):
            if i<1 or i>3:temp_arr_round[i] = round_sig(temp_arr_round[i],4)
            elif i==1 :temp_arr_round[i] = round_sig(temp_arr_round[i],8)
    '''
    return temp_arr_round

def Spectrum_l_diff_g(xx, axis, p1Zo, fs, label, display, fmin_Hz, fmax_Hz,D_global, save_path = save_out_root_path,ramp_time=None,build_time = None,exportname=None, axtable=None):
    display = True ; SaveGraph = True
    friction0 = 6*np.pi*1.e-9*bead_radius       # units pN.s/nm
    
    if axis=='XY' or axis=='R':
        friction = friction0 / ( 1 - (9/16)*(bead_radius/(p1Zo+bead_radius)) + (1/8)*(bead_radius/(p1Zo+bead_radius))**3 )
    elif axis=='Z':
        friction = friction0 / ( 1 - (9/8)*(bead_radius/(p1Zo+bead_radius)) + (1/2)*(bead_radius/(p1Zo+bead_radius))**3 )
 
    if axis=='XY' or axis=='Z':
        #print('Spectrum Model Schäffer')
        def FitSpectrum_global(x, k1,D_g =D_global): return D_g/(np.pi**2)/( x**2 + (k1*D_g/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015
    if axis=='R':
        #print('Spectrum Model Sitter')
        def FitSpectrum_global(x, k1,D_g=D_global): return D_g/(2*np.pi**2)/( x**2 + (k1*D_g/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015

    f, Pxx_spec = signal.periodogram(xx, fs, scaling='density') 
    Pxxt_spec= Pxx_spec[(f>fmin_Hz)&(f<fmax_Hz)]; ft=f[(f>fmin_Hz)&(f<fmax_Hz)]
    nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
    
    for m in range(nbins-1):
        u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
        Pbins[m]=np.mean(u)
        dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
        
    if display:        
        #print("FUCK")
        figname='PowerSpectrum_global'+label
        plt.figure(figname+'bis', figsize=(6,6), dpi=100); ax = plt.gca()
        if axtable!=None:
            wax=[ax, axtable]
        else:
            wax=[ax]   
        for axbis in wax:
            axbis.errorbar(fbins, Pbins, dPbins, marker='.', c='b', alpha=0.2)
            axbis.set_ylim(1e-3, 1e5); axbis.set_xscale('log'); axbis.set_xlim(1e-2, 1e5); axbis.set_yscale('log')    
            axbis.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+label);    #     ax.set_ylabel('spectrum [nm²] '+label); 
            if ramp_time!=None:
                axbis.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r');
                a = '%.2E' % Decimal(ramp_time[1])
                axbis.text(1e0, 1e4, 'Power per second  = '+a,c='r');

            if build_time!=None:axbis.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')

            if axbis==axtable: axbis.axes.get_xaxis().set_visible(False); axbis.axes.get_yaxis().set_visible(False)
    
    while True:
        try:
            pEq, pcovEq=curve_fit(FitSpectrum_global, ft, Pxxt_spec, p0=[1.e-3])
            eEq=np.sqrt(np.diag(pcovEq))
            break
        except RuntimeError:
            print("No convergence"); pEq=[np.nan, np.nan]; eEq=[np.nan, np.nan]; break 
            
            
    pEq = np.abs(pEq[0])
    fc = pEq*D_global/(2*np.pi*kBT_pN_nm)
    ft = np.append(ft,np.array([1e4,5e4]))    #just extending the limits of the predicted curve 
    #print('friction / friction_bulk =', friction/friction0, ' Corner frequency fc=', fc)
    FitPxxt_spec=FitSpectrum_global(ft, pEq, D_global)
    if display:
        for axbis in wax: axbis.plot(ft, FitPxxt_spec, c='r', alpha=0.8)
        if SaveGraph: 
            if os.getcwd() != (save_path) : os.chdir(save_path)
            #plt.savefig(plot_fig_name+".png")
        if CloseAfterSave: plt.close()
    D_Dtheo = D_global*friction/kBT_pN_nm ; dl = 0
    #print('Spectrum'+label, ' k (pN/nm)=',"%.5f" %  (pEq[0]),' D (µm²/s)=',"%.3f" %  (pEq[1]*1.e-6), 'D/Dtheo=', pEq[1]*friction/kBT_pN_nm)
    temp_arr_round = [pEq, D_global, eEq[0], dl,fc, friction ,D_Dtheo]
    for i in range(len(temp_arr_round)):
        #print(temp_arr_round[i])
          if ~np.isnan(temp_arr_round[i]):
            if i<1 or i>3:temp_arr_round[i] = round_sig(temp_arr_round[i],4)
            elif i==1 :temp_arr_round[i] = round_sig(temp_arr_round[i],8)

    return temp_arr_round


#Fit function to determine the force/power voltage. 

def fit_alpha(x,a):
    return a * x


def ramp_extraction(P,T_s):
    #first part of ramp extraction,;figuring out the final power of the ramp 
    start_pt  = 0;ramp_pts = [];count_r=0
    
    for i in range(len(P)-1):
        if P[i] - P[i+1] >0:
            if count_r >0:
                ramp_pts.append((start_pt,i))
            #plt.axvline(T[i])
            count_r+=1;start_pt = i+1    
    if count_r != len(ramp_pts):

        ramp_pts.append((start_pt,len(P)-1 ))
    #correcting the true start of the ramp. 
    #this is need to stop overestimating the loading rates. 
    #print(ramp_pts)
    for i in range(len(ramp_pts)):
        (rx,ry) = ramp_pts[i]
        P_r =P[rx:ry];T_r = T_s[rx:ry]
        #print("check",P_r)
        if len(P_r)>0:
            value, location, lengths = island_props(P_r)
            if value[0] == 0 :
                #rint("cleaning the ramp")
                rx += lengths[0] -1#location[1]-1
                ramp_pts[i] = (rx,ry)
    return(ramp_pts)




def FitHistoGaussian(Length, label, display, delta):

    def gauss_function(x, *p): return p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2))

    bins=np.linspace(-15000, 25000 ,num=20001)
    
    Hy, Hx = np.histogram(Length, bins=bins, density=True)
    Hymax=np.argmax(Hy); xHymax=bins[Hymax]   #    print(Hymax, xHymax)
    indxfit=np.abs(Hx-xHymax)<delta
    Hxt=Hx[indxfit]; Hyt=Hy[indxfit[1:]]
    if len(Hxt)>len(Hyt): Hxt=Hxt[1:]
    pEq, pcovEq=curve_fit(gauss_function, Hxt, Hyt, p0=[1., xHymax,20])
    pEq[2]=np.abs(pEq[2])
    FitHy=gauss_function(Hxt, pEq[0], pEq[1], pEq[2])
    print(label+'Mod',xHymax, '  Gaussian fit: Amp=', "%.3f" % pEq[0], 'Avg=', "%.3f" % pEq[1] , 'SD=', "%.3f" % abs(pEq[2]))
    return pEq[0], pEq[1], pEq[2], xHymax


def FitSpectrum_plot_global(x,kBT_pN_nm, *p): return p[1]/(2*np.pi**2)/( x**2 + (p[0]*p[1]/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015
def DENn(f,fc,fs,n): return 1 + (np.abs(f+n*fs)/fc)**2  
def SINCn(f,te,fs,n): return np.sinc(te*np.abs(f+n*fs))**2 
 
def FitSpectrumGenNew(f, k, D): 
    #te - exp time in seconds 
    #te = 0.0029#; fs = 59.82279832617854
    #print(te,"check the te in s read from meta data and fs : ",fs)
    fc = D*k/(2*np.pi*kBT_pN_nm) 
    PREF = 2*kBT_pN_nm**2/(D*k*k) 
    S=0 
    for n in [0, 1,2]: 
        #print(n)
    #for n in [0, 1]: 

        S+= (SINCn(f,te,fs,n)/DENn(f,fc,fs,n)) 
    return PREF*S   # with correction 
    #return PREF/DENn(f,fc,fs,0)  # no correction

def FitSpectrumGenDaldrop(f, *p): # k = p[0]; D = p[1] 
    return FitSpectrumGenNew(f, p[0], p[1])
def wrapper_glob_alias_spec(D_g):
    def temp_FitSpectrumGenNew(f, k, D =D_g):
        return temp_FitSpectrumGenNew(f, k, D)
    return temp_FitSpectrumGenNew

def FitSpectrumGenDaldrop_glob(f,k,D_g): # k = p[0]; D = p[1] 
    return FitSpectrumGenNew(f, k, D_g)



def Spectrum_l_alias(xx, axis, p1Zo, fs, label, display,plot_fig_name, save_path = save_out_root_path,ramp_time=None,build_time = None,exportname=None, axtable=None):
    display = True ; SaveGraph = True
    friction0 = 6*np.pi*1.e-9*bead_radius       # units pN.s/nm
    
    if axis=='XY' or axis=='R':
        friction = friction0 / ( 1 - (9/16)*(bead_radius/(p1Zo+bead_radius)) + (1/8)*(bead_radius/(p1Zo+bead_radius))**3 )
    elif axis=='Z' :
        friction = friction0 / ( 1 - (9/8)*(bead_radius/(p1Zo+bead_radius)) + (1/2)*(bead_radius/(p1Zo+bead_radius))**3 )
 
    f, Pxx_spec = signal.periodogram(xx, fs, scaling='density') 
    #print("alias spec fmax",fmax_Hz)

    Pxxt_spec= Pxx_spec[(f>fmin_Hz)&(f<fmax_Hz)]; ft=f[(f>fmin_Hz)&(f<fmax_Hz)]
    nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
    
    for m in range(nbins-1):
        u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
        Pbins[m]=np.mean(u)
        dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
        
    if display:        
        #print("FUCK")
        figname = label
        plt.figure(figname+'bis', figsize=(6,6), dpi=100); ax = plt.gca()
        if axtable!=None:
            wax=[ax, axtable]
        else:
            wax=[ax]   
        for axbis in wax:
            axbis.errorbar(fbins, Pbins, dPbins, marker='.', c='b', alpha=0.2)
            axbis.set_ylim(1e-1, 1e5); axbis.set_xscale('log'); axbis.set_xlim(1e-2, 1e3); axbis.set_yscale('log')    
            axbis.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+label);    #     ax.set_ylabel('spectrum [nm²] '+label); 
            if ramp_time!=None:
                axbis.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r');
                a = '%.2E' % Decimal(ramp_time[1])
                axbis.text(1e0, 1e4, 'Power per second  = '+a,c='r');

            if build_time!=None:axbis.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')

            if axbis==axtable: axbis.axes.get_xaxis().set_visible(False); axbis.axes.get_yaxis().set_visible(False)
    
    while True:
        try:
            pEq, pcovEq=curve_fit(FitSpectrumGenDaldrop, ft, Pxxt_spec, p0=[1.e-3, 1.e5])
            eEq=np.sqrt(np.diag(pcovEq))
            break
        except RuntimeError:
            print("No convergence"); pEq=[0, 0]; eEq=[0, 0]; break 
            
            
    pEq[0] = np.abs(pEq[0])
    fc = pEq[0]*pEq[1]/(2*np.pi*kBT_pN_nm)
    
    #print('friction / friction_bulk =', friction/friction0, ' Corner frequency fc=', fc)
    FitPxxt_spec=FitSpectrumGenNew(ft, pEq[0], pEq[1])
    if display:
        for axbis in wax: axbis.plot(ft, FitPxxt_spec, c='r', alpha=0.8)
        if SaveGraph: 
            #print(SaveGraph,os.getcwd() ,os.getcwd() != (save_path))
            
            if os.getcwd() != (save_path) : os.chdir(save_path)
            plt.savefig(plot_fig_name+".png")
        if CloseAfterSave: plt.close()
    D_Dtheo = pEq[1]*friction/kBT_pN_nm
    
    #print('Spectrum'+label, ' k (pN/nm)=',"%.5f" %  (pEq[0]),' D (µm²/s)=',"%.3f" %  (pEq[1]*1.e-6), 'D/Dtheo=', pEq[1]*friction/kBT_pN_nm)
    temp_arr_round = [pEq[0], pEq[1], eEq[0], eEq[1], fc, friction ,D_Dtheo]
    
    #this is code to round the values; I am ignoring for now
    for i in range(len(temp_arr_round)):
        if ~np.isnan(temp_arr_round[i]):
            if i<1 or i>3:temp_arr_round[i] = round_sig(temp_arr_round[i],4)
            elif i==1 :temp_arr_round[i] = round_sig(temp_arr_round[i],8)
    
    return temp_arr_round

def Spectrum_l_D_global(xx, axis, p1Zo, fs, label, display,plot_fig_name,D_global, save_path = save_out_root_path,ramp_time=None,build_time = None,exportname=None, axtable=None):
    display = True ; SaveGraph = True
    friction0 = 6*np.pi*1.e-9*bead_radius       # units pN.s/nm
    if axis=='XY' or axis=='R':
        friction = friction0 / ( 1 - (9/16)*(bead_radius/(p1Zo+bead_radius)) + (1/8)*(bead_radius/(p1Zo+bead_radius))**3 )
    elif axis=='Z':
        friction = friction0 / ( 1 - (9/8)*(bead_radius/(p1Zo+bead_radius)) + (1/2)*(bead_radius/(p1Zo+bead_radius))**3 )
 
    if axis=='XY' or axis=='Z':
        print('Spectrum Model Schäffer')
        def FitSpectrum_global(x, k1,D_g =D_global):
            print(D_g,axis)

            return D_g/(np.pi**2)/( x**2 + (k1*D_g/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015
    if axis=='R':
        #print('Spectrum Model Sitter')
        def FitSpectrum_global(x, k1,D_g=D_global): 
            #print(D_g,axis)
            return D_g/(2*np.pi**2)/( x**2 + (k1*D_g/(2*np.pi*kBT_pN_nm))**2 )   # Sitter 2015
        
    D_Dtheo = D_global*friction/kBT_pN_nm ; dl = 0

    f, Pxx_spec = signal.periodogram(xx, fs, scaling='density') 
    #print("GLobal spec F_max ",fmax_Hz,f)
    
    Pxxt_spec= Pxx_spec[(f>fmin_Hz)&(f<fmax_Hz)]; ft=f[(f>fmin_Hz)&(f<fmax_Hz)]
    #print(ft)
    nbins=101; fbins=np.logspace(-2,3,nbins); Pbins=np.ones(nbins); dPbins=np.zeros(nbins)
    
    for m in range(nbins-1):
        u=Pxx_spec[(f>=fbins[m])&(f<fbins[m+1])]
        Pbins[m]=np.mean(u)
        dPbins[m]=np.std(u)/np.sqrt(len((u[~np.isnan(u)])))
        
    if display or True:        
        #print("FUCK")
        #print(label)
        figname = label
        plt.figure(figname+'bis', figsize=(6,6), dpi=100); ax = plt.gca()
        if axtable!=None:
            wax=[ax, axtable]
        else:
            wax=[ax]   
        for axbis in wax:
            axbis.errorbar(fbins, Pbins, dPbins, marker='.', c='b', alpha=0.2)
            axbis.set_ylim(1e-3, 1e5); axbis.set_xscale('log'); axbis.set_xlim(1e-2, 1e5); axbis.set_yscale('log')    
            axbis.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [nm²/Hz] '+label);    #     ax.set_ylabel('spectrum [nm²] '+label); 
            axbis.text(1e0, 4e4, 'Dglobal _DT = '+str(D_Dtheo),c='r');

            if ramp_time!=None:
                axbis.text(1e0, 5e4, 'Ramp time = '+str(ramp_time[0]),c='r');
                a = '%.2E' % Decimal(ramp_time[1])
                axbis.text(1e0, 1e4, 'Power per second  = '+a,c='r');

            if build_time!=None:axbis.text(1e0, 5e4, 'PSD build time = '+str(build_time),c='r')

            if axbis==axtable: axbis.axes.get_xaxis().set_visible(False); axbis.axes.get_yaxis().set_visible(False)
    
    while True:
        try:
            pEq, pcovEq=curve_fit(FitSpectrum_global, ft, Pxxt_spec, p0=[1.e-2])
            #print("pEq",pEq)
            eEq=np.sqrt(np.diag(pcovEq))
            break
        except RuntimeError:
            print("No convergence"); pEq=[np.nan, np.nan]; eEq=[np.nan, np.nan]; break 
            
            
    pEq = np.abs(pEq[0])
    fc = pEq*D_global/(2*np.pi*kBT_pN_nm)
    ft_global = np.append(ft,np.array([1e4,5e4]))    #just extending the limits of the predicted curve

    FitPxxt_spec=FitSpectrum_plot_global(ft_global,kBT_pN_nm, pEq, D_global)
    if display:
        for axbis in wax: axbis.plot(ft_global, FitPxxt_spec, c='r', alpha=0.8)
        if SaveGraph: 
            if os.getcwd() != (save_path) : os.chdir(save_path)
            plt.savefig(plot_fig_name+".png")
            #if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
            #plt.savefig(plot_fig_name+".png")
        if CloseAfterSave: plt.close()
    #print('Spectrum'+label, ' k (pN/nm)=',"%.5f" %  (pEq[0]),' D (µm²/s)=',"%.3f" %  (pEq[1]*1.e-6), 'D/Dtheo=', pEq[1]*friction/kBT_pN_nm)
    temp_arr_round = [pEq, D_global, eEq[0], dl,fc, friction ,D_Dtheo]
    
    '''
    
    for i in range(len(temp_arr_round)):
        #print(temp_arr_round[i])
          if ~np.isnan(temp_arr_round[i]):
            if i<1 or i>3:temp_arr_round[i] = round_sig(temp_arr_round[i],4)
            elif i==1 :temp_arr_round[i] = round_sig(temp_arr_round[i],8)
    '''
    return temp_arr_round

def find_D_global_alias_all_ramps(trace_arr,arg_list_d_global_ramp,n =20):
    first_spec_otf = True
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,rupture_pos,fs,ramp_loc,L_t_new] = arg_list_d_global_ramp

    print(d_dt_max,d_dt_min,f_c_threshold_global)
    print("Global D all ramps search initiaited ")

    if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)

    try: 
        os.mkdir(refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global))
        save_find_D_global_path = save_out_psd_path +'/'+ refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global)
    except OSError as error: 
        print(error) 
        save_find_D_global_path = save_out_psd_path +'/'+ refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global)
    
    
    if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
    csv_file_spec = 'laurent_D_global_otf_Fc_thresh_'+str(f_c_threshold_global)+"_power_spec_selected.csv"
    csv_file_spec_all = 'laurent_D_global_otf_Fc_thresh_'+str(f_c_threshold_global)+"_power_spec_all.csv"
    D_global_arr = [False,0.0,0.0];D_X_array = [];D_Y_array = []

    #create_csv_file_header(csv_file_spec,header_csv_power_spec)
    for (tx,ty) in ramp_loc:
        print("ramp,",tx,ty)   
        start_time = tx+int(5* fs) ; end_time =  ty - int( 5* fs)
        pos_a = start_time ; 
        axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
    
        while pos_a < end_time and pos_a<rupture_pos:
            pos_b = pos_a +int(n*fs)
            D_arr = np.zeros(2); D_Dt_arr = np.zeros(2)
            avg_power_1 = (P0[pos_a]+P0[pos_b])/2
            fc_l_arr = np.zeros(2)
            temp_force = 0
            for i in range(2):
    
                trace = trace_arr[i]
                temp_label = refname+"_PowerSpectrum_alias_avg_P"+str(avg_power_1)+'_' + str(n) +"_secs_" +axis_arr[i]+'_D_g_otf_Fc_thresh_'+str(f_c_threshold_global)
                
                temp_out_spec_l= Spectrum_l_alias(trace[pos_a:pos_b], temp_arr[i], np.mean(zz[pos_a:pos_b]), fs,temp_label, display_psd,plot_fig_name,save_find_D_global_path) 
                [k_l, D_arr[i], dk_l, dD_l, fc_l_arr[i], friction_l,D_Dt_arr[i]]= temp_out_spec_l
                
                if faxen_correction_glob_bool:
                    print("faxenCorrected glob",D_arr[i])
                    D_arr[i] = faxen_correction_glob(D_Dt_arr[i],L_t_new,axis_arr[i],bead_radius)
                    #print(D_arr[i])
                
                force_l =round_sig( k_l * (np.mean(L_t[pos_a:pos_b])+bead_radius),5)
                
                temp_force+=force_l
    
                if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
                if first_spec_otf: 
                    create_csv_file_header(csv_file_spec_all,header_csv_power_spec)
                    first_spec_otf = False
        
                
                with open(csv_file_spec_all, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf find alais D global", b ,avg_power_1,n,fs,axis_arr[i],k_l,dk_l,
                                                   D_arr[i]*1.e-6,D_Dt_arr[i],fc_l_arr[i],force_l],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
    
            temp_force /=2
    
            if fc_l_arr[0] < f_c_threshold_global and not(D_global_arr[0]):
                if D_Dt_arr[0] >=d_dt_min and D_Dt_arr[0] <=d_dt_max:
                    D_X_array.append(D_arr[0])
                    if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
    
                    with open(csv_file_spec, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                        csv_dict_spec = data_dict([refname,"otf find alias D global", b ,avg_power_1,n,fs,axis_arr[0],k_l,dk_l,
                                                       D_arr[0]*1.e-6,D_Dt_arr[0],fc_l_arr[0],temp_force],header_csv_power_spec)
                        writer.writerow(csv_dict_spec)
    
                
            if fc_l_arr[1] < f_c_threshold_global and not(D_global_arr[0]):
       
                if D_Dt_arr[1] >=d_dt_min and D_Dt_arr[1] <=d_dt_max:
                    D_Y_array.append(D_arr[1])
                    if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
    
                    with open(csv_file_spec, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                        csv_dict_spec = data_dict([refname,"otf find alias D global", b ,avg_power_1,n,fs,axis_arr[1],k_l,dk_l,
                                                       D_arr[1]*1.e-6,D_Dt_arr[1],fc_l_arr[1],temp_force],header_csv_power_spec)
                        writer.writerow(csv_dict_spec)
    
            pos_a+=int( 0.5*n * fs)

    print("final DX", D_X_array);    print("final Dy", D_Y_array)        
        
    if len(D_X_array)>=1 and len(D_Y_array) >=1 :
        D_global_arr = [True , np.mean(D_X_array),np.mean(D_Y_array)]
        print("Global D all ramps alias found ")
    elif len(D_X_array)>=1 :
        D_global_arr = [True , np.mean(D_X_array),np.mean(D_X_array)]
        print("Global D allonly in X ramps alias found ")

    elif len(D_Y_array)>=1 :
        D_global_arr = [True , np.mean(D_Y_array),np.mean(D_Y_array)]
        print("Global D allonly in Y ramps alias found ")


    else :
        D_global_arr = [False , 0,0]
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

        print("D_G not all ramps found: "+refname     )

        with open(fcuk_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_error_log)

            csv_dict = data_dict([refname,"D_g not all ramps found_"+str(len(D_X_array))+"_"+str(len(D_Y_array))],header_error_log)
            writer.writerow(csv_dict)  
    return D_global_arr
      

#%% OTF functions

def on_the_fly_calib_felix(trace_arr,arg_list_spec,n = 20):
    #print(arg_list_spec)
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,min_L_t_bforce,min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor] = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    pos_b = rupture_pos - int(otf_end_T_thresh *fs) ; pos_a = pos_b - int(n * fs)
    axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
    f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
    D_arr =np.array([np.nan,np.nan]);force_arr = np.array([np.nan,np.nan])

    if pos_a > tx:
        temp_force = 0
        avg_power_1 = (P0[pos_a]+P0[pos_b])/2
        avg_power2 = np.average(P0[pos_a:pos_b])
        pow_per_sec = (P0[pos_b] - P0[pos_a])/(T_s[pos_b]-T_s[pos_a])
        ramp_time = T_s[ty] - T_s[tx]

        for i in range(2):

            trace = trace_arr[i]
            temp_label = refname+"_PowerSpectrum_" + str(n) +"_secs_" +axis_arr[i]+'_laurent_felix_otf'
            plot_fig_otf_felix = plot_fig_name+"_felix_otf_"+axis_arr[i]
            temp_out_spec_l= Spectrum_l(trace[pos_a:pos_b], temp_arr[i], np.mean(zz[pos_a:pos_b]),
                                        fs,temp_label, display_psd,plot_fig_otf_felix,save_out_psd_path,[ramp_time,pow_per_sec]) 
            [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
            
            if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                force_arr[i] = k_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                f_c_arr[i] = fc_l ; F_error_arr[i] = dk_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                D_dt_arr[i] = D_Dtheo_l;D_arr[i] =  D_l
            print("OTF flexi bead : radius"  , bead_radius)
            
            #print("On the fly force in axis ",axis_arr[i]," : ",force_l)
            
            D_l *=1.e-6
            
            if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)
            csv_file_spec = "anal_10_felix_otf_power_spec.csv"
            with open(csv_file_spec, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                csv_dict_spec = data_dict([refname,"otf felix ", b ,avg_power_1,n,fs,axis_arr[i],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[i]],header_csv_power_spec)
                writer.writerow(csv_dict_spec)
                f.flush()

           
            
        temp_force = np.nanmean(force_arr);#otf_calib_force_arr.append(temp_force)     
        #computing the quantifiables accordinglt
        #pow_per_sec = np.nan
        
        csv_file = "anal_10_felix_otf_rup_force_spec.csv"
        alpha = (temp_force / avg_power_1);alpha = round_sig(alpha,4);p_error = np.nan ; 
        load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
        loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)
        rupture_force = loading_rate * load_time
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_csv_rup_felix)
            csv_dict = data_dict([refname,"otf", b ,alpha,np.nanmean(f_c_arr),np.nanmean(D_dt_arr),np.nanmean(D_arr),kBT_pN_nm/friction_l,pow_per_sec,load_time,
                                  loading_rate,rupture_force,np.nanmean(F_error_arr),np.nanmean(L_t[pos_a:pos_b]),L_T_bforce_avg,L_T_aforce_avg],header_csv_rup_felix)
            #writer.writerow(csv_dict)
            f.flush()

    
    print("cest felix otf fini")
    D_avg_bool = np.nanmean(f_c_arr) > f_c_threshold_global or np.round(np.nanmean(f_c_arr),10)<=0
    return( D_avg_bool)



def on_the_fly_calib_felix_alias(trace_arr,arg_list_spec,n = 20):
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,min_L_t_bforce,
     min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y]  = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    pos_b = rupture_pos - int(otf_end_T_thresh *fs) ; pos_a = pos_b - int(n * fs)
    axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
    f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
    D_arr =np.array([np.nan,np.nan]);force_arr = np.array([np.nan,np.nan])

    if pos_a > tx:
        temp_force = 0
        avg_power_1 = (P0[pos_a]+P0[pos_b])/2
        avg_power2 = np.average(P0[pos_a:pos_b])
        pow_per_sec = (P0[pos_b] - P0[pos_a])/(T_s[pos_b]-T_s[pos_a])
        ramp_time = T_s[ty] - T_s[tx]


        for i in range(2):

            trace = trace_arr[i]
            temp_label = refname+"_PowerSpectrum_alias_" + str(n) +"_secs_" +axis_arr[i]+'_laurent_felix_otf'
            plot_fig_alais  = plot_fig_name+"_alias_felix_otf_"+axis_arr[i]
            temp_out_spec_l= Spectrum_l_alias(trace[pos_a:pos_b], temp_arr[i], np.mean(zz[pos_a:pos_b]),
                                              fs,temp_label, display_psd,plot_fig_alais,save_out_psd_path,[ramp_time,pow_per_sec]) 
            [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
            
            if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                force_arr[i] = k_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                f_c_arr[i] = fc_l ; F_error_arr[i] = dk_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                D_dt_arr[i] = D_Dtheo_l;D_arr[i] =  D_l
            else:
                print("felix alias the axis fit is bad : ", axis_arr[i],D_Dtheo_l)
            #print("On the fly force in axis ",axis_arr[i]," : ",force_l)
            
            D_l *=1.e-6
            
            if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)
            csv_file_spec = "anal_10_felix_otf_power_spec.csv"
            with open(csv_file_spec, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                csv_dict_spec = data_dict([refname,"otf_alias felix ", b ,avg_power_1,n,fs,axis_arr[i],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[i]],header_csv_power_spec)
                writer.writerow(csv_dict_spec)
                f.flush()

        csv_file = "anal_10_felix_otf_rup_force_spec.csv"
        load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
        xx = trace_arr[0];yy = trace_arr[1]
        avg_x = np.mean(xx[pos_a:pos_b]);avg_y = np.mean(yy[pos_a:pos_b]);
        avg_l_rup = np.mean(L_t[pos_a:pos_b]);max_l_t_rup =np.max(L_t[pos_a:pos_b]) 
        l_t_b_rup_ramp = np.mean(L_t[tx-int(2 * fs):tx])
        rupture_angle = cal_rupture_angle(avg_x,avg_y,avg_l_rup,bead_radius)

           
            
        temp_force =np.nanmean(force_arr);#otf_calib_force_arr.append(temp_force)     
        #computing the quantifiables accordinglt
        if np.isnan(temp_force):
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
    
    
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_rup_felix)
                csv_dict = data_dict([refname,"otf_alais", b ,np.nan,np.nanmean(f_c_arr),fs,np.nanmean(D_dt_arr),np.nanmean(D_arr),kBT_pN_nm/friction_l,pow_per_sec,load_time,
                                      np.nan,np.nan,np.nanmean(F_error_arr),l_t_b_rup_ramp,avg_l_rup,max_l_t_rup,rupture_angle,min_L_t_bforce,min_L_t_aforce,
                                      L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y],header_csv_rup_felix)
                writer.writerow(csv_dict)
                f.flush()

        else:
            
            pow_per_sec = np.nan
            pow_per_sec = (P0[pos_b] - P0[pos_a])/(T_s[pos_b]-T_s[pos_a])
            pow_per_sec = (P0[ty] - P0[tx])/(T_s[ty]-T_s[tx])
            print(temp_force)
            alpha = (temp_force / avg_power_1);alpha = round_sig(alpha,4);p_error = np.nan ; 
            
            loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)
            rupture_force = loading_rate * load_time
    
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
    
    
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_rup_felix)
                csv_dict = data_dict([refname,"otf_alais", b ,alpha,np.nanmean(f_c_arr),fs,np.nanmean(D_dt_arr),np.nanmean(D_arr),kBT_pN_nm/friction_l,pow_per_sec,load_time,
                                      loading_rate,rupture_force,np.nanmean(F_error_arr),l_t_b_rup_ramp,avg_l_rup,max_l_t_rup,rupture_angle,min_L_t_bforce,min_L_t_aforce,
                                      L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y],header_csv_rup_felix)
                writer.writerow(csv_dict)
                f.flush()

            
    
    print("cest felix otf alias fini")
    D_avg_bool = np.nanmean(f_c_arr) > f_c_threshold_global or np.round(np.nanmean(f_c_arr),10)<=0
    return( D_avg_bool)


def make_plot_function(arg_list_plot):
    [xx,yy,zz,L_t,T_s,non_0_force,rupture_pos,zero_force_pos,did_it_unbind,plot_fig_name,b] = arg_list_plot
    L_t_aforce = L_t[non_0_force] ; L_t_bforce = L_t[zero_force_pos]
    T_bf = T_s[:non_0_force[0]]
    plt.figure(1,figsize =(20,10));plt.clf()
    plt.title("Before force ")
    plt.scatter(T_bf,xx[:non_0_force[0]],s = 0.5 ,label = "X")
    plt.scatter(T_s[:non_0_force[0]],zz[:non_0_force[0]],s = 0.5 ,label = "Z")
    plt.scatter(T_bf,yy[:non_0_force[0]],s = 0.5 ,label = "Y")
    plt.legend()
    #plt.savefig(plot_fig_name+"bf_x_y_z_plt.png")

    if CloseAfterSave: plt.close()

    plt.figure(2,figsize =(20,10));plt.clf()
    plt.title("During force ")
    plt.scatter(T_s[non_0_force],xx[non_0_force],s = 0.5 ,label = "X")
    plt.scatter(T_s[non_0_force[0]:],zz[non_0_force[0]:],s = 0.5  ,label = "Z")
    plt.scatter(T_s[non_0_force],yy[non_0_force],s = 0.5 ,label = "Y")
    if did_it_unbind: 
        print("rupture_pos" , rupture_pos)
        plt.axvline(T_s[rupture_pos],label = "Bond rupture time")

    plt.legend();plt.ylim(-1000,1000)
    #plt.savefig(plot_fig_name+"af_x_y_z_plt.png")

    if CloseAfterSave: plt.close()

    plt.figure(4,figsize=(20,10));plt.clf()
    plt.title("length distribution over time and mean L(t) " + str(round_sig(L_t_bforce.mean(),2)) +" nm  for # %1.2d  "%(b))
    plt.scatter(T_s[zero_force_pos],L_t_bforce,s=0.5,label = "before force")
    plt.scatter(T_s[non_0_force],L_t_aforce,s=0.5,label = "during force")
    if did_it_unbind:plt.axvline(T_s[rupture_pos],label = "Bond rupture time")
    plt.xlabel("time");plt.ylabel("Lenght in nm ")
    ymin, ymax = plt.ylim()

    #plt.legend();plt.ylim(-150,500)
    plt.savefig(plot_fig_name+"_L_t .png")
    if CloseAfterSave: plt.close()

    plt.figure(3,figsize =(20,10));plt.clf()
    plt.title("Z trace for bead # %1.2d  "%(b))
    plt.scatter(T_s[:rupture_pos],zz[:rupture_pos])
    plt.ylim(ymin,ymax)
    if did_it_unbind:plt.axvline(T_s[rupture_pos],label = "Bond rupture time");plt.legend()
    plt.savefig(plot_fig_name+"_zplt.png")
    if CloseAfterSave: plt.close()

    plt.figure(50,figsize= (10,10));plt.clf()

    plt.hist(L_t_bforce,histtype = 'step',label= "before force",color = 'blue',density = True)
    plt.hist(L_t_aforce,histtype = 'step',label= "during force",color = 'red',density = True)
    plt.axvline(L_t_bforce.mean(),linestyle = '--',color ='blue' , label = "mean length before force ")
    plt.axvline(L_t_aforce.mean(),linestyle = '--',color = 'red' , label = "mean length during force ")
    plt.xlim(-150,500)
    plt.legend();plt.xlabel("length of tethers in nm")
    plt.title("lenght distribution before and during force")
    if CloseAfterSave: plt.close()

    plt.savefig(plot_fig_name+"l_t_histdist_.png")
    
    
    plt.figure(6,figsize= (10,10));plt.clf()
    plt.scatter(xx[:non_0_force[0]], yy[:non_0_force[0]],s = 0.8)
    plt.axis('square')
    #plt.xlim(-1000,1000);plt.ylim(-1000,1000)
    plt.title("SQUARE PROJECTION X-Y project for bead # %1.2d  "%(b))
    plt.savefig(plot_fig_name+"_sq_xyproj.png")
    if CloseAfterSave: plt.close()
 
    print("TETHER PLOT SUCESS")

def step_ramp_check(P0,T_s,ramp_loc,step_loc,save_out_tether_path):
    plt.figure(1212,figsize=(18,16))
    plt.scatter(T_s,P0)
    for j in range(len(ramp_loc)-1):
        (tx,ty) = ramp_loc[j]
        plt.scatter(T_s[tx:ty],P0[tx:ty],c='r')
    
    for j in range(len(step_loc)):
        (tx,ty) = step_loc[j]
        plt.scatter(T_s[tx:ty],P0[tx:ty],c='b')
    if os.getcwd() != save_out_tether_path : os.chdir(save_out_tether_path)
    plt.savefig("ramp_test_plot_check.png")

def on_the_fly_calib_felix_global(trace_arr,arg_list_spec,D_global_arr,n = 20):
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,min_L_t_bforce,
     min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y] = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    pos_b = rupture_pos - int(otf_end_T_thresh *fs) ; pos_a = pos_b - int(n * fs)
    axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
    f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
    D_arr =np.array([np.nan,np.nan]);force_arr = np.array([np.nan,np.nan])



    if pos_a > tx:
        temp_force = 0
        avg_power_1 = (P0[pos_a]+P0[pos_b])/2
        avg_power2 = np.average(P0[pos_a:pos_b])
        pow_per_sec = (P0[pos_b] - P0[pos_a])/(T_s[pos_b]-T_s[pos_a])
        ramp_time = T_s[ty] - T_s[tx]

        for i in range(2):

            trace = trace_arr[i]
            temp_label = refname+"_PowerSpectrum_" + str(n) +"_secs_" +axis_arr[i]+'_laurent_felix_otf_global'
            plot_fig_glob = plot_fig_name +"_global_felix_otf_"+axis_arr[i]
            temp_out_spec_l= Spectrum_l_D_global(trace[pos_a:pos_b], temp_arr[i], np.mean(zz[pos_a:pos_b]),
                                                 fs,temp_label, display_psd,plot_fig_glob,D_global_arr[i+1],save_out_psd_path,[ramp_time,pow_per_sec]) 
            [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
            if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                force_arr[i] = k_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                f_c_arr[i] = fc_l ; F_error_arr[i] = dk_l * (np.mean(L_t[pos_a:pos_b])+bead_radius)
                D_dt_arr[i] = D_Dtheo_l;D_arr[i] =  D_l
            else:
                print("the axis fit is bad : ", axis_arr[i])
            D_l *=1.e-6
            
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
            csv_file_spec = "anal_10_felix_otf_power_spec.csv"
            with open(csv_file_spec, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                csv_dict_spec = data_dict([refname,"otf felix D global", b ,avg_power_1,n,fs,axis_arr[i],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[i]],header_csv_power_spec)
                writer.writerow(csv_dict_spec)
                f.flush()

            if  F_error_arr[i]==np.inf :
                if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

                with open(fcuk_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_error_log)
    
                    csv_dict = data_dict([refname+"_"+axis_arr[i],"GLob non convergent"],header_error_log)
                    writer.writerow(csv_dict)
                    f.flush()


           
        
        temp_force = np.nanmean(force_arr);#otf_calib_force_arr.append(temp_force)     
        #computing the quantifiables accordinglt
        if np.isnan(temp_force):
            print("force is not converged in any axis")
        else :
            pow_per_sec = np.nan
            pow_per_sec = (P0[pos_b] - P0[pos_a])/(T_s[pos_b]-T_s[pos_a])
            
            csv_file = "anal_10_felix_otf_rup_force_spec.csv"
            alpha = (temp_force / avg_power_1);alpha = round_sig(alpha,4);p_error = np.nan ; 
            load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
            loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)
            rupture_force = loading_rate * load_time
            xx = trace_arr[0];yy = trace_arr[1]
            avg_x = np.mean(xx[pos_a:pos_b]);avg_y = np.mean(yy[pos_a:pos_b])
            avg_l_rup = np.mean(L_t[pos_a:pos_b]);max_l_t_rup =np.max(L_t[pos_a:pos_b]) 
            l_t_b_rup_ramp = np.mean(L_t[tx-int(2 * fs):tx])
    
            rupture_angle = cal_rupture_angle(avg_x,avg_y,avg_l_rup,bead_radius)
            
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
    
    
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_rup_felix)
                csv_dict = data_dict([refname,"otf global ", b ,alpha,np.mean(f_c_arr),fs,np.mean(D_dt_arr),np.mean(D_arr),kBT_pN_nm/friction_l,pow_per_sec,load_time,
                                      loading_rate,rupture_force,np.mean(F_error_arr),l_t_b_rup_ramp,avg_l_rup,max_l_t_rup,rupture_angle,min_L_t_bforce,min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,
                                      max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y],header_csv_rup_felix)
    
                writer.writerow(csv_dict)
                f.flush()

    print("cest global felix otf fini")


def find_D_global_alias(trace_arr,arg_list_d_global,n =20):
    first_spec_otf = True
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,rupture_pos,fs,tx,L_t_new] = arg_list_d_global

    print(d_dt_max,d_dt_min)
    print("Global D search initiaited ")

    if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)

    try: 
        os.mkdir(refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global))
        save_find_D_global_path = save_out_psd_path +'/'+ refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global)
    except OSError as error: 
        print(error) 
        save_find_D_global_path = save_out_psd_path +'/'+ refname +"_PSD_plots_D_global_Fc_thresh_"+str(f_c_threshold_global)
    
    
    if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
    csv_file_spec = 'laurent_D_global_otf_Fc_thresh_'+str(f_c_threshold_global)+"_power_spec_selected.csv"
    csv_file_spec_all = 'laurent_D_global_otf_Fc_thresh_'+str(f_c_threshold_global)+"_power_spec_all.csv"

    #create_csv_file_header(csv_file_spec,header_csv_power_spec)

    start_time = tx+int(5* fs) ; end_time =  rupture_pos - int( 5* fs)
    start_time = tx+int(0* fs) ; end_time =  rupture_pos - int( 0* fs)

    pos_a = start_time ; 
    D_global_arr = [False,0.0,0.0];D_X_array = [];D_Y_array = []
    axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']

    while pos_a < end_time and D_global_arr[0] == False:
        pos_b = pos_a +int(n*fs)
        D_arr = np.zeros(2); D_Dt_arr = np.zeros(2)
        avg_power_1 = (P0[pos_a]+P0[pos_b])/2
        fc_l_arr = np.zeros(2)
        temp_force = 0
        for i in range(2):

            trace = trace_arr[i]
            temp_label = refname+"_PowerSpectrum_alias_avg_P"+str(avg_power_1)+'_' + str(n) +"_secs_" +axis_arr[i]+'_D_g_otf_Fc_thresh_'+str(f_c_threshold_global)
            plot_fig_glob = plot_fig_name +"_global_alias_felix_otf_"+axis_arr[i]+"_P_"+str(avg_power_1)

            temp_out_spec_l= Spectrum_l_alias(trace[pos_a:pos_b], temp_arr[i], np.mean(zz[pos_a:pos_b]), fs,temp_label, display_psd,plot_fig_glob,save_find_D_global_path) 
            [k_l, D_arr[i], dk_l, dD_l, fc_l_arr[i], friction_l,D_Dt_arr[i]]= temp_out_spec_l
            
            if faxen_correction_glob_bool:
                print("faxenCorrected glob",D_arr[i])
                D_arr[i] = faxen_correction_glob(D_Dt_arr[i],L_t_new,axis_arr[i],bead_radius)
                #print(D_arr[i])
            
            force_l =round_sig( k_l * (np.mean(L_t[pos_a:pos_b])+bead_radius),5)
            
            temp_force+=force_l

            if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)
            if first_spec_otf: 
                create_csv_file_header(csv_file_spec_all,header_csv_power_spec)
                first_spec_otf = False
    
            
            with open(csv_file_spec_all, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                csv_dict_spec = data_dict([refname,"otf find alais D global", b ,avg_power_1,n,fs,axis_arr[i],k_l,dk_l,
                                               D_arr[i]*1.e-6,D_Dt_arr[i],fc_l_arr[i],force_l],header_csv_power_spec)
                writer.writerow(csv_dict_spec)
                f.flush()


        temp_force /=2

        if fc_l_arr[0] < f_c_threshold_global and not(D_global_arr[0]):
            if d_dt_min <= D_Dt_arr[0] <=d_dt_max:
                D_X_array.append(D_arr[0])
                if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)

                with open(csv_file_spec, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf find alias D global", b ,avg_power_1,n,fs,axis_arr[0],k_l,dk_l,
                                                   D_arr[0]*1.e-6,D_Dt_arr[0],fc_l_arr[0],temp_force],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
                    f.flush()


            
        if fc_l_arr[1] < f_c_threshold_global and not(D_global_arr[0]):
   
            if d_dt_min <= D_Dt_arr[1] <=d_dt_max:
                D_Y_array.append(D_arr[1])
                if os.getcwd() != (save_find_D_global_path) : os.chdir(save_find_D_global_path)

                with open(csv_file_spec, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf find alias D global", b ,avg_power_1,n,fs,axis_arr[1],k_l,dk_l,
                                                   D_arr[1]*1.e-6,D_Dt_arr[1],fc_l_arr[1],temp_force],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
                    f.flush()



        elif f_c_threshold_global < fc_l_arr[1]and f_c_threshold_global < fc_l_arr[0] and not(D_global_arr[0]):
            D_global_arr = [True , np.mean(D_X_array),np.mean(D_Y_array)]
            #print(D_X_array,D_Y_array)

        pos_a += int( 0.5*n * fs)
        
    if len(D_X_array)>=1 and len(D_Y_array) >=1 :
        D_global_arr = [True , np.mean(D_X_array),np.mean(D_Y_array)]
        print("Global D alias found ")
    elif len(D_X_array)>=1 :
        print("Global D only X  found ")

        D_global_arr = [True , np.mean(D_X_array),np.mean(D_X_array)]
    elif len(D_Y_array)>=1 :
        print("Global D only Y  found ")

        D_global_arr = [True , np.mean(D_Y_array),np.mean(D_Y_array)]

    else :
        D_global_arr = [False , 0,0]
        print("NOT FOUNDGlobal D ")

        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

        #print("D_G not found: "+refname     )

        with open(fcuk_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_error_log)

            csv_dict = data_dict([refname,"D_g not found_"+str(len(D_X_array))+"_"+str(len(D_Y_array))],header_error_log)
            writer.writerow(csv_dict)
            f.flush()



    

    return D_global_arr
     

     
def on_the_fly_calib_claire(trace_arr,arg_list_spec,n = 20):
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,
     min_L_t_bforce,min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y] = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    n_segs =4 
    load_thresh_time  = T_s[rupture_pos] - otf_end_T_thresh - T_s[tx]
    load_time = T_s[rupture_pos]  - T_s[tx]
    otf_force_evol_arr = [] ; avg_power_arr = [] ; time_arr = [];ldr_claire_arr = []
    seg_f_c_arr = np.array([np.nan]*4);seg_F_error_arr = np.array([np.nan]*4)
    seg_D_dt_arr =np.array([np.nan]*4)

    seg_len = int(load_thresh_time/(n_segs))
    if seg_len >= n:
        for i in range(n_segs):
            otf_leg_a = tx + int((seg_len-n)*0.5*fs + i*seg_len*fs) 
            otf_leg_b = otf_leg_a +int(n*fs)
            time_arr.append(np.average(T_s[otf_leg_a:otf_leg_b]))
            avg_pow= (P0[otf_leg_a]+P0[otf_leg_b])/2
            axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
            f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
            force_arr = np.array([np.nan,np.nan])
            pow_per_sec = (P0[otf_leg_b] - P0[otf_leg_a])/(T_s[otf_leg_b]-T_s[otf_leg_a])
            ramp_time = T_s[ty] - T_s[tx]

            for j in range(2):

                trace = trace_arr[j]
                temp_label = refname+"_PowerSpectrum_" + str(n) +"_secs_" +axis_arr[j]+'_laurent_claire_otf_'+str(i+1)+'_segment#'
                plot_fig_claire = plot_fig_name+"_claire_otf_"+axis_arr[i]
                temp_out_spec_l= Spectrum_l(trace[otf_leg_a:otf_leg_b], temp_arr[j], np.mean(zz[otf_leg_a:otf_leg_b]),
                                            fs,temp_label, display_psd,plot_fig_claire,save_out_psd_path,[ramp_time,pow_per_sec]) 
                [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
                if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                    force_arr[j] =  k_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    f_c_arr[j] = fc_l ; F_error_arr[j] = dk_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    D_dt_arr[j] = D_Dtheo_l
               
                #print("On the fly force in axis ",axis_arr[j]," : ",force_l)

                D_l *=1.e-6
                
                if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
                csv_file_spec = "anal_10_claire_otf_power_spec.csv"
                with open(csv_file_spec, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf claire_#seg_"+str(i+1), b ,avg_pow,n,fs,axis_arr[j],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[j]],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
                    f.flush()


            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            temp_force = np.nanmean(force_arr) ;#otf_calib_force_arr.append(temp_force)     
            otf_force_evol_arr.append(temp_force);avg_power_arr.append(avg_pow)
            seg_f_c_arr[i] = np.nanmean(f_c_arr);seg_F_error_arr[i] =np.nanmean(F_error_arr)
            seg_D_dt_arr[i] = np.nanmean(D_dt_arr)

        for i in range(n_segs-1):
            #ldr_temp_2 = (otf_force_evol_arr[i+1] - otf_force_evol_arr[i]) / (avg_power_arr[i+1] - avg_power_arr[i])
            #ldr_claire_arr.append(ldr_temp_2*pow_rate)   

            ldr_temp = (otf_force_evol_arr[i+1] - otf_force_evol_arr[i]) / (time_arr[i+1] - time_arr[i])
            ldr_claire_arr.append(ldr_temp)

        csv_file = "anal_10_claire_otf_rup_force_spec.csv"
        load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
        pow_per_sec = round_sig((P0[rupture_pos]-P0[tx])/(T_s[rupture_pos] - T_s[tx]),6)
        #loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)

        rupture_force_arr = np.array(ldr_claire_arr) * load_time
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_csv_rup_claire_otf)
            csv_dict_arr = data_dict([refname,"otf array", b,seg_f_c_arr,seg_D_dt_arr ,pow_per_sec,load_time,ldr_claire_arr,rupture_force_arr,seg_F_error_arr],header_csv_rup_claire_otf)
            csv_dict_avg = data_dict([refname,"otf avg", b ,np.nanmean(seg_f_c_arr),np.nanmean(seg_D_dt_arr) ,pow_per_sec,load_time,np.nanmean(ldr_claire_arr),np.nanmean(rupture_force_arr),np.nanmean(seg_F_error_arr)],header_csv_rup_claire_otf)

            writer.writerow(csv_dict_arr);writer.writerow(csv_dict_avg)
            f.flush()


    print("cest claire otf fini")
def on_the_fly_calib_claire_alias(trace_arr,arg_list_spec,n = 20):
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,
     min_L_t_bforce,min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y] = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    n_segs =4 
    load_thresh_time  = T_s[rupture_pos] - otf_end_T_thresh - T_s[tx]
    load_time = T_s[rupture_pos]  - T_s[tx]
    otf_force_evol_arr = [] ; avg_power_arr = [] ; time_arr = [];ldr_claire_arr = []
    seg_f_c_arr = np.array([np.nan]*4);seg_F_error_arr = np.array([np.nan]*4)
    seg_D_dt_arr =np.array([np.nan]*4)
    seg_rup_angle = np.array([np.nan]*4)
    
    seg_len = int(load_thresh_time/(n_segs))
    if seg_len >= n:
        for i in range(n_segs):
            otf_leg_a = tx + int((seg_len-n)*0.5*fs + i*seg_len*fs) 
            otf_leg_b = otf_leg_a +int(n*fs)
            time_arr.append(np.average(T_s[otf_leg_a:otf_leg_b]))
            avg_pow= (P0[otf_leg_a]+P0[otf_leg_b])/2
            axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
            f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
            force_arr = np.array([np.nan,np.nan])
            pow_per_sec = (P0[otf_leg_b] - P0[otf_leg_a])/(T_s[otf_leg_b]-T_s[otf_leg_a])
            ramp_time = T_s[ty] - T_s[tx]

            for j in range(2):

                trace = trace_arr[j]
                temp_label = refname+"_PowerSpectrum_alias_" + str(n) +"_secs_" +axis_arr[j]+'_laurent_claire_otf_'+str(i+1)+'_segment#'
                plot_fig_claire = plot_fig_name+"_claire_alias_otf_"+str(i+1)+'_seg_'+axis_arr[j]

                temp_out_spec_l= Spectrum_l_alias(trace[otf_leg_a:otf_leg_b], temp_arr[j], np.mean(zz[otf_leg_a:otf_leg_b]), 
                                                  fs,temp_label, display_psd,plot_fig_claire,save_out_psd_path,[ramp_time,pow_per_sec]) 
                [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
                if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                    force_arr[j] =  k_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    f_c_arr[j] = fc_l ; F_error_arr[j] = dk_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    D_dt_arr[j] = D_Dtheo_l
                #print("On the fly force in axis ",axis_arr[j]," : ",force_l)

                D_l *=1.e-6
                
                if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
                csv_file_spec = "anal_10_claire_otf_power_spec.csv"
                with open(csv_file_spec, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf_alias claire_#seg_"+str(i+1), b ,avg_pow,n,fs,axis_arr[j],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[j]],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
                    f.flush()


            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            temp_force= np.nanmean(force_arr);#otf_calib_force_arr.append(temp_force)     
            otf_force_evol_arr.append(temp_force);avg_power_arr.append(avg_pow)
            seg_f_c_arr[i] = np.nanmean(f_c_arr);seg_F_error_arr[i] =np.nanmean(F_error_arr)
            seg_D_dt_arr[i] = np.nanmean(D_dt_arr)
            xx = trace_arr[0];yy = trace_arr[1]
            avg_x_seg = np.mean(xx[otf_leg_a:otf_leg_b])
            avg_y_seg = np.mean(yy[otf_leg_a:otf_leg_b])
            avg_L_seg = np.mean(L_t[otf_leg_a:otf_leg_b])

            seg_rup_angle[i] = cal_rupture_angle(avg_x_seg, avg_y_seg, avg_L_seg, bead_radius)    
        for i in range(n_segs-1):
            #ldr_temp_2 = (otf_force_evol_arr[i+1] - otf_force_evol_arr[i]) / (avg_power_arr[i+1] - avg_power_arr[i])
            #ldr_claire_arr.append(ldr_temp_2*pow_rate)   

            ldr_temp = (otf_force_evol_arr[i+1] - otf_force_evol_arr[i]) / (time_arr[i+1] - time_arr[i])
            ldr_claire_arr.append(ldr_temp)

        csv_file = "anal_10_claire_otf_rup_force_spec.csv"
        load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
        pow_per_sec = round_sig((P0[rupture_pos]-P0[tx])/(T_s[rupture_pos] - T_s[tx]),6)
        #loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)

        rupture_force_arr = np.array(ldr_claire_arr) * load_time
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_csv_rup_claire_otf)
            csv_dict_arr = data_dict([refname,"otf_alaias array", b,seg_f_c_arr,seg_D_dt_arr ,pow_per_sec,load_time,ldr_claire_arr,rupture_force_arr,seg_F_error_arr],header_csv_rup_claire_otf)
            csv_dict_avg = data_dict([refname,"otf_alaias avg", b ,np.nanmean(seg_f_c_arr),np.nanmean(seg_D_dt_arr) 
                                      ,pow_per_sec,load_time,np.nanmean(ldr_claire_arr),np.nanmean(rupture_force_arr),
                                      np.nanmean(seg_F_error_arr)],header_csv_rup_claire_otf)

            writer.writerow(csv_dict_arr);writer.writerow(csv_dict_avg)
            f.flush()


    print("cest claire otfalias fini")

     
def on_the_fly_calib_claire_global(trace_arr,arg_list_spec,D_global_arr,n = 20):
    [b,P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,
     min_L_t_bforce,min_L_t_aforce,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y]  = arg_list_spec
    temp_force = 0
    first_spec_otf = True
    n_segs =4 
    load_thresh_time  = T_s[rupture_pos] - otf_end_T_thresh - T_s[tx]
    load_time = T_s[rupture_pos]  - T_s[tx]
    otf_force_evol_arr = [] ; avg_power_arr = [] ; time_arr = [];ldr_claire_arr = []
    seg_f_c_arr = np.array([np.nan]*4);seg_F_error_arr = np.array([np.nan]*4)
    seg_D_dt_arr =np.array([np.nan]*4)


    seg_len = int(load_thresh_time/(n_segs))
    if seg_len >= n:
        for i in range(n_segs):
            otf_leg_a = tx + int((seg_len-n)*0.5*fs + i*seg_len*fs) 
            otf_leg_b = otf_leg_a +int(n*fs)
            time_arr.append(np.average(T_s[otf_leg_a:otf_leg_b]))
            avg_pow= (P0[otf_leg_a]+P0[otf_leg_b])/2
            axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
            f_c_arr = np.array([np.nan,np.nan]);F_error_arr = np.array([np.nan,np.nan]);D_dt_arr = np.array([np.nan,np.nan])
            force_arr = np.array([np.nan,np.nan])
            pow_per_sec = (P0[otf_leg_b] - P0[otf_leg_a])/(T_s[otf_leg_b]-T_s[otf_leg_a])
            ramp_time = T_s[ty] - T_s[tx]


            for j in range(2):

                trace = trace_arr[j]
                temp_label = refname+"_PowerSpectrum_" + str(n) +"_secs_" +axis_arr[j]+'_laurent_claire_otf_global_'+str(i+1)+'_segment#'
                plot_fig_claire = plot_fig_name+"_claire_global_otf_"+str(i+1)+'_seg_'+axis_arr[j]
                temp_out_spec_l= Spectrum_l_D_global(trace[otf_leg_a:otf_leg_b], temp_arr[j], np.mean(zz[otf_leg_a:otf_leg_b]), 
                                                     fs,temp_label, display_psd,plot_fig_claire,D_global_arr[j+1],save_out_psd_path,[ramp_time,pow_per_sec]) 
                [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l
                if (d_dt_min <= D_Dtheo_l<= d_dt_max):
                    force_arr[j] =  k_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    f_c_arr[j] = fc_l ; F_error_arr[j] = dk_l * (np.mean(L_t[otf_leg_a:otf_leg_b])+bead_radius)
                    D_dt_arr[j] = D_Dtheo_l
                D_l *=1.e-6
                
                if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
                csv_file_spec = "anal_10_claire_otf_power_spec.csv"
                with open(csv_file_spec, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                    csv_dict_spec = data_dict([refname,"otf D global claire_#seg_"+str(i+1), b ,avg_pow,n,fs,axis_arr[j],k_l,dk_l,
                                               D_l,D_Dtheo_l,fc_l,force_arr[j]],header_csv_power_spec)
                    writer.writerow(csv_dict_spec)
                    f.flush()



            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)


            temp_force =np.nanmean(force_arr);#otf_calib_force_arr.append(temp_force)     
            otf_force_evol_arr.append(temp_force);avg_power_arr.append(avg_pow)
            seg_f_c_arr[i] = np.nanmean(f_c_arr);seg_F_error_arr[i] =np.nanmean(F_error_arr)
            seg_D_dt_arr[i] = np.nanmean(D_dt_arr)

        for k in range(n_segs-1):
            #ldr_temp_2 = (otf_force_evol_arr[i+1] - otf_force_evol_arr[i]) / (avg_power_arr[i+1] - avg_power_arr[i])
            #ldr_claire_arr.append(ldr_temp_2*pow_rate)   

            ldr_temp = (otf_force_evol_arr[k+1] - otf_force_evol_arr[k]) / (time_arr[k+1] - time_arr[k])
            ldr_claire_arr.append(ldr_temp)

        csv_file = "anal_10_claire_otf_rup_force_spec.csv"
        load_time = round_sig(T_s[rupture_pos] - T_s[tx],6)
        pow_per_sec = round_sig((P0[rupture_pos]-P0[tx])/(T_s[rupture_pos] - T_s[tx]),6)
        #loading_rate = alpha * pow_per_sec;loading_rate = round_sig(loading_rate,10)

        rupture_force_arr = np.array(ldr_claire_arr) * load_time
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_csv_rup_claire_otf)
            csv_dict_arr = data_dict([refname,"otf global array", b,seg_f_c_arr,seg_D_dt_arr ,pow_per_sec,load_time,ldr_claire_arr,rupture_force_arr,seg_F_error_arr],header_csv_rup_claire_otf)
            csv_dict_avg = data_dict([refname,"otf Global avg", b ,np.nanmean(seg_f_c_arr),np.nanmean(seg_D_dt_arr) 
                                      ,pow_per_sec,load_time,np.nanmean(ldr_claire_arr),np.nanmean(rupture_force_arr),
                                      np.nanmean(seg_F_error_arr)],header_csv_rup_claire_otf)

            writer.writerow(csv_dict_arr);writer.writerow(csv_dict_avg)
            f.flush()

        
    print("cest claire global otf fini")

def qt_anchor_Z(Z_bf,low_level=0.05):
    z_qt_5_bf = np.quantile(Z_bf,low_level)
    print(z_qt_5_bf)
    count =0
    for i in range(len(Z_bf)):
        if Z_bf[i]<=z_qt_5_bf:
            Z_bf[i] = z_qt_5_bf
            count+=1
    return(Z_bf,count)
#%% main analysis 9 function
def analysis_9(log_afs_df,i_file,CleanOutlier_med = True):
    N_psd_build = 1200

    ori_path = log_afs_df['File path'][i_file]
    tdms_file_name = log_afs_df['TDMS file name '][i_file]
    rup_pos_offset_read =log_afs_df['Rupture_pos_ofset_(s)'][i_file]
    rup_pos_read =log_afs_df['Rupture pos'][i_file]
    first_z  =True ; saveZ = False ;power_spec_bool = True
    out_Z =False;


    #create_csv_file_header(tdms_file_name[:8] +"_rup_force_spec.csv",header_csv_rup)

    SaveGraph =True;CloseAfterSave = True;display_psd = False
    save_alpha_fit = True
    b, bref = bead_extract_from_df(log_afs_df,i_file)

    step_pos = str(log_afs_df['Step_position'][i_file])
    
    step_pos = np.array([int(e) if e.isdigit() else e for e in step_pos.split(',')])
    display_psd = True
    print(step_pos)
    os.chdir(save_out_root_path);

    #creating differnt folders for outputs. 
    try: 
        os.mkdir(tdms_file_name[:8])
    except OSError as error: 
        print(error)
    save_out_path = save_out_root_path+"/"+tdms_file_name[:8]
    os.chdir(save_out_path);
    try: 

        os.mkdir(tdms_file_name[9:]+"_PSD_plots")
        save_out_psd_path = save_out_path +"/"+tdms_file_name[9:]+"_PSD_plots"
        os.mkdir(tdms_file_name[9:]+"_tether_plots")
        save_out_tether_path = save_out_path +"/"+tdms_file_name[9:]+"_tether_plots"

    except OSError as error: 
        print(error)  
        save_out_psd_path = save_out_path +"/"+tdms_file_name[9:]+"_PSD_plots"
        save_out_tether_path = save_out_path +"/"+tdms_file_name[9:]+"_tether_plots"

    nbead = len(b);fname = [""]*nbead 
    start = np.zeros(nbead ,dtype = int) ;stop = np.zeros(nbead ,dtype = int) 
    load = np.zeros(nbead ,dtype = int)
    start_min = np.zeros(nbead ,dtype = int) ;stop_min = np.zeros(nbead ,dtype = int)
    bead_diameter = 3101 #diameter in nm
    bead_radius = bead_diameter /2.0
    
    L_p = 50 # persistance lenght of DNA in nm 
    range_anchorpoint_min=(0,100); range_spectrum_min=(0, 100); Select_range_spectrum=False

    for nf in range(nbead): fname[nf] = tdms_file_name+".tdms"; start_min[nf]=6; stop_min[nf]=306*60*1000/20; load[nf]= nf== 0  # stop[nf]=1600000
    #In this code block i shall try to obtain the step values for each data file
    #the fs and the exposure time are also read from the meta data 
    os.chdir(ori_path)
    tdms_file = TdmsFile.open("" + fname[0])  # alternative TdmsFile.read(path1+fname[ibead])
    tdms_groups = tdms_file.groups()   
    tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
    (TE0,NE0,DE0) = events(tdms_event)
    (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[0]) # test bead
    global fs
    T0, fs = timet(tdms_track, False)
    
    global te 
    te = find_te_tdms_index(fname[0],ori_path)


    #extracting the step and ramps values for this TDMS file. 
    P0 = BuildP(T0, TE0, NE0, DE0)
    value, location, lengths = island_props(P0)

    global fmax_Hz
    fmin_Hz=0.1; fmax_Hz=fs/2.   # frequency range for spectrum fit (Hz)


    step_values = [] ;step_loc = []
    for i in step_pos:
        step_values.append(value[i])
        step_loc += [(location[i],location[i+1]-1)]
    print("CHECK THE STEPS :",step_values,step_loc)
    step_check_bool = len(step_values) == len(np.nonzero(step_values)[0])
    ramp_loc = ramp_extraction(P0,T_s)


    #extracting and setting hte average temperature of the chamber dyanmicsally 

    temperature = tdms_temp['Actual Temp'][:]
    Temperature_C = np.mean(temperature); global kBT_pN_nm
    kBT_pN_nm= 1.38e-23*(Temperature_C+273)*1.e12*1.e9

    step_ramp_check(P0,T_s,ramp_loc,step_loc,save_out_tether_path)
    
    #importatnt parameters for the D_avg_ global method of calibrating the forces OTF.
    global f_c_threshold_global 

    d_dt_min ,d_dt_max = [0.8,1.2];f_c_threshold_global = fs/6 ; n_sec_otf = N_psd_build/fs

    #maybe will be updated over time and functionality. 
    loaddata = True 
    pprint = True
    RangeNZlavg=[400]       # size of moving average for smoothing z before anchor point determination
    NZrefavg=5000
    mode='reflect' #'nearest'  # mode for rolling average   # edge modes = ['full', 'same', 'valid', 'nearst']
    # see details on https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    AnchorPointStateList=['custom'] # correction with AnchorPoint as determined by anchor point
    alpha_arr = np.zeros(nbead)


    for ibead in range(nbead):
        plot_fig_name = fname[ibead][-11:-5] #+"_R_"+str(bead_radius) + "_bead_"+str(b[ibead]) 
        os.chdir(ori_path)

        if loaddata and load[ibead]:
            print('Loading data', fname[ibead])
            tdms_file = TdmsFile.open("" + fname[ibead])  # alternative TdmsFile.read(path1+fname[ibead])
        print('======================BEAD ', b[ibead] ,' ===========================')
        tdms_groups = tdms_file.groups()        
        tdms_track = tdms_groups[0]; tdms_temp = tdms_groups[1]; tdms_event = tdms_groups[2]
        (TE0,NE0,DE0) = events(tdms_event)
        P0 = BuildP(T0, TE0, NE0, DE0)


        dmax=1000000.; imax=(len(tdms_track)-1)//5; dX=1500; dY=1500    
        print('Nb of traj=', imax)
        refname=fname[ibead][:-5]+'_Bead'+str(b[ibead])+'_'; print(refname)
        (X0,Y0,Z0,MinLUT0,SSIM0,T_s)=track(tdms_track, b[ibead]) # test bead
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
            print(len(j0))
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

        if load[ibead] or True :
            print("Calculating pool of reference bead(s)", bref )
            if len(bref)>1:

                (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])
                for i in range(1,len(bref)):

                    (Xr0_,Yr0_,Zr0_,MinLUTr0_,SSIMr0_,Temp_t)=track(tdms_track, bref[i])
                    Xr0=Xr0+Xr0_; Yr0=Yr0+Yr0_; Zr0=Zr0+Zr0_
                Xr0=Xr0/len(bref); Yr0=Yr0/len(bref); Zr0=Zr0/len(bref)
                #Zr0 = Zr0[temp_out_Z_pos]
            elif len(bref)==1:
                (Xr0,Yr0,Zr0,MinLUTr0,SSIMr0,Temp_t)=track(tdms_track, bref[0])   # reference bead
        else: print("Pool of reference bead(s) already calculated") 

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
        if rup_pos_offset_read.isdigit():
            print("Updated the offset ");rup_pos_offset = int(int(rup_pos_offset_read) * fs)
        else:rup_pos_offset = step_loc[-1][1]
    
        
        rupture_pos = rupture_probe_func(zz[rup_pos_offset:],2)
        #rupture_pos,lol = rupture_probe_func1(zz[t:],-4)
        rupture_pos += rup_pos_offset
        
        if rup_pos_read!= None and rup_pos_read.isdigit():
            rupture_pos = int(rup_pos_read)
            print("Rup pos updated ")

        did_it_unbind = ~np.isnan(rupture_pos)
        
        #Compute the lenghts distribution for the whole range now 
        anchor_pt_arr = [(np.mean(xx[:non_0_force[0]]),np.mean(yy[:non_0_force[0]]))]
        print(anchor_pt_arr)
        for a in anchor_pt_arr :
            xx -= a[0];yy-=a[1]
            if did_it_unbind : 
                non_0_force = pre_rup_func(non_0_force,rupture_pos-int(fs*otf_end_T_thresh))
                zero_force_pos = pre_rup_func(zero_force_pos,rupture_pos)
                
                zz = zz - zz[zero_force_pos].min()
                
            #print(zero_force_pos)
            sigma_x = np.quantile(X_bf- np.mean(X_bf),0.975) - np.quantile(X_bf- np.mean(X_bf),0.025)
            sigma_y = np.quantile(Y_bf- np.mean(Y_bf),0.975) - np.quantile(Y_bf- np.mean(Y_bf),0.025)

            L_t_planar = np.sqrt((xx[zero_force_pos])**2 + (yy[zero_force_pos] )**2)
            max_L_t_plane = np.quantile(L_t_planar,0.95)

            L_t_bforce = (np.sqrt((xx[zero_force_pos] )**2 + (yy[zero_force_pos] )**2+
                                 (zz[zero_force_pos] + bead_radius )**2 ) - bead_radius)

            L_t_aforce = (np.sqrt((xx[non_0_force])**2 + (yy[non_0_force])**2+
                                 (zz[non_0_force] + bead_radius )**2 ) - bead_radius)
            L_t = (np.sqrt((xx )**2 + (yy )**2+
                                 (zz + bead_radius )**2 ) - bead_radius)
            
            L_T_bforce_avg = L_t_bforce.mean();L_T_aforce_avg = L_t_aforce.mean()
            AnchorCov=np.cov(xx[:non_0_force[0]],yy[:non_0_force[0]])
            w, v = np.linalg.eig(AnchorCov); symmetry_factor = np.sqrt(np.amax(w)/np.amin(w))

            csv_file = fname[ibead][:8] +".csv"
            #a = (xx.mean()-a[0],yy.mean()-a[1])
            
            max_L_t_bforce = np.quantile(L_t_bforce,0.95);max_L_t_aforce = np.quantile(L_t_aforce,0.95)
            min_L_t_bforce_qt = np.quantile(L_t_bforce,0.05);min_L_t_aforce_qt = np.quantile(L_t_aforce,0.05)
            min_L_t_bforce = L_t_bforce.min();min_L_t_aforce =L_t_aforce.min()
            min_L_t_aforce_rup_ramp,qt_L_t_aforce_rup_ramp = np.nan ,np.nan
            #Z_rup_avg = np.mean(zz[rupture_pos-int((otf_end_T_thresh+n_sec_otf)*fs):rupture_pos-int((otf_end_T_thresh)*fs)])
            Z_rup_avg = np.nan
            #Z_rup_avg = np.mean(zz[rupture_pos-int((otf_end_T_thresh+n_sec_otf)*fs):rupture_pos-int((otf_end_T_thresh)*fs)])

            if os.getcwd() != (save_out_path) : os.chdir(save_out_path)
            for j in range(len(ramp_loc)):
                (tx,ty) = ramp_loc[j]
                if rupture_pos >tx and rupture_pos<ty:
                    min_L_t_aforce_rup_ramp = L_t[tx:rupture_pos].min()
                    qt_L_t_aforce_rup_ramp =np.quantile(L_t[tx:rupture_pos].min(),0.05) 

            if ibead==0:create_csv_file_header(csv_file,header_csv)    
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv)
                temp_arr = [refname , bref, a , b[ibead],did_it_unbind,step_check_bool ,non_0_force[0] , L_T_bforce_avg,L_T_aforce_avg
                        ,max_L_t_bforce,max_L_t_aforce,min_L_t_bforce_qt,min_L_t_aforce_qt,min_L_t_bforce,min_L_t_aforce]

                csv_dict = data_dict(temp_arr,header_csv)
                writer.writerow(csv_dict)
                f.flush()

            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            csv_file = "master_tether_data_all_TDMS.csv"
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_csv)
                
                temp_arr = [refname , bref, a , b[ibead],did_it_unbind,step_check_bool ,non_0_force[0] , L_t_bforce.mean(),L_t_aforce.mean()
                        ,max_L_t_bforce,max_L_t_aforce,min_L_t_bforce_qt,min_L_t_aforce_qt,min_L_t_bforce,min_L_t_aforce,]
                csv_dict = data_dict(temp_arr,header_csv)
                writer.writerow(csv_dict)
                f.flush()



        #the power spectrum building and force calibration happens below. 
        
        #Step 10 : random plots that is needed for various visualisations
            
        if os.getcwd() != (save_out_tether_path) : os.chdir(save_out_tether_path)
        arg_list_plot=[xx,yy,zz,L_t,T_s,non_0_force,rupture_pos,zero_force_pos,did_it_unbind,refname,b[ibead]]
        make_plot_function(arg_list_plot)
        if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)
        make_plot_function(arg_list_plot)
        
        if np.isnan(rupture_pos):
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            with open(fcuk_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_error_log)

                csv_dict = data_dict([refname,"Rup pos error"],header_error_log)
                writer.writerow(csv_dict)
                f.flush()


            continue;
        if step_check_bool==False:
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            print("Yo u are fucked rup ops: "+refname     )

            with open(fcuk_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_error_log)

                csv_dict = data_dict([refname,"step error"],header_error_log)
                writer.writerow(csv_dict)
                f.flush()

            print("You are fucked : "+refname,file=fcuk_file); break;
        plot_fig_name = fname[ibead][-11:-5]+ "_bead_"+str(b[ibead]) 

        power_spec_bool = True ; first_spec = True
        if power_spec_bool:
            bead_radius = 1550
            trace_arr = [xx,yy,zz] ; axis_arr = ['X','Y','Z'] ; temp_arr =['R','R','Z']
            force_calib = np.zeros(len(step_values))     
            for j in range(len(step_values)):
                (tx,ty) = step_loc[j];temp_force = 0 
                #step_values[j] = round_sig(step_values[j],3)
                for i in range(2) :
                    trace = trace_arr[i];temp_label = refname+'_PowerSpectrum_' + str(step_values[j]) +"_" +axis_arr[i]+'_laurent'

                    temp_out_spec_l= Spectrum_l(trace[tx:ty], temp_arr[i], np.mean(zz[tx:ty]), fs,temp_label, display_psd,plot_fig_name,save_out_psd_path) 
                    [k_l, D_l, dk_l, dD_l, fc_l, friction_l,D_Dtheo_l]= temp_out_spec_l

                    force_l =round_sig( k_l * (np.mean(L_t[tx:ty])+bead_radius),5)
                    temp_force += force_l;D_l *=1.e-6  
                    spec_l_dict = data_dict([ step_values[j] ,axis_arr[i],k_l, D_l,D_Dtheo_l,fc_l,force_l],['Power ','axis',' k (pN/nm) ',' D (µm²/s)' ,'D/Dtheo',"corner freq","force (pN)"])
                    
                    if os.getcwd() != (save_out_path) : os.chdir(save_out_path)
                    csv_file_spec = fname[ibead][:8]+"_power_spec_l.csv"
                    create_csv_file_header(csv_file_spec,header_csv_power_spec)
                    with open(csv_file_spec, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header_csv_power_spec)
                        csv_dict_spec = data_dict([refname,"Trad", b[ibead] ,step_values[j],T_s[ty]-T_s[tx],fs,axis_arr[i],k_l,dk_l,
                                                   D_l,D_Dtheo_l,fc_l,force_l],header_csv_power_spec)
                        writer.writerow(csv_dict_spec)
                        f.flush()


           
                force_calib[j] = temp_force/2.0

            popt, pcov = curve_fit(fit_alpha, step_values, force_calib)
            alpha_arr[ibead] = popt
            if os.getcwd() != (save_out_root_path) : os.chdir(save_out_root_path)

            #compute_rup_force_trad(b[ibead],P0,T_s,ramp_loc,pcov,alpha_arr[ibead],rupture_pos,fname[ibead][-11:-5]) 
            '''
            #following code is for step evolve; uncomment it if needed.  
            trace_arr =[xx,yy,zz]
            for j in range(len(step_values)):
                tx,ty = step_loc[j]
                arg_list_spec_evol=[b[ibead],P0,T_s,L_t,kBT_pN_nm,refname,plot_fig_name,save_out_psd_path,save_out_root_path,fs,tx,ty] 
                start_time = tx ; end_time =  ty
                temp_probe_time = start_time
                i =2
                while temp_probe_time < end_time:
                    #step_calib_evol(trace_arr,arg_list_spec_evol,(i-1)*5)
                    temp_probe_time=tx +int( i*5 * fs);i+=1

            
            
            '''
            
            
            for j in range(len(ramp_loc)):
                (tx,ty) = ramp_loc[j]
                if rupture_pos >tx and rupture_pos<ty :
                    pos_b = rupture_pos - int(otf_end_T_thresh *fs) ; pos_a = pos_b - int(n_sec_otf * fs)

                    L_t_new = np.mean(L_t[pos_a:pos_b])
                    arg_list_spec = [b[ibead],P0,T_s,L_t,zz,refname,refname,save_out_psd_path,save_out_root_path,rupture_pos,fs,tx,ty,
                                     min_L_t_bforce_qt,min_L_t_aforce_qt,L_T_bforce_avg,L_T_aforce_avg,max_L_t_bforce,max_L_t_aforce,symmetry_factor,max_L_t_plane,sigma_x,sigma_y] 
                    arg_list_d_global=[b[ibead],P0,T_s,L_t,zz,refname,refname,save_out_psd_path,rupture_pos,fs,tx,L_t_new]  

                    print("hellya")
                    do_i_need_DG = on_the_fly_calib_felix_alias(trace_arr,arg_list_spec,n_sec_otf)
                    
                    print("do i need global",do_i_need_DG)
                    #on_the_fly_calib_claire(trace_arr,arg_list_spec,n_sec_otf)
                    on_the_fly_calib_claire_alias(trace_arr,arg_list_spec,n_sec_otf)
                    
                    if do_i_need_DG or True:
                        #D_global_arr = find_D_global(trace_arr,arg_list_d_global,n_sec_otf)
                        D_global_arr_alias = find_D_global_alias(trace_arr,arg_list_d_global,n_sec_otf)
                        print("D_glob parameters ",D_global_arr_alias)
                        if D_global_arr_alias[0]:
                            on_the_fly_calib_felix_global(trace_arr,arg_list_spec,D_global_arr_alias,n_sec_otf)
                            on_the_fly_calib_claire_global(trace_arr,arg_list_spec,D_global_arr_alias,n_sec_otf)
                        else:
                            arg_list_d_global_all_ramps = [b[ibead],P0,T_s,L_t,zz,refname,plot_fig_name,save_out_psd_path,rupture_pos,fs,ramp_loc,L_t_new]  
                            D_global_arr_alias_all_ramps = find_D_global_alias_all_ramps(trace_arr,arg_list_d_global_all_ramps,n_sec_otf)
                            if D_global_arr_alias_all_ramps[0]:
                                
                                on_the_fly_calib_felix_global(trace_arr,arg_list_spec,D_global_arr_alias_all_ramps,n_sec_otf)
                                on_the_fly_calib_claire_global(trace_arr,arg_list_spec,D_global_arr_alias_all_ramps,n_sec_otf)
    

            if save_alpha_fit :
                if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)

                plt.figure(1020);plt.clf()
                plt.scatter(step_values, force_calib,label ="expt value")
                plt.plot(step_values, fit_alpha(step_values, popt),"r--",label = "Fitted linear func")
                plt.xlabel("Power value in %");plt.ylabel("Force in pN");plt.title("Force calibration")
                plt.legend();plt.savefig(plot_fig_name+"_alpha_fit_force_calib.png")
        
        n_otf_high_fs = n_sec_otf
    
        for ri in range(len(ramp_loc)):
            (tx,ty) = ramp_loc[ri]
            if os.getcwd() != (save_out_psd_path) : os.chdir(save_out_psd_path)
    
            create_csv_file_header("anal_10_felix_otf_power_spec_evol.csv",header_arr_power_spec_otr_evol)
            create_csv_file_header("anal_10_felix_otf_power_spec_evol_glob.csv",header_arr_power_spec_otr_evol)
    
            arg_list_spec_evol=[b[ibead],P0,T_s,L_t,kBT_pN_nm,refname,ri,plot_fig_name,save_out_psd_path,save_out_root_path,fs,tx,ty] 
            start_time = tx+ int(fs*5) ; end_time =  ty
            temp_probe_time = start_time
            i =2
            while (temp_probe_time < end_time) and (temp_probe_time < rupture_pos):
                arg_list_spec_evol = [b[ibead],P0,T_s,L_t,zz,refname,ri,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,temp_probe_time,tx,ty] 
                arg_list_spec_evol_glob = [b[ibead],P0,T_s,L_t,zz,refname,ri,plot_fig_name,save_out_psd_path,save_out_root_path,rupture_pos,fs,temp_probe_time,tx,ty,D_global_arr_alias] 
                otr_evol_felix_alais(trace_arr,arg_list_spec_evol,n_otf_high_fs)
                otr_evol_felix_alais_glob(trace_arr,arg_list_spec_evol_glob,n_otf_high_fs)
                temp_probe_time=tx +int( i*5 * fs);i+=1

        print("cest analysis fini pour tdms file : ",i_file+1)
        gc.collect()



def set_f_max(temp):
    global fmax_Hz
    fmax_Hz = temp



#%% some global parameters that i am setting outisde the main loop .
# 
RangeNZlavg=[400]       # size of moving average for smoothing z before anchor point determination
NZrefavg=5000
mode='reflect' #'nearest'  # mode for rolling average   # edge modes = ['full', 'same', 'valid', 'nearst']
# see details on https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
AnchorPointStateList=['custom'] # correction with AnchorPoint as determined by anchor point
    
bead_diameter = 3101 #diameter in nm
bead_radius = bead_diameter /2.0
#global fmin_Hz;global fmax_Hz

fmin_Hz= 0.1; fmax_Hz= 25.   # frequency range for spectrum fit (Hz)
SaveGraph =True;CloseAfterSave = True;display_psd = False
save_alpha_fit = True

first_file_of_day = True
#%% header file names 



import warnings

warnings.filterwarnings('ignore')

header_arr_power_spec_otr_evol = ["TDMS file name","Calib type","ramp_number","power rate %/s","time from ramp start", "Tether bead",'Power ','number of sec','fs'
                         ,'axis',' k (pN/nm) ','k_error (pN/nm)',' D (µm²/s)' ,'D/Dtheo',"corner freq","Force (pN)"]


header_csv = ['tdms_file_name'  , 'ref_beads',"Z anchor pt method",'anchor points' , 'tether_bead',"Did it unbind ","step_chekc" ,
              ' Mean L_dist_bforce' ,'Mean L_dist_aforce' ,'Max L_dist_bforce','Max L_dist_aforce','min_qt L_dist_bforce',
              'min_qt L_dist_aforce','min L_dist_bforce','min L_dist_aforce' ]

header_csv_rup = ["TDMS file name" ,"Calib type", "Tether bead" , "Force per power(pN/percent)","P_error in alpha","Power percent per second (%/sec)", "Loaded timebefore rupture(s)", "Loading rate","Rupture force (pN)"]
header_csv_rup_felix = ["TDMS file name" ,"Calib type", "Tether bead" , "Force per power(pN/percent)","Avg F_c","fs","Avg D_Dtheo","Avg D_fitted","Avg D_t","Power percent per second (%/sec)", 
                        "Loaded timebefore rupture(s)", "Loading rate","Rupture force (pN)","Error in force calib(pN)","mean_pre_rup_ramp_length","Mean_rup_lenght","max_rup_lenght","Ruputre angle (deg)",
                        "L_T_bforce_min(nm)","L_T_aforce_min(nm)","L_T_bforce_avg(nm)","L_T_aforce_avg(nm)","L_T_bforce_max(nm)","L_T_aforce_max(nm)","Symmetry_factor","max_planar_L_T_95","sigma_X","sigma_Y"]


header_csv_rup_claire_otf = ["TDMS file name" ,"Calib type", "Tether bead" , 
                             "Avg F_c","Avg D_Dtheo","Power percent per second (%/sec)", "Loaded timebefore rupture(s)", 
                             "Loading rate","Rupture force (pN)","Error in force calib(pN)"]

header_csv_rup_evol_global = ["TDMS file name" ,"Calib type", "Tether bead" 
                       ,"Power percent per second (%/sec)",
                       "Loaded timebefore rupture(s)", "Loading rate","Rupture force (pN)","Time from ramp start","N_sec_step"]

header_csv_power_spec_evol_avg = ["TDMS file name","Calib type", "Tether bead",'Power ','number of sec','fs'
                         ,'D/Dtheo',"corner freq","Force (pN)","error in Force (pN)"]
header_error_log = ["ref name ","error _description"]
header_csv_power_spec = ["TDMS file name","Calib type", "Tether bead",'Power ','number of sec','fs'
                         ,'axis',' k (pN/nm) ','k_error (pN/nm)',' D (µm²/s)' ,'D/Dtheo',"corner freq","Force (pN)"]
print(header_csv)

#%%loading the log AFS





