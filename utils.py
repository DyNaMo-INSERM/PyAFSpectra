#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:04:12 2024

@author: yogehs
"""

#import tweezepy as tp
import numpy as np
from nptdms import TdmsFile #from nptdms import tdms  # pip install nptdms
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
import os
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d
import matplotlib
from math import log10, floor


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




def qt_anchor_Z(Z_bf,low_level=0.05):
    z_qt_5_bf = np.quantile(Z_bf,low_level)
    print(z_qt_5_bf)
    count =0
    for i in range(len(Z_bf)):
        if Z_bf[i]<=z_qt_5_bf:
            Z_bf[i] = z_qt_5_bf
            count+=1
    return(Z_bf,count)

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


