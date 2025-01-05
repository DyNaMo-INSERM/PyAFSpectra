#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:02:57 2024

@author: yogehs
"""
import os 
import pandas as pd
import datetime
import sys
from utils import *
from LAS_GUI_CLASS import Lumicks_sucks

from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton

sheet_name_excel = 'sample_log'

df_log = pd.read_excel('/Users/yogehs/Documents/ATIP_PhD/PyAFSpectra/sample_AFS_analysis_log.xlsx',header = 0 ,keep_default_na=False,sheet_name=sheet_name_excel)



save_path ='/Users/yogehs/Documents/MS_thesis_all_files /analysis_files/rupture_force_data_september_22/AFS_FS_23/troubleshoot_gui/'
temp = datetime.datetime.now()

save_path_for_output = save_path+ temp.strftime("%m_%d_%Y__%H:%M_")+f'AFS_FS_{sheet_name_excel}_all_clean_final_.xlsx'
# Launch the GUI
#%%
global do_we_want_a_gui_bool 
do_we_want_a_gui_bool = True

if __name__ == '__main__': 

    class_LAS = Lumicks_sucks(df_log,save_path_for_output)

    if do_we_want_a_gui_bool:
        
        app = QApplication(sys.argv)
        class_LAS.launch_gui()

        sys.exit(app.exec_())
         
    else:
        for i in range(len(df_curr)):
            
            class_LAS.i_file = i
            class_LAS.load_bead()
            t =  class_LAS.find_temp()
            print(t,class_LAS.file_label)
       
