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

temp_sheet_name = '22_23'

df_curr = pd.read_excel('/Users/yogehs/Documents/MS_thesis_all_files /LAS_GUI/pre_clean_serial_log/AFS_all_data_pre_clean.xlsx',header = 0 ,keep_default_na=False,sheet_name=temp_sheet_name)

#df_24_high_pulling= pd.read_csv('/Users/yogehs/Documents/MS_thesis_all_files /analysis_files/rupture_force_data_september_22/AFS_24_high_pulling_all_serial_rup_pos.csv',keep_default_na=False)


save_path ='/Users/yogehs/Documents/MS_thesis_all_files /analysis_files/rupture_force_data_september_22/AFS_FS_23/troubleshoot_gui/'
temp = datetime.datetime.now()

save_path = save_path+ temp.strftime("%m_%d_%Y__%H:%M_")+f'AFS_FS_{temp_sheet_name}_all_clean_final_.xlsx'
# Launch the GUI
#obj_LAS = Lumicks_sucks(log_afs_df)
#%%
global gui_bool 
gui_bool = True

if __name__ == '__main__': 

    class_LAS = Lumicks_sucks(df_curr,save_path,314)

    if gui_bool:
        
        app = QApplication(sys.argv)
        class_LAS.load_gui()
        #class_LAS.controller_cont()

        sys.exit(app.exec_())
         
    else:
        for i in range(len(df_curr)):
            
            class_LAS.i_file = i
            class_LAS.load_bead()
            t =  class_LAS.find_temp()
            print(t,class_LAS.file_label)
       
        #class_LAS.search_DG(class_LAS.ramp_loc[0])

#%%doing std 
class_LAS.i_file = 3
class_LAS.load_bead()
plt.plot(class_LAS.L_t)
class_LAS.s
#%%
df = class_LAS.log_trouble_sess
plt.hist(df['L_t_rup_corrected'])
plt.show()

sns.scatterplot(df,x ='LDR_corrected',y = 'rup_force_corrected' )