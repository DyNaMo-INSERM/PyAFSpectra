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
from CLASS_AFSDataAnalyzer import AFSDataAnalyzer

from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton

# Define the sheet name to read from the Excel file

sheet_name_excel = 'sample_log'

# Read the log data from the Excel file
df_log = pd.read_excel('/Users/yogehs/Documents/ATIP_PhD/PyAFSpectra/sample_AFS_analysis_log.xlsx', header=0, keep_default_na=False, sheet_name=sheet_name_excel)

# Define the path to save the analysis results
save_path = '/Users/yogehs/Documents/MS_thesis_all_files/analysis_files/rupture_force_data_september_22/AFS_FS_23/troubleshoot_gui/'
temp = datetime.datetime.now()

# Create a unique save path for the output file based on the current date and time
save_path_for_output = save_path + temp.strftime("%m_%d_%Y__%H:%M_") + f'AFS_FS_{sheet_name_excel}_all_clean_final_.xlsx'

# Flag to determine whether to launch the GUI
global do_we_want_a_gui_bool 
do_we_want_a_gui_bool = True

if __name__ == '__main__': 
    # Initialize the AFSDataAnalyzer with the log data and save path
    session_afs_anlaysis = AFSDataAnalyzer(df_log, save_path_for_output)

    if do_we_want_a_gui_bool:
        # Launch the GUI application
        app = QApplication(sys.argv)
        session_afs_anlaysis.launch_gui()
        sys.exit(app.exec_())
    else:
        # Process each file without GUI
        for i in range(len(df_log)):
            session_afs_anlaysis.i_file = i
            session_afs_anlaysis.load_bead()
            t = session_afs_anlaysis.find_temp()
            print(t, session_afs_anlaysis.file_label)

