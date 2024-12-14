import tkinter as tk
from tkinter import ttk
import tkinter.ttk as ttk
from tkinter import messagebox
import sys
import os
import subprocess
from tkinter import filedialog
from tkinter import font
from PIL import Image, ImageTk
import time
import numpy as np
import platform
import tkinter.font as tkfont
#------------------------
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=Warning)
import argparse
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data   import InMemoryDataset, download_url
from torch_geometric.data   import Data
from torch_geometric.data   import DataLoader
from torch_geometric.loader import DataLoader
from torch.nn               import Linear
from torch_geometric.nn     import GCNConv
from torch_geometric.nn     import GATConv
from torch_geometric.nn     import GATv2Conv
from torch_geometric.nn     import global_mean_pool
import matplotlib.pyplot as plt
import shutil
import datetime
import math
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--imp',   default='2'  )  #0: zero imputation, 1: peak value replacement, 2: Pesudo PMU measurement
parser.add_argument('--mdl',   default='GAT')  #GCN5 or GCN6 or GAT
parser.add_argument('--enc',   default='Y'  )  #Yes or No
parser.add_argument('--per',   default='0'  )  #error percentage
args = parser.parse_args()
Wstatus   = 0
Wstatus73 = 0
var1 = 4 # Chosen DEFAULT NN model,4 is GAT V2-L2, 0 is GCN2
txt_64 = ""
ContName = ""
NumBus=0
GM2024 = 0
file_path4  = os.getcwd()
file_path22 = os.getcwd() #"" #Folder name of the Input files
file_pathML2= os.getcwd() # ML Model Path during Training
temp        = os.path.join(file_path22, 'Sample_Datasets', 'tmp_files') # same as os.getcwd()+"tmp_files"
file_nameML = "dummy.pth"
file_name3  = "dummy.lst"
if os.path.exists(temp):
    shutil.rmtree(temp)
os.makedirs(temp)
args_logits = 1
WindowW, WindowH = 580, 500
Index_map = {0:"A0index",1:"A1index",2:"F0index",3:"F1index",4:"Vindex",5:"V0index",6:"V1index",7:"V2index",8:"V3index"}
MDL_map = {0:"GCN2",1:"GCN5",2:"GCN6",3:"GCN3",4:"SGA2",5:"SGA3",6:"SGA4",7:"SGA5",8:"GCN4",9:"MLP"}
# Initialization
def tee(text, file):
    print(text) # Write to console
    print(text, file=file) # Write to file
def tee1e(text1, file):
    print(text1, end="") # Write to console
    print(text1, end="", file=file) # Write to file
def tee2(text1, test2, file):
    print(text1, test2) # Write to console
    print(text1, test2, file=file) # Write to file
def tee2e(text1, test2, file):
    print(text1, test2, end="") # Write to console
    print(text1, test2, end="", file=file) # Write to file
def tee3(text1, test2, test3, file):
    print(text1, test2, test3) # Write to console
    print(text1, test2, test3, file=file) # Write to file
def tee3e(text1, test2, test3, file):
    print(text1, test2, test3, end="") # Write to console
    print(text1, test2, test3, end="", file=file) # Write to file
def tee4(text1, test2, test3, text4, file):
    print(text1, test2, test3, text4) # Write to console
    print(text1, test2, test3, text4, file=file) # Write to file
def tee4e(text1, test2, test3, text4, file):
    print(text1, test2, test3, text4, end="") # Write to console
    print(text1, test2, test3, text4, end="", file=file) # Write to file
def tee5(text1, test2, test3, text4, text5, file):
    print(text1, test2, test3, text4, text5) # Write to console
    print(text1, test2, test3, text4, text5, file=file) # Write to file
def add_timestamp_to_filename(filename):
    current_datetime = datetime.datetime.now() # Get the current date and time
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")# Format the date and time as a string (e.g., "2023-08-04_12-34-56")
    root, extension = os.path.splitext(filename)# Extract the file extension (if any) from the original file name
    timestamped_filename = f"{formatted_datetime}_{root}{extension}"# Concatenate the formatted datetime with the original file name and extension (if any)
    return timestamped_filename
if not os.path.exists(os.path.join('Sample_Datasets', 'Logs')):
    os.makedirs(os.path.join('Sample_Datasets', 'Logs'))
ConsoleFile = open(os.path.join("Sample_Datasets", "Logs", add_timestamp_to_filename('log.txt')),'w')
def torch_fix_seed(seed=42):
    random.seed(seed)# Python random
    np.random.seed(seed)# Numpy
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
def update_font_size(event, label):
    new_font_size = int(event.width / 20)  # You can adjust the scaling factor as needed
    label.config(font=("Arial", new_font_size))  # Update the label's font size
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
def App_Close():
    tki.quit()
def btn9_click(counter):
    global lbl_5, lbl_50, lbl_51, lbl_55, lbl_59, lbl_64, lbl_80, lbl_81, lbl_90, lbl_91, lbl_5Vx, lbl_MVR, lblCont, lblPMUCov
    global rdo2F0inx, rdo2F1inx, rdo2A0inx, rdo2A1inx, rdo5F0inx, rdo5F1inx, rdo5A0inx, rdo5A1inx, rdo5V0inx, rdo5V1inx, rdo5V2inx, rdo5V3inx
    global rdo5EncN, rdo5EncY, rdoMVR0, rdoMVR1, rdoMVR2, rdoCont1, rdoCont2, rdoPMUFul, rdoPMUPar
    global txt_1, txt_2, txt_3, txt_50, txt_57, txt_59, txt_64, txt_80, txt_81, txt_90, txt_91  # Make txt_1 and txt_2
    global btn02, btn021, btn22, btn23, btn26, btn50, btn52, btn54, btn56, btn59, btn80, btn81, btn90, btn054
    global var1, var26, var27, var56, var68, var86
    global model, GAT1model,GAT2model,GAT3model,GAT4model,GCN1model,GCN2model,GCN3model,GCN4model,GCN5model,GCN6model,SGA2model,SGA3model,SGA4model,SGA5model,MLPmodel
    global cmp_list3, MsmchList
    global args_Ainx, args_logits
    global NumBus
    global file_path4, file_pathML, file_nameML, file_name3, file_path22, file_path52
    global WindowW, WindowH
    ### For GitHub ###
    from tkinter import Tk, IntVar
    var26 = IntVar()
    var56 = IntVar()
    var86 = IntVar()
    var56.set(2)
    var86.set(0)
    var68 = IntVar()
    txt_59 = tk.Entry()
    txt_59.pack()
    txt_59.delete(0, tk.END)
    txt_91 = tk.Entry()
    txt_91.pack()
    txt_91.insert(tk.END, "100")
    ### For GitHub ###
    var26.set(var56.get()) # synchronize the selected index between Training and Testing tabs
    args_Ainx = var56.get()
    args_inx  = Index_map.get(args_Ainx)
    if not counter:
        tee2("args_index:", args_inx, ConsoleFile)
    if os.path.exists(os.path.join(temp, "MdlType.npy")):
        args_mdl = MDL_map.get(int(np.load(os.path.join(temp, 'MdlType.npy'))), "GCN2")
    else:
        args_mdl = "GCN2"
    if not counter:
        tee2(args_mdl, " is selected.", ConsoleFile)
    if os.path.exists(os.path.join(temp, "OrdEnc.npy")):
        args_logits = (np.load(os.path.join(temp, 'OrdEnc.npy'))) #--logits
    else:
        args_logits = 1
    if not counter:
        tee2("Ordinal Encoder Usage Flag at the Beginning of btn9_click Subroutine:", args_logits, ConsoleFile)
    if os.path.exists(os.path.join(temp, "GpuID.npy")):
        args_gpu =    (np.load(os.path.join(temp, 'GpuID.npy')))  #--gpu
    else:
        args_gpu = 4
    if os.path.exists(os.path.join(temp, "Neuron.npy")):
        args_neur =   (np.load(os.path.join(temp, 'Neuron.npy')))
    else:
        args_neur = 0
    if os.path.exists(os.path.join(temp, "GATHead.npy")):
        args_head =   (np.load(os.path.join(temp, 'GATHead.npy')))
    else:
        args_head = 0
    if os.path.exists(os.path.join(temp, "Batch.npy")):
        args_batch =  (np.load(os.path.join(temp, 'Batch.npy')))
    else:
        args_batch = 504
    if os.path.exists(os.path.join(temp, "EdgeNeuron.npy")):
        args_e_neur = (np.load(os.path.join(temp, 'EdgeNeuron.npy')))
    else:
        args_e_neur = 32
    args_bus = "0"             #--bus
    sys.stout=ConsoleFile

    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super().__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
            #tee(self.data, ConsoleFile) #  Output torch.load Loaded dataset data
        @property
        def raw_file_names(self):
            # pass #  Out of commission pass, Will be submitted to the join() argument must be str or bytes, not 'NoneType' error 
            return []
        @property
        def processed_file_names(self):
            return ['datas.pt']
        def download(self):
            pass
        def process(self):
            if os.path.exists("data/processed/datas.pt"):
                #tee("Processed files found. Loading the dataset...", ConsoleFile)
                self.data, self.slices = torch.load("data/processed/datas.pt")
            else:
                def open_error_window7(text_msg):
                    global nwn66
                    nwn66 = tk.Toplevel()
                    winW, winH = 450, 40
                    winSize = str(winW)+"x"+str(winH)
                    nwn66.geometry(winSize)
                    nwn66.title('Error')
                    lbl_166 = tk.Label(nwn66, text=text_msg, fg='red', font=font0)
                    lbl_166.place(x=20, y=10)
                    btn166 = tk.Button(nwn66, text='OK', command=nwn66.destroy, font=font0)
                    btn166.place(x=400, y= 7)
                def open_warn_window8(text_msg1,text_msg2,text_msg3):
                    global nwn86
                    nwn86 = tk.Toplevel()
                    winW, winH = 550, 80
                    winSize = str(winW)+"x"+str(winH)
                    nwn86.geometry(winSize)
                    nwn86.title('Warning')
                    lbl_186 = tk.Label(nwn86, text=text_msg1, fg='red', font=font0)
                    lbl_186.place(x=20, y=10)
                    lbl_187 = tk.Label(nwn86, text=text_msg2, fg='red', font=font0)
                    lbl_187.place(x=20, y=30)
                    lbl_188 = tk.Label(nwn86, text=text_msg3, fg='red', font=font0)
                    lbl_188.place(x=20, y=50)
                    btn186 = tk.Button(nwn86, text='OK', command=nwn86.destroy, font=font0)
                    btn186.place(x=450, y=10)

                edge_index0 = np.genfromtxt(os.path.join(file_path22, 'EdgeIndex.csv'), delimiter=',')
                NumBus = int(np.max(edge_index0))+1
                src1 = np.hstack([edge_index0[:,0],edge_index0[:,1]])
                dst1 = np.hstack([edge_index0[:,1],edge_index0[:,0]])
                edge_index1 = torch.tensor([src1, dst1], dtype=torch.long)
                ###########################################################################################
                # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE #
                ###########################################################################################
                if os.path.exists(file_path52):
                    with open(file_path52, 'r') as file_N:
                        NSel = np.array([int(line.strip()) for line in file_N])
                    if os.path.exists(os.path.join(temp, 'NodeFeatureList.npy')):
                        open_warn_window8("Two different sources are found for node feature selection.",\
                        "File-based selection has been employed.",\
                        "Please remove 'node_feature.lst' when you want to use manual selections.")
                    if not counter:
                        tee2("Reading from ",file_path52,ConsoleFile)
                elif os.path.exists(os.path.join(temp, 'NodeFeatureList.npy')):
                    NSel = np.load(os.path.join(temp, 'NodeFeatureList.npy'))
                    if not counter:
                        tee2("Reading from NodeFeatureList.npy",ConsoleFile)
                else:
                    if   int(var86.get()) == 0: # IEEE 14 Finx
                        NSel = np.array([0,1,1,1,1,1,0,1,0,0,0])
                    elif int(var86.get()) == 1: # IEEE 14 Vinx
                        NSel = np.array([0,1,1,1,1,1,0,0,0,0,0])
                    else:                       # IEEE 118
                        NSel = np.array([1,1,1,1,1,1,1,1,1,1,1])
                    if not counter:
                        tee("Default node feature selection has been employed",ConsoleFile)
                if not counter:
                    tee2("NodeSwitch", NSel, ConsoleFile)
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.DV.node.csv')):
                    DV  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.DV.node.csv'), delimiter=',')
                elif NSel[0] == 1:
                    tee("DV (voltage drop) not found", ConsoleFile)
                    open_error_window7("DV (hypothesized volt. drop) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.PG.node.csv')):
                    PG  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.PG.node.csv'), delimiter=',')
                elif NSel[1] == 1:
                    tee("PG not found", ConsoleFile)
                    open_error_window7("PG (active power gen.) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.PL.node.csv')):
                    PL  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.PL.node.csv'), delimiter=',')
                elif NSel[2] == 1:
                    tee("PL not found", ConsoleFile)
                    open_error_window7("PL (active power load) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.QG.node.csv')):
                    QG  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.QG.node.csv'), delimiter=',')
                elif NSel[3] == 1:
                    tee("QG not found", ConsoleFile)
                    open_error_window7("QG (reactive power gen.) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.QL.node.csv')):
                    QL  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.QL.node.csv'), delimiter=',')
                elif NSel[4] == 1:
                    tee("QL not found", ConsoleFile)
                    open_error_window7("QL (reactive power load) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.VM.node.csv')):
                    VM  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.VM.node.csv'), delimiter=',')
                elif NSel[5] == 1:
                    tee("VM not found", ConsoleFile)
                    open_error_window7("VM (voltage magnitude) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.VA.node.csv')):
                    VA  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.VA.node.csv'), delimiter=',')
                elif NSel[6] == 1:
                    tee("VA not found", ConsoleFile)
                    open_error_window7("VA (voltage angle) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.DPG.nod.csv')):
                    DPG = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.DPG.nod.csv'), delimiter=',')
                elif NSel[7] == 1:
                    tee("DPG not found", ConsoleFile)
                    open_error_window7("DPG (active power gen. change) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.DPL.nod.csv')):
                    DPL = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.DPL.nod.csv'), delimiter=',')
                elif NSel[8] == 1:
                    tee("DPL not found", ConsoleFile)
                    open_error_window7("DPL (active power load change) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.DQG.nod.csv')):
                    DQG = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.DQG.nod.csv'), delimiter=',')
                elif NSel[9] == 1:
                    tee("DQG not found", ConsoleFile)
                    open_error_window7("DQG (reactive power gen. change) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.DQL.nod.csv')):
                    DQL = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.DQL.nod.csv'), delimiter=',')
                elif NSel[10] == 1:
                    tee("DQL not found", ConsoleFile)
                    open_error_window7("DQL (reactive power load change) file not found.")
                if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.FD.node.csv')):
                    FDN = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.FD.node.csv'), delimiter=',')
                elif NSel[11] == 1:
                    tee("FDN not found", ConsoleFile)
                    open_error_window7("FD (fault duration) file not found.")
                # Missing data reflection in original data
                NumGraph = len(VM)
                # MANUALLY SPECIFIED STARTS **************************************************************************************************************************
                np.save(os.path.join(temp, 'ImputationMeasureSelect.npy'), np.array(int(args.imp))) # 0:Zero Imputation, 1:Peak replacement, 2: Pesudo measurement
                np.save(os.path.join(temp, 'ErrorPercent.npy'), np.array(int(args.per)))
                np.random.seed() 
                # MANUALLY SPECIFIED ENDS ****************************************************************************************************************************
                if os.path.exists(os.path.join(temp, 'ImputationMeasureSelect.npy')):
                    ### For GitHub ###var68 = tk.IntVar(master=tab5)
                    var68.set(np.load(os.path.join(temp, 'ImputationMeasureSelect.npy')))
                    if not counter:
                        tee2("Imputation Measure Type: ", var68.get(),ConsoleFile)
                print("Imputation Measure Type: ", var68.get())
                # Making missing data nodes as 1, otherwise 0
                if os.path.exists(os.path.join(temp, 'PMUmiss.npy')):
                    PMUmiss = np.load(os.path.join(temp, 'PMUmiss.npy'))
                    if NumBus == 118 or NumBus == 264: # IEEE 118-bus
                        PMUmissLoad = PMUmiss[:118] + np.insert(np.zeros(91),[4,6,6,6,19,19,19,22,28,28,50,51,51,51,53,53,54,54,54,61,66,67,68,75,86,87,89],2)
                        PMUmissGen  = PMUmiss[:118] + np.insert(np.zeros(53),[1,1,2,3,4,5,6,6,7,7,9,9,9,9,13,13,13,15,16,17,17,17,18,19,19,19,20,20,20,21,21,21,24,24,25,27,27,28,28,28,30,33,35,35,36,36,36,36,37,38,42,42,42,42,42,42,44,44,47,48,48,52,52,53,53],2)
                        PMUmissALL = np.concatenate((PMUmiss[:118], PMUmissLoad[PMUmissLoad <= 1], PMUmissGen[PMUmissGen <= 1], np.atleast_1d(PMUmiss[99])))
                        if not counter:
                            tee2("PMU Coverage [%]: ", (1-np.sum(PMUmiss)/118)*100, ConsoleFile)
                    else:                # IEEE 14-bus
                        PMUmissALL = PMUmiss
                        if not counter:
                            tee2("PMU Coverage [%]: ", (1-np.sum(PMUmiss)/NumBus)*100, ConsoleFile)
                if not counter:
                    tee2("# of Graphs",NumGraph, ConsoleFile)
                if os.path.exists(os.path.join(temp, 'ErrorPercent.npy')) and os.path.exists(os.path.join(temp, 'PMUmiss.npy')):
                    ErrFrac = np.load(os.path.join(temp, 'ErrorPercent.npy'))
                    if not counter:
                        tee2("Pesudo measurement with equivalent percent error of ", ErrFrac, ConsoleFile)
                if os.path.exists(os.path.join(temp, 'PMUmiss.npy')):
                    if   var68.get() == 1: # Peak value imputation
                        if not counter:
                            tee("Peak value replacement is currently selected.", ConsoleFile)
                        print("Peak value replacement is currently selected.")
                        for kk in range(len(PMUmissALL)):
                            if PMUmissALL[kk] == 1:
                                if not counter:
                                    pass #tee2(kk,"th Bus has no PMUs", ConsoleFile)
                                if NumBus == 118 or NumBus == 264: # IEEE 118-bus
                                    if NSel[1] == 1:
                                        PG[:,kk] = PG [NumGraph-7:NumGraph-6,kk]
                                    if NSel[2] == 1:
                                        PL[:,kk] = PL [NumGraph-7:NumGraph-6,kk]
                                    if NSel[3] == 1:
                                        QG[:,kk] = QG [NumGraph-7:NumGraph-6,kk]
                                    if NSel[4] == 1:
                                        QL[:,kk] = QL [NumGraph-7:NumGraph-6,kk]
                                    if NSel[5] == 1:
                                        VM[:,kk] = VM [NumGraph-7:NumGraph-6,kk]
                                    if NSel[6] == 1:
                                        VA[:,kk] = VA [NumGraph-7:NumGraph-6,kk]
                                    if NSel[7] == 1:
                                        DPG[:,kk]= DPG[NumGraph-7:NumGraph-6,kk]
                                    if NSel[8] == 1:
                                        DPL[:,kk]= DPL[NumGraph-7:NumGraph-6,kk]
                                    if NSel[9] == 1:
                                        DQG[:,kk]= DQG[NumGraph-7:NumGraph-6,kk]
                                    if NSel[10] == 1:
                                        DQL[:,kk]= DQL[NumGraph-7:NumGraph-6,kk]
                                else: # IEEE 14-bus
                                    if NSel[1] == 1:
                                        PG[:,kk] = PG [NumGraph-1:NumGraph,kk]
                                    if NSel[2] == 1:
                                        PL[:,kk] = PL [NumGraph-1:NumGraph,kk]
                                    if NSel[3] == 1:
                                        QG[:,kk] = QG [NumGraph-1:NumGraph,kk]
                                    if NSel[4] == 1:
                                        QL[:,kk] = QL [NumGraph-1:NumGraph,kk]
                                    if NSel[5] == 1:
                                        VM[:,kk] = VM [NumGraph-1:NumGraph,kk]
                                    if NSel[6] == 1:
                                        VA[:,kk] = VA [NumGraph-1:NumGraph,kk]
                                    if NSel[7] == 1:
                                        DPG[:,kk]= DPG[NumGraph-1:NumGraph,kk]
                                    if NSel[8] == 1:
                                        DPL[:,kk]= DPL[NumGraph-1:NumGraph,kk]
                                    if NSel[9] == 1:
                                        DQG[:,kk]= DQG[NumGraph-1:NumGraph,kk]
                                    if NSel[10] == 1:
                                        DQL[:,kk]= DQL[NumGraph-1:NumGraph,kk]
                    elif var68.get() == 2: # error
                        ErrFrac = np.load(os.path.join(temp, 'ErrorPercent.npy'))
                        if not counter:
                            tee2("Pesudo measurement is currently selected. Error percentage is ", ErrFrac, ConsoleFile)
                        print("Pesudo measurement is currently selected. Error percentage is ", ErrFrac)
                        for kk in range(len(PMUmissALL)):
                            if PMUmissALL[kk] == 1:
                                if NumBus == 118 or NumBus == 264: # IEEE 118-bus

                                    if NSel[1] == 1:
                                        PG  = np.hstack([PG [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=PG [:,kk:kk+1], scale=np.max(np.abs(PG [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),PG [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[2] == 1:
                                        PL  = np.hstack([PL [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=PL [:,kk:kk+1], scale=np.max(np.abs(PL [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),PL [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[3] == 1:
                                        QG  = np.hstack([QG [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=QG [:,kk:kk+1], scale=np.max(np.abs(QG [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),QG [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[4] == 1:
                                        QL  = np.hstack([QL [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=QL [:,kk:kk+1], scale=np.max(np.abs(QL [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),QL [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[5] == 1:
                                        VM  = np.hstack([VM [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=VM [:,kk:kk+1], scale=np.max(np.abs(VM [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),VM [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[6] == 1:
                                        VA  = np.hstack([VA [:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=VA [:,kk:kk+1], scale=np.max(np.abs(VA [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),VA [:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[7] == 1:
                                        DPG = np.hstack([DPG[:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=DPG[:,kk:kk+1], scale=np.max(np.abs(PG [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),DPG[:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[8] == 1:
                                        DPL = np.hstack([DPL[:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=DPL[:,kk:kk+1], scale=np.max(np.abs(PL [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),DPL[:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[9] == 1:
                                        DQG = np.hstack([DQG[:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=DQG[:,kk:kk+1], scale=np.max(np.abs(QG [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),DQG[:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                    if NSel[10] == 1:
                                        DQL = np.hstack([DQL[:,0:kk].reshape(NumGraph,kk),np.random.normal(loc=DQL[:,kk:kk+1], scale=np.max(np.abs(QL [:, kk:kk+1]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),DQL[:,kk+1:264].reshape(NumGraph,264-kk-1)])
                                else: # IEEE 14-bus
                                    if NSel[1] == 1:
                                        PG  = np.hstack((PG [:,0:kk],np.random.normal(loc=PG [:,kk:kk+1], scale=np.abs(PG [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),PG [:,kk+1:NumBus]))
                                    if NSel[2] == 1:
                                        PL  = np.hstack([PL [:,0:kk],np.random.normal(loc=PL [:,kk:kk+1], scale=np.abs(PL [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),PL [:,kk+1:NumBus]])
                                    if NSel[3] == 1:
                                        QG  = np.hstack([QG [:,0:kk],np.random.normal(loc=QG [:,kk:kk+1], scale=np.abs(QG [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),QG [:,kk+1:NumBus]])
                                    if NSel[4] == 1:
                                        QL  = np.hstack([QL [:,0:kk],np.random.normal(loc=QL [:,kk:kk+1], scale=np.abs(QL [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),QL [:,kk+1:NumBus]])
                                    if NSel[5] == 1:
                                        VM  = np.hstack([VM [:,0:kk],np.random.normal(loc=VM [:,kk:kk+1], scale=np.abs(VM [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),VM [:,kk+1:NumBus]])
                                    if NSel[6] == 1:
                                        VA  = np.hstack([VA [:,0:kk],np.random.normal(loc=VA [:,kk:kk+1], scale=np.abs(VA [:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),VA [:,kk+1:NumBus]])
                                    if NSel[7] == 1:
                                        DPG = np.hstack([DPG[:,0:kk],np.random.normal(loc=DPG[:,kk:kk+1], scale=np.abs(DPG[:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),DPG[:,kk+1:NumBus]])
                                    if NSel[8] == 1:
                                        DPL = np.hstack([DPL[:,0:kk],np.random.normal(loc=DPL[:,kk:kk+1], scale=np.abs(DPL[:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),DPL[:,kk+1:NumBus]])
                                    if NSel[9] == 1:
                                        DQG = np.hstack([DQG[:,0:kk],np.random.normal(loc=DQG[:,kk:kk+1], scale=np.abs(DQG[:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),DQG[:,kk+1:NumBus]])
                                    if NSel[10] == 1:
                                        DQL = np.hstack([DQL[:,0:kk],np.random.normal(loc=DQL[:,kk:kk+1], scale=np.abs(DQL[:, kk:kk+1]*float(ErrFrac)*0.01/3.0)),DQL[:,kk+1:NumBus]])
                    elif var68.get() == 0: # Zero value imputation
                        if not counter:
                            tee("Zero value imputation is currently selected.", ConsoleFile)
                        for kk in range(len(PMUmissALL)):
                            if PMUmissALL[kk] == 1:
                                if not counter:
                                    print(kk+1,"th Bus has no PMUs")
                                if NSel[1] == 1:
                                    PG[:,kk]=0.0
                                if NSel[2] == 1:
                                    PL[:,kk]=0.0
                                if NSel[3] == 1:
                                    QG[:,kk]=0.0
                                if NSel[4] == 1:
                                    QL[:,kk]=0.0
                                if NSel[5] == 1:
                                    VM[:,kk]=0.0
                                if NSel[6] == 1:
                                    VA[:,kk]=0.0
                                if NSel[7] == 1:
                                    DPG[:,kk]=0.0
                                if NSel[8] == 1:
                                    DPL[:,kk]=0.0
                                if NSel[9] == 1:
                                    DQG[:,kk]=0.0
                                if NSel[10] == 1:
                                    DQL[:,kk]=0.0
                    #else:
                        #tee("Warning: No PMU coverage method is going to be applied.", ConsoleFile)
                if   NSel[0]==0 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([      PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==0 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],      PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==0 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],      QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==0 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],      QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],      VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==0 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],      VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==0 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],      DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],       DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==0 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],       DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==0 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],       DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i]       ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],              DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==1 and NSel[9]==0 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],       DPL[i],       DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==0 and NSel[9]==1 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],       DQG[i]       ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i]              ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i]                            ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==0 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],            VM[i],VA[i]                            ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==0 and NSel[2]==0 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],            QG[i],QL[i],VM[i],VA[i]                            ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==0 and NSel[3]==1 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],PG[i],      QG[i],      VM[i],VA[i]                            ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==0 and NSel[2]==1 and NSel[3]==0 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([DV[i],      PL[i],      QL[i],VM[i],VA[i]                            ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==0 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],            VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==0 and NSel[3]==1 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],      QG[i],      VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==1 and NSel[3]==0 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],            VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==0 and NSel[2]==0 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],            QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==1 and NSel[2]==0 and NSel[3]==1 and NSel[4]==0 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],PG[i],      QG[i],      VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==1 and NSel[1]==0 and NSel[2]==1 and NSel[3]==0 and NSel[4]==1 and NSel[5]==1 and NSel[6]==1 and NSel[7]==1 and NSel[8]==1 and NSel[9]==1 and NSel[10]==1:
                    NODE = [torch.tensor([DV[i],      PL[i],      QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                # for IEEE 14 (Voltage)
                elif NSel[0]==0 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==0 and NSel[7]==0 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([      PG[i],PL[i],QG[i],QL[i],VM[i]                                  ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                # for IEEE 14 (Frequency)
                elif NSel[0]==0 and NSel[1]==1 and NSel[2]==1 and NSel[3]==1 and NSel[4]==1 and NSel[5]==1 and NSel[6]==0 and NSel[7]==1 and NSel[8]==0 and NSel[9]==0 and NSel[10]==0:
                    NODE = [torch.tensor([      PG[i],PL[i],QG[i],QL[i],VM[i],      DPG[i]                     ],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif NSel[0]==0 or NSel[1]==0 or NSel[2]==0 or NSel[3]==0 or NSel[4]==0 or NSel[5]==0 or NSel[6]==0 or NSel[7]==0 or NSel[8]==0 or NSel[9]==0 or NSel[10]==0:
                    #tee("WARNING: Node feature selections are not in the list.", ConsoleFile)
                    open_error_window7("These node selections are not in the list.")
                else:
                    #ODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i],FDN[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                    #tee("All node features will be used.",ConsoleFile)
                if not counter:
                    tee ([VM.shape], ConsoleFile)
                    tee2("Node:     ",NODE[0].size(), ConsoleFile)
                ConsoleFile.flush()
                ###########################################################################################
                # EDGE FEATURE # EGE FEATURE # EDGE FEATURE # EDGE FEATURE # EDGE FEATURE # EDGE FEATURE #
                ###########################################################################################
                if os.path.exists(file_path53):
                    with open(file_path53, 'r') as file_B:
                        BSel = np.array([int(line.strip()) for line in file_B])
                    if os.path.exists(os.path.join(temp, 'BranchFeatureList.npy')):
                        open_warn_window8("Two different sources are found for branch feature selection.",\
                        "File-based selection has been employed.",\
                        "Please remove 'branch_feature.lst' when you want to use manual selections.")
                    if not counter:
                        tee2("Reading from ",file_path53,ConsoleFile)
                elif os.path.exists(os.path.join(temp, 'BranchFeatureList.npy')):
                    BSel = np.load(os.path.join(temp, 'BranchFeatureList.npy'))
                    if not counter:
                        tee2("Reading from BranchFeatureList.npy",ConsoleFile)
                else:
                    if   int(var86.get()) == 0: # IEEE 14 Finx
                        BSel = np.array([1,1,0,1,0])
                    elif int(var86.get()) == 1: # IEEE 14 Vinx
                        BSel = np.array([1,1,0,1,1])
                    else:                  # IEEE 118
                        BSel = np.array([1,1,1,0,0])
                    if not counter:
                        tee("Default branch feature selection has been employed",ConsoleFile)
                if not counter:
                    tee2("BranchSwitch", BSel, ConsoleFile)
                if   BSel[0]==1: #PS and PR
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.PS.edge.csv')):
                        PS  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.PS.edge.csv'), delimiter=',')
                    else:
                        tee("PS not found", ConsoleFile)
                        open_error_window7("PS (active power) file not found.")
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.PR.edge.csv')):
                        PR  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.PR.edge.csv'), delimiter=',')
                    else:
                        tee("PR not found", ConsoleFile)
                        open_error_window7("PR (active power) file not found.")
                    Ptie= np.hstack((PS,PR))
                if   BSel[1]==1: # QS and QR
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.QS.edge.csv')):
                        QS  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.QS.edge.csv'), delimiter=',')
                    else:
                        tee("QS not found", ConsoleFile)
                        open_error_window7("QR (reactive power) file not found.")
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.QR.edge.csv')):
                        QR  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.QR.edge.csv'), delimiter=',')
                    else:
                        tee("QR not found", ConsoleFile)
                        open_error_window7("QR (reactive power) file not found.")
                    Qtie= np.hstack((QS,QR))
                if   BSel[2]==1: # FD
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.FD.edge.csv')):
                        FD  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.FD.edge.csv'), delimiter=',')
                    else:
                        tee("FD not found", ConsoleFile)
                        open_error_window7("FD (fault duration) file not found.")
                    Dtie= np.hstack((FD,FD))
                if   BSel[3]==1: # AS and AR
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.AS.edge.csv')):
                        AS  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.AS.edge.csv'), delimiter=',')
                    else:
                        tee("AS not found", ConsoleFile)
                        open_error_window7("AS (voltage angle) file not found.")
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.AR.edge.csv')):
                        AR  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.AR.edge.csv'), delimiter=',')
                    else:
                        tee("AR not found", ConsoleFile)
                        open_error_window7("AR (voltage angle) file not found.")
                    Atie= np.hstack((AS,AR))
                if   BSel[4]==1: # SW
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.CS.edge.csv')):
                        CS  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.CS.edge.csv'), delimiter=',')
                    else:
                        tee("CS not found", ConsoleFile)
                        open_error_window7("CS (voltage angle) file not found.")
                    if os.path.exists(os.path.join(file_path22, 'AllFeature_GCN.CR.edge.csv')):
                        CR  = np.genfromtxt(os.path.join(file_path22, 'AllFeature_GCN.CR.edge.csv'), delimiter=',')
                    else:
                        tee("CR not found", ConsoleFile)
                        open_error_window7("CR (voltage angle) file not found.")
                    Stie= np.hstack((CS,CR))
                if os.path.exists(os.path.join(temp, "ContingencyCase.prn")):
                    with open(os.path.join(temp, "ContingencyCase.prn"), "r") as file: ContingencyCaseName = file.read()
                    if ContingencyCaseName[0:3] == "050" or ContingencyCaseName[0:3] == "200" or ContingencyCaseName[0:3] == "367":
                        pass
                    else:
                        FD[FD==1.000] = int(ContingencyCaseName[0:3])*0.005
                        #tee("Fault Duration of 50 [ms] is adjusted to "+ContingencyCaseName[0:3]+" [ms].", ConsoleFile)
                #Ptie= np.hstack((PS,PR))
                #Qtie= np.hstack((QS,QR))
                #Dtie= np.hstack((FD,FD))
                #tee ([PS.shape, PR.shape], ConsoleFile)
                if os.path.exists(os.path.join(temp, 'PMUmiss.npy')):
                    #src = np.array([0,0,1,3,1,1,2,3,3, 5, 5,4, 5,6,8, 8,6, 9,11,12])
                    #dst = np.array([1,4,2,6,3,4,3,8,4,10,11,5,12,7,9,13,8,10,12,13])
                    if NumBus == 14:
                        NoPMUedge = [None] * 14 
                        NoPMUedge[  0] = np.array([  1,  4])
                        NoPMUedge[  1] = np.array([  0,  2,  3,  4])
                        NoPMUedge[  2] = np.array([  1,  3])
                        NoPMUedge[  3] = np.array([  1,  2,  4,  6,  8])
                        NoPMUedge[  4] = np.array([  0,  1,  3,  5])
                        NoPMUedge[  5] = np.array([  4, 10, 11, 12])
                        NoPMUedge[  6] = np.array([  3,  7,  8])
                        NoPMUedge[  7] = np.array([  6    ])
                        NoPMUedge[  8] = np.array([  3,  6,  9, 13])
                        NoPMUedge[  9] = np.array([ 10    ])
                        NoPMUedge[ 10] = np.array([  5,  9])
                        NoPMUedge[ 11] = np.array([  5, 12])
                        NoPMUedge[ 12] = np.array([  5, 11, 13])
                        NoPMUedge[ 13] = np.array([  8, 12])
                        if   var68.get() == 1: # Peak value imputation
                            if not counter:
                                tee("Peak value replacement is currently selected.", ConsoleFile)
                            for kk in range(len(NoPMUedge)):
                                if PMUmissALL[kk] == 1:
                                    for jj in range(len(NoPMUedge[kk])):
                                        PS[:,NoPMUedge[kk][jj]-1]=PS[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                                        PR[:,NoPMUedge[kk][jj]-1]=PR[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                                        QS[:,NoPMUedge[kk][jj]-1]=QS[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                                        QR[:,NoPMUedge[kk][jj]-1]=QR[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                                        AS[:,NoPMUedge[kk][jj]-1]=AS[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                                        AR[:,NoPMUedge[kk][jj]-1]=AR[NumGraph-1:NumGraph,NoPMUedge[kk][jj]-1]
                        elif var68.get() == 2: # error
                            if not counter:
                                tee("Pesudo measurement is currently selected.", ConsoleFile)
                            ErrFrac = np.load(os.path.join(temp, 'ErrorPercent.npy'))
                            if not ErrFrac == 0:
                                for kk in range(len(NoPMUedge)):
                                    if PMUmissALL[kk] == 1:
                                        for jj in range(len(NoPMUedge[kk])):
                                            PS=np.hstack([PS[:,0:NoPMUedge[kk][jj]-1].reshape(NumGraph,NoPMUedge[kk][jj]-1),np.random.normal(loc=PS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.abs(PS[:,kk:kk+1]*float(ErrFrac)*0.01/3.0)).reshape(NumGraph,1),PS[:,NoPMUedge[kk][jj]: 20].reshape(NumGraph, 20-NoPMUedge[kk][jj])])
                                            QS=np.hstack([QS[:,0:NoPMUedge[kk][jj]-1].reshape(NumGraph,NoPMUedge[kk][jj]-1),np.random.normal(loc=QS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.abs(QS[:,kk:kk+1]*float(ErrFrac)*0.01/3.0)).reshape(NumGraph,1),QS[:,NoPMUedge[kk][jj]: 20].reshape(NumGraph, 20-NoPMUedge[kk][jj])])
                                            AS=np.hstack([AS[:,0:NoPMUedge[kk][jj]-1].reshape(NumGraph,NoPMUedge[kk][jj]-1),np.random.normal(loc=AS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.abs(AS[:,kk:kk+1]*float(ErrFrac)*0.01/3.0)).reshape(NumGraph,1),AS[:,NoPMUedge[kk][jj]: 20].reshape(NumGraph, 20-NoPMUedge[kk][jj])])
                            Ptie= np.hstack((PS,PR))
                            Qtie= np.hstack((QS,QR))
                            Atie= np.hstack((AS,AR))
                        else: #   var68.get() == 0: # Zero value imputation
                            for kk in range(len(NoPMUedge)):
                                if PMUmissALL[kk] == 1:
                                    for jj in range(len(NoPMUedge[kk])):
                                        PS[:,NoPMUedge[kk][jj]-1]=PR[:,NoPMUedge[kk][jj]-1]=0.0
                                        QS[:,NoPMUedge[kk][jj]-1]=QR[:,NoPMUedge[kk][jj]-1]=0.0
                                        AS[:,NoPMUedge[kk][jj]-1]=AR[:,NoPMUedge[kk][jj]-1]=0.0
                            Ptie= np.hstack((PS,PR))
                            Qtie= np.hstack((QS,QR))
                            Atie= np.hstack((AS,AR))
                    else: # IEEE 118-bus
                        NoPMUedge = [None] * 118 
                        NoPMUedge[  0] = np.array([  1,  2,171,262])
                        NoPMUedge[  1] = np.array([  1,  3,172])
                        NoPMUedge[  2] = np.array([  2,  4,  5,173])
                        NoPMUedge[  3] = np.array([  6,  7,174,263])
                        NoPMUedge[  4] = np.array([  4,  6,  8,  9,317])
                        NoPMUedge[  5] = np.array([  8, 10,175,264])
                        NoPMUedge[  6] = np.array([ 10, 11,176])
                        NoPMUedge[  7] = np.array([ 12, 13,265,317])
                        NoPMUedge[  8] = np.array([ 12, 14])
                        NoPMUedge[  9] = np.array([ 14,266])
                        NoPMUedge[ 10] = np.array([  7,  9, 15, 16,177])
                        NoPMUedge[ 11] = np.array([  3,  5, 11, 15, 17, 18, 19,178,267])
                        NoPMUedge[ 12] = np.array([ 16, 20,179])
                        NoPMUedge[ 13] = np.array([ 17, 21,180])
                        NoPMUedge[ 14] = np.array([ 20, 21, 22, 23, 24,181,268])
                        NoPMUedge[ 15] = np.array([ 18, 25,182])
                        NoPMUedge[ 16] = np.array([ 22, 25, 26, 27, 28,183,319])
                        NoPMUedge[ 17] = np.array([ 26, 29,184,269])
                        NoPMUedge[ 18] = np.array([ 23, 29, 30, 31,185,270])
                        NoPMUedge[ 19] = np.array([ 30, 32,186])
                        NoPMUedge[ 20] = np.array([ 32, 33,187])
                        NoPMUedge[ 21] = np.array([ 33, 34,188])
                        NoPMUedge[ 22] = np.array([ 34, 35, 36, 37,189])
                        NoPMUedge[ 23] = np.array([ 35, 38, 39])
                        NoPMUedge[ 24] = np.array([ 36, 40,272,318])
                        NoPMUedge[ 25] = np.array([ 41,273])
                        NoPMUedge[ 26] = np.array([ 40, 42, 43, 44,190,274])
                        NoPMUedge[ 27] = np.array([ 42, 45,191])
                        NoPMUedge[ 28] = np.array([ 45, 46,192])
                        NoPMUedge[ 29] = np.array([ 13, 41, 47,319])
                        NoPMUedge[ 30] = np.array([ 27, 46, 48,193,275])
                        NoPMUedge[ 31] = np.array([ 37, 43, 48, 49, 50,194,276])
                        NoPMUedge[ 32] = np.array([ 24, 51,195])
                        NoPMUedge[ 33] = np.array([ 31, 52, 53, 54,196,277])
                        NoPMUedge[ 34] = np.array([ 55, 56,197])
                        NoPMUedge[ 35] = np.array([ 52, 55,198,278])
                        NoPMUedge[ 36] = np.array([ 51, 53, 56, 57, 58,320])
                        NoPMUedge[ 37] = np.array([ 47, 59,320])
                        NoPMUedge[ 38] = np.array([ 57, 60,199])
                        NoPMUedge[ 39] = np.array([ 58, 60, 61, 62,200,279])
                        NoPMUedge[ 40] = np.array([ 61, 63,201])
                        NoPMUedge[ 41] = np.array([ 62, 63, 64,202,280])
                        NoPMUedge[ 42] = np.array([ 54, 65,203])
                        NoPMUedge[ 43] = np.array([ 65, 66,204])
                        NoPMUedge[ 44] = np.array([ 66, 67, 68,205])
                        NoPMUedge[ 45] = np.array([ 67, 69, 70,206,281])
                        NoPMUedge[ 46] = np.array([ 69, 71, 72,207])
                        NoPMUedge[ 47] = np.array([ 70, 73,208])
                        NoPMUedge[ 48] = np.array([ 64, 68, 71, 73, 74, 75, 76, 77, 78,209,282])
                        NoPMUedge[ 49] = np.array([ 74, 79,210])
                        NoPMUedge[ 50] = np.array([ 75, 80, 81,211])
                        NoPMUedge[ 51] = np.array([ 80, 82,212])
                        NoPMUedge[ 52] = np.array([ 82, 83,213])
                        NoPMUedge[ 53] = np.array([ 76, 83, 84, 85, 86,214,283])
                        NoPMUedge[ 54] = np.array([ 84, 87, 88,215,284])
                        NoPMUedge[ 55] = np.array([ 85, 87, 89, 90, 91,216,285])
                        NoPMUedge[ 56] = np.array([ 79, 89,217])
                        NoPMUedge[ 57] = np.array([ 81, 90,218])
                        NoPMUedge[ 58] = np.array([ 86, 88, 91, 92, 93,219,286,321])
                        NoPMUedge[ 59] = np.array([ 92, 94, 95,220])
                        NoPMUedge[ 60] = np.array([ 93, 94, 96,287,322])
                        NoPMUedge[ 61] = np.array([ 95, 96, 97, 98,221,288])
                        NoPMUedge[ 62] = np.array([ 99,321])
                        NoPMUedge[ 63] = np.array([ 99,100,322])
                        NoPMUedge[ 64] = np.array([ 59,100,101,289])
                        NoPMUedge[ 65] = np.array([ 77, 97,102,222,290,323])
                        NoPMUedge[ 66] = np.array([ 98,102,223])
                        NoPMUedge[ 67] = np.array([101,103,104,324])
                        NoPMUedge[ 68] = np.array([ 72, 78,105,106,107,291,324])
                        NoPMUedge[ 69] = np.array([ 38,105,108,109,110,224,292])
                        NoPMUedge[ 70] = np.array([108,111,112])
                        NoPMUedge[ 71] = np.array([ 39,111,293])
                        NoPMUedge[ 72] = np.array([112,294])
                        NoPMUedge[ 73] = np.array([109,113,225,295])
                        NoPMUedge[ 74] = np.array([106,110,113,114,115,226])
                        NoPMUedge[ 75] = np.array([116,117,227,296])
                        NoPMUedge[ 76] = np.array([107,114,116,118,119,120,228,297])
                        NoPMUedge[ 77] = np.array([118,121,229])
                        NoPMUedge[ 78] = np.array([121,122,230])
                        NoPMUedge[ 79] = np.array([119,122,123,124,125,126,231,298,325])
                        NoPMUedge[ 80] = np.array([103,325])
                        NoPMUedge[ 81] = np.array([120,127,128,232])
                        NoPMUedge[ 82] = np.array([127,129,130,233])
                        NoPMUedge[ 83] = np.array([129,131,234])
                        NoPMUedge[ 84] = np.array([130,131,132,133,134,235,299])
                        NoPMUedge[ 85] = np.array([132,135,236])
                        NoPMUedge[ 86] = np.array([135,300])
                        NoPMUedge[ 87] = np.array([133,136,237])
                        NoPMUedge[ 88] = np.array([134,136,137,138,301])
                        NoPMUedge[ 89] = np.array([137,139,238,302])
                        NoPMUedge[ 90] = np.array([139,140,303])
                        NoPMUedge[ 91] = np.array([138,140,141,142,143,144,239,304])
                        NoPMUedge[ 92] = np.array([141,145,240])
                        NoPMUedge[ 93] = np.array([142,145,146,147,148,241])
                        NoPMUedge[ 94] = np.array([146,149,242])
                        NoPMUedge[ 95] = np.array([123,128,147,149,150,243])
                        NoPMUedge[ 96] = np.array([124,150,244])
                        NoPMUedge[ 97] = np.array([125,151,245])
                        NoPMUedge[ 98] = np.array([126,152,305])
                        NoPMUedge[ 99] = np.array([143,148,151,152,153,154,155,156,246,306,307])
                        NoPMUedge[100] = np.array([153,157,247])
                        NoPMUedge[101] = np.array([144,157,248])
                        NoPMUedge[102] = np.array([154,158,159,160,249,308])
                        NoPMUedge[103] = np.array([155,158,161,250,309])
                        NoPMUedge[104] = np.array([159,161,162,163,164,251,310])
                        NoPMUedge[105] = np.array([156,162,165,252])
                        NoPMUedge[106] = np.array([163,165,253,311])
                        NoPMUedge[107] = np.array([164,166,254])
                        NoPMUedge[108] = np.array([166,167,255])
                        NoPMUedge[109] = np.array([160,167,168,169,256,312])
                        NoPMUedge[110] = np.array([168,313])
                        NoPMUedge[111] = np.array([169,257,314])
                        NoPMUedge[112] = np.array([ 28, 49,315])
                        NoPMUedge[113] = np.array([ 50,170,258])
                        NoPMUedge[114] = np.array([ 44,170,259])
                        NoPMUedge[115] = np.array([104,316])
                        NoPMUedge[116] = np.array([ 19,260])
                        NoPMUedge[117] = np.array([115,117,261])
                        if   var68.get() == 1: # Peak value imputation
                            if not counter:
                                tee("Peak value replacement is currently selected.", ConsoleFile)
                            for kk in range(len(NoPMUedge)):
                                if PMUmissALL[kk] == 1:
                                    for jj in range(len(NoPMUedge[kk])):
                                        PS[:,NoPMUedge[kk][jj]-1]=PS[NumGraph-7:NumGraph-6,NoPMUedge[kk][jj]-1]
                                        #PR[:,NoPMUedge[kk][jj]-1]=PR[NumGraph-7:NumGraph-6,NoPMUedge[kk][jj]-1]
                                        QS[:,NoPMUedge[kk][jj]-1]=QS[NumGraph-7:NumGraph-6,NoPMUedge[kk][jj]-1]
                                        #QR[:,NoPMUedge[kk][jj]-1]=QR[NumGraph-7:NumGraph-6,NoPMUedge[kk][jj]-1]
                            Ptie= np.hstack((PS,PR))
                            Qtie= np.hstack((QS,QR))
                        elif var68.get() == 2: # error
                            ErrFrac = np.load(os.path.join(temp, 'ErrorPercent.npy'))
                            if not counter:
                                tee2("Pesudo measurement is currently selected with the error percentage of ", ErrFrac, ConsoleFile)
                            for kk in range(len(NoPMUedge)):
                                if PMUmissALL[kk] == 1:
                                    for jj in range(len(NoPMUedge[kk])):
                                        #PS=np.hstack([PS[:,0:NoPMUedge[kk][jj]-1].reshape(NumGraph,NoPMUedge[kk][jj]-1),np.random.normal(loc=PS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.max(np.abs(PS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),PS[:,NoPMUedge[kk][jj]:325].reshape(NumGraph,325-NoPMUedge[kk][jj])])
                                        #QS=np.hstack([PS[:,0:NoPMUedge[kk][jj]-1].reshape(NumGraph,NoPMUedge[kk][jj]-1),np.random.normal(loc=QS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.max(np.abs(QS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph,1),QS[:,NoPMUedge[kk][jj]:325].reshape(NumGraph,325-NoPMUedge[kk][jj])])
                                        PS[:,NoPMUedge[kk][jj]-1]=np.random.normal(loc=PS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.max(np.abs(PS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph)
                                        QS[:,NoPMUedge[kk][jj]-1]=np.random.normal(loc=QS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]],scale=np.max(np.abs(QS[:,NoPMUedge[kk][jj]-1:NoPMUedge[kk][jj]]))*float(ErrFrac)*0.01/3.0).reshape(NumGraph)
                            Ptie= np.hstack((PS,PR))
                            Qtie= np.hstack((QS,QR))
                        else: #   var68.get() == 0: # Zero value imputation
                            for kk in range(len(NoPMUedge)):
                                if PMUmissALL[kk] == 1:
                                    for jj in range(len(NoPMUedge[kk])):
                                        PS[:,NoPMUedge[kk][jj]-1]=0.0
                                        QS[:,NoPMUedge[kk][jj]-1]=0.0
                            Ptie= np.hstack((PS,PR))
                            Qtie= np.hstack((QS,QR))
                        #lse:
                        #   tee("Warning: probably failing to get PMU coverage method.", ConsoleFile)
                if   BSel[0]==0 and BSel[1]==1 and BSel[2]==1 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([          Qtie[i],  Dtie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==0 and BSel[2]==1 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],            Dtie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==0 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],  Qtie[i]          ], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==0 and BSel[1]==0 and BSel[2]==1 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([                    Dtie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==0 and BSel[1]==1 and BSel[2]==0 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([          Qtie[i]          ], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==0 and BSel[2]==0 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i]                    ], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==1 and BSel[3]==0 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],  Qtie[i], Dtie[i]          ], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==0 and BSel[2]==0 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],                     Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==0 and BSel[1]==1 and BSel[2]==0 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([          Qtie[i],           Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==0 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],  Qtie[i],           Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==1 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],  Qtie[i], Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==0 and BSel[2]==1 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([Ptie[i],           Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==0 and BSel[1]==1 and BSel[2]==1 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([          Qtie[i], Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==0 and BSel[1]==0 and BSel[2]==1 and BSel[3]==1 and BSel[4]==0:
                    EDGE  = [torch.tensor([                   Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==0 and BSel[3]==1 and BSel[4]==1:
                    EDGE  = [torch.tensor([Ptie[i],  Qtie[i], Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                else:
                    #tee("WARNING: Bran feature selections are not in the list.", ConsoleFile)
                    open_error_window7("These bran. selections are not in the list.")
                if not counter:
                    tee3("Edge:     ", EDGE[0].size(), Ptie.shape, ConsoleFile)
                ConsoleFile.flush()
                ###########################################################################################
                #  Label # Label # Label # Label # Label # Label # Label # Label # Label # Label # Label # 
                ###########################################################################################
                if   int(args_Ainx) == 0:
                    if not counter:
                        tee("Max of pp-Angle Label", ConsoleFile)
                    #BL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Angle2.inx.csv'),  delimiter=',') #Amax
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.AngleDiff.inx.csv'),  delimiter=',') #Amax
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
                        '''
                        if   180.0 > LBL0[kk] and LBL0[kk] >= 90.0:
                            OUTPUT0[kk] = 1
                        elif 90.0  > LBL0[kk] and LBL0[kk] >= 65.0:
                            OUTPUT0[kk] = 2
                        elif 65.0  > LBL0[kk] and LBL0[kk] >= 55.0:
                            OUTPUT0[kk] = 3
                        elif 55.0  > LBL0[kk] and LBL0[kk] >= 45.0:
                            OUTPUT0[kk] = 4
                        elif 45.0  > LBL0[kk] and LBL0[kk] >=-90.0:
                            OUTPUT0[kk] = 5
                        '''
                        if   180.0 > LBL0[kk] and LBL0[kk] >= 90.0:
                            OUTPUT0[kk] = 1
                        elif 90.0  > LBL0[kk] and LBL0[kk] >= 60.0:
                            OUTPUT0[kk] = 2
                        elif 60.0  > LBL0[kk] and LBL0[kk] >= 45.0:
                            OUTPUT0[kk] = 3
                        elif 45.0  > LBL0[kk] and LBL0[kk] >= 35.0:
                            OUTPUT0[kk] = 4
                        elif 35.0  > LBL0[kk] and LBL0[kk] >= 25.0:
                            OUTPUT0[kk] = 5
                        elif 25.0  > LBL0[kk] and LBL0[kk] >= 17.0:
                            OUTPUT0[kk] = 6
                        elif 17.0  > LBL0[kk] and LBL0[kk] >= 11.0:
                            OUTPUT0[kk] = 7
                        elif 11.0  > LBL0[kk] and LBL0[kk] >=  8.0:
                            OUTPUT0[kk] = 8
                        elif  8.0  > LBL0[kk] and LBL0[kk] >=  5.0:
                            OUTPUT0[kk] = 9
                        elif  5.0  > LBL0[kk] and LBL0[kk] >=  3.0:
                            OUTPUT0[kk] = 10
                        elif  3.0  > LBL0[kk] and LBL0[kk] >=  2.0:
                            OUTPUT0[kk] = 11
                        elif  2.0  > LBL0[kk] and LBL0[kk] >=  1.0:
                            OUTPUT0[kk] = 12
                        elif  1.0  > LBL0[kk] and LBL0[kk] >=-90.0:
                            OUTPUT0[kk] = 13
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                elif int(args_Ainx) == 1:
                    if not counter:
                        tee("Median of pp-Angle Label", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Angle2m.inx.csv'), delimiter=',') #Median of Amax
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
                        if   40.0  > LBL0[kk] and LBL0[kk] >= 30.0:
                            OUTPUT0[kk] = 1
                        elif 30.0  > LBL0[kk] and LBL0[kk] >= 25.0:
                            OUTPUT0[kk] = 2
                        elif 25.0  > LBL0[kk] and LBL0[kk] >= 20.0:
                            OUTPUT0[kk] = 3
                        elif 20.0  > LBL0[kk] and LBL0[kk] >= 15.0:
                            OUTPUT0[kk] = 4
                        elif 15.0  > LBL0[kk] and LBL0[kk] >=-90.0:
                            OUTPUT0[kk] = 5
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                elif   int(args_Ainx) == 2: # Frequency (large grid)
                    if not counter:
                        tee("Frequency nadir Label (large grid)", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Freq.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
                        if   -0.025> LBL0[kk] and LBL0[kk] >= -0.05:
                            OUTPUT0[kk] = 1
                        elif -0.05 > LBL0[kk] and LBL0[kk] >= -0.15:
                            OUTPUT0[kk] = 2
                        elif -0.15 > LBL0[kk] and LBL0[kk] >= -0.30:
                            OUTPUT0[kk] = 3
                        elif -0.30 > LBL0[kk] and LBL0[kk] >= -0.50:
                            OUTPUT0[kk] = 4
                        elif -0.50 > LBL0[kk] and LBL0[kk] >= -3.00:
                            OUTPUT0[kk] = 5
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                elif   int(args_Ainx) == 3: # Frequency (small grid)
                    if not counter:
                        tee("Frequency nadir Label (small grid)", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Freq.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
                        if   -0.25 > LBL0[kk] and LBL0[kk] >= -0.50:
                            OUTPUT0[kk] = 1
                        elif -0.50 > LBL0[kk] and LBL0[kk] >= -0.70:
                            OUTPUT0[kk] = 2
                        elif -0.70 > LBL0[kk] and LBL0[kk] >= -0.90:
                            OUTPUT0[kk] = 3
                        elif -0.90 > LBL0[kk] and LBL0[kk] >= -1.10:
                            OUTPUT0[kk] = 4
                        elif -1.10 > LBL0[kk] and LBL0[kk] >= -1.30:
                            OUTPUT0[kk] = 5
                        elif -1.30 > LBL0[kk] and LBL0[kk] >= -1.50:
                            OUTPUT0[kk] = 6
                        elif -1.50 > LBL0[kk] and LBL0[kk] >= -1.70:
                            OUTPUT0[kk] = 7
                        elif -1.70 > LBL0[kk] and LBL0[kk] >= -2.00:
                            OUTPUT0[kk] = 8
                        elif -2.00 > LBL0[kk]:
                            OUTPUT0[kk] = 9
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                elif   int(args_Ainx) == 5: # Voltage (post steady-state)
                    NumLbl0,NumLbl1,NumLbl2,NumLbl3,NumLbl4,NumLbl5,NumLbl6,NumLbl7,NumLbl8,NumLbl9 = 0,0,0,0,0,0,0,0,0,0
                    if not counter:
                        tee("Voltage post steady-state", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Volt0.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph,VM.shape[1]), dtype='int64')
                    # for 5 classes (Volt0)
                    for kk in range (NumGraph):# For Volt0
                        for jj in range (14):
                            if   1.070 >= LBL0[kk][jj] and LBL0[kk][jj] > 1.050:
                                OUTPUT0[kk][jj] = 1
                                NumLbl1 += 1
                            elif 1.050 >= LBL0[kk][jj] and LBL0[kk][jj] > 1.030:
                                OUTPUT0[kk][jj] = 2
                                NumLbl2 += 1
                            elif 1.030 >= LBL0[kk][jj] and LBL0[kk][jj] > 1.000:
                                OUTPUT0[kk][jj] = 3
                                NumLbl3 += 1
                            elif 1.000 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.900:
                                OUTPUT0[kk][jj] = 4
                                NumLbl4 += 1
                            elif 0.900 >= LBL0[kk][jj]:
                                OUTPUT0[kk][jj] = 5
                                NumLbl5 += 1
                    LABEL   = torch.tensor(OUTPUT0)
                elif   int(args_Ainx) == 6: # Voltage (post steady-state deviation)
                    NumLbl0,NumLbl1,NumLbl2,NumLbl3,NumLbl4,NumLbl5,NumLbl6,NumLbl7,NumLbl8,NumLbl9 = 0,0,0,0,0,0,0,0,0,0
                    if not counter:
                        tee("Voltage post steady-state deviation", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Volt1.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph,VM.shape[1]), dtype='int64')
                    # for 5 classes (Volt1)
                    for kk in range (NumGraph):# For Volt1
                        for jj in range (14):
                            if  LBL0[kk][jj] > 0.010:
                                OUTPUT0[kk][jj] = 0
                                NumLbl0 += 1
                            elif  0.010 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.005:
                                OUTPUT0[kk][jj] = 1
                                NumLbl1 += 1
                            elif  0.005 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.000:
                                OUTPUT0[kk][jj] = 2
                                NumLbl2 += 1
                            elif  0.000 >= LBL0[kk][jj] and LBL0[kk][jj] >-0.030:
                                OUTPUT0[kk][jj] = 3
                                NumLbl3 += 1
                            elif -0.030 >= LBL0[kk][jj] and LBL0[kk][jj] >-0.130:
                                OUTPUT0[kk][jj] = 4
                                NumLbl4 += 1
                            elif -0.130 >= LBL0[kk][jj]:
                                OUTPUT0[kk][jj] = 5
                                NumLbl5 += 1
                    LABEL   = torch.tensor(OUTPUT0)
                elif   int(args_Ainx) == 7: # Voltage (peak-to-peak)
                    NumLbl0,NumLbl1,NumLbl2,NumLbl3,NumLbl4,NumLbl5,NumLbl6,NumLbl7,NumLbl8,NumLbl9 = 0,0,0,0,0,0,0,0,0,0
                    if not counter:
                        tee("Voltage pp (Vmax-Vmin)", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Volt2.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph,VM.shape[1]), dtype='int64')
                    for kk in range (NumGraph):# For Volt2
                        for jj in range (14):
                            if  LBL0[kk][jj] > 0.400:
                                OUTPUT0[kk][jj] = 0
                                NumLbl0 += 1
                            elif  0.400 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.300:
                                OUTPUT0[kk][jj] = 1
                                NumLbl1 += 1
                            elif  0.300 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.200:
                                OUTPUT0[kk][jj] = 2
                                NumLbl2 += 1
                            elif  0.200 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.100:
                                OUTPUT0[kk][jj] = 3
                                NumLbl3 += 1
                            elif  0.100 >= LBL0[kk][jj] and LBL0[kk][jj] >-0.000:
                                OUTPUT0[kk][jj] = 4
                                NumLbl4 += 1
                            elif  0.000 >= LBL0[kk][jj]:
                                OUTPUT0[kk][jj] = 5
                                NumLbl5 += 1
                    LABEL = torch.tensor(OUTPUT0)
                elif   int(args_Ainx) == 8: # Voltage (dynamic deviation)
                    NumLbl0,NumLbl1,NumLbl2,NumLbl3,NumLbl4,NumLbl5,NumLbl6,NumLbl7,NumLbl8,NumLbl9 = 0,0,0,0,0,0,0,0,0,0
                    if not counter:
                        tee("Voltage dynamic deviation", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Volt4.inx.csv'), delimiter=',')
                    OUTPUT0= np.zeros((NumGraph,VM.shape[1]), dtype='int64')
                    for kk in range (NumGraph):# For Volt4
                         for jj in range (14):
                             if  LBL0[kk][jj] > 1.250:
                                 OUTPUT0[kk][jj] = 0
                                 NumLbl0 += 1
                             elif  1.250 >= LBL0[kk][jj] and LBL0[kk][jj] > 1.000:
                                 OUTPUT0[kk][jj] = 1
                                 NumLbl1 += 1
                             elif  1.000 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.100:
                                 OUTPUT0[kk][jj] = 2
                                 NumLbl2 += 1
                             elif  0.100 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.050:
                                 OUTPUT0[kk][jj] = 3
                                 NumLbl3 += 1
                             elif  0.050 >= LBL0[kk][jj] and LBL0[kk][jj] > 0.025:
                                 OUTPUT0[kk][jj] = 4
                                 NumLbl4 += 1
                             elif  0.025 >= LBL0[kk][jj]:
                                 OUTPUT0[kk][jj] = 5
                                 NumLbl5 += 1
                    LABEL   = torch.tensor(OUTPUT0)
                else:
                    if not counter:
                        tee("Post-ontingency steady-state frequency Label (provisional)", ConsoleFile)
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.Angle.inx.csv'),  delimiter=',') #Post-steady-state Angle
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
                        if   180.0 > LBL0[kk] and LBL0[kk] >= 30.0:
                            OUTPUT0[kk] = 1
                        elif 30.0  > LBL0[kk] and LBL0[kk] >= 25.0:
                            OUTPUT0[kk] = 2
                        elif 25.0  > LBL0[kk] and LBL0[kk] >= 20.0:
                            OUTPUT0[kk] = 3
                        elif 20.0  > LBL0[kk] and LBL0[kk] >= 15.0:
                            OUTPUT0[kk] = 4
                        elif 15.0  > LBL0[kk] and LBL0[kk] >=-90.0:
                            OUTPUT0[kk] = 5
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                if not counter:
                    tee3("Labels:   ",LABEL[0].size(),OUTPUT0.shape, ConsoleFile)
                #  establish data data 
                data       = [Data(x=NODE[i], edge_index=edge_index1, edge_attr=EDGE[i], y=LABEL[i]) for i in range(NumGraph)]
                data_list  = [data[i] for i in range(NumGraph)]
                if not counter:
                    tee2("Data_list:",data_list[0], ConsoleFile)
                if self.pre_filter is not None: # pre_filter Function can manually filter out data objects before saving . Use cases may involve restrictions that data objects belong to a particular class . Default None
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None: # pre_transform The function applies the conversion before saving the data object to disk ( Therefore, it is best used for a large number of precomputations that need to be performed only once ), Default None
                    data_list = [self.pre_transform(data) for data in data_list]
                data, slices = self.collate(data_list) #  Use collate Convert function to large torch_geometric.data.Data object
                if not counter:
                    tee(self.processed_paths[0], ConsoleFile)
                    tee(type(data), ConsoleFile)
                torch.save((data, slices), self.processed_paths[0])
    ConsoleFile.flush()
    # Dataset object operation
    if os.path.exists(os.path.join(os.getcwd(),"data")):
        shutil.rmtree("data") #delete data folder

    b = MyOwnDataset("data") #Create a dataset object
    shutil.rmtree("data") #delete data folder
    #tee("Start creating testing dataset with or without PMU coverage ...", ConsoleFile)
    #torch_fix_seed(42)
    dataset = b
    data=dataset[0]
    FileList = open(os.path.join(file_path22,"N-1FullList.txt"), "r")
    N1_list  = FileList.readlines()
    FileList.close()
    N1_list = [line.strip() for line in N1_list]

    NumIndx = dataset.indices()
    ForIndx = [f"{index}" for index in NumIndx] #ForIndx = [f"{index + 1}" for index in NumIndx]
    if os.path.exists(os.path.join(temp, "ContingencyCase.prn")):
        ContList = open(os.path.join(temp, "ContingencyCase.prn"), "r")
        ContingencyCaseName  = ContList.readlines()
        ContList.close()
        ContingencyCaseName = [line.strip() for line in ContingencyCaseName]
        ContCaseList = np.empty(len(ContingencyCaseName))
        InitCounter = 0
        for pp in range(len(ContingencyCaseName)):
            for kk in range(len(ForIndx)):
                if ContingencyCaseName[pp][0:3] == "050" or ContingencyCaseName[pp][0:3] == "200" or ContingencyCaseName[pp][0:3] == "367":
                    if ContingencyCaseName[pp] == N1_list[int(ForIndx[kk])]:
                        if InitCounter == 0:
                            ConstCaseList = kk
                            #tee4(kk, ConstCaseList, N1_list[int(ForIndx[kk])], ContingencyCaseName[pp], ConsoleFile)
                            InitCounter = 1
                        else:
                            ConstCaseList = np.append(ConstCaseList,kk)
                            #tee4(kk, ConstCaseList, N1_list[int(ForIndx[kk])], ContingencyCaseName[pp], ConsoleFile)
                else:
                    if "050"+ContingencyCaseName[pp][3:] == N1_list[int(ForIndx[kk])]:
                        if InitCounter == 0:
                            ConstCaseList = kk
                            #tee4(kk, ConstCaseList, N1_list[int(ForIndx[kk])], ContingencyCaseName[pp], ConsoleFile)
                            InitCounter = 1
                        else:
                            ConstCaseList = np.append(ConstCaseList,kk)
                            #tee4(kk, ConstCaseList, N1_list[int(ForIndx[kk])], ContingencyCaseName[pp], ConsoleFile)
        test_dataset = b
        ConstCaseList = ConstCaseList[np.argsort(ConstCaseList)]
    else:
        #torch_fix_seed(42)
        NumGraph = len(b) # same as len(dataset)
        #if len(b)   < 1000:  #Frequency in IEEE 14-Bus
        if int(var86.get()) == 0:
            dataset = b
            if not txt_59.get() == "": # if contingency scenario file is existed ...
                #tee("Manual Sample Selection is used for contingency analysis.", ConsoleFile)
                test_dataset = dataset
            else:
                #tee3("Percentage Selection of ", txt_91.get(), "% is used for contingency analysis.", ConsoleFile)
                test_dataset = dataset[0:int(int(txt_91.get())/100.0*len(b))]
            Data10p  = len(test_dataset)
            NumGraph = len(test_dataset)
        elif int(var86.get()) == 1:
            dataset = b # previously dataset = b.shuffle()
            if not txt_59.get() == "": # if contingency scenario file is existed ...
                tee("Manual Sample Selection is used for contingency analysis.", ConsoleFile)
                test_dataset = dataset
            else:
                tee3("Percentage Selection of ", txt_91.get(), "% is used for contingency analysis.", ConsoleFile)
                test_dataset = dataset[0:int(int(txt_91.get())/100.0*len(b))]
            Data10p  = len(test_dataset)
            NumGraph = len(test_dataset)
            #tee2("Vinx in IEEE 14-bus Testing Dataset with Samples of ", len(test_dataset), ConsoleFile)
        else:                  # IEEE 118
            dataset = b # previously dataset = b.shuffle()
            Data10p = math.floor(NumGraph*.01)*10 #2520
            Data30p = math.floor(NumGraph*.01)*30 #7560
            #ata80p = NumGraph-Data10p*2
            test_dataset = dataset # previously dataset[NumGraph-Data30p:] # 30% data for testing
            #tee2("Health index in IEEE 118-bus Testing Dataset with Samples of ", len(test_dataset), ConsoleFile)
    if not counter:
        tee2("# of graphs (dataset):", len(test_dataset),                    ConsoleFile)      #25225
        tee2("# of featues (dataset):", test_dataset.num_features,           ConsoleFile)      #11
        tee2("# of node featues (dataset):", test_dataset.num_node_features, ConsoleFile)      #11
        tee2("# of edge featues (dataset):", test_dataset.num_edge_features, ConsoleFile)      #3
        tee2("# of classes/labels (dataset):", test_dataset.num_classes,     ConsoleFile)      #6
        tee2("# of nodes (data/one graph):", test_dataset[0].num_nodes,      ConsoleFile)      #25225
    ConsoleFile.flush()
    #torch_fix_seed(42)#torch.random.seed()
    class MLP2model(nn.Module): #MLP2 model
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature, int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.fc0  (data.x)# for Linear
            x = x.relu()          # for Linear
            x = self.fc0A (x)     # for Linear
            x = x.relu()          # for Linear
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class MLP3model(nn.Module): #MLP3 model
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature, int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0B  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.fc0  (data.x)# for Linear
            x = x.relu()          # for Linear
            x = self.fc0A (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0B (x)     # for Linear
            x = x.relu()          # for Linear
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class MLP4model(nn.Module): #MLP4 model
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature,  int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0B  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0C  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.fc0  (data.x)# for Linear
            x = x.relu()          # for Linear
            x = self.fc0A (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0B (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0C (x)     # for Linear
            x = x.relu()          # for Linear
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class MLP5model(nn.Module): #MLP5 model
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature, int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0B  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0C  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0D  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.fc0  (data.x)# for Linear
            x = x.relu()          # for Linear
            x = self.fc0A (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0B (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0C (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0D (x)     # for Linear
            x = x.relu()          # for Linear
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class MLPmodel(nn.Module): #MLP4 model
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature,  int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0B  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0C  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.fc0  (data.x)# for Linear
            x = x.relu()          # for Linear
            x = self.fc0A (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0B (x)     # for Linear
            x = x.relu()          # for Linear
            x = self.fc0C (x)     # for Linear
            x = x.relu()          # for Linear
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y
    class SGA2model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features
            num_classes  = dataset.num_classes
            edge_feature = dataset.num_edge_features
            Nhead = int(args_head)
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature, int(args_neur), nn1, aggr='mean')
            self.conv2 = GATv2Conv(int(args_neur),        int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 128) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  64) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  32)  # (128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv2(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)  # For graph classification only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x)  # For ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x), dim=1)
            return y
    class SGA3model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features
            num_classes  = dataset.num_classes
            edge_feature = dataset.num_edge_features
            Nhead = int(args_head)
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature, int(args_neur), nn1, aggr='mean')
            self.conv2 = GATv2Conv(int(args_neur),        int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv3 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 128) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  64) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  32)  # (128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv2(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv3(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)  # For graph classification only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x)  # For ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x), dim=1)
            return y
    class SGA4model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features
            num_classes  = dataset.num_classes
            edge_feature = dataset.num_edge_features
            Nhead = int(args_head)
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature, int(args_neur), nn1, aggr='mean')
            self.conv2 = GATv2Conv(int(args_neur),        int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv3 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv4 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 128) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  64) #(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,  32)  # (128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv2(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv3(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv4(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)  # For graph classification only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x)  # For ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x), dim=1)
            return y
    class SGA5model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features
            num_classes  = dataset.num_classes
            edge_feature = dataset.num_edge_features
            Nhead = int(args_head)
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature, int(args_neur), nn1, aggr='mean')
            self.conv2 = GATv2Conv(int(args_neur),        int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv3 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv4 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            self.conv5 = GATv2Conv(int(args_neur) * Nhead,int(args_neur),heads=Nhead,edge_dim=edge_feature)  # Pass edge_dim
            #   self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 32)  # (128,32)#128-32,32-16,16-class
            #   self.fc4 = torch.nn.Linear( 32, num_classes)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead,128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur) * Nhead, 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv2(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv3(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv4(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            x = self.conv5(x,      data.edge_index, data.edge_attr)  # Pass edge_attr
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)  # For graph classification only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x)  # For ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x), dim=1)
            return y
    # Try Graph attention network
    class GCN1model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature =  dataset.num_node_features # 6
            num_classes  =  dataset.num_classes       # 6 => 10
            edge_feature =  dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, 128), nn.ReLU(), nn.Linear(128, node_feature *int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.fc1 = torch.nn.Linear(int(args_neur),32)#(128,32)#128-32,32-16,16-class
            self.fc4 = torch.nn.Linear(32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = global_mean_pool(x, data.batch)#only GCN, no need for MLP
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                return self.fc4(x) #for ordinal encoder
            else:
                return F.log_softmax(self.fc4(x), dim=1)
    class GCN2model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.conv2 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = x.relu()
            x = self.conv2(x, data.edge_index)
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class GCN3model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.conv2 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv3 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = x.relu()
            x = self.conv2(x, data.edge_index)
            x = x.relu()
            x = self.conv3(x, data.edge_index)
            x = x.relu()
            x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class GCN4model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.conv2 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv3 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv4 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = x.relu()
            x = self.conv2(x, data.edge_index)
            x = x.relu()
            x = self.conv3(x, data.edge_index)
            x = x.relu()
            x = self.conv4(x, data.edge_index)
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class GCN5model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.conv2 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv3 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv4 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv5 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = x.relu()
            x = self.conv2(x, data.edge_index)
            x = x.relu()
            x = self.conv3(x, data.edge_index)
            x = x.relu()
            x = self.conv4(x, data.edge_index)
            x = x.relu()
            x = self.conv5(x, data.edge_index)
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    class GCN6model(nn.Module):
        def __init__(self):
            super().__init__()
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, int(args_e_neur)), nn.ReLU(), nn.Linear(int(args_e_neur), node_feature * int(args_neur)))
            self.conv1 = NNConv(node_feature,int(args_neur), nn1, aggr='mean')
            self.conv2 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv3 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv4 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv5 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            self.conv6 = GCNConv(int(args_neur),int(args_neur))#(128,128)
            if   int(args_neur) >= 256:
                self.fc1 = torch.nn.Linear(int(args_neur),128)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear(128, num_classes)
            elif int(args_neur) >= 128:
                self.fc1 = torch.nn.Linear(int(args_neur), 64)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 64, num_classes)
            else:
                self.fc1 = torch.nn.Linear(int(args_neur), 32)#(128,32)#128-32,32-16,16-class
                self.fc4 = torch.nn.Linear( 32, num_classes)
        def forward(self, data):
            x = self.conv1(data.x, data.edge_index, data.edge_attr)
            x = x.relu()
            x = self.conv2(x, data.edge_index)
            x = x.relu()
            x = self.conv3(x, data.edge_index)
            x = x.relu()
            x = self.conv4(x, data.edge_index)
            x = x.relu()
            x = self.conv5(x, data.edge_index)
            x = x.relu()
            x = self.conv6(x, data.edge_index)
            x = x.relu()
            if int(args_Ainx) < 5: # skip global mean pool for Vinx (node classificataion)
                x = global_mean_pool(x, data.batch)# for Graph Classification Only
            x = self.fc1(x)
            x = x.relu()
            if int(args_logits) > 0:
                y = self.fc4(x) #for ordinal encoder
            else:
                y = F.log_softmax(self.fc4(x),  dim=1)
            return  y 
    if int(args_gpu) == 4:
        torch.cuda.is_available = lambda : False#This is for disabling cuda and force us to use CPU.
    device = torch.device('cuda:'+str(args_gpu) if torch.cuda.is_available() else 'cpu')
    if not counter:
        tee(device, ConsoleFile)
        tee2("Ordinal Encoder Status Right Before Model Selection", args_logits, ConsoleFile)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # (1) GCN: GCN model, (2) MLP: MLP model
    if   str(args_mdl) == "GCN":
        model  = GCNmodel().to(device)
    elif str(args_mdl) == "GCN1":
        model  = GCN1model().to(device)
    elif str(args_mdl) == "GCN2":
        model  = GCN2model().to(device)
    elif str(args_mdl) == "GCN3":
        model  = GCN3model().to(device)
    elif str(args_mdl) == "GCN4":
        model  = GCN4model().to(device)
    elif str(args_mdl) == "GCN5":
        model  = GCN5model().to(device)
    elif str(args_mdl) == "GCN6":
        model  = GCN6model().to(device)
    elif str(args_mdl) == "GAT1":
        model  = GAT1model().to(device)
    elif str(args_mdl) == "GAT2":
        model  = GAT2model().to(device)
    elif str(args_mdl) == "GAT3":
        model  = GAT3model().to(device)
    elif str(args_mdl) == "GAT4":
        model  = GAT4model().to(device)
    elif str(args_mdl) == "GNN3":
        model  = GNN3model().to(device)
    elif str(args_mdl) == "SGA2":
        model  = SGA2model().to(device)
    elif str(args_mdl) == "SGA3":
        model  = SGA3model().to(device)
    elif str(args_mdl) == "SGA4":
        model  = SGA4model().to(device)
    elif str(args_mdl) == "SGA5":
        model  = SGA5model().to(device)
    elif str(args_mdl) == "MLP2":
        model  = MLP2model().to(device)
    elif str(args_mdl) == "MLP3":
        model  = MLP3model().to(device)
    elif str(args_mdl) == "MLP4":
        model  = MLP4model().to(device)
    elif str(args_mdl) == "MLP5":
        model  = MLP5model().to(device)
    elif str(args_mdl) == "MLP":
        model  = MLPmodel().to(device)
    if not counter:
        tee("Selected model structure:", ConsoleFile)
        tee(model, ConsoleFile)
    Bsize = int(args_batch)
    #tee2("Batch size (not used): ", Bsize, ConsoleFile)
    ConsoleFile.flush()
    def NewLogits(x,cls):
        y = torch.sigmoid(x).to(device)
        z = torch.ones_like(y)-y
        L = torch.mm(torch.log(y+1e-7),torch.triu(torch.ones((cls,cls))).to(device))\
          + torch.mm(torch.log(z+1e-7),torch.ones((cls,cls)).to(device)-torch.triu(torch.ones((cls,cls))).to(device))
        return(L)
    ConsoleFile.flush()
    def f1_score_multiclass(pred_labels, true_labels, num_classes):
        f1_scores = []
        for class_label in range(num_classes):
            # Calculate true positives, false positives, and false negatives for the current class
            tp = torch.sum((pred_labels == class_label) & (true_labels == class_label)).float()
            fp = torch.sum((pred_labels == class_label) & (true_labels != class_label)).float()
            fn = torch.sum((pred_labels != class_label) & (true_labels == class_label)).float()
            # Calculate precision, recall, and F1 score for the current class
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
            if isinstance(f1,  float):
                f1_scores.append(f1       ) # Append the F1 score for the current class to the list
            else:
                f1_scores.append(f1.item()) # Append the F1 score for the current class to the list
        return sum(f1_scores) / num_classes # Calculate the average F1 score across all classes
    def mcc_multiclass(pred_labels, true_labels, num_classes):
        mcc_scores = []
        for class_label in range(num_classes):
            # Calculate true positives, false positives, and false negatives for the current class
            tp = torch.sum((pred_labels == class_label) & (true_labels == class_label)).float()
            fp = torch.sum((pred_labels == class_label) & (true_labels != class_label)).float()
            fn = torch.sum((pred_labels != class_label) & (true_labels == class_label)).float()
            tn = torch.sum((pred_labels != class_label) & (true_labels != class_label)).float()
            # Calculate Matthews Correlation Coefficient
            numerator = tp * tn - fp * fn
            denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numerator / denominator if denominator != 0 else 0.0
            if isinstance(mcc, float):
                mcc_scores.append(mcc       )
            else:
                mcc_scores.append(mcc.item())
        return sum(mcc_scores) / num_classes
    ########################################################
    # TESTING #TESTING #TESTING #TESTING #TESTING #TESTING #
    ########################################################
    args_logits = np.loadtxt(file_pathML[:-4]+".enc", dtype=int, encoding='utf-8')
    #if args_logits == 0:
        #tee2("The model without ord. enc. ", file_pathML, ConsoleFile)
    #else:
        #tee2("The model with ord. enc. ",    file_pathML, ConsoleFile)
    if torch.cuda.is_available():
        model = torch.load(file_pathML, map_location=lambda storage, loc: storage.cuda(int(args_gpu)))
    else:
        model = torch.load(file_pathML, map_location='cpu')
    model.eval() #Ensure that the model is in evaluation mode
    if not counter:
        tee(model, ConsoleFile)
    test_loss, test_acc, test_f1, test_mcc = 0, 0, 0, 0
    cmp_test_cls, AllTestOuts = [], []
    if os.path.exists(os.path.join(temp, "ContingencyCase.prn")):
        test_loader  = DataLoader(test_dataset)
    else:
        edge_index0 = np.genfromtxt(os.path.join(file_path22, 'EdgeIndex.csv'), delimiter=',')
        NumBus = int(np.max(edge_index0))+1
        if   NumBus == 14 and int(args_Ainx) < 5: # Finx (graph classificataion)
            test_loader   = DataLoader(test_dataset) #86
        elif NumBus == 14 and int(args_Ainx) > 5: # Vinx (node classificataion)
            test_loader   = DataLoader(test_dataset,  batch_size=Data10p, shuffle=False, drop_last=True) #827
        else:
            test_loader   = DataLoader(test_dataset,  batch_size=int(Data10p/21), shuffle=False, drop_last=True) #2520
            #est_loader   = DataLoader(test_dataset,  batch_size=Data10p, shuffle=False, drop_last=True)
    test_indices = test_dataset.indices()
    formatted_indices = [f"{index    }" for index in test_indices]
    #tee("Start testing process ...", ConsoleFile)
    if os.path.exists(os.path.join(temp, "ContingencyCase.prn")):
        TestDataCounter, ConstListCounter = 0, 0
        with torch.no_grad(): 
            for data in test_loader:
                if TestDataCounter == int(ConstCaseList[ConstListCounter]):
                    data=data.to(device)
                    if int(args_logits) == 0:
                        out     = model(data)
                    else:
                        x       = model(data)
                        y       = NewLogits(x,dataset.num_classes)
                        out     = F.log_softmax(y, dim=-1)
                    AllTestOuts.append(softmax(out))
                    loss3       = F.nll_loss(out, data.y)
                    test_loss  += loss3.item()
                    pred        = out.argmax(dim=1)  # Use the class with highest probability.
                    test_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
                    cmp_test_cls.append([pred,data.y])
                    if ConstListCounter == len(ConstCaseList) - 1:
                        ConstListCounter = -1
                    if ConstListCounter < len(ConstCaseList):
                        #print("Contingency Case: ", ConstListCounter, "/", len(ConstCaseList))
                        ConstListCounter = ConstListCounter + 1
                TestDataCounter = TestDataCounter + 1
            if int(args_Ainx) < 5: # Finx (graph classificataion)
                batch_test_loss = test_loss / len(ContingencyCaseName)#len(test_loader.dataset)
                batch_test_acc  = test_acc  / len(ContingencyCaseName)#len(test_loader.dataset)
            else:                  # Vinx (node classificataion)
                #atch_test_loss = test_loss / (len(ContingencyCaseName) * dataset[0].num_nodes)
                #atch_test_acc  = test_acc  / (len(ContingencyCaseName) * dataset[0].num_nodes)
                batch_test_loss = test_loss / (len(ContingencyCaseName))# * dataset[0].num_nodes)
                batch_test_acc  = test_acc  / (len(ContingencyCaseName))# * dataset[0].num_nodes)
            sc3_            = cmp_test_cls
    else:
        all_preds, all_labels = [], []
        with torch.no_grad(): 
            for data in test_loader: 
                data=data.to(device)
                if int(args_logits) == 0:
                    out         = model(data)
                else:
                    x           = model(data)
                    y           = NewLogits(x,dataset.num_classes)
                    out         = F.log_softmax(y, dim=-1)
                AllTestOuts.append(softmax(out))
                loss3       = F.nll_loss(out, data.y)
                test_loss  += loss3.item()
                pred        = out.argmax(dim=1)  # Use the class with highest probability.
                all_preds.append(pred)
                all_labels.append(data.y)
                test_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
                cmp_test_cls.append([pred,data.y])
                #est_f1  += f1_score_multiclass(pred, data.y, dataset.num_classes)
                #est_mcc += mcc_multiclass     (pred, data.y, dataset.num_classes)
            if int(args_Ainx) < 5: # Finx (graph classificataion)
                batch_test_loss = test_loss / len(test_loader.dataset)
                batch_test_acc  = test_acc  / len(test_loader.dataset)
            else:                  # Vinx (node classificataion)
                batch_test_loss = test_loss / (len(test_loader.dataset) * dataset[0].num_nodes)
                batch_test_acc  = test_acc  / (len(test_loader.dataset) * dataset[0].num_nodes)
            sc3_            = cmp_test_cls
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            test_f1  = f1_score_multiclass(all_preds, all_labels, dataset.num_classes)
            test_mcc = mcc_multiclass     (all_preds, all_labels, dataset.num_classes)
    if int(args_Ainx) < 5: # Finx (graph classificataion)
        tee('Testing Loss: %.8f | Testing Acc: %.8f | Testing F1 Score: %.8f | Testing MCC: %.8f' % (batch_test_loss, batch_test_acc, test_f1, test_mcc), ConsoleFile)
    #else:                  # Vinx (node classificataion)
    #    if os.path.exists(os.path.join(temp, "ContingencyCase.prn")):
    #        tee('Testing Loss: %.8f | Testing Acc: %.8f' % (batch_test_loss, batch_test_acc), ConsoleFile)
    #    else:
    #        tee('Testing Loss: %.8f | Testing Acc: %.8f' % (batch_test_loss/dataset[0].num_nodes, batch_test_acc/dataset[0].num_nodes), ConsoleFile)
    ConsoleFile.flush()
def pmuerr_std(counter):
    global args_logits
    print("PMU coverage study starts ...")
    for jj in range(counter):
        print(jj)
        btn9_click(jj)
    current_datetime = datetime.datetime.now()
    tee2("Imputation type:", np.load(os.path.join(temp, 'ImputationMeasureSelect.npy')), ConsoleFile)
    tee("Type0: Zero imputation, Type1: Peak value replacement, Type2: Pesudo PMU measurement", ConsoleFile)
    if os.path.exists(os.path.join(temp, 'ErrorPercent.npy')) and os.path.exists(os.path.join(temp, 'PMUmiss.npy')):
        ErrFrac = np.load(os.path.join(temp, 'ErrorPercent.npy'))
        tee2("When Type 2, pesudo measurement with equivalent percent error of ", ErrFrac, ConsoleFile)
    if args_logits == 0:
        formatted_datetime = current_datetime.strftime(__file__+" with no ord. enc. is finished on %m/%d at %I:%M %p")
    else:
        formatted_datetime = current_datetime.strftime(__file__+" with ord. enc. is finished on %m/%d at %I:%M %p")
    tee(formatted_datetime, ConsoleFile)
    ConsoleFile.close()
# Create the main window

### GitHub ###
##global GM2024
GM2024 = 1
print("GM2024 Flag:",GM2024)
if os.path.exists(os.path.join(os.getcwd(), 'Sample_Datasets', 'tmp_files')):
    shutil.rmtree(os.path.join(os.getcwd(), 'Sample_Datasets', 'tmp_files'))
os.makedirs(os.path.join(os.getcwd(), 'Sample_Datasets', 'tmp_files'))
##global txt_24, file_path22 #Input File Path
file_path22 = os.path.join(os.getcwd(),'Sample_Datasets','Freq','TestingDataset')
tee2("Input folder path is ", file_path22, ConsoleFile)
##global txt_26, file_path52 # Node/Bus feature file
file_path52 = os.path.join(os.getcwd(),'Sample_Datasets','Freq','node_feature_default.lst')
tee2("Node feature file name that users will use is ", file_path52, ConsoleFile)
##global txt_28, file_path53 # Branch feature file
file_path53 = os.path.join(os.getcwd(),'Sample_Datasets','Freq','branch_feature_default.lst')
tee2("Branch feature file name that users will use is ", file_path53, ConsoleFile)
##global file_pathML


# MANUALLY SPECIFIED STARTS **************************************************************************************************************************
file_pathML = os.path.join(os.getcwd(),'Sample_Datasets','Freq','F1index','ENN32BNN64GAT4H4LR0.0001Enc1A.pth')
np.save(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files', 'MdlType.npy'), np.array(6))#0:"GCN2",1:"GCN5",2:"GCN6",3:"GCN3",4:"SGA2",5:"SGA3",6:"SGA4",7:"SGA5",8:"GCN4",9:"MLP"
np.save(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files', 'Neuron.npy'), np.array(64))     # Number of Neurons for Nodes
np.save(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files', 'EdgeNeuron.npy'), np.array(32)) # Number of Neurons for Edges
np.save(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files', 'GpuID.npy'), np.array(0))       # GPU ID, 4: CPU
np.save(os.path.join(temp, 'GATHead.npy'), np.array(4)) # Number of Heads of GAT Model
tee("PMU missing file is: Sample_Datasets/Freq/PMUmissSW.lst", ConsoleFile)
os.remove(os.path.join(os.getcwd(),   'Sample_Datasets','Freq','PMUmiss.lst'))
shutil.copy2(os.path.join(os.getcwd(),'Sample_Datasets','Freq','PMUmissSW.lst'), os.path.join(os.getcwd(),'Sample_Datasets','Freq','PMUmiss.lst'))
# MANUALLY SPECIFIED ENDS ****************************************************************************************************************************
tee2("Loaded ML model will be ", file_pathML, ConsoleFile)
# PMU Coverage
if os.path.exists(os.path.join(os.getcwd(), 'Sample_Datasets','Freq','PMUmiss.lst')):
    edge_index0 = np.genfromtxt(os.path.join(file_path22, 'EdgeIndex.csv'), delimiter=',')
    file_nameLST = os.path.join(os.getcwd(),'Sample_Datasets','Freq','PMUmiss.lst')
    NonePMU = [line.strip() for line in open(file_nameLST, 'r')]
    NoPMU = np.zeros(int(np.max(edge_index0))+1)
    for kk in range(len(NonePMU)):
        for pp in range(int(np.max(edge_index0))+1):
            if pp+1 == int(NonePMU[kk]):
                NoPMU[pp] = 1
    np.save(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files','PMUmiss.npy'), NoPMU)
    tee(np.load(os.path.join(os.getcwd(),'Sample_Datasets','tmp_files','PMUmiss.npy')),ConsoleFile)

tki = tk.Tk()
menu = tk.Menu(tki)
tki.config(menu=menu)

if GM2024==1:
    pmuerr_std(1)
