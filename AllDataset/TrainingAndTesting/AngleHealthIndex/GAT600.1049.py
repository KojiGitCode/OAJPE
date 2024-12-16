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
#import threading
import csv

Wstatus   = 0
Wstatus73 = 0
var1 = 0 # Chosen DEFAULT NN model, 6 is GAT V2, 0 is GCN2
### For GitHub ###
var_opt = 1 # Chosen DEFAULT optimizer as AdamX
OAJPE = 0
### For GitHub ###
txt_64 = ""
ContName = ""
NumBus=0
nwn1   = None
nwn2   = None
nwnLbl = None
file_path4  = os.getcwd()
file_path22 = os.getcwd() #"" #Folder name of the Input files
file_pathML2= os.getcwd() # ML Model Path during Training
temp        = os.path.join(file_path22, 'Sample_Datasets', 'tmp_files') # same as os.getcwd()+"tmp_files"
file_nameML = "dummy.pth"
file_name3  = "dummy.lst"
if os.path.exists(temp):
    shutil.rmtree(temp)
os.makedirs(temp)
args_logits = 0
WindowW, WindowH = 580, 500
Index_map = {0:"A0index",1:"A1index",2:"F0index",3:"F1index",4:"Vindex",5:"V0index",6:"V1index",7:"V2index",8:"V3index"}
MDL_map = {0:"GCN2",1:"GCN5",2:"GCN6",3:"GCN3",4:"SGA2",5:"SGA3",6:"SGA4",7:"SGA5",8:"GCN4",9:"MLP5",10:"MLP4",11:"MLP3",12:"MLP2"}

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
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)

def btn2_click(): #TRAINING#
    global txt_1, txt_2, txt_3, txt_24, txt_25, txt_27, txt_59, txt_57, txt_50, txt_80, txt_81, txt_90, txt_91, file_path22 # Make txt_1 and txt_2 global
    global var1, var26, var27, var56, var58, var_opt
    global lbl_5, lbl_5r, rlbl_29, lbl_25, lbl_55, lbl_59, lbl_50, lbl_51, lbl_80, lbl_81, lbl_90, lbl_91, lbl_5Vx, lbl_2Vx
    global lbl_Te011,lbl_Te012,lbl_Te013,lbl_Te021,lbl_Te022,lbl_Te023,lbl_Te031,lbl_Te032,lbl_Te033
    global lbl_Tr011,lbl_Tr012,lbl_Tr013,lbl_Tr021,lbl_Tr022,lbl_Tr023,lbl_Tr031,lbl_Tr032,lbl_Tr033
    global progress_label, progbar, Wstatus, nwn
    global model, GAT1model,GAT2model,GAT3model,GAT4model,GCN1model,GCN2model,GCN3model,GCN4model,GCN5model,GCN6model,SGA2model,SGA3model,SGA4model,SGA5model,MLP5model,MLP4model,MLP3model,MLP2model 
    global cmp_list3, MsmchList
    global btn01, btn02, btn021, btn22, btn23, btn024, btn025, btn026, btn028, btn26, tn29, btn50, btn52, btn054, btn56, btn59, btn80, btn81, btn90
    #lobal dataset
    global rdo2F0inx, rdo2F1inx, rdo2A0inx, rdo2V0inx, rdo2V1inx, rdo2V2inx, rdo2V3inx
    global rdo5F0inx, rdo5F1inx, rdo5A0inx, rdo5V0inx, rdo5V1inx, rdo5V2inx, rdo5V3inx
    global rdo5EncN,rdo5EncY
    global args_Ainx, args_logits, args_inx
    global file_path52, file_path53
    global do2F1inx,rdo2A0inx,lbl_2Vx,rdo2V0inx,rdo2V1inx,rdo2V2inx,rdo2V3inx,lbl_29,txt_24,lbl_25,btn23,btn22
    global btn01,btn024
    global WindowW, WindowH
    global OAJPE
    ### For GitHub ###
    from tkinter import Tk, IntVar
    var26 = IntVar()
    var56 = IntVar()
    var26.set(0)
    var_opt = IntVar()
    var_opt.set(1) # Adamx is used as the optimizer
    ### For GitHub ###
    if Wstatus == 0:
        Pref_Close(0)
    var56.set(var26.get()) # synchronize the selected index between Training and Testing tabs
    args_Ainx = var26.get()
    args_inx  = Index_map.get(args_Ainx)
    print("args_inx ",args_inx)
    args_mdl    = MDL_map.get(int(np.load(os.path.join(temp, 'MdlType.npy'))), "SGA4")
    print(str(args_mdl)+" ML model is selected.")
    args_logits = (np.load(os.path.join(temp, 'OrdEnc.npy'))) #--logits
    args_gpu    = (np.load(os.path.join(temp, 'GpuID.npy')))  #--gpu
    args_lr     = (np.load(os.path.join(temp, 'LrRate.npy'))) #lr
    args_epoch  = (np.load(os.path.join(temp, 'Epochs.npy'))) #
    args_neur   = (np.load(os.path.join(temp, 'Neuron.npy'))) #
    args_head   = (np.load(os.path.join(temp, 'GATHead.npy')))#
    args_batch  = (np.load(os.path.join(temp, 'Batch.npy')))  #
    args_e_neur = (np.load(os.path.join(temp, 'EdgeNeuron.npy'))) #
    args_bus = "0"             #--bus
    if int(args_logits) == 0:
        if var_opt.get() == 0:
            tee("GNN model with no ord. enc. and optimizer of ADAM is going to be used.",   ConsoleFile)
        else:
            tee("GNN model with no ord. enc. and optimizer of ADAMax is going to be used.", ConsoleFile)
    else:
        if var_opt.get() == 0:
            tee("GNN model with ord. enc. and optimizer of ADAM is going to be used.",      ConsoleFile)
        else:
            tee("GNN model with ord. enc. and optimizer of ADAMax is going to be used.",    ConsoleFile)
    if not os.path.exists(os.path.join(temp, "F0index")) and int(args_Ainx) == 2:
        os.makedirs(os.path.join(temp, "F0index"))
    if not os.path.exists(os.path.join(temp, "F1index")) and int(args_Ainx) == 3:
        os.makedirs(os.path.join(temp, "F1index"))
    if not os.path.exists(os.path.join(temp, "V0index")) and int(args_Ainx) == 5:
        os.makedirs(os.path.join(temp, "V0index"))
    if not os.path.exists(os.path.join(temp, "V1index")) and int(args_Ainx) == 6:
        os.makedirs(os.path.join(temp, "V1index"))
    if not os.path.exists(os.path.join(temp, "V2index")) and int(args_Ainx) == 7:
        os.makedirs(os.path.join(temp, "V2index"))
    if not os.path.exists(os.path.join(temp, "V3index")) and int(args_Ainx) == 8:
        os.makedirs(os.path.join(temp, "V3index"))
    if not os.path.exists(os.path.join(temp, "A0index")) and int(args_Ainx) == 0:
        os.makedirs(os.path.join(temp, "A0index"))
    if not os.path.exists(os.path.join(temp, "A1index")) and int(args_Ainx) == 1:
        os.makedirs(os.path.join(temp, "A1index"))
    sys.stout=ConsoleFile
    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super().__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
            tee(self.data, ConsoleFile) #  Output torch.load Loaded dataset data
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
            if os.path.exists(os.path.join(temp, "data", "processed", "datas.pt")):
                tee("Processed files found. Loading the dataset...", ConsoleFile)
                self.data, self.slices = torch.load(os.path.join(temp, "data", "processed", "datas.pt"))
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
                print("loading ", os.path.join(file_path22, 'EdgeIndex.csv'))
                edge_index0 = np.genfromtxt(os.path.join(file_path22, 'EdgeIndex.csv'), delimiter=',')
                src1 = np.hstack([edge_index0[:,0],edge_index0[:,1]])
                dst1 = np.hstack([edge_index0[:,1],edge_index0[:,0]])
                edge_index1 = torch.tensor([src1, dst1], dtype=torch.long)
                ###########################################################################################
                # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE # NODE FEATURE #
                ###########################################################################################
                if os.path.exists(file_path52):
                    with open(file_path52, 'r') as file_N:
                        NSel = np.array([int(line.strip()) for line in file_N])
                    tee2("Reading from ",os.path.join(file_path22, 'node_feature.lst'),ConsoleFile)
                else:
                    open_warn_window7(file_path52+" not found.")
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
                    tee("FD not found", ConsoleFile)
                    open_error_window7("FD (fault duration) file not found.")
                NumGraph = len(VM)
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
                    tee("WARNING: Node feature selections are not in the list.", ConsoleFile)
                    open_error_window7("These node selections are not in the list.")
                else:
                    #NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)

                    NODE = [torch.tensor([DV[i],PG[i],PL[i],QG[i],QL[i],VM[i],VA[i],DPG[i],DPL[i],DQG[i],DQL[i],FDN[i]],dtype=torch.float).permute(1,0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                tee ([VM.shape], ConsoleFile)
                tee2("Node:     ",NODE[0].size(), ConsoleFile)
                ###########################################################################################
                # EDGE FEATURE # EGE FEATURE # EDGE FEATURE # EDGE FEATURE # EDGE FEATURE # EDGE FEATURE #
                ###########################################################################################
                if os.path.exists(file_path53):
                    with open(file_path53, 'r') as file_B:
                        BSel = np.array([int(line.strip()) for line in file_B])
                    tee2("Reading from ",os.path.join(file_path22, 'branch_feature.lst'),ConsoleFile)
                else:
                    open_warn_window7(file_path53+" not found.")
                print("BranchSwitch", BSel, BSel.shape)
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
                #Ptie= np.hstack((PS,PR))
                #Qtie= np.hstack((QS,QR))
                #Dtie= np.hstack((FD,FD))
                tee ([PS.shape, PR.shape], ConsoleFile)
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
                    EDGE  = [torch.tensor([                             Dtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                elif BSel[0]==1 and BSel[1]==1 and BSel[2]==0 and BSel[3]==1 and BSel[4]==1:
                    EDGE  = [torch.tensor([Stie[i],  Ptie[i], Qtie[i],  Atie[i]], dtype=torch.float).permute(1, 0) for i in range(NumGraph)]#.transpose with permute(1, 0)
                else:
                    tee("WARNING: Bran feature selections are not in the list.", ConsoleFile)
                    open_error_window7("These bran. selections are not in the list.")
                tee3("Edge:     ", EDGE[0].size(), Ptie.shape, ConsoleFile)
                ###########################################################################################
                #  Label # Label # Label # Label # Label # Label # Label # Label # Label # Label # Label # 
                ###########################################################################################
                if   int(args_Ainx) == 0:
                    tee("Max of pp-Angle Label", ConsoleFile)
                    ### GitHub ###
                    #BL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.AngleDiff1012.inx.csv'),  delimiter=',') #Amax Bet. Buses 10-12
                    LBL0   = np.genfromtxt(os.path.join(file_path22, 'Labels_GCN.AngleDiff1049.inx.csv'),  delimiter=',') #Amax Bet. Buses 10-49
                    ### GitHub ###
                    OUTPUT0= np.zeros((NumGraph))
                    for kk in range (NumGraph):
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
                    LABEL = [torch.tensor(int(OUTPUT0[i]),dtype=torch.long) for i in range(NumGraph)]
                elif int(args_Ainx) == 1:
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
                tee3("Labels:   ",LABEL[0].size(),OUTPUT0.shape, ConsoleFile)
                # establish data data 
                data       = [Data(x=NODE[i], edge_index=edge_index1, edge_attr=EDGE[i], y=LABEL[i]) for i in range(NumGraph)]
                data_list  = [data[i] for i in range(NumGraph)]
                tee2("Data_list:",data_list[0], ConsoleFile)
                if self.pre_filter is not None: # pre_filter Function can manually filter out data objects before saving . Use cases may involve restrictions that data objects belong to a particular class . Default None
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None: # pre_transform The function applies the conversion before saving the data object to disk ( Therefore, it is best used for a large number of precomputations that need to be performed only once ), Default None
                    data_list = [self.pre_transform(data) for data in data_list]
                data, slices = self.collate(data_list) #  Use collate Convert function to large torch_geometric.data.Data object
                tee(self.processed_paths[0], ConsoleFile)
                tee(type(data), ConsoleFile)
                torch.save((data, slices), self.processed_paths[0])
    # Dataset object operation
    if os.path.exists(os.path.join("data")):
        shutil.rmtree(os.path.join("data"))
    b = MyOwnDataset("data") #Create a dataset object
    shutil.rmtree(os.path.join("data"))
    tee2("Dataset ==> ", b, ConsoleFile)
    tee("====", ConsoleFile)
    torch_fix_seed(42)
    dataset = b.shuffle()
    data=dataset[0]
    tee2("# of graphs (dataset):", len(dataset),                    ConsoleFile)      #25225
    tee2("# of featues (dataset):", dataset.num_features,           ConsoleFile)      #11
    tee2("# of node featues (dataset):", dataset.num_node_features, ConsoleFile)      #11
    tee2("# of edge featues (dataset):", dataset.num_edge_features, ConsoleFile)      #3
    tee2("# of classes/labels (dataset):", dataset.num_classes,     ConsoleFile)      #6
    tee2("# of nodes (data/one graph):", data.num_nodes,            ConsoleFile)      #25225
    NumGraph = len(dataset)
    if NumGraph   < 1000:  #Frequency in IEEE 14-Bus
        if not OAJPE==1:
            Data10p = int(math.floor(NumGraph*0.1)) #86
            Data80p = int(NumGraph-Data10p*2)       #694
        else:
            Data10p = 173 # 20% of whole dataset
            Data80p = 520 # 60% of whole dataset
        train_dataset, val_dataset = torch.utils.data.random_split(dataset[:NumGraph-Data10p],[Data80p, Data10p])
        test_dataset  = dataset[NumGraph-Data10p:] # 10% data for testing
    elif NumGraph < 10000: #Voltage in IEEE 14-Bus
        Data10p = int(NumGraph - round(NumGraph*.9, -1)) #827
        Data80p = int(round((NumGraph-Data10p)*8/9,-1))    #6580
        train_dataset, val_dataset = torch.utils.data.random_split(dataset[:NumGraph-Data10p],[Data80p, (NumGraph-Data10p)-Data80p])
        test_dataset  = dataset[NumGraph-Data10p:] # 10% data for testing
    else:                  # IEEE 118
        Data10p = math.floor(NumGraph*.01)*10 # 2520
        Data30p = math.floor(NumGraph*.01)*30 # 7560
        Data60p = NumGraph-Data10p-Data30p    #15145
        Data80p = NumGraph-Data10p*2 #20185
        train_dataset, val_dataset = torch.utils.data.random_split(dataset[:NumGraph-Data30p],[Data60p, Data10p])
        test_dataset  = dataset[NumGraph-Data30p:] # 10% data for testing

    torch_fix_seed(42)#torch.random.seed()
    tee4("Dataset:",type(train_dataset),type(val_dataset),type(dataset), ConsoleFile)
    class MLP2model(nn.Module): #MLP2 model
        def __init__(self):
            super().__init__()
            torch_fix_seed(42)
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
            torch_fix_seed(42)
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature, int(args_neur))# for Linear
            self.fc0A  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            self.fc0B  = torch.nn.Linear(int(args_neur),int(args_neur))         # for Linear
            #elf.fc1 = torch.nn.Linear(128,64)#(128,32)#128-32,32-16,16-class
            #elf.fc4 = torch.nn.Linear(64, num_classes)
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
            torch_fix_seed(42)
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            self.fc0   = torch.nn.Linear(node_feature, int(args_neur))# for Linear
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
            torch_fix_seed(42)
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
    class SGA2model(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
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
            torch.manual_seed(42)
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
            torch.manual_seed(42)
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
            torch.manual_seed(42)
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
            torch_fix_seed(42)
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature,  32), nn.ReLU(), nn.Linear( 32, node_feature * int(args_neur)))
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
            torch_fix_seed(42)
            node_feature = dataset.num_node_features # 6 => 5
            num_classes  = dataset.num_classes       # 10
            edge_feature = dataset.num_edge_features # 3
            nn1 = nn.Sequential(nn.Linear(edge_feature, 32), nn.ReLU(), nn.Linear(32, node_feature * int(args_neur)))
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
            torch_fix_seed(42)
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
            torch_fix_seed(42)
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
            torch_fix_seed(42)
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
    tee(device, ConsoleFile)
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
    elif str(args_mdl) == "SGA2":
        model  = SGA2model().to(device)
    elif str(args_mdl) == "SGA3":
        model  = SGA3model().to(device)
    elif str(args_mdl) == "SGA4":
        model  = SGA4model().to(device)
    elif str(args_mdl) == "SGA5":
        model  = SGA5model().to(device)
    elif str(args_mdl) == "MLP5":
        model  = MLP5model().to(device)
    elif str(args_mdl) == "MLP4":
        model  = MLP4model().to(device)
    elif str(args_mdl) == "MLP3":
        model  = MLP3model().to(device)
    elif str(args_mdl) == "MLP2":
        model  = MLP2model().to(device)
    Bsize = int(args_batch)
    tee2("Batch size: ", Bsize, ConsoleFile)
    tee(model, ConsoleFile)
    torch.save(model, os.path.join(temp, args_inx, 'model_tmp_acc.pth'))
    ConsoleFile.flush()

    if var_opt.get() == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args_lr))
    else:
        optimizer = torch.optim.Adamax(model.parameters(), lr=float(args_lr))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[50000], gamma=0.7)
    tee(scheduler.__dict__, ConsoleFile)
    def NewLogits(x,cls):
        y = torch.sigmoid(x).to(device)
        z = torch.ones_like(y)-y
        L = torch.mm(torch.log(y+1e-7),torch.triu(torch.ones((cls,cls))).to(device))\
          + torch.mm(torch.log(z+1e-7),torch.ones((cls,cls)).to(device)-torch.triu(torch.ones((cls,cls))).to(device))
        return(L)
    if int(args_epoch) < 4500:
        ManualPatience =  500
    else:
        ManualPatience = 1000
    patience    = 500 #10 
    triggertime = 0
    val_loss_stock, min_val_loss, max_val_acc = 100., 100., 0

    model_save_flg, save_flag = 0, 0
    loss, loss2, loss3 = 0.0, 0.0, 0.0
    train_loss_list, train_acc_list = [], []
    val_loss_list,   val_acc_list  = [], []
    train_acc_stock,  val_acc_stock  = 0,  0
    Stock_list1, Stock_list2 = np.zeros((1,5)),  np.zeros((1,5))
    Stock_out1,  Stock_out2  = np.zeros((1,dataset.num_classes)), np.zeros((1,dataset.num_classes))
    dots = 0
    epoch_stock = 0
    ###############################################################################
    # TRAINING AND VALIDATION # TRAINING AND VALIDATION # TRAINING AND VALIDATION #
    ###############################################################################
    if   NumGraph < 1000:  # Frequency index in IEEE 14-Bus
        val_loader       = DataLoader(val_dataset,   batch_size=Data10p,                    shuffle=False, drop_last=True)
        tee2("batch size during validation: ", Data10p, ConsoleFile)
    elif NumGraph < 10000: # Voltage index in IEEE 14-Bus
        val_loader       = DataLoader(val_dataset,   batch_size=(NumGraph-Data10p)-Data80p, shuffle=False, drop_last=True)
        tee2("batch size during validation (IEEE14): ", (NumGraph-Data10p)-Data80p, ConsoleFile)
    else:                 # Health index in IEEE 118-Bus
        val_loader       = DataLoader(val_dataset,   batch_size=int(Data10p/7),             shuffle=False, drop_last=True) #num_workers=4
        #al_loader       = DataLoader(val_dataset,   batch_size=Data10p,                      shuffle=False, drop_last=True) #num_workers=4
        tee4("batch size during training/validation (IEEE118): ", Bsize, "/", Data10p/7, ConsoleFile)
    for epoch in range(int(args_epoch)):#15001
        if os.path.exists(os.path.join(temp, "stop.txt")):
            break
        ConsoleFile.flush()
        model.train()
        optimizer.zero_grad()
        scheduler.step() #optimizer
        train_loss, train_acc = 0, 0
        if NumGraph < 1000: # Health index in IEEE 14-Bus
            train_loader = DataLoader(train_dataset, batch_size=Data80p, shuffle=True, drop_last=True) #num_workers=4
        elif NumGraph < 10000: # Health index in IEEE 14-Bus
            train_loader = DataLoader(train_dataset, batch_size=6580,    shuffle=True, drop_last=True) #num_workers=4
        else:               # Health index in IEEE 118-Bus
            #rain_loader = DataLoader(train_dataset, batch_size=Bsize,   shuffle=True, drop_last=True) #num_workers=4
            train_loader = DataLoader(train_dataset, batch_size=int(Data10p/7), shuffle=True, drop_last=True) #num_workers=4
        for data in train_loader:
            data=data.to(device)
            if int(args_logits) == 0:
                out  = model(data)  #data.x, data.edge_index, data.batch
            else:
                x    = model(data)  #data.x, data.edge_index, data.batch
                y    = NewLogits(x,dataset.num_classes)
                out  = F.log_softmax(y, dim=-1)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item()
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            train_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
        if int(args_Ainx) < 5: # Finx (graph classificataion)
            batch_train_loss = train_loss / len(train_loader.dataset)
            batch_train_acc  = train_acc  / len(train_loader.dataset)
        else:                  # Vinx (node classificataion)
            batch_train_loss = train_loss / (len(train_loader.dataset) * dataset[0].num_nodes)
            batch_train_acc  = train_acc  / (len(train_loader.dataset) * dataset[0].num_nodes)
        train_loss_list.append(batch_train_loss)
        train_acc_list.append(batch_train_acc)
        
        model.eval()
        val_loss, val_acc = 0, 0
        cmp_cls, AllOuts = [], []
        for data in val_loader:   # Iterate in batches over the validation dataset.
            data=data.to(device)
            if int(args_logits) == 0:
                out         = model(data)#data.x, data.edge_index, data.batch)
            else:
                x           = model(data)#data.x, data.edge_index, data.batch)
                y           = NewLogits(x,dataset.num_classes)
                out         = F.log_softmax(y, dim=-1)
            AllOuts.append(softmax(out))
            loss2      = F.nll_loss(out, data.y)
            val_loss  += loss2.item()
            pred       = out.argmax(dim=1)  # Use the class with highest probability.
            val_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
            cmp_cls.append([pred,data.y])
        if int(args_Ainx) < 5: # Finx (graph classificataion)
            batch_val_loss = val_loss / len(val_loader.dataset)
            batch_val_acc  = val_acc  / len(val_loader.dataset)
        else:                  # Vinx (node classificataion)
            batch_val_loss = val_loss / (len(val_loader.dataset) * dataset[0].num_nodes)
            batch_val_acc  = val_acc  / (len(val_loader.dataset) * dataset[0].num_nodes)
        val_loss_list.append(batch_val_loss)
        val_acc_list.append(batch_val_acc)
        if epoch - epoch_stock > ManualPatience: #if no model update in ManualPatience epochs, it is treated as overfitting.
            tee2("No more accuracy and loss improvement. Stopped at Epoch ", epoch, ConsoleFile)
            if torch.cuda.is_available():
                command = "nvidia-smi --id="+str(args_gpu)+" --query-gpu=memory.used --format=csv,noheader"
                output = subprocess.check_output(command, shell=True, text=True)
                tee2("GPU memory used [MB]: ", int(output.strip().split()[0]), ConsoleFile)
            break
        if batch_val_acc > val_acc_stock and batch_train_acc > 0.1: #0.7:#0.93 for GCN
            epoch_stock = epoch
            Stock_list2     = [epoch, batch_train_loss, batch_train_acc, batch_val_loss,  batch_val_acc]
            val_acc_stock   = batch_val_acc
            sc2_            = cmp_cls
            Stock_out2      = AllOuts[0][0:]
        if batch_train_acc > train_acc_stock:
            Stock_list1     = [epoch, batch_train_loss, batch_train_acc, batch_val_loss,  batch_val_acc]
            train_acc_stock = batch_train_acc
            sc1_            = cmp_cls
            Stock_out1      = AllOuts[0][0:]
        epoch_skip = min(max(1, int(int(args_epoch) / 180)), 50)
        if epoch % epoch_skip == 0:        
            tee('Epoch %5d/Epoch %5d | Training Loss: %.8f | Training Acc: %.8f | Validation Loss: %.8f | Validation Acc: %.8f' % (epoch_stock,epoch,batch_train_loss,batch_train_acc,batch_val_loss, batch_val_acc),ConsoleFile)
            ConsoleFile.flush()
        if batch_val_loss > val_loss_stock:
            triggertime += 1
            if triggertime >= patience:
                tee('Epoch %5d | Training Loss: %.8f | Training Acc: %.8f | Validation Loss: %.8f | Validation Acc: %.8f' % (epoch, batch_train_loss, batch_train_acc, batch_val_loss, batch_val_acc), ConsoleFile)
                tee(" Early stopping at epoch of", epoch, ConsoleFile)
                tee("Start to test process.", ConsoleFile)
                break
        else:
            if save_flag == 0:
                if args_logits == 0:
                    tee("First model with no ord. enc. is saved.", ConsoleFile)
                else:
                    tee("First model with ord. enc. is saved.",    ConsoleFile)
                if int(args_logits) == 0:
                    torch.save(model, os.path.join(temp, args_inx, 'model_enc_no.pth'))
                    save_flag   = 1
                else:
                    torch.save(model, os.path.join(temp, args_inx, 'model_enc_ay.pth'))
                    save_flag   = 1

            if save_flag ==1 and min_val_loss > batch_val_loss:
                min_loss_val_acc = batch_val_acc
                min_val_loss     = batch_val_loss
                if  batch_train_acc > 0.1:
                    epoch_stock = epoch
                    print("The best model1 is updated with the loss of {:.6f} and the accuracy of {:.6f} at Epoch {:.6g}.".format(batch_val_loss, batch_val_acc, epoch))
                if int(args_logits) == 0:
                    torch.save(model, os.path.join(temp, args_inx, 'model_enc_no.pth'))
                else:
                    torch.save(model, os.path.join(temp, args_inx, 'model_enc_ay.pth'))

            if save_flag ==1 and batch_val_acc > max_val_acc:
                max_val_acc  = batch_val_acc
                max_acc_val_loss = batch_val_loss # store validation loss
                epoch_stock2 = epoch
                print("The best model2 is updated with the loss of {:.6f} and the accuracy of {:.6f} at Epoch {:.6g}.".format(batch_val_loss, batch_val_acc, epoch))
                torch.save(model, os.path.join(temp, args_inx, 'model_tmp_acc.pth'))
            triggertime = 0
        val_loss_stock  = batch_val_loss
        if torch.cuda.is_available() and epoch == int(args_epoch)-1:
            command = "nvidia-smi --id="+str(args_gpu)+" --query-gpu=memory.used --format=csv,noheader"
            output = subprocess.check_output(command, shell=True, text=True)
            tee2("GPU memory used [MB]: ", int(output.strip().split()[0]), ConsoleFile)
    if save_flag == 0:
        tee("No best model", ConsoleFile)
        if int(args_logits) == 0:
            torch.save(model, os.path.join(temp, args_inx, 'model_enc_no.pth'))
        else:
            torch.save(model, os.path.join(temp, args_inx, 'model_enc_ay.pth'))
    else:
        tee4("Best model was saved at epoch of ", [epoch_stock,epoch_stock2], "with validation loss [LossMin/AccMax] of ", [min_val_loss,max_acc_val_loss], ConsoleFile)
    ConsoleFile.flush()
    cpu_data = []
    for i, lst in enumerate(sc1_):
        cpu_data.append(torch.stack(lst).cpu().numpy())
    sc1=np.transpose(np.array(cpu_data), (1, 0, 2)).reshape(-1,len(val_dataset))
    cpu_data = []
    for i, lst in enumerate(sc2_):
        cpu_data.append(torch.stack(lst).cpu().numpy())
    sc2=np.transpose(np.array(cpu_data), (1, 0, 2)).reshape(-1,len(val_dataset))

    #if not Stock_out2.sum  == 0:
    cmp_list1, cmp_list2 = np.array([0,0]), np.array([0,0])
    cmp_out1,  cmp_out2  = np.array([0,0]), np.array([0,0])
    for kk in range (len(val_dataset)):#if prediction and label are not matched ..., kk denotes number of training dataset
        if not sc1[0][kk] == sc1[1][kk]:
            cmp_list1 = np.vstack((cmp_list1,np.array([sc1[0][kk],sc1[1][kk]])))
        if not sc2[0][kk] == sc2[1][kk]:
            cmp_list2 = np.vstack((cmp_list2,np.array([sc2[0][kk],sc2[1][kk]])))
    AveHighAcc = 0;
    for jj in range(len(train_loss_list)):
        if train_loss_list[jj] == 1:
            AveHighAcc = val_loss_list[jj]
    AveHighAcc = [val_acc_list[jj] for jj in range(len(train_acc_list)) if train_acc_list[jj] == 1]
    if len(AveHighAcc) > 0:
        tee3("Averaged High Validation Accuracy", len(AveHighAcc), sum(AveHighAcc)/len(AveHighAcc), ConsoleFile)
    else:
        tee("No Averaged High Validation Accuracy", ConsoleFile)
    #ee4("Best Training Accuracy", Stock_list1[0], Stock_list1[2], Stock_list1[4], ConsoleFile)
    tee4("Best Training Accuracy", Stock_list1[0], Stock_list1[3], Stock_list1[4], ConsoleFile)
    ConsoleFile.flush()
    ########################################################
    # TESTING #TESTING #TESTING #TESTING #TESTING #TESTING #
    ########################################################
    # Highest Accuracy Model
    if torch.cuda.is_available():
        model = torch.load(os.path.join(temp, args_inx, 'model_tmp_acc.pth'), map_location=lambda storage, loc: storage.cuda(int(args_gpu)))
    else:
        model = torch.load(os.path.join(temp, args_inx, 'model_tmp_acc.pth'), map_location='cpu')
    test_acc, test_loss, max_acc_model, min_loss_model = 0, 0, 0, 0
    if NumGraph < 1000:
        test_loader   = DataLoader(test_dataset,  batch_size=Data10p, shuffle=False, drop_last=True) #86
    elif NumGraph < 10000:
        test_loader   = DataLoader(test_dataset,  batch_size=Data10p, shuffle=False, drop_last=True) #827
    else:
        test_loader   = DataLoader(test_dataset,  batch_size=int(Data30p/21), shuffle=False, drop_last=True) #2520
    with torch.no_grad(): 
        for data in test_loader: 
            data=data.to(device)
            if int(args_logits) == 0:
                out     = model(data)
            else:
                x       = model(data)
                y       = NewLogits(x,dataset.num_classes)
                out     = F.log_softmax(y, dim=-1)
            loss3       = F.nll_loss(out, data.y)
            test_loss  += loss3.item()
            pred        = out.argmax(dim=1)  # Use the class with highest probability.
            test_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
        if int(args_Ainx) < 5:
            max_acc_model   = test_acc  / len(test_loader.dataset)
        else:
            max_acc_model   = test_acc  / (len(test_loader.dataset) * dataset[0].num_nodes)
    # Lowest Loss Model
    if torch.cuda.is_available():
        if int(args_logits) == 0:
            tee("model with no ord. enc. is used for testing", ConsoleFile)
            model = torch.load(os.path.join(temp, args_inx, 'model_enc_no.pth'), map_location=lambda storage, loc: storage.cuda(int(args_gpu)))
        else:
            tee("model with ord. enc. is used for testing", ConsoleFile)
            model = torch.load(os.path.join(temp, args_inx, 'model_enc_ay.pth'), map_location=lambda storage, loc: storage.cuda(int(args_gpu)))
    else:
        if int(args_logits) == 0:
            tee("model with no ord. enc. is used for testing", ConsoleFile)
            model = torch.load(os.path.join(temp, args_inx, 'model_enc_no.pth'), map_location='cpu')
        else:
            tee("model with ord. enc. is used for testing", ConsoleFile)
            model = torch.load(os.path.join(temp, args_inx, 'model_enc_ay.pth'), map_location='cpu')
    test_loss, test_acc = 0, 0
    with torch.no_grad(): 
        for data in test_loader: 
            data=data.to(device)
            if int(args_logits) == 0:
                out     = model(data)
            else:
                x       = model(data)
                y       = NewLogits(x,dataset.num_classes)
                out     = F.log_softmax(y, dim=-1)
            loss3       = F.nll_loss(out, data.y)
            test_loss  += loss3.item()
            pred        = out.argmax(dim=1)  # Use the class with highest probability.
            test_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
        if int(args_Ainx) < 5:
            min_loss_model  = test_acc  / len(test_loader.dataset)
        else:                 
            min_loss_model  = test_acc  / (len(test_loader.dataset) * dataset[0].num_nodes)
    # Model Selection
    tee4("Accuracy of model with min. loss:     ", [min_val_loss, min_loss_val_acc], " at the epoch of ", epoch_stock,  ConsoleFile)
    tee4("Accuracy of model with max. accuracy: ", [max_acc_val_loss, max_val_acc],  " at the epoch of ", epoch_stock2, ConsoleFile)
    if min_val_loss  > max_acc_val_loss and epoch_stock < epoch_stock2:
        tee("Model with max. accuracy is selected.", ConsoleFile)
        if torch.cuda.is_available():
            model = torch.load(os.path.join(temp, args_inx, 'model_tmp_acc.pth'), map_location=lambda storage, loc: storage.cuda(int(args_gpu)))
        else:
            model = torch.load(os.path.join(temp, args_inx, 'model_tmp_acc.pth'), map_location='cpu')
    else:
        tee("Model with min. loss is selected.", ConsoleFile)
    torch.save(model, file_nameML)  #The model that is used for testing is saved as the specified file name.
    # "Follogin provisionally reecovered and activated" Start
    ##### MAIN Testing Process Starts #####
    test_loss, test_acc = 0, 0
    cmp_test_cls, AllTestOuts = [], []
    #est_loader  = DataLoader(test_dataset,  batch_size=Bsize, shuffle=False, drop_last=True) #2520

    test_indices = test_dataset.indices()
    formatted_indices = [f"{index + 1}" for index in test_indices]
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
            test_acc   += torch.sum(pred == data.y).item()# / dataset.num_classes
            cmp_test_cls.append([pred,data.y])
        if int(args_Ainx) < 5: # Finx (graph classificataion)
            batch_test_loss = test_loss / len(test_loader.dataset)
            batch_test_acc  = test_acc  / len(test_loader.dataset)
        else:                  # Vinx (node classificataion)
            batch_test_loss = test_loss / (len(test_loader.dataset) * dataset[0].num_nodes)
            batch_test_acc  = test_acc  / (len(test_loader.dataset) * dataset[0].num_nodes)
        sc3_            = cmp_test_cls

    tee('Testing Loss: %.8f | Testing Acc: %.8f' % (batch_test_loss, batch_test_acc), ConsoleFile)
    ConsoleFile.flush()
    if int(args_Ainx) < 5: 
        cpu_data = []
        for i, lst in enumerate(sc3_):
            cpu_data.append(torch.stack(lst).cpu().numpy())
        sc3=np.transpose(np.array(cpu_data), (1, 0, 2)).reshape(-1,len(test_dataset))
    else:                  
        sc3 = []
        for tensor in cmp_test_cls[0]:
            #c3.append (tensor.cpu())
            sc3.append (tensor.cpu().tolist())

    Stock_out3 = AllTestOuts[0:][0:]

    ttt0 = []
    for kk in range(len(Stock_out3)):
        ttt0.append(Stock_out3[kk].to('cpu').detach().numpy().copy())
    ttt = np.concatenate(ttt0, axis=0)
    FileList = open(os.path.join(file_path22, 'N-1FullList.txt'),"r")
    N1_list  = FileList.readlines()
    FileList.close()
    N1_list  = [line.strip() for line in N1_list]

    cmp_list3 = np.array([0,0])
    cmp_out3  = np.array([0,0])
    MsmchList = [[] for _ in range(3)]
    if int(args_Ainx) < 5: # graph classification
        for kk in range (len(test_dataset)):#if prediction and label are not matched ..., kk denotes number of Testing dataset
            if not sc3[0][kk] == sc3[1][kk]:
                cmp_list3 = np.vstack((cmp_list3,np.array([sc3[0][kk],sc3[1][kk]])))
                MsmchList[0].append(N1_list[int(formatted_indices[kk])])
                MsmchList[1].append(sc3[0][kk])
                MsmchList[2].append(sc3[1][kk])
    else:                  # node classification
        for kk in range (len(test_dataset)):
            for jj in range(dataset[0].num_nodes):
                pp = kk*dataset[0].num_nodes+jj
                if not sc3[0][pp] == sc3[1][pp]:
                    cmp_list3 = np.vstack((cmp_list3,np.array([sc3[0][pp],sc3[1][pp]])))
                    MsmchList[0].append(N1_list[int(formatted_indices[kk])])
                    MsmchList[1].append(sc3[0][pp])
                    MsmchList[2].append(sc3[1][pp])
    ConsoleFile.flush()
    np.set_printoptions(threshold=np.inf)
    tee2("Prediction:   ",cmp_list3.T[0,1:], ConsoleFile)
    tee2("Ground Truth: ",cmp_list3.T[1,1:], ConsoleFile)
    current_datetime = datetime.datetime.now()
    if args_logits == 0:
        formatted_datetime = current_datetime.strftime(__file__+" with no ord. enc. is finished on %m/%d at %I:%M %p")
    else:
        formatted_datetime = current_datetime.strftime(__file__+" with ord. enc. is finished on %m/%d at %I:%M %p")
    tee(formatted_datetime, ConsoleFile)
    #ConsoleFile.close()

def file_select52(): #btn80-lbl_80-txt_80, node/bus feature file selection in Testing
    global txt_80, file_path52
    txt_80.delete(0, tk.END)
    fTyp = [("","*.lst")]
    idir = os.path.join(os.getcwd(),'Sample_Datasets')
    file_path52 = tk.filedialog.askopenfilename(filetypes=[("Node Feature", "*.lst")], defaultextension=".lst", initialdir=idir, initialfile="node_feature_default.lst")
    if platform.system() == "Linux":
        pass
    else: #Windows
        file_path52 = file_path52.replace('/', '\\')
    tee2("Node feature file name that users will use is ", file_path52, ConsoleFile)
    txt_80.insert(tk.END, file_path52) #
def file_select53(): #btn81-lbl_81-txt_81, branch feature file selection in Testing
    global txt_81, file_path53
    txt_81.delete(0, tk.END)
    fTyp = [("","*.lst")]
    idir = os.path.join(os.getcwd(),'Sample_Datasets')
    file_path53 = tk.filedialog.askopenfilename(filetypes=[("Branch Feature", "*.lst")], defaultextension=".lst", initialdir=idir, initialfile="branch_feature_default.lst")
    if platform.system() == "Linux":
        pass
    else: #Windows
        file_path53 = file_path53.replace('/', '\\')
    tee2("Branch feature file name that users will use is ", file_path53, ConsoleFile)
    txt_81.insert(tk.END, file_path53) #

def Pref_Close(Flag):
    global Wstatus
    global var1, var2, var26, var31, var51, var86
    global txt_1, txt_2, txt_3, txt_14, txt_15, txt_16 
    global var_node_dict, var_bran_dict, var_opt
    if Flag == 1:
        pass
    else:
        ### For GitHub ###
        #p.save(os.path.join(temp, 'MdlType.npy'),        np.array(5))     # GNN Model Type for Angle between Buses 10-12
        np.save(os.path.join(temp, 'MdlType.npy'),        np.array(6))     # GNN Model Type for Angle between Buses 10-49
        ### For GitHub ###
        np.save(os.path.join(temp, 'OrdEnc.npy'),         np.array(1))     # Ordinary Encoder Usage
        np.save(os.path.join(temp, 'GpuID.npy'),          np.array(0))     # GPU ID
        np.save(os.path.join(temp, 'LrRate.npy'),         np.array(0.0001)) # Learning Rate
        np.save(os.path.join(temp, 'Epochs.npy'),         np.array(451))  # Number of Epochs
        np.save(os.path.join(temp, 'Neuron.npy'),         np.array(64))   # Number of Neurons
        np.save(os.path.join(temp, 'GATHead.npy'),        np.array(2))     # Number of heads of GAT
        np.save(os.path.join(temp, 'Batch.npy'),          np.array(504))   # Batch Size
        args_logits = 0                # Ord. enc.
        np.save(os.path.join(temp, 'NodeFeatureList.npy'),   np.ones(11))  # Node features
        np.save(os.path.join(temp, 'BranchFeatureList.npy'), np.ones( 3))  # Branch features
        np.save(os.path.join(temp, 'EdgeNeuron.npy'),        np.array(32))# Number of Neurons

print("OAJPE Flag:",OAJPE)
file_path22 = os.path.join(os.getcwd(),'Sample_Datasets','Angl','TrainingDataset')
tee2("Input folder path is ", file_path22, ConsoleFile)
global txt_26, file_path52 # Node/Bus feature file
file_path52 = os.path.join(os.getcwd(),'Sample_Datasets','Angl','node_feature_default.lst')
tee2("Node feature file name that users will use is ", file_path52, ConsoleFile)
global txt_28, file_path53 # Branch feature file
file_path53 = os.path.join(os.getcwd(),'Sample_Datasets','Angl','branch_feature_default.lst')
tee2("Branch feature file name that users will use is ", file_path53, ConsoleFile)

### For GitHub ### global file_nameML
ini_dir = os.path.join(os.getcwd(),'Sample_Datasets','Angl','A0index')
#IniFname = "GAT3H2LR.0001Enc1.pth" # For Angle Buses 10-12
IniFname = "GAT4H2LR.0001Enc1.pth" # For Angle Buses 10-49
file_nameML = IniFname ### For GitHub ### Skip specifying Model Name
### For GitHub ### global file_nameML

tki = tk.Tk()
btn2_click()
