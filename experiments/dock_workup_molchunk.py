

import os
import copy
import sys
import pathlib
from io import StringIO
import math
import time
from functools import reduce
from datetime import timedelta
from timeit import time

import pandas as pd
import pyarrow.feather as feather
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
import math
from rich import print
import pathlib
from rich.console import Console
import subprocess


import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetWriter

import numpy as np
import numpy.ma as ma

from scipy import sparse

from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from openbabel import pybel
from mdtraj.utils.delay_import import import_
import mols2grid

import re
import xml.etree.ElementTree as ET

import tqdm

import datamol as dm
import operator


output_docking_poses_ext = ".dlg"
scores_output_ext = ".xml"


def _collect_docking_outfiles_and_data(i, row):
#     print('hello')
#     try:
    name = row[11]
    pdbqt_block_string = row[9]
    dlg = f'{working_dir}/{name}_docked{output_docking_poses_ext}'
    xml_path = f'{working_dir}/{name}_docked{scores_output_ext}'

    tree = ET.parse(xml_path)
    root = tree.getroot()
    docking_score = [x.get('lowest_binding_energy') for x in root.findall(".//*[@cluster_rank='1']")]
    target_run = [y.get('run') for y in root.findall(".//*[@cluster_rank='1']")]
    target_run = target_run[0]
    file = open(dlg, "r")
#         print(file)
    lines = file.readlines()
#         print(len(lines))
#         target_dir = str(file_name.parent)
#     target_run = target_runs[count]


    starting_line = 0
    ending_line = 0
#         print(f'Extracting Run:   {target_run} / {num_runs}')
    run_target = str(f'Run:   {target_run} / {num_runs}')
#         print(run_target)
    for line in lines:
        if line.startswith(run_target):
#                 print('found starting line of target')
            starting_line= lines.index(line)
#                 print(f'The startomg line is {starting_line}')
            break

    if target_run != num_runs:
        # print('the target was not 10')
        for line in lines[starting_line:]:
            end_run_target = f'Run:   {(int(target_run) + 1)} / {num_runs}'
            if line.startswith(str(end_run_target)):
#                     print('found ending line of target')
                ending_line = lines.index(line)
#                     print(f'The ending line is {ending_line}')
                break

#             print(f' the starting line is {starting_line} and the ending line is {ending_line}')
        pdbqt_content = lines[(starting_line + 4):(ending_line - 9)]
#             print(f' the pdbqt content is found on lines {pdbqt_content}')
        stripped_pdbqt_content = [line.rstrip('\n') for line in pdbqt_content]


    if target_run == num_runs:
        # print('the target was 10')
        for line in lines[starting_line:]:
            if line.startswith('    CLUSTERING HISTOGRAM'):
                ending_line = lines.index(line)
                break
#             print(f' the starting line is {starting_line} and the ending line is {ending_line}')
        pdbqt_content = lines[(starting_line + 4):(ending_line - 5)]
#             print(f' the pdbqt content is found on lines {pdbqt_content}')
        stripped_pdbqt_content = [line.rstrip('\n') for line in pdbqt_content]

    clean_pdbqt_content = []
#         print(f'there are {len(clean_pdbqt_content)} lines in the clean_pdbqt_content')
    for line in stripped_pdbqt_content:
        cleaned_line = line[max(line.find('D'), 8):]
        if not cleaned_line.startswith('USER'):
            clean_pdbqt_content.append(cleaned_line)
#         print(f'there are now {len(clean_pdbqt_content)} lines in the clean_pdbqt_content')

    pdbqt_block_content = []
#         print(f'there are {len(pdbqt_block_content)} lines in the pdbqt content')
    for line_item in clean_pdbqt_content:
        pdbqt_block_content.append("%s\n" % line_item)
#         print(f'there are now {len(clean_pdbqt_content)} lines in the pdbqt_block_content')

    pdbqt_name = f'{working_dir}/{name}temp.pdbqt'
    with open(pdbqt_name, 'w') as f:
        for line_item in clean_pdbqt_content:
            f.write("%s\n" % line_item)


    stringpdbqt = ' '.join(map(str,  pdbqt_block_content))
    pdbqt_pybel = list(pybel.readfile("pdbqt", pdbqt_name))
#         print(list(pdbqt_pybel))
    mol2_blocks_docked = pdbqt_pybel[0].write(format='mol2')
    
    row["mol2_blocks_docked"] = mol2_blocks_docked
    row["target_run"] = target_run
    row["docking_score"] = target_run
    
    return row

def _generate_pdbqt_outfiles_for_docking(i, row):
#     print('hello')
#     try:
    name = row[11]
    pdbqt_block_string = row[9]
    pdbqt_pybel = pybel.readstring(format='pdbqt', string=pdbqt_block_string)
    filename = f'{working_dir}/{name}_{col_to_dock}.pdbqt'
    pdbqt_pybel.write(format='pdbqt', filename=filename)


    row["pdbqt_out_path"] = filename
    
    return row




@stopwatch        
def run_autodock_gpu(df, col_to_dock, autodock_gpu, lsmet, num_runs, working_dir, receptor_path, dev_num):
    
    working_dir = str(working_dir)
    print(working_dir)
    names = list(df['names'])
    pdbqt = df[col_to_dock]
    
    filenames = []
    names_to_dock = []
    for count,m in enumerate(pdbqt):
        pdbqt_block_string = m
        pdbqt_pybel = pybel.readstring(format='pdbqt', string=pdbqt_block_string)
#         print(pdbqt_pybel)
#         print(pdbqt_pybel.write(format='pdbqt'))
#         print(type(pdbqt_pybel))
        filename = f'{working_dir}/{names[count]}_{col_to_dock}.pdbqt'
        filenames.append(filename)
        names_to_dock.append(names[count])
        pdbqt_pybel.write(format='pdbqt', filename=filename)
        
    
    output_prefix_paths = []
    docked_names = []

    
    program_exe = '''\
    {autodock_gpu} \
    -filelist {batch_list} \
    -lsmet {lsmet} \
    -devnum {dev_num} \
    -nrun {num_runs}
    '''
    
#         -autostop 0 \
#     -heuristics 0 \

    exe_cmd = program_exe.format(autodock_gpu=autodock_gpu, receptor_path=receptor_path, batch_list=batch_list, lsmet=lsmet, dev_num=dev_num, num_runs=num_runs)
    shell_script = '''\
    {exe_cmd}
    '''.format(exe_cmd=exe_cmd)
    print(shell_script)
    
    x = subprocess.Popen(shell_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_out, std_err = x.communicate()
    if std_err :
        print(std_err)

    else:
        x.wait()
        console.print('docking_complete')
        
    output_docking_poses_ext = ".dlg"
    scores_output_ext = ".xml"
    output_docking_pose_paths = [(f'{working_dir}/{name}_docked{output_docking_poses_ext}') for i,name in enumerate(docked_names)]
    output_docking_scores = [(f'{working_dir}/{name}_docked{scores_output_ext}') for i,name in enumerate(docked_names)]

        
    docked_df = pd.DataFrame(list(zip(names_to_dock, filenames, output_docking_pose_paths, output_docking_scores)), \
                             columns = ['names', 'input_pdbqt_path', 'output_docking_pose_paths', 'output_docking_scores'])
    
    
    try:
        out_df = pd.merge(df, docked_df, on="names")
        return out_df 
    except:
        print("merging df, failed")
        return None


col_to_dock = 'pdbqt_ambcc'
working_dir = '/data/dockop_glide_d3/dock_test'

smiles_column = 'standard_smiles'
df2 = dm.parallelized(_generate_pdbqt_outfiles_for_docking, list(df2.iterrows()), arg_type='args', progress=True)
df2 = pd.DataFrame(df2)

autodock_gpu = '/home/schrogpu/ADFRsuite-1.0/AutoDock-GPU/bin/autodock_gpu_128wi'
receptor_path = '/home/schrogpu/ADFRsuite-1.0/d3_docking/pocket2fixer/rigidReceptor.maps.fld'
lsmet = 'sw'
num_runs = 50
dev_num = 0

names_to_dock = list(df2['canonical_id'])
filenames = list(df2['pdbqt_out_path'])
batch_list = f'{working_dir}/{col_to_dock}_batch.txt'    
with open(batch_list, 'w') as f:
    f.write(f'{receptor_path}\n')
    for i, filepath in enumerate(filenames):
        f.write(f'{filepath}\n')
        output_prefix = f'{working_dir}/{names_to_dock[i]}_docked'
        f.write(f'{working_dir}/{names_to_dock[i]}_docked\n')

program_exe = '''\
{autodock_gpu} \
-filelist {batch_list} \
-lsmet {lsmet} \
-devnum {dev_num} \
-nrun {num_runs}
'''

#         -autostop 0 \
#     -heuristics 0 \

exe_cmd = program_exe.format(autodock_gpu=autodock_gpu, receptor_path=receptor_path, batch_list=batch_list, lsmet=lsmet, dev_num=dev_num, num_runs=num_runs)
shell_script = '''\
{exe_cmd}
'''.format(exe_cmd=exe_cmd)
print(shell_script)

x = subprocess.Popen(shell_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
std_out, std_err = x.communicate()
if std_err :
    print(std_err)

else:
    x.wait()
    console.print('docking_complete')


smiles_column = 'standard_smiles'
df3 = dm.parallelized(_collect_docking_outfiles_and_data, list(df2.iterrows()), arg_type='args', progress=True)
df3 = pd.DataFrame(df3)
