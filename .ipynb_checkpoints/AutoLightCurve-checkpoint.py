from aart_func import *
from params import * 
import subprocess
import pandas as pd
import itertools

# Params
SpinCase = (0.1, 0.5, 0.94)
iCase = (17, 45, 75)
dxCase = (0.02, 0.04, 0.08)  # dx0=dx1=dx2, así que es solo un conjunto

# params map
combinations = list(itertools.product(SpinCase, iCase, dxCase))
N_com = len(combinations)

for i in range(N_com):
    spin_case = combinations[i][0]
    i_case = combinations[i][1]
    dx0 = dx1 = dx2 = combinations[i][2]

    print("Working with the parameters a = %s , theta = %s , dx = %s"%(spin_case,i_case,dx0))
    subprocess.run(["python", "LightCurve.py"]) 

    