from aart_func import *
from params import * 

spin_case = float(sys.argv[1])
i_case = float(sys.argv[2])
dx0 = float(sys.argv[3])

print("Computing the lensing bands")
lb.lb()