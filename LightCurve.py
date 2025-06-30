from aart_func import *
from params import * 
import subprocess
import pandas as pd
# LightCurve Function generation
def LightCurve(I_0,I_1,I_2, cor = 0):
    light_curve = np.zeros(snapshots)
    I_total = I_0 + I_1 + I_2
    for tsnap in range(snapshots):
        light_curve[tsnap] = np.sum(I_0[tsnap,:,:]) + np.sum(I_1[tsnap,:,:]) + np.sum(I_2[tsnap,:,:])
    max_lc = np.max(light_curve)
    return light_curve/(max_lc + cor*max_lc)

# Creation of the .h5 data
subprocess.run(["python", "lensingbands.py"])
subprocess.run(["python", "raytracing.py"])
subprocess.run(["python", "iMovies.py"])
subprocess.run(["python", "FastLightMovie.py"])


# Importation of the slow-light movie 
fimages="./Results/Images_dx%s_a%s_i%s_%s.h5"%(dx0,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
Is0=h5f['bghts0'][:]
Is1=h5f['bghts1'][:]
Is2=h5f['bghts2'][:]
h5f.close()

# Importation of the fast-light movie
fimages="./Results/FastLight_Images_dx%s_a%s_i%s_%s.h5"%(dx0,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
I0=h5f['bghts0'][:]
I1=h5f['bghts1'][:]
I2=h5f['bghts2'][:]
h5f.close() 

# Light Curve Generation 
LightCurve_SlowLight = LightCurve(Is0,Is1,Is2)
LightCurve_FastLight = LightCurve(I0,I1,I2)
Time = np.linspace(i_tM,f_tM,snapshots)

df = pd.DataFrame({
    'SlowLight': LightCurve_SlowLight,
    'FastLight': LightCurve_FastLight,
    'Time': Time})

df_name = path + 'LightCurve_datas_dx%s_a%s_i%s_%s.csv'%(dx0,spin_case,i_case,i_fname[:-3])
df.to_csv(df_name, index=False)


