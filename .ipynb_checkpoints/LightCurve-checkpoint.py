import subprocess
import pandas as pd
from aart_func import *
from params import * 

# LightCurve Function generation
def LightCurve(I_0,I_1,I_2, cor = 0):
    light_curve = np.zeros(snapshots)
    I_total = I_0 + I_1 + I_2
    for tsnap in range(snapshots):
        light_curve[tsnap] = np.sum(I_0[tsnap,:,:]) + np.sum(I_1[tsnap,:,:]) + np.sum(I_2[tsnap,:,:])
   
    return light_curve

# Creation of the .h5 data
filename = "LensingBands_a%s_i%s_dx%s.h5"%(spin_case, i_case, dx0)
filepath = os.path.join(path_lb, filename)
if os.path.exists(filepath):
    print(f"File {filename} already exists. Skipping...")
else:
    print(f"File {filename} not found. Running lensingbands.py...")
    subprocess.run(["python", "lensingbands.py"])

filename = "Rays_a%s_i%s_dx%s.h5"%(spin_case,i_case,dx0)
filepath = os.path.join(path_rt, filename)
if os.path.exists(filepath):
    print(f"File {filename} already exists. Skipping...")
else:
    print(f"File {filename} not found. Running raytracing.py...")
    subprocess.run(["python", "raytracing.py"])
    
subprocess.run(["python", "iMovies.py"])
subprocess.run(["python", "FastLightMovie.py"])

# Lecuture of inoisy
print("Reading file: ",path_InoisyEnvelope+i_fname)
h5f = h5py.File(path_InoisyEnvelope+i_fname, 'r')
data_lc = np.array(h5f['data/lightcurve_env'])
h5f.close()

# Importation of the slow-light movie 
fimages= path_sl + "Images_dt%s_a%s_i%s_%s.h5"%(dt,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
Is0=h5f['bghts0'][:]
Is1=h5f['bghts1'][:]
Is2=h5f['bghts2'][:]
h5f.close()

# Importation of the fast-light movie
#fimages= path_fl + "FastLight_Images_noise%s_i%s_a%s_%s.h5"%(noise,i_case,spin_case,i_fname[:-3])
fimages= path_fl + "FastLight_Images_dt%s_a%s_i%s_%s.h5"%(dt,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
I0=h5f['bghts0'][:]
I1=h5f['bghts1'][:]
I2=h5f['bghts2'][:]
h5f.close() 

# Light Curve Generation 
LightCurve_inoisy    = data_lc
LightCurve_SlowLight = LightCurve(Is0,Is1,Is2)
LightCurve_FastLight = LightCurve(I0,I1,I2)
Time = np.linspace(i_tM,f_tM,snapshots)

##################################Change Emission Rate##################################
## You can delate this section and nothing happen, this is only for the variation on the
## Emission rate


#def LowerDimension(df):
#    selected_indices = np.linspace(0, snapshots_inoisy-1, int(snapshots_source), dtype=int)
#
#    data_N = np.zeros(len(selected_indices))
#    for i in range(len(selected_indices)):
#        j = selected_indices[i]
#        data_N[i] =  df[j]
#        
#    return data_N
#
#Time_dinoisy  = LowerDimension(Time)
########################################################################################

df = pd.DataFrame({
    'SlowLight': LightCurve_SlowLight,
    'FastLight': LightCurve_FastLight,
    'Time': Time})

df_name = path_lc + 'ProveLightCurve_datas_dt%s_a%s_i%s_%s.csv'%(dt,spin_case,i_case,i_fname[:-3])
df.to_csv(df_name, index=False)

#df = pd.DataFrame({
#    'inoisy': LightCurve_inoisy,
#    'Time': Time})
#
#
#df_name = path_lc + i_fname
#df.to_csv(df_name, index=False)


print("The Light Curves was created!!!")

