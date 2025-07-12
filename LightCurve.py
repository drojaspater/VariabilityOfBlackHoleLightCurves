import subprocess
import pandas as pd
#
#spin_case = float(sys.argv[1])
#i_case = float(sys.argv[2])
#dx0 = float(sys.argv[3])
#i_fname = str(sys.argv[4])
#
#
#params_doc = f"""   """
#
#nombre_archivo = r"'/projects/bekt/drojaspaternina/VariabilityOfBlackHoleLightCurves/params.py"
#
#
## Crear y escribir el archivo
#with open(nombre_archivo, "w", encoding="utf-8") as archivo:
#    archivo.write(params_doc)

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
subprocess.run(["python", "lensingbands.py"])
subprocess.run(["python", "raytracing.py"])
subprocess.run(["python", "iMovies.py"])
#subprocess.run(["python", "FastLightMovie.py"])

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
#fimages= path_fl + "FastLight_Images_dt%s_a%s_i%s_%s.h5"%(dt,spin_case,i_case,i_fname[:-3])
#print("Reading file: ",fimages)
#h5f = h5py.File(fimages,'r')
#I0=h5f['bghts0'][:]
#I1=h5f['bghts1'][:]
#I2=h5f['bghts2'][:]
#h5f.close() 

# Light Curve Generation 
LightCurve_inoisy    = data_lc
LightCurve_SlowLight = LightCurve(Is0,Is1,Is2)
#LightCurve_FastLight = LightCurve(I0,I1,I2)
Time = np.linspace(i_tM,f_tM,snapshots_inoisy)

##################################Change Emission Rate##################################
## You can delate this section and nothing happen, this is only for the variation on the
## Emission rate


def LowerDimension(df):
    selected_indices = np.linspace(0, snapshots_inoisy-1, snapshots, dtype=int)

    data_N = np.zeros(len(selected_indices))
    for i in range(len(selected_indices)):
        j = selected_indices[i]
        data_N[i] =  df[j]
        
    return data_N

Time  = LowerDimension(Time)
########################################################################################

df = pd.DataFrame({
    'SlowLight': LightCurve_SlowLight,
    'FastLight': LightCurve_FastLight,
    'Time': Time})

dt_str = f"{dt:.2f}"
df_name = path_lc + 'LightCurve_datas_dt%s_a%s_i%s_%s.csv'%(dt_str,spin_case,i_case,i_fname[:-3])
df.to_csv(df_name, index=False)

print("The Light Curves was created!!!")

