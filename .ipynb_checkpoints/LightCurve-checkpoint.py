import subprocess
import pandas as pd
#
#spin_case = float(sys.argv[1])
#i_case = float(sys.argv[2])
#dx0 = float(sys.argv[3])
#i_fname = str(sys.argv[4])
#
#
#params_doc = f"""
#from aart_func import *
#
#print("\nThanks for using AART")
##print("Copyright (C) 2025, A. Cardenas-Avendano, H. Zhu & A. Lupsasca\n")
#
##BH's Spin
#spin_case={spin_case}
##Observer's inclination  
#i_case={i_case}
#
## Distance to the BH in meters (default: M87)
#dBH=5.214795112e23  
## BH mass-to-distance ratio (default: 1/psi= 6.2e9 Kg)
#psi=1.07473555940836 
##Observer's distance in units of M
#D_obs=1e5
#
##Velocity Profile for the gas
#
##Sub-Kepleniarity param
#sub_kep=1.0#0.95
##Radial velocity param
#betar=1.0#0.95
##Angular velocity param
#betaphi=1.0#0.95
#
## If equal to 1, the radon cuts profiles will be stored   
#radonfile=0
## If equal to 1, the Beloborodov approximation will also be computed
#bvapp=0
# 
##For the Image resolution in the Bardeen coordinates
##Limits for the image [M]. It should coincide with the inoisy if used.
##If equal to 1, the sizes of the grids will be equal and an image can be computed
##by summing the contributions    
#p_image=1
#limits=25
##Resolution for the n=0 image [M]
#dx0={dx0}
##Resolution for the n=1 image [M]
#dx1={dx0}
##Resolution for the n=2 image [M]
#dx2={dx0}
#
## Projection angles for the radon transformation
#radonangles=[0,90]
#
## Image treatment 
#fudge=1.5 #Fudge factor (For n>0)
#
## Sample Equatorial Profile
#i_fname={i_fname}
##i_fname="hotspot.h5"
#
## Stationary assumes a single inoisy frame. "stationary" or "dynamical" 
#disk="dynamical" 
#
## inoisy initial time frame for single images
#i_frame=0
#
## Initial and final times in units of M
#i_tM=0
##Makes sense when is less than the inosy temporal length 
#f_tM=2500
##Number of snapshots in that range    
#snapshots=2048
#
#isco = rms(spin_case)
## SU's parameters for the envelope 
## Just used for the analytical profiles
#gammap=-3/2
#mup=1-sqrt(1-spin_case**2)
#sigmap=1/2 
#
##Magnetic field parametrs
#cr=0.0
#cphi=1.0
#
##Hotspot
##Radius of the hotspot
#radhs=8#2.2
##Radius of the hotspot
#velhs=0.05#0.2
#rwidth = 0.5 #0.1
#
## Useful for disk visualizations or when studying truncated disks.
## 0: Neglected
## 1: Radii computed up to thar radius
## 2: Adds 5M to r_cutoff for interpolation purposes. 
#imag_cut=0
## Cutoff radius   
#r_cutoff=20
#
##The power of the redshift factor
#gfactor=3
#
## Max baseline in G\lambda
#maxbaseline=500 
#
## Number of points in the critical curve 
#npointsS=100    
#
##For the parallel generation of images (movies)
#nthreads= 1 ############
#
##Observer's inclination in radians
#thetao=i_case*np.pi/180
##Disk's inclination  
##Current version is just implemented for equatorial models   
#i_disk=90    
#thetad=i_disk*np.pi/180
#
#Gc=6.67e-11 # G constant [m^3 kg^-1 s^-2]
#cc= 2.99792458e8 # c constant [m/s]
#Msc=1.988435e30 # Solar Mass [Kg]
#
#MMkg= 6.2e9*psi*Msc # [Kg]
#MM=MMkg *Gc/cc**2 # Mass of the BH in meters, i.e., for M87(psi*6.2*10^9) psi ("Best fit") Solar Masses 
#
## Size of the real image in meters
#sizeim_Real=(limits)*MM 
##1 microarcsec in radians
#muas_to_rad = np.pi/648000 *1e-6 
#fov_Real=np.arctan(sizeim_Real/(dBH))/muas_to_rad #muas
##print("FOV= ",np.round(2*fov,2),"muas")
#
##Path where the results will be stored
#path = '/projects/bekt/drojaspaternina/Results/'
#path_lb = path + "LensingBands/"
#path_rt = path + "RayTracing/"
#path_fl = path + "FastLight/"
#path_sl = path + "SlowLight/"
#path_lc = path + "LightCurves/"
#
#path_inoisy = '/projects/bekt/inoisy/'
#path_InoisyEnvelope = path + 'Inoisy_files/'
#
## Create a directory for the results
#isExist = os.path.exists(path)
#if not isExist:
#    os.makedirs(path)
#    print("A directory (Results) was created to store the results")
#
#'''
#MIT license
#Permission is hereby granted, free of charge, to any person obtaining a copy of this 
#software and associated documentation files (the "Software"), to deal in the Software 
#without restriction, including without limitation the rights to use, copy, modify, merge, 
#publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
#persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies 
#or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
#PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
#FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
#THE SOFTWARE.
#'''
#"""
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
Time_inoisy
df = pd.DataFrame({
    'SlowLight': LightCurve_SlowLight,
    'FastLight': LightCurve_FastLight,
    'Time': Time})

Time_inoisy = np.linspace(0,2500,2048)

df_inoisy = pd.DataFrame({
    'inoisy'   : LightCurve_inoisy,
    'Time' : Time_inoisy})


df_name = path_lc + 'LightCurve_datas_dt%s_a%s_i%s_%s.csv'%(dt,spin_case,i_case,i_fname[:-3])
df.to_csv(df_name, index=False)

df_nameI = path_lc + 'LightCurve_%s.csv'%(i_fname[:-3])
df.to_csv(df_nameI, index=False)

print("The Light Curves was created!!!")

