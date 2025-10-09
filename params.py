from aart_func import *

print("\nThanks for using AART")
print("Copyright (C) 2023, A. Cardenas-Avendano, H. Zhu & A. Lupsasca\n")

#BH's Spin
spin_case=0.94
#Observer's inclination
i_case=60

# Distance to M87 in meters
dM=5.214795112e23  
# Mass of M87 1 /psi= 6.2e9 Kg
psi=1.07473555940836 
#Observer's distance [M]
D_obs=10000

#Sub-Kepleniarity param
sub_kep=1.0
#Radial velocity param
betar=1.0
#Angular velocity param
betaphi=1.0
#Anisotropy direction
armangle=0.349
#Noise Scale
noise=0.4

# If equal to 1, an inoisy single file will be produced     
iplots=0
# If equal to 1, the radon cuts profiles will be stored   
radonfile=0
# If equal to 1, the Beloborodov approximation will also be computed
bvapp=0
 
#For the Image resolution in the Bardeen coordinates
#Limits for the image [M]. It should coincide with the inoisy if used.
#If equal to 1, images will be computed (This will make the sizes of the grids equal)    
p_image=1
limits=25
#Resolution for the n=0 image [M]
dx0 =0.1
#Resolution for the n=1 image [M]
dx1 =0.1
#Resolution for the n=2 image [M]
dx2 =0.1

# Projection angle for the radon transformation

radonangles=range(0,180,5)
#radonangles=[0,90]

#How many lensing bands are computed
lbs=2.0
padding=32

# Image treatment 
fudge=1.5 #Fudge factor (For n>0)

#To cut the intensity of the inoisy snapshots for comparison (>10)
inoisylims=0
# Stationary assumes a single inoisy frame. "stationary" or "dynamical" 
disk="dynamical"

# inoisy frame for single images
i_frame=0 

# Initial and final times in units of M
i_tM=0  
#Makes sense when is less than the inoisy temporal length 
f_tM= 2500
#Number of snapshots in that range   
snapshots= 1024
snapshots_inoisy = 2048
n = 1
snapshots_source = snapshots_inoisy // n
#Parameter for change the number of snapshots

dt = f_tM/snapshots_source
dt_movie = f_tM/snapshots

isco = rms(spin_case)
horizon = 1+np.sqrt(1-spin_case**2)

# SU's parameters for the envelope 
# Just used for the profiles computed within AART
# If we use the synchroton emission model then
# gammap-> \zeta ~(2.5,4) and mup-> r_{inner} ~ ISCO 
# gammap=2.5
# mup= isco #1-np.sqrt(1-spin_case**2)
# sigmap=0

#GLM
gammap=-1.5
mup=1-np.sqrt(1-spin_case**2)
sigmap=0.5

#Other #P1 in Apolito (Best Match)
#gammap=0.0
#mup=1.5*(1+np.sqrt(1-spin_case**2))
#sigmap=1.0

# With an equatorial profile from inoisy
i_spatial=1024
i_temporal=2048
inoisyduration=2500
i_spatialcorr=5.0
i_spatialcorrxy=0.1
inoisylimsgrid=30
taucorr=12.0
tauxcorr=5.0
seed=662003

#path = r"C:\Users\danyp\OneDrive\Escritorio\CosasPater\UNAL\TrabajoGradoAlejandro\Results\\"
#path_lb = path + r"LensignBands\\"
#path_rt = path + r"RayTracing\\"
path = r'/projects/bekt/drojaspaternina/Results/'
path_lb = path + r"LensingBands/"
path_rt = path + r"RayTracing/"
path_fl = path + r"FastLight/"
path_sl = path + r"SlowLight/"
path_lc = path + r"LightCurves/"
path_im = path + r"Images/"


path_inoisy = r'/projects/bekt/inoisy/'
path_InoisyEnvelope = path + r'Inoisy_files/'

##i_spatial=1024, i_temporal(snapshots_inoisy) = 2048, inoisyduration(f_tM)= 15000
#i_source = r"inoisy_1024_2048_30_15000_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_148123.0.h5"

##i_spatial=1024, i_temporal(snapshots_inoisy) = 2048, inoisyduration(f_tM)= 2500
#i_source = r"inoisy_1024_2048_30_2500_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_3459.0.h5"

##i_spatial=512, i_temporal(snapshots_inoisy) = 8192, inoisyduration(f_tM)= 5000
#i_source = r"inoisy_512_8192_30_5000_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_87648.0.h5"
#i_source = r"inoisy_512_8192_30_5000_5.00_0.10_0.5000_1.00_1.00_1.00_0.349_137.0_137.0_54287.0.h5"

##i_spatial=1024, i_temporal(snapshots_inoisy) = 4096, inoisyduration(f_tM)= 2500
#i_source = r"inoisy_1024_4096_30_2500_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_87435.0.h5"
#i_source = r"inoisy_1024_4096_30_2500_5.00_0.10_0.5000_1.00_1.00_1.00_0.349_137.0_137.0_67897.0.h5"

##i_spatial=1024, i_temporal(snapshots_inoisy) = 4096, inoisyduration(f_tM)= 10000
#i_source = r"inoisy_1024_4096_30_10000_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_558979.0.h"
#i_source = r"inoisy_1024_4096_30_10000_5.00_0.10_0.5000_1.00_1.00_1.00_0.349_137.0_137.0_670867.0.h"

i_fname = r"inoisy_n%s_i%s_ft%s_spatial%s_snap%s.h5"%(noise,spin_case,f_tM,i_spatial,snapshots_inoisy)


#Smooth profile
#Setting this value to 1 is a very conservative way to smooth the last bit of the file. 
####smoothradon=1

#speed_p=1.0
#cutoff_p=1000.0

ulim1_fit=42
ulim2_fit=67
dmin=20
#Goodnes of fit
gfit_min=0.2

######################
######################

#The power of the redshift factor
gfactor=3

# Max baseline in G\lambda
maxbaseline=500

# Number of points in the critical curve 
npointsS=100    

#For the generation of the images (movies)
nthreads=1 ##

# Useful for disk visualizations or when studying truncated disks.
imag_cut=0
# Cutoff radius   
r_cutoff=20.25 

thetao=i_case*np.pi/180
#Disk's inclination  
#Current version just implemented for equatorial models     
i_disk=90
thetad=i_disk*np.pi/180

Gc=6.67e-11 # G constant [m^3 kg^-1 s^-2]
cc= 2.99792458e8 # c constant [m/s]
Msc=1.988435e30 # Solar Mass [Kg]
MMkg= 6.2e9*psi*Msc # [Kg]

MM=MMkg *Gc/cc**2 # Mass of M87 in meters, i.e., (psi*6.2*10^9) psi ("Best fit") Solar Masses 

# Size of the real image in meters
sizeim_Real=(limits)*MM
#1 microarcsec in radians
muas_to_rad = np.pi/648000 *1e-6 
fov_Real=np.arctan(sizeim_Real/(dM))/muas_to_rad #muas
M_to_muas=np.arctan(MM/(dM))/muas_to_rad
unitfact=1/(muas_to_rad*1e9)
#print("FOV= ",np.round(2*fov_Real,2),"muas")

#Path where the results will be stored
#path = './Results/'
#MAC
#path = '/Users/alejo/Desktop/'
#Workstation
#path = '/home/alejo/Desktop/AARTData/'
#path = '/media/datadrive/AARTData/'
#path = '/home/alejo/Desktop/SALTUS/'

#If the code is run several times
##production=1

# Create a directory for the results
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    print("A directory was created to store the results")

'''
MIT license
Permission is hereby granted, free of charge, to any person obtaining a copy of this 
software and associated documentation files (the "Software"), to deal in the Software 
without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
'''
