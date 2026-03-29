import subprocess
import pandas as pd
from aart_func import *
from params import * 

# Creation of the .h5 data
filename = "LensingBands_a%s_i%s_dx%s.h5"%(spin_case,i_case,dx0)
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

# Lecuture of inoisy
print("Reading inoisy file: ",path_InoisyEnvelope+i_fname)

hf = h5py.File(path_InoisyEnvelope+i_fname, 'r')
try:
    data = np.array(hf['data/data_env'])
except:
    data = np.array(hf['data/data_raw'])
#inoisy has periodic boudaries, so we need to copy wrap the data with one frame
data=np.concatenate((data,data[0,:,:][np.newaxis,:,:]),axis=0)


##################################Change Emission Rate##################################
## You can delate this section and nothing happen, this is only for the variation on the
## Emission rate


def LowerDimension(df):
    selected_indices = np.linspace(0, snapshots_inoisy-1, int(snapshots_source), dtype=int)

    data_N = np.zeros((len(selected_indices),data.shape[1],data.shape[2]))
    for i in range(len(selected_indices)):
        j = selected_indices[i]
        data_N[i] =  df[j]
        
    return data_N

data = LowerDimension(data)
########################################################################################

nt = data.shape[0] #inoisy time resolution
ni = data.shape[1] #inoisy x resolution
nj = data.shape[2] #inoisy y resolution

try: 
	xtstart = np.array(hf['params/x0start'])[0]
	xtend = np.array(hf['params/x0end'])[0]

	x1start = np.array(hf['params/x1start'])[0]
	x2start = np.array(hf['params/x2start'])[0]

	x1end = np.array(hf['params/x1end'])[0]
	x2end = np.array(hf['params/x2end'])[0]

except:
	xtstart = np.array(hf['params/x0start'])
	xtend = np.array(hf['params/x0end'])

	x1start = np.array(hf['params/x1start'])
	x2start = np.array(hf['params/x2start'])

	x1end = np.array(hf['params/x1end'])
	x2end = np.array(hf['params/x2end'])


x1 = np.linspace(x1start, x1end, ni) 
x2 = np.linspace(x2start, x2end, nj)

times = np.linspace(xtstart, xtend, nt) 

h5py.File.close(hf)



#Fast-Light two modes 
print("Computing the fast-light single picture")

fnbands=path_lb+"LensingBands_a%s_i%s_dx%s.h5"%(spin_case,i_case,dx0)

print("Reading file: ",fnbands)

h5f = h5py.File(fnbands,'r')

supergrid0=h5f['grid0'][:]
mask0=h5f['mask0'][:]
N0=int(h5f["N0"][0])

supergrid1=h5f['grid1'][:]
mask1=h5f['mask1'][:]
N1=int(h5f["N1"][0])
	
supergrid2=h5f['grid2'][:]
mask2=h5f['mask2'][:]
N2=int(h5f["N2"][0])

h5f.close()

fnbands=path_rt+"Rays_a%s_i%s_dx%s.h5"%(spin_case,i_case,dx0)

print("Reading file: ",fnbands)

h5f = h5py.File(fnbands,'r')

rs0=h5f['rs0'][:]
sign0=h5f['sign0'][:]
t0=h5f['t0'][:]
phi0=h5f['phi0'][:]

rs1=h5f['rs1'][:]
sign1=h5f['sign1'][:]
t1=h5f['t1'][:]
phi1=h5f['phi1'][:]

rs2=h5f['rs2'][:]
sign2=h5f['sign2'][:]
t2=h5f['t2'][:]
phi2=h5f['phi2'][:]

h5f.close()






print("Calculating interpolator and fundamental variables")

fact=-(D_obs+2*np.log(D_obs))

t0-=fact
t1-=fact
t2-=fact


i_dt = times[1] - times[0]
timeconversion = i_dt*MMkg*Gc/cc**3/(3600*24)  # [days]

maxintensity = np.nanmax(data)

# Mode of the lensing band
mode0, left0, right0, _, _, _ = obsint.modal_hdi_kde(t0, p_brisk)
mode1, left1, right1, _, _, _ = obsint.modal_hdi_kde(t1, p_brisk)
mode2, left2, right2, _, _, _ = obsint.modal_hdi_kde(t2, p_brisk)

dt_src = times[1] - times[0]

shift_01 = int(np.round((mode1 - mode0)/dt_src))
shift_02 = int(np.round((mode2 - mode0)/dt_src))

k0 = i_frame
k1 = (k0 + shift_01) % (nt - 1)
k2 = (k0 + shift_02) % (nt - 1)

t0_fast = times[k0]
t1_fast = times[k1]
t2_fast = times[k2]


interpolated3_R = RegularGridInterpolator((times,x1,x2), data, fill_value=0, bounds_error=False, method='linear')

t_obs = times[i_frame]






##### displaced fast-light  #####  
print("Displaced fast-light starts!")

#Aquí debes meterle los tiempo k0, k1 y k2 
i_bghts0 = obsint.fast_light(supergrid0,mask0,sign0,spin_case,isco,rs0,phi0,t0_fast,interpolated3_R,thetao)
i_bghts1 = obsint.fast_light(supergrid0,mask0,sign0,spin_case,isco,rs0,phi0,t1_fast,interpolated3_R,thetao)
i_bghts2 = obsint.fast_light(supergrid0,mask0,sign0,spin_case,isco,rs0,phi0,t2_fast,interpolated3_R,thetao)

i_I0 = (i_bghts0).reshape(N0,N0).T
i_I1 = (i_bghts1).reshape(N0,N0).T
i_I2 = (i_bghts2).reshape(N0,N0).T

filename=path+"ImageDisplacedFastLight_a_%s_i_%s_%frame.h5"%(spin_case,i_case,i_frame)
h5f = h5py.File(filename, 'w')
h5f.create_dataset('bghts0', data=i_I0)
h5f.create_dataset('bghts1', data=i_I1)
h5f.create_dataset('bghts2', data=i_I2)
h5f.close()
print("Single image file ",filename," created.\n")





#####  Fast-light  #####  
print("Fast-light starts!")
i_bghts0 = obsint.fast_light(supergrid0,mask0,sign0,spin_case,isco,rs0,phi0,t0_fast,interpolated3_R,thetao)
i_bghts1 = obsint.fast_light(supergrid1,mask1,sign1,spin_case,isco,rs1,phi1,t0_fast,interpolated3_R,thetao)
i_bghts2 = obsint.fast_light(supergrid2,mask2,sign2,spin_case,isco,rs2,phi2,t0_fast,interpolated3_R,thetao)

i_I0 = (i_bghts0).reshape(N0,N0).T
i_I1 = (i_bghts1).reshape(N1,N1).T
i_I2 = (i_bghts2).reshape(N2,N2).T

filename=path+"ImageFastLight_a_%s_i_%s_%frame.h5"%(spin_case,i_case,i_frame)
h5f = h5py.File(filename, 'w')
h5f.create_dataset('bghts0', data=i_I0)
h5f.create_dataset('bghts1', data=i_I1)
h5f.create_dataset('bghts2', data=i_I2)
h5f.close()
print("Single image file ",filename," created.\n")





#####  Slow-light  #####  
print("Using all the available inoisy frames")
#interpolated3_R = RegularGridInterpolator((times,x1,x2), data, fill_value=0, bounds_error=False, method='linear')

print("Slow-light starts!")
i_bghts0 = obsint.slow_light(supergrid0,mask0,sign0,spin_case,isco,rs0,phi0,np.mod(t0 + t_obs, xtend), interpolated3_R,thetao)
i_bghts1 = obsint.slow_light(supergrid1,mask1,sign1,spin_case,isco,rs1,phi1,np.mod(t1 + t_obs, xtend), interpolated3_R,thetao)
i_bghts2 = obsint.slow_light(supergrid2,mask2,sign2,spin_case,isco,rs2,phi2,np.mod(t2 + t_obs, xtend), interpolated3_R,thetao)

i_I0 = (i_bghts0).reshape(N0,N0).T
i_I1 = (i_bghts1).reshape(N1,N1).T
i_I2 = (i_bghts2).reshape(N2,N2).T

filename=path+"Dynamical_Image_a_%s_i_%s.h5"%(spin_case,i_case)
h5f = h5py.File(filename, 'w')
h5f.create_dataset('bghts0', data=i_I0)
h5f.create_dataset('bghts1', data=i_I1)
h5f.create_dataset('bghts2', data=i_I2)
h5f.close()
print("Images file ",filename," created.")



