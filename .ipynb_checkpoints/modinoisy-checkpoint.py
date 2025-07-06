from aart_func import *
from params_modinoisy import * 

noise = float(sys.argv[1])  
spin_case = float(sys.argv[2])
f_tM = float(sys.argv[3])
snapshots = float(sys.argv[4])
i_source = str(sys.argv[5])


horizon = 1+np.sqrt(1-spin_case**2)
mup=1-np.sqrt(1-spin_case**2)

print(f"[Script] Recibí noise = {noise}, spin_case = {spin_case}, f_tM = {f_tM}, snapshots = {snapshots}, i_source = {i_source}")

print("Reading source inoisy file "+ path_inoisy + i_source)

hf = h5py.File(path_inoisy + i_source, 'r')

data_raw = np.array(hf['data/data_raw'])
#inoisy has periodic boudaries, so we need to copy wrap the data with one frame
#data_raw=np.concatenate((data_raw,data_raw[0,:,:][np.newaxis,:,:]),axis=0)

xtstart = np.array(hf['params/x0start'])
xtend = np.array(hf['params/x0end'])

xstart = np.array(hf['params/x1start'])
ystart = np.array(hf['params/x2start'])

xend = np.array(hf['params/x1end'])
yend = np.array(hf['params/x2end'])

nt = data_raw.shape[0]  
ni = data_raw.shape[1]
nj = data_raw.shape[2]

dx = np.array(hf['params/dx1'])
dy = np.array(hf['params/dx2'])

x1 = np.linspace(xstart, xend, ni) 
x2 = np.linspace(ystart, yend, nj)

times = np.linspace(xtstart, xtend, nt)

h5py.File.close(hf)

avg_raw=np.average(data_raw)
std_raw=np.std(data_raw)

Xs=np.arange(xstart,xend,dx)
Ys=np.arange(ystart,yend,dy)
xx, yy = np.meshgrid(Xs, Ys)

radius=np.sqrt(xx**2 + yy**2)

rows, cols =np.where(radius<horizon)

envelope=ilp.profile(radius,spin_case,gammap,mup,sigmap)

envelope[rows,cols]=0.0

data_raw=(data_raw-avg_raw)/std_raw

#inoisy light curve
lightcurver=np.zeros(nt)
#Resulting light curve
lightcurve=np.zeros(nt)

print("Computing light curves and the modified data file")

for i in range(nt):
    lightcurver[i]=np.sum(data_raw[i,:,:])
    #Here we apply the envelope to the source inoisy data
    data_raw[i,:,:]=envelope*np.exp(noise*data_raw[i,:,:]-noise**2/2)
    lightcurve[i]=np.sum(data_raw[i,:,:])

lightcurver=lightcurver*dx*dy
lightcurve=lightcurve*dx*dy

i_fname = r"inoisy_n%s_i%s_ft%s_snap%s.h5"%(noise,spin_case,f_tM,snapshots)

print("Creating modified inoisy file " + path_InoisyEnvelope +  i_fname)

h5f = h5py.File(path_InoisyEnvelope + i_fname, 'w')

h5f.create_dataset('data/data_env', data=data_raw)
h5f.create_dataset('data/lightcurve_raw', data=lightcurver)
h5f.create_dataset('data/lightcurve_env', data=lightcurve)

h5f.create_dataset('params/x0start', data=np.array([xtstart]))
h5f.create_dataset('params/x0end', data=np.array([xtend]))

h5f.create_dataset('params/x1start', data=np.array([xstart]))
h5f.create_dataset('params/x2start', data=np.array([ystart]))

h5f.create_dataset('params/x1end', data=np.array([xend]))
h5f.create_dataset('params/x2end', data=np.array([yend]))

h5f.create_dataset('params/dx1', data=np.array([dx]))
h5f.create_dataset('params/dx2', data=np.array([dy]))

h5f.create_dataset('params/gammap', data=np.array([gammap]))
h5f.create_dataset('params/mup', data=np.array([mup]))
h5f.create_dataset('params/sigmap', data=np.array([sigmap]))

h5f.create_dataset('params/spin', data=np.array([spin_case]))
h5f.create_dataset('params/sub_kep', data=np.array([sub_kep]))
h5f.create_dataset('params/betar', data=np.array([betar]))

h5f.create_dataset('params/betaphi', data=np.array([betaphi]))
h5f.create_dataset('params/armangle', data=np.array([armangle]))

h5f.create_dataset('params/spatialcorr', data=np.array([i_spatialcorr]))
h5f.create_dataset('params/spatialcorrxy', data=np.array([i_spatialcorrxy]))

h5f.close()
