from skimage.metrics import structural_similarity as ssim
from params import *
from aart_func import *
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

def similarity_by_ssim(img1, img2):
    # Asegúrate de especificar el rango de datos (data_range)
    score, _ = ssim(img1, img2, full=True, data_range=img1.max() - img1.min())
    return score


######### Descargar Archivo inoisy ##############
inoisyfile = h5py.File(path_InoisyEnvelope+i_fname, 'r')


data = np.array(inoisyfile['data/data_env'])

#Limits of the figure
xystart = np.array(inoisyfile['params/x1start'])[0]

print("There are %s snapshots in this inoisy data set"%data.shape[0])
inoisyfile.close()

########### Descargar primer frame ##################

#fig, ax = plt.subplots(figsize=[5,5],dpi=400)
#
##You can select different snapshots by changing the slicing 
#ax.imshow(np.log(data[13,:,:]),cmap="plasma",origin="lower",extent=[-xystart,xystart,-xystart,xystart])
#
#ax.set_facecolor('xkcd:black')
#ax.set_xlabel(r"$X$"+" "+"(M)")
#ax.set_ylabel(r"$Y$"+" "+"(M)")
##plt.savefig(f'InoisySnapshot.png',dpi=400,bbox_inches='tight')
#plt.savefig(path + "inoisy13frame.png")


#### Analisis de Correlación ####

#inoisy_13frame = data[13,:,:]
#
#def CurveSim(dtau):
#    rango = np.arange(0, snapshots_inoisy , dtau, dtype=int)
#    
#    lis = []
#    
#    for i in rango:
#        inoisy_iframe = data[i,:,:]
#        sim = similarity_by_ssim(inoisy_iframe,inoisy_13frame)
#        lis.append(sim)
#    
#    df = pd.DataFrame(lis, columns=['ssim'])
#    df["Frame"] = rango
#    
#    # Guardar el DataFrame como archivo CSV
#    df.to_csv(path + f'ssim{dtau}.csv', index=False)
#    return print(f"Tienes la curva de {dtau}")
#
#CurveSim(13)
#CurveSim(38)

############### Obtención de imagines par el modelo ###############

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


########### Descarga imagen Slow-Light ###########3
lim0 = 30
VMax=np.max(Is0+Is1+Is2)

fig, ax = plt.subplots(figsize=[5,5],dpi=400)

#You can select different snapshots by changing the slicing 
ax.imshow(Is0[13,:,:] + Is1[13,:,:] + Is2[13,:,:],vmax=VMax,origin="lower",cmap="plasma",extent=[-lim0,lim0,-lim0,lim0])


ax.set_facecolor('xkcd:black')
ax.set_xlabel(r"$\alpha$"+" "+"(M)")
ax.set_ylabel(r"$\beta$"+" "+"(M)")
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
plt.savefig(path + "SlowLight13frame.png")

########### Descarga imagen Flow-Light ###########
lim0 = 30
VMax=np.max(I0+I1+I2)

fig, ax = plt.subplots(figsize=[5,5],dpi=400)

#You can select different snapshots by changing the slicing 
ax.imshow(I0[13,:,:] + I1[13,:,:] + I2[13,:,:],vmax=VMax,origin="lower",cmap="plasma",extent=[-lim0,lim0,-lim0,lim0])


ax.set_facecolor('xkcd:black')
ax.set_xlabel(r"$\alpha$"+" "+"(M)")
ax.set_ylabel(r"$\beta$"+" "+"(M)")
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
plt.savefig(path + "FastLight13frame.png")

################### Importación Curvas de luz #################
dt_str = f"{dt:.2f}"
df_name = path_lc + 'LightCurve_datas_dt%s_a%s_i%s_%s.csv'%(dt_str,spin_case,i_case,i_fname[:-3])
ruta_lc = path_lc + r"LightCurve_inoisy_n0.4_i0.94_ft2500_snap2048.h5"

lc_inoisy = pd.read_csv(ruta_lc)
lc_aart =  pd.read_csv(df_name)

LightCurve_inoisy = lc_inoisy["inoisy"].to_numpy()
Time_inoisy       = lc_inoisy["Time"].to_numpy()

LightCurve_sl = lc_aart["SlowLight"].to_numpy()
LightCurve_fl = lc_aart["FastLight"].to_numpy()
Time_aart     = lc_aart["Time"].to_numpy()

LightCurve_inoisy = (LightCurve_inoisy-np.mean(LightCurve_inoisy))/np.std(LightCurve_inoisy)
LightCurve_sl = (LightCurve_sl-np.mean(LightCurve_sl))/np.std(LightCurve_sl)
LightCurve_fl = (LightCurve_fl-np.mean(LightCurve_fl))/np.std(LightCurve_fl)

FracDif = (LightCurve_sl-LightCurve_fl)/LightCurve_sl
########### Imagen TOTAL ##########
# Create figure and axes using make_figure_from_design (you can define or adapt this if needed)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
plt.subplots_adjust(hspace=0.5)  # Adjust space between plots

# Unpack axes for convenience
ax_top_left = axes[0]
ax_lc = axes[1]
x_diff = axes[2]

# Plot 1: Top Left (Inoisy)
ax_top_left.imshow(np.log(data[13,:,:]), cmap="plasma", origin="lower", extent=[-xystart, xystart, -xystart, xystart])
ax_top_left.set_facecolor('xkcd:black')
ax_top_left.set_xlim(-10, 10)
ax_top_left.set_ylim(-10, 10)
ax_top_left.set_xlabel(f"$X$ $(M)$")
ax_top_left.set_ylabel(f"$Y$ $(M)$")
ax_top_left.set_title(r"inoisy")

# Plot 2: Top Center (Slow-Light)
VMAX = np.max(Is0 + Is1 + Is2)
ax_top_center = fig.add_axes([0.33, 0.65, 0.33, 0.3])  # Customize location and size
ax_top_center.imshow(Is0[13,:,:] + Is1[13,:,:] + Is2[13,:,:], vmax=VMAX, origin="lower", cmap="plasma", extent=[-lim0, lim0, -lim0, lim0])
ax_top_center.set_xlim(-10, 10)
ax_top_center.set_ylim(-10, 10)
ax_top_center.set_facecolor('xkcd:black')
ax_top_center.set_xlabel(r"$\alpha$"+" "+"(M)")
ax_top_center.set_ylabel(r"$\beta$"+" "+"(M)")
ax_top_center.set_title(r"Slow-Light")

# Plot 3: Top Right (Fast-Light)
VMax = np.max(I0 + I1 + I2)
ax_top_right = fig.add_axes([0.66, 0.65, 0.33, 0.3])  # Customize location and size
ax_top_right.imshow(I0[13,:,:] + I1[13,:,:] + I2[13,:,:], vmax=VMax, origin="lower", cmap="plasma", extent=[-lim0, lim0, -lim0, lim0])
ax_top_right.set_xlim(-10, 10)
ax_top_right.set_ylim(-10, 10)
ax_top_right.set_facecolor('xkcd:black')
ax_top_right.set_xlabel(r"$\alpha$"+" "+"(M)")
ax_top_right.set_ylabel(r"$\beta$"+" "+"(M)")
ax_top_right.set_title(r"Fast-Light")

# Plot 4: Normalized Light Curve (Center)
ax_lc.plot(Time_aart, LightCurve_fl , label='inoisy + aart in fast-light', color='#0072B2')
ax_lc.plot(Time_aart, LightCurve_sl, label='inoisy + aart in slow-light', color='#E69F00')
ax_lc.plot(Time_inoisy, LightCurve_inoisy, linestyle="--", label='inoisy', color='#009E73')
ax_lc.set_xlabel(r"$T$"+" "+"(M)")
ax_lc.set_ylabel(r"Normalized Light Curve")
ax_lc.set_xlim(0, 1000)
ax_lc.set_ylim(np.nanmin(LightCurve_inoisy), np.nanmax(LightCurve_inoisy) + 0.1)
ax_lc.grid(linestyle="--", alpha=0.2)
ax_lc.legend()

# Plot 5: Fractional Difference (Bottom)
x_diff.plot(Time_aart, FracDif, color='#0072B2')
x_diff.set_xlabel(r"$T$"+" "+"(M)")
x_diff.set_ylabel(r"$\Delta\,(\%)$")
x_diff.set_xlim(0, 1000)
x_diff.set_ylim(-np.nanmax(FracDif)-0.01, np.nanmax(FracDif)+0.01)
x_diff.grid(linestyle="--", alpha=0.2)

plt.savefig(path + "TotalFigure13frame.png")
