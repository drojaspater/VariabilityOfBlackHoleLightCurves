from skimage.metrics import structural_similarity as ssim
from params import *
from aart_func import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.rcParams['text.usetex'] = True

def similarity_by_ssim(img1, img2):
    # Asegúrate de especificar el rango de datos (data_range)
    #score, _ = ssim(img1, img2, full=True, data_range=img1.max() - img1.min()) #Índice de Similitud Estructural (SSIM)
    
    #score = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    num = np.sum((img1 - img1.mean()) * (img2 - img2.mean()))
    den = np.sqrt(np.sum((img1 - img1.mean())**2) * np.sum((img2 - img2.mean())**2))
    
    ncc = num / den #Coeficiente de correlación normalizada (NCC)
    return ncc


######### Descargar Archivo inoisy ##############
inoisyfile = h5py.File(path_InoisyEnvelope+i_fname, 'r')


data = np.array(inoisyfile['data/data_env'])
data_lc = np.array(inoisyfile['data/lightcurve_env'])
#Limits of the figure
xystart = np.array(inoisyfile['params/x1start'])[0]

print("There are %s snapshots in this inoisy data set"%data.shape[0])
inoisyfile.close()

df_name = path_lc + 'LightCurve_datas_dx%s_%s.csv'%(dx0,i_fname[:-3])

#df = pd.DataFrame(data_lc)
#df.to_csv(df_name, index=False)
########### Descargar primer frame ##################
#List_Im = np.arange(16,snapshots,500)
#for frame_Im in List_Im:
#    fig, ax = plt.subplots(figsize=[5,5],dpi=400)
#    
#    #You can select different snapshots by changing the slicing 
#    ax.imshow(np.log(data[frame_Im,:,:]),cmap="plasma",origin="lower",extent=[-xystart,xystart,-xystart,xystart])
#    
#    ax.set_facecolor('xkcd:black')
#    ax.set_xlabel(r"$X$"+" "+"(M)")
#    ax.set_ylabel(r"$Y$"+" "+"(M)")
#    #plt.savefig(f'InoisySnapshot.png',dpi=400,bbox_inches='tight')
#    plt.savefig(path_im + f"inoisy{frame_Im}frame.png")



#### Analisis de Correlación ####


def CurveSim(dtau):
    dframe = int(dtau//dt)
    rango = np.arange(0, snapshots_inoisy , dframe, dtype=int)
    inoisy_dtauframe = data[dframe,:,:]
    
    lis = []
    
    for i in rango:
        inoisy_iframe = data[i,:,:]
        sim = similarity_by_ssim(inoisy_iframe,inoisy_dtauframe)
        lis.append(sim)
    
    df = pd.DataFrame(lis, columns=['ssim'])
    df["Frame"] = rango
    
    # Guardar el DataFrame como archivo CSV
    df.to_csv(path + f'ssim{dtau}.csv', index=False)
    return print(f"Tienes la curva de {dtau}")


dtau_17 = 16
dtau_60 = 84

CurveSim(dtau_17)
CurveSim(dtau_60)

############### Obtención de imagines par el modelo ###############

# Importation of the slow-light movie 
#fimages= path_sl + "Images_dx%s_a%s_i%s_%s.h5"%(dx0,spin_case,i_case,i_fname[:-3])
#print("Reading file: ",fimages)
#h5f = h5py.File(fimages,'r')
#Is0=h5f['bghts0'][:]
#Is1=h5f['bghts1'][:]
#Is2=h5f['bghts2'][:]
#h5f.close()
#
## Importation of the fast-light movie
#fimages= path_fl + "FastLight_Images_dx%s_a%s_i%s_%s.h5"%(dx0,spin_case,i_case,i_fname[:-3])
#print("Reading file: ",fimages)
#h5f = h5py.File(fimages,'r')
#I0=h5f['bghts0'][:]
#I1=h5f['bghts1'][:]
#I2=h5f['bghts2'][:]
#h5f.close() 
#
#
############ Descarga imagen Slow-Light ###########3
#lim0 = 30
#VMax=np.max(Is0+Is1+Is2)
#
#fig, ax = plt.subplots(figsize=[5,5],dpi=400)
#
##You can select different snapshots by changing the slicing 
#ax.imshow(Is0[13,:,:] + Is1[13,:,:] + Is2[13,:,:],vmax=VMax*1.2,origin="lower",cmap="afmhot",extent=[-lim0,lim0,-lim0,lim0])
#
#
#ax.set_facecolor('xkcd:black')
#ax.set_xlabel(r"$\alpha$"+" "+"(M)")
#ax.set_ylabel(r"$\beta$"+" "+"(M)")
#ax.set_xlim(-10,10)
#ax.set_ylim(-10,10)
#plt.savefig(path + "SlowLight13frame.png")
#
############ Descarga imagen Slow-Light ###########
#lim0 = 30
#VMax=np.max(I0+I1+I2)
#
#fig, ax = plt.subplots(figsize=[5,5],dpi=400)
#
##You can select different snapshots by changing the slicing 
#ax.imshow(I0[13,:,:] + I1[13,:,:] + I2[13,:,:],vmax=VMax*1.2,origin="lower",cmap="afmhot",extent=[-lim0,lim0,-lim0,lim0])
#
#
#ax.set_facecolor('xkcd:black')
#ax.set_xlabel(r"$\alpha$"+" "+"(M)")
#ax.set_ylabel(r"$\beta$"+" "+"(M)")
#ax.set_xlim(-10,10)
#ax.set_ylim(-10,10)
#plt.savefig(path + "FastLight13frame.png")
#
#################### Importación Curvas de luz #################
#df_name = path_lc + 'LightCurve_datas_dx%s_a%s_i%s_%s.csv'%(dx0,spin_case,i_case,i_fname[:-3])
#ruta_lc = path_lc + r"inoisy_n0.4_i0.94_ft2500_snap2048.h5"
#
#lc_inoisy = pd.read_csv(ruta_lc)
#lc_aart =  pd.read_csv(df_name)
#
#LightCurve_inoisy = lc_inoisy["inoisy"].to_numpy()
#Time_inoisy       = lc_inoisy["Time"].to_numpy()
#
#LightCurve_sl = lc_aart["SlowLight"].to_numpy()
#LightCurve_fl = lc_aart["FastLight"].to_numpy()
#Time_aart     = lc_aart["Time"].to_numpy()
#
#def FluxFactor(fluxes):
#    maxflux = np.max(fluxes)
#    minflux = np.min(fluxes)
#    avgflux = np.mean(fluxes)
#    stdflux = np.std(fluxes)
#
#    target_flux = 0.6
#    fluxfactor = target_flux/avgflux
#    return fluxes*fluxfactor
#    
#LightCurve_inoisy = FluxFactor(LightCurve_inoisy)
#LightCurve_sl     = FluxFactor(LightCurve_sl)
#LightCurve_fl     = FluxFactor(LightCurve_fl)
#
#FracDif = (LightCurve_sl-LightCurve_fl)/LightCurve_sl
############ Imagen TOTAL ##########
#
## Crear la figura con el diseño específico
#fig = plt.figure(figsize=(12, 10))
#
## Configurar GridSpec con 3 filas y 3 columnas
## height_ratios: [F1-F3 row, F4 row, F5 row] = [2, 2, 1]
#gs = plt.GridSpec(3, 3, figure=fig, height_ratios=[2, 2, 1], hspace=0.5, wspace=0.4)
#
## Plot 1: Top Left (Inoisy) - F1
#ax_top_left = fig.add_subplot(gs[0, 0])
#ax_top_left.imshow(np.log(data[13,:,:]), cmap="plasma", origin="lower", extent=[-xystart, xystart, -xystart, xystart])
#ax_top_left.set_facecolor('xkcd:black')
#ax_top_left.set_xlim(-10, 10)
#ax_top_left.set_ylim(-10, 10)
#ax_top_left.set_xlabel(f"$X$ $(M)$")
#ax_top_left.set_ylabel(f"$Y$ $(M)$")
#ax_top_left.set_title(r"inoisy")
#
## Plot 2: Top Center (Slow-Light) - F2
#VMAX = np.max(Is0 + Is1 + Is2)
#ax_top_center = fig.add_subplot(gs[0, 1])
#ax_top_center.imshow(Is0[13,:,:] + Is1[13,:,:] + Is2[13,:,:], vmax=VMAX, origin="lower", cmap="afmhot", extent=[-lim0, lim0, -lim0, lim0])
#ax_top_center.set_xlim(-10, 10)
#ax_top_center.set_ylim(-10, 10)
#ax_top_center.set_facecolor('xkcd:black')
#ax_top_center.set_xlabel(r"$\alpha$"+" "+"(M)")
#ax_top_center.set_ylabel(r"$\beta$"+" "+"(M)")
#ax_top_center.set_title(r"Slow-Light")
#
## Plot 3: Top Right (Fast-Light) - F3
#VMax = np.max(I0 + I1 + I2)
#ax_top_right = fig.add_subplot(gs[0, 2])
#ax_top_right.imshow(I0[13,:,:] + I1[13,:,:] + I2[13,:,:], vmax=VMax, origin="lower", cmap="afmhot", extent=[-lim0, lim0, -lim0, lim0])
#ax_top_right.set_xlim(-10, 10)
#ax_top_right.set_ylim(-10, 10)
#ax_top_right.set_facecolor('xkcd:black')
#ax_top_right.set_xlabel(r"$\alpha$"+" "+"(M)")
#ax_top_right.set_ylabel(r"$\beta$"+" "+"(M)")
#ax_top_right.set_title(r"Fast-Light")
#
## Hacer que F1-F3 sean cuadrados
#for ax in [ax_top_left, ax_top_center, ax_top_right]:
#    ax.set_box_aspect(1)  # Esto fuerza la relación de aspecto 1:1
#
## Plot 4: Normalized Light Curve (Center) - F4 (ocupa todo el ancho)
#ax_lc = fig.add_subplot(gs[1, :])
#ax_lc.plot(Time_aart, LightCurve_fl, label='inoisy + aart in fast-light', color='#0072B2')
#ax_lc.plot(Time_aart, LightCurve_sl, label='inoisy + aart in slow-light', color='#E69F00')
#ax_lc.plot(Time_inoisy, LightCurve_inoisy, linestyle="--", label='inoisy', color='#009E73')
#ax_lc.set_xlabel(r"$T$"+" "+"(M)")
#ax_lc.set_ylabel(r"Normalized Light Curve")
#ax_lc.set_xlim(0, 2500)
#ax_lc.set_ylim(np.nanmin(LightCurve_inoisy), np.nanmax(LightCurve_inoisy) + 0.1)
#ax_lc.grid(linestyle="--", alpha=0.2)
#ax_lc.legend()
#
## Plot 5: Fractional Difference (Bottom) - F5 (ocupa todo el ancho, mitad de altura que F4)
#x_diff = fig.add_subplot(gs[2, :])
#x_diff.plot(Time_aart, FracDif, color='#0072B2')
#x_diff.set_xlabel(r"$T$"+" "+"(M)")
#x_diff.set_ylabel(r"$\Delta\,(\%)$")
#x_diff.set_xlim(0, 2500)
#x_diff.set_ylim(-np.nanmax(FracDif)-0.01, np.nanmax(FracDif)+0.01)
#x_diff.grid(linestyle="--", alpha=0.2)
#
#plt.savefig(path + "TotalFigure13frame.png")
