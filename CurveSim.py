from skimage.metrics import structural_similarity as ssim
from params import *
from aart_func import *
import pandas as pd

def similarity_by_ssim(img1, img2):
    # Asegúrate de especificar el rango de datos (data_range)
    score, _ = ssim(img1, img2, full=True, data_range=img1.max() - img1.min())
    return score


#### Descargar Archivo inoisy
inoisyfile = h5py.File(path_InoisyEnvelope+i_fname, 'r')


data = np.array(inoisyfile['data/data_env'])

#Limits of the figure
xystart = np.array(inoisyfile['params/x1start'])[0]

print("There are %s snapshots in this inoisy data set"%data.shape[0])
inoisyfile.close()

#### Descargar primer frame ####
fig, ax = plt.subplots(figsize=[5,5],dpi=400)

#You can select different snapshots by changing the slicing 
ax.imshow(np.log(data[13,:,:]),cmap="plasma",origin="lower",extent=[-xystart,xystart,-xystart,xystart])

ax.set_facecolor('xkcd:black')
ax.set_xlabel(r"$X$"+" "+"(M)")
ax.set_ylabel(r"$Y$"+" "+"(M)")
#plt.savefig(f'InoisySnapshot.png',dpi=400,bbox_inches='tight')
plt.savefig(path + "inoisy13frame.png")


#### Analisis de Correlación ####
inoisy_13frame = data[13,:,:]

def CurveSim(dtau):
    rango = np.arange(0, snapshots_inoisy , dtau, dtype=int)
    
    lis = []
    
    for i in rango:
        inoisy_iframe = data[i,:,:]
        sim = similarity_by_ssim(inoisy_iframe,inoisy_13frame)
        lis.append(sim)
    
    df = pd.DataFrame(lis, columns=['ssim'])
    df["Frame"] = rango
    
    # Guardar el DataFrame como archivo CSV
    df.to_csv(path + f'ssim{dtau}.csv', index=False)
    return print(f"Tienes la curva de {dtau}")

CurveSim(13)
CurveSim(38)