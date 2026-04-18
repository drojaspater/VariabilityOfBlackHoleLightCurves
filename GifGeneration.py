import subprocess
import pandas as pd
from aart_func import *
from params import * 

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
# LightCurve Function generation


#def LightCurve(I_0, I_1=None, I_2=None, cor=0):
#    I_total = I_0.copy()
#    if I_1 is not None:
#        I_total += I_1
#    if I_2 is not None:
#        I_total += I_2
#    light_curve = np.sum(I_total, axis=(1, 2))
#    return light_curve

        
# Importation of the slow-light movie 
fimages= path_sl + "Images_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
Is0=h5f['bghts0'][:]
Is1=h5f['bghts1'][:]
Is2=h5f['bghts2'][:]
h5f.close()


# Importation of the Brisk-light movie 
fimages= path_bl + "BriskLight_p%s_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(p_brisk,dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
Ib0=h5f['bghts0'][:]
Ib1=h5f['bghts1'][:]
Ib2=h5f['bghts2'][:]
h5f.close()

# Importation of the fast-light movie
#fimages= path_fl + "FastLight_Images_noise%s_i%s_a%s_%s.h5"%(noise,i_case,spin_case,i_fname[:-3])
fimages= path_fl + "FastLight_Images_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
I0=h5f['bghts0'][:]
I1=h5f['bghts1'][:]
I2=h5f['bghts2'][:]
h5f.close() 

#print("Starting light curve generation")
#LightCurve_BriskLight = LightCurve(Ib0,Ib1,Ib2)
#LightCurve_FastLight = LightCurve(I0,I1,I2)
#LightCurve_SlowLight = LightCurve(Is0,Is1,Is2)


Time = np.linspace(i_tM,f_tM,snapshots)


from matplotlib.gridspec import GridSpec

# ============================================================
# NORMALIZACIÓN DE CURVAS
# ============================================================
#def FluxFactor(fluxes):
#    target_flux = 0.6
#    return fluxes * (target_flux / np.mean(fluxes))

#LCb = FluxFactor(LightCurve_BriskLight)
#LCs = FluxFactor(LightCurve_SlowLight)
#LCf = FluxFactor(LightCurve_FastLight)

# ============================================================
# CONFIG (USANDO path_im)
# ============================================================
#extent = [-30, 30, -30, 30]
#
#frames_dir = os.path.join(path_im, "frames_gif")
#os.makedirs(frames_dir, exist_ok=True)
#
#gif_path = os.path.join(path_im, "movie.gif")
#
#images = []

# ============================================================
# ESCALA LOG GLOBAL
# ============================================================
from matplotlib.colors import LogNorm

#eps = 1e-12
#all_slow = Is0 + Is1 + Is2
#VMAX = np.percentile(all_slow, 99.5)
#VMIN = max(np.percentile(all_slow[all_slow > 0], 1), eps)

# ============================================================
# GENERACIÓN DE FRAMES
# ============================================================
#print("Starting frame pictures generation")
#
#step = 32  # 1 full resolución, 32 faster prube
#
##for tsnap in range(0, snapshots, step):
#for tsnap in [snapshots-1]:
#
#    fig = plt.figure(figsize=(12, 7))
#    gs = GridSpec(2, 3, height_ratios=[1, 1.2])
#
#    ax1 = fig.add_subplot(gs[0, 0])
#    ax2 = fig.add_subplot(gs[0, 1])
#    ax3 = fig.add_subplot(gs[0, 2])
#    ax4 = fig.add_subplot(gs[1, :])
#
#    # ============================================================
#    # IMÁGENES + CONTEO
#    # ============================================================
#    img_b = Ib0[tsnap] + Ib1[tsnap] + Ib2[tsnap]
#    img_s = Is0[tsnap] + Is1[tsnap] + Is2[tsnap]
#    img_f = I0[tsnap] + I1[tsnap] + I2[tsnap]
#
#    # Slow
#    ax1.imshow(img_s + eps, origin="lower", cmap="plasma",
#               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
#    ax1.set_xlim(-10, 10)
#    ax1.set_ylim(-10, 10)
#    ax1.set_facecolor("xkcd:black")
#    ax1.set_title("Slow-Light")
#    ax1.text(0.02, 0.95,
#             f"N={img_b.size}\nNZ={np.count_nonzero(img_s)}",
#             transform=ax1.transAxes, color="white",
#             fontsize=9, va="top",
#             bbox=dict(facecolor="black", alpha=0.6, pad=3))
#    ax1.set_xlabel(r"$\alpha$"+" "+"(M)")
#    ax1.set_ylabel(r"$\beta$"+" "+"(M)")
#
#    # Brisk
#    ax2.imshow(img_b + eps, origin="lower", cmap="plasma",
#               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
#    ax2.set_xlim(-10, 10)
#    ax2.set_ylim(-10, 10)
#    ax2.set_facecolor("xkcd:black")
#    ax2.set_title("Brisk-Light")
#    ax2.text(0.02, 0.95,
#             f"N={img_s.size}\nNZ={np.count_nonzero(img_b)}",
#             transform=ax2.transAxes, color="white",
#             fontsize=9, va="top",
#             bbox=dict(facecolor="black", alpha=0.6, pad=3))
#    ax2.set_xlabel(r"$\alpha$"+" "+"(M)")
#    ax2.set_ylabel(r"$\beta$"+" "+"(M)")
#
#    # Fast
#    ax3.imshow(img_f + eps, origin="lower", cmap="plasma",
#               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
#    ax3.set_xlim(-10, 10)
#    ax3.set_ylim(-10, 10)
#    ax3.set_facecolor("xkcd:black")
#    ax3.set_title("Fast-Light")
#    ax3.text(0.02, 0.95,
#             f"N={img_f.size}\nNZ={np.count_nonzero(img_f)}",
#             transform=ax3.transAxes, color="white",
#             fontsize=9, va="top",
#             bbox=dict(facecolor="black", alpha=0.6, pad=3))
#    ax3.set_xlabel(r"$\alpha$"+" "+"(M)")
#    ax3.set_ylabel(r"$\beta$"+" "+"(M)")
#
#    # ============================================================
#    # CURVAS
#    # ============================================================
#    ax4.plot(Time[:tsnap+1], LCb[:tsnap+1], label="\texttt{aart} in brisk-light",color="#1b9e77")
#    ax4.plot(Time[:tsnap+1], LCs[:tsnap+1], label="\texttt{aart} in slow-light",color="#d95f02")
#    ax4.plot(Time[:tsnap+1], LCf[:tsnap+1], label="\texttt{aart} in fast-light",color="#7570b3")
#    #ax4.axvline(Time[tsnap], color="k", ls="--", lw=1)
#
#    ax4.set_xlim(Time[0], Time[-1])
#    ax4.set_xlabel("T (M)")
#    ax4.set_ylabel("Flux (normalized)")
#    ax4.legend()
#
#    # ============================================================
#    # GUARDAR FRAME
#    # ============================================================
#    fname = os.path.join(frames_dir, f"frame_{tsnap:04d}.png")
#    plt.savefig(fname, dpi=150)
#    plt.close(fig)
#
#    #images.append(imageio.imread(fname))
#
## ============================================================
## CREAR GIF
## ============================================================
##imageio.mimsave(gif_path, images, fps=15)
#
#print("The GIF of the black holes modes was made")
#

print("🚀 Iniciando comparación de imágenes - Versión Cluster")

# ============================================================
# SNAPSHOT - CARGAR TUS DATOS REALES
# ============================================================
tsnap = 500

# IMPORTANTE: Ajusta estos nombres de variables según tu archivo
# Si tus datos están en otro formato, modifica esta sección
img_s = Is0[tsnap] + Is1[tsnap] + Is2[tsnap]  # Slow
img_b = Ib0[tsnap] + Ib1[tsnap] + Ib2[tsnap]  # Brisk
img_f = I0[tsnap]  + I1[tsnap]  + I2[tsnap]   # Fast

print(f"✅ Imágenes cargadas - Snapshot t={tsnap}")
print(f"   Dimensiones: {img_s.shape}")

# ============================================================
# FUNCIÓN: CALCULA NMSE GLOBAL
# ============================================================
def calculate_nmse(img_a, img_b):
    """Calcula NMSE global: sum((a-b)^2) / sum(a^2)"""
    denominador = np.sum(img_a**2)
    if denominador == 0:
        return np.inf
    return np.sum((img_a - img_b)**2) / denominador

# ============================================================
# CALCULAR NMSE PARA CADA PAR
# ============================================================
print("📊 Calculando NMSE...")
nmse_sf = calculate_nmse(img_s, img_f)  # Slow vs Fast
nmse_sb = calculate_nmse(img_s, img_b)  # Slow vs Brisk
nmse_fb = calculate_nmse(img_f, img_b)  # Fast vs Brisk

print(f"   Slow vs Fast:  {nmse_sf:.3e}")
print(f"   Slow vs Brisk: {nmse_sb:.3e}")
print(f"   Fast vs Brisk: {nmse_fb:.3e}")

# ============================================================
# MAPAS DE DIFERENCIA (para visualizar)
# ============================================================
print("🎨 Generando mapas de diferencia...")
diff_sf = (img_s - img_f)**2
diff_sb = (img_s - img_b)**2
diff_fb = (img_f - img_b)**2

# Escala común para los mapas de diferencia
all_diff = np.concatenate([diff_sf.ravel(), diff_sb.ravel(), diff_fb.ravel()])
pos_diff = all_diff[all_diff > 0]
eps = 1e-12
if len(pos_diff) > 0:
    DMIN = max(np.percentile(pos_diff, 1), eps)
    DMAX = np.percentile(pos_diff, 99.5)
else:
    DMIN, DMAX = eps, 1.0

# ============================================================
# ESCALA PARA IMÁGENES FÍSICAS
# ============================================================
print("📏 Ajustando escalas...")
all_imgs = np.concatenate([img_s.ravel(), img_f.ravel(), img_b.ravel()])
positive_mask = all_imgs > 0
VMAX = np.percentile(all_imgs, 99.5)
VMIN = max(np.percentile(all_imgs[positive_mask], 1), eps)



# ============================================================
# EXTENT (ajusta según tus coordenadas)
# ============================================================
# Si tienes coordenadas específicas, modifica esta línea
extent = [-12, 12, -12, 12]  # [alpha_min, alpha_max, beta_min, beta_max]

# ============================================================
# CREAR FIGURA 3x3
# ============================================================
print("🖼️ Generando figura...")
fig, axes = plt.subplots(3, 3, figsize=(13, 11))

# FILA 1: Slow | Fast | Diferencia
im00 = axes[0,0].imshow(img_s + eps, origin='lower', cmap='plasma', 
                        extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[0,0].text(0.05, 0.95, 'Slow-Light', transform=axes[0,0].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
axes[0,0].set_ylabel(r'$\beta$ (M)')

axes[0,1].imshow(img_f + eps, origin='lower', cmap='plasma', 
                 extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[0,1].text(0.05, 0.95, 'Fast-Light', transform=axes[0,1].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

im02 = axes[0,2].imshow(diff_sf + eps, origin='lower', cmap='viridis', 
                        extent=extent, norm=LogNorm(vmin=DMIN, vmax=DMAX))
axes[0,2].text(0.05, 0.95, f'Slow vs Fast\nNMSE = {nmse_sf:.3e}', 
               transform=axes[0,2].transAxes, fontsize=9, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

# FILA 2: Slow | Brisk | Diferencia
axes[1,0].imshow(img_s + eps, origin='lower', cmap='plasma', 
                 extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[1,0].text(0.05, 0.95, 'Slow-Light', transform=axes[1,0].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
axes[1,0].set_ylabel(r'$\beta$ (M)')

axes[1,1].imshow(img_b + eps, origin='lower', cmap='plasma', 
                 extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[1,1].text(0.05, 0.95, 'Brisk-Light', transform=axes[1,1].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

im12 = axes[1,2].imshow(diff_sb + eps, origin='lower', cmap='viridis', 
                        extent=extent, norm=LogNorm(vmin=DMIN, vmax=DMAX))
axes[1,2].text(0.05, 0.95, f'Slow vs Brisk\nNMSE = {nmse_sb:.3e}', 
               transform=axes[1,2].transAxes, fontsize=9, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

# FILA 3: Fast | Brisk | Diferencia
axes[2,0].imshow(img_f + eps, origin='lower', cmap='plasma', 
                 extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[2,0].text(0.05, 0.95, 'Fast-Light', transform=axes[2,0].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
axes[2,0].set_xlabel(r'$\alpha$ (M)')
axes[2,0].set_ylabel(r'$\beta$ (M)')

axes[2,1].imshow(img_b + eps, origin='lower', cmap='plasma', 
                 extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
axes[2,1].text(0.05, 0.95, 'Brisk-Light', transform=axes[2,1].transAxes, 
               fontsize=11, fontweight='bold', color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
axes[2,1].set_xlabel(r'$\alpha$ (M)')

im22 = axes[2,2].imshow(diff_fb + eps, origin='lower', cmap='viridis', 
                        extent=extent, norm=LogNorm(vmin=DMIN, vmax=DMAX))
axes[2,2].text(0.05, 0.95, f'Fast vs Brisk\nNMSE = {nmse_fb:.3e}', 
               transform=axes[2,2].transAxes, fontsize=9, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
axes[2,2].set_xlabel(r'$\alpha$ (M)')

# Configurar límites y aspecto para todas las subplots
for i in range(3):
    for j in range(3):
        axes[i,j].set_xlim(-12, 12)
        axes[i,j].set_ylim(-12, 12)
        axes[i,j].set_facecolor('black')
        axes[i,j].set_aspect('equal')

# ============================================================
# SOLO UNA COLORBAR PARA LAS DIFERENCIAS (tercera columna)
# ============================================================
cbar_diff = fig.colorbar(im02, ax=axes[:,2], fraction=0.03, pad=0.1, aspect=40)
# Si quieres agregar label a la colorbar, descomenta la línea siguiente:
# cbar_diff.set_label('(a-b)²', fontsize=10)

# ============================================================
# AJUSTAR LAYOUT Y GUARDAR
# ============================================================
fig.subplots_adjust(wspace=0.25, hspace=0.25, right=0.88)

# Crear directorio si no existe
out_dir = "plots_comparison"
os.makedirs(out_dir, exist_ok=True)

# Guardar la figura (sin mostrarla para optimizar en cluster)
fname = os.path.join(path_im, f"comparison_triplets_tsnap_{tsnap:04d}_p{p_brisk}.png")
fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"\n✅ FIGURA GUARDADA: {fname}")
print("="*50)
print("RESUMEN DE MÉTRICAS NMSE:")
print(f"   Slow vs Fast:  {nmse_sf:.6e}")
print(f"   Slow vs Brisk: {nmse_sb:.6e}")
print(f"   Fast vs Brisk: {nmse_fb:.6e}")
print("="*50)