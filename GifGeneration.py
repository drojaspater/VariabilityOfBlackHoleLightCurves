import subprocess
import pandas as pd
from aart_func import *
from params import * 

# LightCurve Function generation

#def LightCurve(I_0,I_1 = np.zeros(snapshots),I_2 = np.zeros(snapshots), cor = 0):
#    light_curve = np.zeros(snapshots)
#    for tsnap in range(snapshots):
#        light_curve[tsnap] = np.sum(I_0[tsnap,:,:]) + np.sum(I_1[tsnap,:,:]) + np.sum(I_2[tsnap,:,:])
#   
#    return light_curve

def LightCurve(I_0, I_1=None, I_2=None, cor=0):
    I_total = I_0.copy()
    if I_1 is not None:
        I_total += I_1
    if I_2 is not None:
        I_total += I_2
    light_curve = np.sum(I_total, axis=(1, 2))
    return light_curve

        
# Importation of the slow-light movie 
fimages= path_sl + "Images_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])
print("Reading file: ",fimages)
h5f = h5py.File(fimages,'r')
Is0=h5f['bghts0'][:]
Is1=h5f['bghts1'][:]
Is2=h5f['bghts2'][:]
h5f.close()


# Importation of the Brisk-light movie 
fimages= path_bl + "LCBriskLight_p%s_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(p_brisk,dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])
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

print("Starting light curve generation")
LightCurve_BriskLight = LightCurve(Ib0,Ib1,Ib2)
LightCurve_FastLight = LightCurve(I0,I1,I2)
LightCurve_SlowLight = LightCurve(Is0,Is1,Is2)


Time = np.linspace(i_tM,f_tM,snapshots)


from matplotlib.gridspec import GridSpec

# ============================================================
# NORMALIZACIÓN DE CURVAS
# ============================================================
def FluxFactor(fluxes):
    target_flux = 0.6
    return fluxes * (target_flux / np.mean(fluxes))

LCb = FluxFactor(LightCurve_BriskLight)
LCs = FluxFactor(LightCurve_SlowLight)
LCf = FluxFactor(LightCurve_FastLight)

# ============================================================
# CONFIG (USANDO path_im)
# ============================================================
extent = [-30, 30, -30, 30]

frames_dir = os.path.join(path_im, "frames_gif")
os.makedirs(frames_dir, exist_ok=True)

gif_path = os.path.join(path_im, "movie.gif")

images = []

# ============================================================
# ESCALA LOG GLOBAL
# ============================================================
from matplotlib.colors import LogNorm

eps = 1e-12
all_slow = Is0 + Is1 + Is2
VMAX = np.percentile(all_slow, 99.5)
VMIN = max(np.percentile(all_slow[all_slow > 0], 1), eps)

# ============================================================
# GENERACIÓN DE FRAMES
# ============================================================
print("Starting frame pictures generation")

step = 32  # 1 full resolución, 32 faster prube

for tsnap in range(0, snapshots, step):

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 3, height_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # ============================================================
    # IMÁGENES + CONTEO
    # ============================================================
    img_b = Ib0[tsnap] + Ib1[tsnap] + Ib2[tsnap]
    img_s = Is0[tsnap] + Is1[tsnap] + Is2[tsnap]
    img_f = I0[tsnap] + I1[tsnap] + I2[tsnap]

    # Brisk
    ax1.imshow(img_b + eps, origin="lower", cmap="plasma",
               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_facecolor("xkcd:black")
    ax1.set_title("Brisk-Light")
    ax1.text(0.02, 0.95,
             f"N={img_b.size}\nNZ={np.count_nonzero(img_b)}",
             transform=ax1.transAxes, color="white",
             fontsize=9, va="top",
             bbox=dict(facecolor="black", alpha=0.6, pad=3))

    # Slow
    ax2.imshow(img_s + eps, origin="lower", cmap="plasma",
               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_facecolor("xkcd:black")
    ax2.set_title("Slow-Light")
    ax2.text(0.02, 0.95,
             f"N={img_s.size}\nNZ={np.count_nonzero(img_s)}",
             transform=ax2.transAxes, color="white",
             fontsize=9, va="top",
             bbox=dict(facecolor="black", alpha=0.6, pad=3))

    # Fast
    ax3.imshow(img_f + eps, origin="lower", cmap="plasma",
               extent=extent, norm=LogNorm(vmin=VMIN, vmax=VMAX))
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax3.set_facecolor("xkcd:black")
    ax3.set_title("Fast-Light")
    ax3.text(0.02, 0.95,
             f"N={img_f.size}\nNZ={np.count_nonzero(img_f)}",
             transform=ax3.transAxes, color="white",
             fontsize=9, va="top",
             bbox=dict(facecolor="black", alpha=0.6, pad=3))

    # ============================================================
    # CURVAS
    # ============================================================
    ax4.plot(Time[:tsnap+1], LCb[:tsnap+1], label="Brisk")
    ax4.plot(Time[:tsnap+1], LCs[:tsnap+1], label="Slow")
    ax4.plot(Time[:tsnap+1], LCf[:tsnap+1], label="Fast")
    ax4.axvline(Time[tsnap], color="k", ls="--", lw=1)

    ax4.set_xlim(Time[0], Time[-1])
    ax4.set_xlabel("T (M)")
    ax4.set_ylabel("Flux (normalized)")
    ax4.legend()

    # ============================================================
    # GUARDAR FRAME
    # ============================================================
    fname = os.path.join(frames_dir, f"frame_{tsnap:04d}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    images.append(imageio.imread(fname))

# ============================================================
# CREAR GIF
# ============================================================
imageio.mimsave(gif_path, images, fps=15)

print("The GIF of the black holes modes was made")