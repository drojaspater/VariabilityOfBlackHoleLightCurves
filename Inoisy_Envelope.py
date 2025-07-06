from aart_func import *
from params_modinoisy import *
import subprocess

def extract_variables(filename):
    parts = filename.split("_")
    # Extraer las partes principales
    name = parts[0]
    N_xy = int(parts[1])
    N_t = int(parts[2])
    spatial_ext = float(parts[3])
    temporal_ext = float(parts[4])
    corr_ratio = float(parts[5])
    spatial_corr_ratio = float(parts[6])
    spin = float(parts[7])
    keplerianity = float(parts[8])
    radial_velocity = float(parts[9])
    angular_velocity = float(parts[10])
    arm_angle = float(parts[11])
    time_corr = float(parts[12])
    spatial_corr = float(parts[13])
    
    # Procesar la semilla y la extensión
    seed_ext = parts[14].split(".")
    seed = int(float(seed_ext[0]))  # Convertir "1662.0" a 1662
    extension = seed_ext[1]         # "h5"


    return {
        "name": name,
        "N_xy": N_xy,
        "N_t": N_t,
        "spatial_extension": spatial_ext,
        "temporal_extension": temporal_ext,
        "correlation_ratio": corr_ratio,
        "spatial_correlation_ratio": spatial_corr_ratio,
        "spin": spin,
        "keplerianity_factor": keplerianity,
        "radial_velocity": radial_velocity,
        "angular_velocity": angular_velocity,
        "arm_angle": arm_angle,
        "time_correlation": time_corr,
        "spatial_correlation": spatial_corr,
        "seed": seed,
        "extension": extension
    }

# Listar y recorrer solo archivos .h5 (sin necesidad de if)
#for filepath in glob.glob('/projects/bkt/inoisy/*.h5'):
#    print(filepath)  # <- Ruta completa del archivo




i_source = "i_inoisy_1024_2048_30_2500_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_3459.0.h5"

variables = extract_variables(i_source)

snapshots = Variable["N_t"]
f_tM = Variable["temporal_extension"]
spin_case = Variable["spin"]

Noise = [0.2,0.4] 

for i in range(2):
    noise = Noise[i]
    print("Working with the noise = %s"%(noise))
    subprocess.run(["python", "modinoisy.py", str(noise), str(spin_case), str(f_tM), str(snapshots), str(i_source)]) 

