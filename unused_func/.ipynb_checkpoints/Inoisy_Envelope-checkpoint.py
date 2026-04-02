from aart_func import *
from params_modinoisy import *
import subprocess

### Original Brisk Light

def brisk_light(grid, mask, redshift_sign, a, isco, rs, th, ts,interpolation, thetao, left_s, right_s):
    """
    Calculate the black hole image including the time delay due to lensing and geometric effect but with a restriction in the source
    (Eq. 50 P1)

    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param ts: time of emission at the source
    :param interpolation: a time series of 2 dimensional brightness function of the source, 3d interpolation object
    :param thetao: observer inclination
    :param left_s: left boundary of the modal HDI
    :param right_s: right boundary of the modal HDI 

    :return: image of a lensed equitorial source with only radial dependence. 
    """

    alpha = grid[:,0][mask]
    beta  = grid[:,1][mask]
    rs    = rs[mask]
    th    = th[mask]
    ts    = ts[mask]

    time_mask = periodic_interval_mask(ts, left_s, right_s)

    cond_disk = (rs >= isco)
    cond_gas  = (rs < isco)

    interp_disk = cond_disk & time_mask
    interp_gas  = cond_gas & time_mask

    lamb, eta = rt.conserved_quantities(alpha, beta, thetao, a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]

    x_aux = rs*np.cos(th)
    y_aux = rs*np.sin(th)

    if np.any(cond_disk):
        g_factors = gDisk(rs[cond_disk], a, redshift_sign[cond_disk],
                          lamb[cond_disk], eta[cond_disk])**gfactor
        interp_values = np.zeros(np.sum(cond_disk))
        if np.any(interp_disk):
            disk_idx = np.where(cond_disk)[0]
            interp_idx = np.where(interp_disk)[0]
            local_mask = np.isin(disk_idx, interp_idx)
            interp_values[local_mask] = interpolation(
                np.vstack([ts[interp_disk], x_aux[interp_disk], y_aux[interp_disk]]).T
            )
        brightness[cond_disk] = g_factors * interp_values

    if np.any(cond_gas):
        g_factors = gGas(rs[cond_gas], a, redshift_sign[cond_gas],
                         lamb[cond_gas], eta[cond_gas])**gfactor
        interp_values = np.zeros(np.sum(cond_gas))
        if np.any(interp_gas):
            gas_idx = np.where(cond_gas)[0]
            interp_idx = np.where(interp_gas)[0]
            local_mask = np.isin(gas_idx, interp_idx)
            interp_values[local_mask] = interpolation(
                np.vstack([ts[interp_gas], x_aux[interp_gas], y_aux[interp_gas]]).T
            )
        brightness[cond_gas] = g_factors * interp_values

    r_p = 1 + np.sqrt(1 - a**2)
    brightness[rs <= r_p] = 0

    I = np.zeros(mask.shape)
    I[mask] = brightness
    return I



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




i_source = "inoisy_1024_2048_30_2500_5.00_0.10_0.9400_1.00_1.00_1.00_0.349_137.0_137.0_3459.0.h5"

variables = extract_variables(i_source)

snapshots = Variable["N_t"]
f_tM = Variable["temporal_extension"]
spin_case = Variable["spin"]

Noise = [0.2,0.4] 

for i in range(2):
    noise = Noise[i]
    print("Working with the noise = %s"%(noise))
    subprocess.run(["python", "modinoisy.py", str(noise), str(spin_case), str(f_tM), str(snapshots), str(i_source)]) 

