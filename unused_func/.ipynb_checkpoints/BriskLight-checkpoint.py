def Mask_FilterTime(T,porcentage):
    """
    Filter of the time data wth higher frequency and Create a mask for 
    data falling into the selected containers
    bins: number of bins of the distribution
    T: Time coordinate data"""
    Bins = freedman_diaconis_bins(T)
    # frequency per bin 
    hist, bins = np.histogram(T, bins=Bins) 
    
    # Here we search for the most frequent bin
    n_top_bins = max(1, int(len(hist) * porcentage)) #Numbers of bins with the higher frequency
    #Wee choose the index of bins with the higher frequency
    top_bins_indices = heapq.nlargest(n_top_bins, range(len(hist)), key=lambda i: hist[i])
    
    mascara = np.zeros(len(T), dtype=bool)
    
    for idx in  top_bins_indices:
        bin_min = bins[idx]
        bin_max = bins[idx + 1]
        
        #Create the mask
        if idx == len(bins) - 2:  
            mascara |= (T >= bin_min) & (T <= bin_max)
        else:
            mascara |= (T >= bin_min) & (T < bin_max)
            
    return mascara

def freedman_diaconis_bins(data):
    """
    Calculates the optimal number of bins using the Freedman-Diaconis rule.
    Ideal for approximately normal distributions with anomalies.
    :param data: array-like
    :param n_bins: int
    
    :return: Optimal number of bins for histogram
    """
    data = np.asarray(data)
    n = len(data)
    
    # Data range
    data_range = data.max() - data.min()
    
    # Interquartile range (robust to outliers)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # Avoid division by zero
    if IQR == 0:
        IQR = data_range / 10  # fallback
    
    # Bin width according to Freedman-Diaconis
    bin_width = 2 * IQR * n ** (-1/3)
    
    # Number of bins
    n_bins = max(1, int(np.ceil(data_range / bin_width)))
    
    # Upper limit to avoid excessive bins
    n_bins = min(n_bins, 500)
    
    return n_bins

def periodic_interval_mask(t, left, right):
    if left <= right:
        return (t >= left) & (t <= right)
    else:
        return (t >= left) | (t <= right)


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

    # Restrict everything to the active lensing-band pixels
    alpha = grid[:, 0][mask]
    beta = grid[:, 1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]
    redshift_sign = redshift_sign[mask]

    lamb, eta = rt.conserved_quantities(alpha, beta, thetao, a)

    brightness = np.zeros_like(rs, dtype=float)

    # Temporal selection
    time_mask = periodic_interval_mask(ts, left_s, right_s)

    # Final masks: only interpolate where both the region and time are valid
    disk_mask = (rs >= isco) & time_mask 
    gas_mask = (rs < isco) & time_mask 

    if np.any(disk_mask):
        x_disk = rs[disk_mask] * np.cos(th[disk_mask])
        y_disk = rs[disk_mask] * np.sin(th[disk_mask])

        interp_disk = interpolation(np.column_stack((ts[disk_mask], x_disk, y_disk)))

        g_disk = gDisk(rs[disk_mask],a,redshift_sign[disk_mask],lamb[disk_mask],eta[disk_mask]) ** gfactor

        brightness[disk_mask] = g_disk * interp_disk

    if np.any(gas_mask):
        x_gas = rs[gas_mask] * np.cos(th[gas_mask])
        y_gas = rs[gas_mask] * np.sin(th[gas_mask])

        interp_gas = interpolation(np.column_stack((ts[gas_mask], x_gas, y_gas)))

        g_gas = gGas(rs[gas_mask],a,redshift_sign[gas_mask],lamb[gas_mask],eta[gas_mask]) ** gfactor

        brightness[gas_mask] = g_gas * interp_gas

    # Horizon cutoff
    r_p = 1 + np.sqrt(1 - a**2)
    brightness[rs <= r_p] = 0
    # Rebuild full image array
    I = np.zeros(mask.shape, dtype=float)
    I[mask] = brightness

    return I


def zoom_slow_light(grid,mask,redshift_sign,a,isco,rs,th,ts,interpolation,thetao,zoom):
    """
    Calculate the black hole image including the time delay due to lensing and geometric effect
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
    :param zoom: porcentage of interest data for the zoom in the BH

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    #######################################
    # without nan
    valid_mask = np.isfinite(ts)
    
    # time filtered 
    ts_valid = ts[valid_mask]
    time_mask_valid = Mask_FilterTime(ts_valid,zoom)
    
    # mask reconstruction
    time_mask = np.zeros_like(ts, dtype=bool)
    time_mask[valid_mask] = time_mask_valid
    
    # final mask 
    combined_mask = mask & time_mask
    #######################################
    
    alpha = grid[:,0][combined_mask]
    beta = grid[:,1][combined_mask]
    rs = rs[combined_mask]
    th = th[combined_mask]
    ts = ts[combined_mask]
    
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[combined_mask]
    
    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)

    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*interpolation(np.vstack([ts[rs>=isco],x_aux[rs>=isco],y_aux[rs>=isco]]).T)
    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*interpolation(np.vstack([ts[rs<isco],x_aux[rs<isco],y_aux[rs<isco]]).T)

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape) 
    I[combined_mask] = brightness
    return(I)


print("BriskLight calculation starts!")

i_dt = xtend/nt
timeconversion=i_dt*MMkg*Gc/cc**3/(3600*24) # [days]

interpolated3_R=RegularGridInterpolator((times,x1,x2),data,fill_value=0,bounds_error=False,method='linear')

# Calcular ancho para n=0
mode0, left0, right0, _, _, _ = obsint.modal_hdi_kde(t0, p_brisk)
mode1, left1, right1, _, _, _ = obsint.modal_hdi_kde(t1, p_brisk)
mode2, left2, right2, _, _, _ = obsint.modal_hdi_kde(t2, p_brisk)

#widthleft0  = np.abs(mode0-left0)
#widthright0 = np.abs(right0-mode0)
#widthleft1  = np.abs(mode1-left1)
#widthright1 = np.abs(right1-mode1)
#widthleft2  = np.abs(mode2-left2)
#widthright2 = np.abs(right2-mode2)




I0s = []
I1s = []
I2s = []


def mp_worker(tsnap):
    ts0 = np.mod(t0 + tsnap, xtend)
    ts1 = np.mod(t1 + tsnap, xtend)
    ts2 = np.mod(t2 + tsnap, xtend)

    left0_snap  = np.mod(left0 + tsnap, xtend)
    right0_snap = np.mod(right0 + tsnap, xtend)

    left1_snap  = np.mod(left1 + tsnap, xtend)
    right1_snap = np.mod(right1 + tsnap, xtend)

    left2_snap  = np.mod(left2 + tsnap, xtend)
    right2_snap = np.mod(right2 + tsnap, xtend)
    
    i_bghts0 = obsint.brisk_light(
        supergrid0, mask0, sign0, spin_case, isco, rs0, phi0, ts0,
        interpolated3_R, thetao, left0_snap, right0_snap
    )

    i_bghts1 = obsint.brisk_light(
        supergrid1, mask1, sign1, spin_case, isco, rs1, phi1, ts1,
        interpolated3_R, thetao, left1_snap, right1_snap
    )

    i_bghts2 = obsint.brisk_light(
        supergrid2, mask2, sign2, spin_case, isco, rs2, phi2, ts2,
        interpolated3_R, thetao, left2_snap, right2_snap
    )

    i_I0 = (i_bghts0).reshape(N0,N0).T
    i_I1 = (i_bghts1).reshape(N1,N1).T
    i_I2 = (i_bghts2).reshape(N2,N2).T

    print("Calculating an image at time t=%s (M)"%np.round(tsnap,5))
    return(i_I0,i_I1,i_I2)


###Opción for Windows
def main():
    p = get_context("spawn").Pool(nthreads)
    I0s, I1s, I2s = zip(*p.map(mp_worker, np.linspace(i_tM + i_frame, f_tM, snapshots)))
    filename=path_bl+"BriskLight_p%s_dx%s_dt%s_dtM%s_a%s_i%s_%s.csv"%(p_brisk,dx0,dt,dt_movie,spin_case,i_case,i_fname[:-3])

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('bghts0', data=np.array(I0s))
    h5f.create_dataset('bghts1', data=np.array(I1s))
    h5f.create_dataset('bghts2', data=np.array(I2s))
    h5f.create_dataset('tc', data=np.array([timeconversion]))
    h5f.create_dataset('limits', data=np.array([limits]))

    print(h5f['bghts0'])
    print("Images ",filename," created.\n")
    h5f.close()
    p.close()
    # Aquí puedes agregar lo que quieras hacer con I0s, I1s, I2s

if __name__ == '__main__':
    main()