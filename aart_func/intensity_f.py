from aart_func import *
from params import * 

def Delta(r,a):
    """
    Calculates the Kerr metric function Delta(t)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return r**2-2*r+a**2

def PIF(r,a):
    """
    Calculates PI(r) (Eq. B6 P1)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return (r**2+a**2)**2-a**2*Delta(r,a)

def urbar(r,a):
    """
    Calculates the r (contravariant) component of the four velocity for radial infall
    (Eq. B34b P1)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return -np.sqrt(2*r*(r**2+a**2))/(r**2)

def Omegabar(r,a):
    """
    Calculates the angular velocity of the radial infall
    (Eq. B32a P1)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return (2*a*r)/PIF(r,a)

def Omegahat(r,a,laux):
    """
    Calculates the angular velocity of the sub-Keplerian orbit
    (Eq. B39 P1)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return (a+(1-2/r)*(laux-a))/(PIF(r,a)/(r**2)-(2*a*laux)/r)

def uttilde(r, a,urT,OT):
    """
    Calculates the t (contravariant) component of the general four velocity
    (Eq. B52 P1)
    :param r: radius of the source
    :param a: spin of the black hole
    :param urT: r (contravariant) component of the general four velocity
    :param OT: Angular velocity of the general four velocity
    """
    return np.sqrt((1 + urT**2*r**2/Delta(r,a))/(1-(r**2+a**2)*OT**2-(2/r)*(1-a*OT)**2))

def Ehat(r,a,laux):
    """
    Calculates the orbital energy of the sub-Keplerian flow
    (Eq. B44a P1)
    :param r: radius of the source
    :param a: spin of the black hole
    :param laux: sub-Keplerian specific angular momentum
    """
    return np.sqrt(Delta(r,a)/(PIF(r,a)/(r**2)-(4*a*laux)/r-(1-2/r)*laux**2))

def nuhat(r,a,laux,Ehataux):
    """
    Calculates the radial velocity of the sub-Keplerian flow
    (Eq. B45 P1)
    :param r: radius of the source
    :param a: spin of the black hole
    :param laux: sub-Keplerian specific angular momentum
    :param Ehataux: sub-Keplerian orbital energy
    """
    return r/Delta(r,a)*np.sqrt(np.abs(PIF(r,a)/(r**2)-(4*a*laux)/r-(1-2/r)*laux**2-Delta(r,a)/(Ehataux**2)))

def lhat(r,a):
    """
    Calculates the rspecific angular momentum of the sub-Keplerian flow
    (Eq. B44b P1)
    :param r: radius of the source
    :param a: spin of the black hole
    """
    return sub_kep*(r**2+a**2-2*a*np.sqrt(r))/(np.sqrt(r)*(r-2)+a)

def Rint(r,a,lamb,eta):
    """
    Evaluates the "radial potential", for calculating the redshift factor for infalling material
    :param r: radius of the source
    :param a: spin of the black hole
    :param lamb: angular momentum
    :param eta: carter constant

    :return: radial potential evaluated at the source
    """
    #Eqns (P2 5)
    return (r**2 + a**2 - a*lamb)**2 - (r**2 - 2*r + a**2)*(eta + (lamb - a)**2)

def gDisk(r,a,b,lamb,eta):
    """
    Calculates the redshift factor for a photon outside the inner-most stable circular orbit(isco) (assume circular orbit)
    (Eq. B13 P1)
    :param r: radius of the source
    :param b: The +- sign of p^r
    :param a: spin of the black hole
    :param lamb: angular momentum
    :param eta: Carter constant

    :return: the redshift factor associated with the ray
    """

    OH=Omegahat(r,a,lhat(r,a))
    OT=OH+(1-betaphi)*(Omegabar(r,a)-OH)
    ur=(1-betar)*urbar(r,a)
    ut=uttilde(r,a,ur,OT)
    uphi=ut*OT
    
    return 1/(ut*(1-b*np.sign(ur)*sqrt(np.abs(Rint(r,a,lamb,eta)*ur**2))/Delta(r,a)/ut-lamb*uphi/ut))

def gGas(r,a,b,lamb,eta):
    """
    Calculates the redshift factor for a photon inside the isco (assume infalling orbit)
    (Eq. B13 P1)
    :param r: radius of the source
    :param a: spin of the black hole
    :param b: sign for the redshift
    :param lamb: angular momentum
    :param eta: carter constant

    :return: the redshift factor associated with the ray
    """
    #Calculate radius of the inner-most stable circular orbit
    isco=rms(a)

    lms=lhat(isco,a)
    OH=Omegahat(r,a,lms)
    OT=OH+(1-betaphi)*(Omegabar(r,a)-OH)

    Ems=Ehat(isco,a,lms)
    urhat=-Delta(r,a)/(r**2)*nuhat(r, a, lms ,Ems)*Ems
    ur=urhat+(1-betar)*(urbar(r,a)-urhat)
    ut=uttilde(r,a,ur,OT)
    uphi=OT*ut

    return 1/(ut*(1-b*np.sign(ur)*sqrt(np.abs(Rint(r,a,lamb,eta)*ur**2))/Delta(r,a)/ut-lamb*uphi/ut))

#TODO: This expression will just work for sure for the Keplerian velocity. 
# I need to check if it has to be modified for the general four-velocity
def CosAng(r,a,b,lamb,eta):
    """
    Calculates the cosine of the emission angle
    :param r: radius of the source
    :param a: spin of the black hole
    :param b: sign for the redshift
    :param lamb: angular momentum
    :param eta: Carter constant

    :return: the  cosine of the emission angle
    """
    #From eta, solve for Sqrt(p_\theta/p_t)
    kthkt=np.sqrt(eta)
    #Sqrt(g^{\theta\theta}) Evaluated at the equatorial plane
    thth=1/r
    return thth*gDisk(r,a,b,lamb,eta)*kthkt

#calculate the observed brightness for a purely radial profile
def bright_radial(grid,mask,redshift_sign,a,rs,isco,thetao):
    """
    Calculate the brightness of a rotationally symmetric disk
    (Eq. 50 P1)
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param rs: source radius
    :param isco: radius of the inner-most stable circular orbit
    :param thetao: observer inclination

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]

    rs = rs[mask]

    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)

    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]

    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*ilp.profile(rs[rs>=isco],a,gammap,mup,sigmap)
    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*ilp.profile(rs[rs<isco],a,gammap,mup,sigmap)
    
    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    
    return(I)

#########################################################################################
#########################################################################################
##########################################################################################
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
#########################################################################################
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
#########################################################################################
def modal_hdi_kde(data, p=0.68, gridsize=4000):
    """
    Compute a modal highest-density interval (HDI) from a kernel density estimate (KDE).
    This function estimates the probability density of the input data using a Gaussian
    KDE, identifies the mode as the location of maximum estimated density, and then
    finds the connected interval around that mode that contains approximately a target
    probability mass `p`.
    :param data: One-dimensional sample of data points.
    :param p: Target probability mass to be enclosed by the modal HDI. Must be between 0 and 1.
    :param gridsize: Number of grid points used to evaluate the KDE between the minimum and maximum of the data.
    
    :returns: A tuple containing:
              - mode: location of the KDE mode
              - left: left boundary of the modal HDI
              - right: right boundary of the modal HDI
              - width: interval width, computed as right - left
              - mass: approximate probability mass enclosed in the interval
              - threshold: density threshold defining the interval
    """
    data = np.asarray(data)
    # Filtrar valores no finitos
    data_finite = data[np.isfinite(data)]
    
    kde = gaussian_kde(data_finite)
    xgrid = np.linspace(data_finite.min(), data_finite.max(), gridsize)
    dens = kde(xgrid)

    dx = xgrid[1] - xgrid[0]
    probs = dens * dx
    mode_idx = np.argmax(dens)
    mode = xgrid[mode_idx]
    
    # Binary search over the density threshold
    lo, hi = 0.0, dens[mode_idx]

    best = None

    for _ in range(60):
        c = (lo + hi) / 2.0
        mask = dens >= c

        # Connected component containing the mode
        if not mask[mode_idx]:
            hi = c
            continue

        i = mode_idx
        l = i
        while l > 0 and mask[l - 1]:
            l -= 1
        r = i
        while r < len(mask) - 1 and mask[r + 1]:
            r += 1

        mass = probs[l:r+1].sum()

        if mass >= p:
            lo = c
            best = (mode, xgrid[l], xgrid[r], xgrid[r] - xgrid[l],  mass, c)
        else:
            hi = c

    return best  # (mode, left, right, width = right - left, mass, threshold)

#########################################################################################
#########################################################################################
##########################################################################################

#calculate the observed brightness for an arbitrary profile, passed in as the interpolation object
#but ignoring the time delay due to lensing
def fast_light(grid,mask,redshift_sign,a,isco,rs,th,ts,interpolation,thetao):
    """
    Calculate the black hole image ignoring the time delay due to lensing or geometric effect
    (Eq. 116 P1)
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param ts: time of emission at the source (for fast-light need to be and scalar value)
    :param interpolation: 2 dimensional brightness function of the source, interpolation object
    :param thetao: observer inclination

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]

    
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]

    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)
    
    K_1 =np.count_nonzero(rs>=isco)
    tcol_1 = np.full(K_1, ts)

    K_2 =np.count_nonzero(rs<isco)
    tcol_2 = np.full(K_2, ts)
    
    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*interpolation(np.vstack([tcol_1,x_aux[rs>=isco],y_aux[rs>=isco]]).T)
    
    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*interpolation(np.vstack([tcol_2,x_aux[rs<isco],y_aux[rs<isco]]).T)
    
    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    return(I)

def periodic_interval_mask(t, left, right, period):
    if left <= right:
        return (t >= left) & (t <= right)
    else:
        return (t >= left) | (t <= right)

def brisk_light(grid, mask, redshift_sign, a, isco, rs, th, ts,interpolation, thetao, left_s, right_s, period):
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
    :param period: snapshot time (observation time)

    :return: image of a lensed equitorial source with only radial dependence. 
    """

    alpha = grid[:,0][mask]
    beta  = grid[:,1][mask]
    rs    = rs[mask]
    th    = th[mask]
    ts    = ts[mask]

    time_mask = periodic_interval_mask(ts, left_s, right_s, period)

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

    
#def brisk_light(grid,mask,redshift_sign,a,isco,rs,th,ts,interpolation,thetao,width,tsnap):
#    """
#    Calculate the black hole image including the time delay due to lensing and geometric effect but with a restriction in the source
#    (Eq. 50 P1)
#
#    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
#    :param mask: mask out the lensing band, see lb_f.py for detail
#    :param redshift_sign: sign of the redshift
#    :param a: black hole spin
#    :param isco: radius of the inner-most stable circular orbit
#    :param rs: source radius
#    :param th: source angle, polar coordinate
#    :param ts: time of emission at the source
#    :param interpolation: a time series of 2 dimensional brightness function of the source, 3d interpolation object
#    :param thetao: observer inclination
#    :param width: Width of the interest time distribution and restriction of the source
#    :param tsnap: snapshot time (observation time)
#
#    :return: image of a lensed equitorial source with only radial dependence. 
#    """
#    alpha = grid[:,0][mask]
#    beta = grid[:,1][mask]
#    rs = rs[mask]
#    th = th[mask]
#    ts = ts[mask]
#
#    time_mask = (ts >= tsnap - width/2) & (ts <= tsnap + width/2)
#    
#    combined_mask1 = (rs>=isco) & time_mask
#    combined_mask2 = (rs<isco) & time_mask
#    
#    
#    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
#    brightness = np.zeros(rs.shape[0])
#    redshift_sign = redshift_sign[mask]
#    
#    x_aux=rs*np.cos(th)
#    y_aux=rs*np.sin(th)
#
#    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*interpolation(np.vstack([ts[combined_mask1],x_aux[rs>=isco],y_aux[rs>=isco]]).T)
#    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*interpolation(np.vstack([ts[combined_mask2],x_aux[rs<isco],y_aux[rs<isco]]).T)
#
#    r_p = 1+np.sqrt(1-a**2)
#    brightness[rs<=r_p] = 0
#    
#    I = np.zeros(mask.shape) 
#    I[mask] = brightness
#    return(I)


#calculate the observed brightness for an arbitrary, evolving profile, passed in as the interpolation object
def slow_light(grid,mask,redshift_sign,a,isco,rs,th,ts,interpolation,thetao):
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

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]
    
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)

    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*interpolation(np.vstack([ts[rs>=isco],x_aux[rs>=isco],y_aux[rs>=isco]]).T)
    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*interpolation(np.vstack([ts[rs<isco],x_aux[rs<isco],y_aux[rs<isco]]).T)

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape) 
    I[mask] = brightness
    return(I)
    
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

def br(supergrid0,mask0,N0,rs0,sign0,supergrid1,mask1,N1,rs1,sign1,supergrid2,mask2,N2,rs2,sign2):
    """
    Calculate and save the radial brightness profile
    """
    bghts0 = bright_radial(supergrid0,mask0,sign0,spin_case,rs0,isco,thetao)
    bghts1 = bright_radial(supergrid1,mask1,sign1,spin_case,rs1,isco,thetao)
    bghts2 = bright_radial(supergrid2,mask2,sign2,spin_case,rs2,isco,thetao)

    I0 = bghts0.reshape(N0,N0).T
    I1 = bghts1.reshape(N1,N1).T
    I2 = bghts2.reshape(N2,N2).T

    filename=path+"Intensity_a_%s_i_%s.h5"%(spin_case,i_case)
    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('bghts0', data=I0)
    h5f.create_dataset('bghts1', data=I1)
    h5f.create_dataset('bghts2', data=I2)

    h5f.close()

    print("File ",filename," created.")

def br_bv(supergrid0,mask0,N0,rs0,sign0):
    """
    Calculate and save the radial brightness profile
    """
    bghts0 = bright_radial(supergrid0,mask0,sign0,spin_case,rs0,isco,thetao)

    I0 = bghts0.reshape(N0,N0).T

    filename=path+"Intensity_bv_a_%s_i_%s.h5"%(spin_case,i_case)
    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('bghts0', data=I0)

    h5f.close()

    print("File ",filename," created.")

def gfactorf(grid,mask,redshift_sign,a,isco,rs,thetao):
    """
    Calculate the redshift factor
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param thetao: observer inclination

    :return: redshift factor at each point.

    """
    
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    gfact = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    gfact[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])
    gfact[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])
    
    r_p = 1+np.sqrt(1-a**2)
    gfact[rs<=r_p] = 0
    
    gs = np.zeros(mask.shape)
    gs[mask] = gfact
    return(gs)

# orbit for the centroid with radhs=Radius of the hotspot and velhs = 0.01 (angular frequency)
# one may put an arbitrary orbit
def x0(t):
    return(radhs*np.cos(t*velhs))

def y0(t):
    return(radhs*np.sin(t*velhs))

def flare_model(grid,mask,redshift_sign,a,rs,th,ts,thetao,rwidth,delta_t):

    """
    Calculate the black hole image including the time delay due to lensing and geometric effect
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param mbar: lensing band index 0,1,2,...
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param ts: time of emission at the source
    :param interpolation: a time series of 2 dimensional brightness function of the source, 3d interpolation object
    :param thetao: observer inclination
    
    :return: image of a lensed equitorial source with only radial dependence. 
    """

    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    x_aux = rs*np.cos(th)
    y_aux = rs*np.sin(th)
    # x0 and y0 is now a function of t, where one can specify an arbitrary equitorial orbit
    brightness = np.exp(-(x_aux-x0(ts+delta_t))**2/rwidth**2-(y_aux-y0(ts+delta_t))**2/rwidth**2)
    
    brightness[rs>=isco]*= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor
    brightness[rs<isco]*= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    return(np.nan_to_num(I))
