##### Brisk Light Functions 
from aart_func import *
from params import * 
#The functions work in intensity_f.py

def clip_ts_to_interval(ts, left_s, right_s):
    """
    Clip finite values of a 1D time array to an interval on a periodic domain.

    Case 1: left_s <= right_s
        Valid interval is [left_s, right_s].
        Values below are mapped to left_s, values above to right_s.

    Case 2: left_s > right_s
        The valid interval wraps around the period boundary:
            [left_s, T) U [0, right_s]
        Then the excluded gap is (right_s, left_s).
        Values inside that gap are mapped to the nearest boundary,
        using the midpoint of the gap.

    Non-finite values (NaN, inf, -inf) are preserved.

    :param ts array_like, 1D: Array of emission times.
    :param left_s float: Left boundary.
    :param right_s float: Right boundary.

    :Returns ts_out : ndarray Output array with clipped values.
    """
    ts = np.asarray(ts, dtype=float)

    if ts.ndim != 1:
        raise ValueError("ts must be a 1D array.")
    if ts.size == 0:
        raise ValueError("ts cannot be empty.")

    ts_out = ts.copy()
    finite_mask = np.isfinite(ts_out)

    if left_s <= right_s:
        # Standard interval: [left_s, right_s]
        ts_out[finite_mask] = np.clip(ts_out[finite_mask], left_s, right_s)

    else:
        # Wrapped interval: [left_s, T) U [0, right_s]
        # Gap to collapse: (right_s, left_s)
        center = 0.5 * (left_s + right_s)

        gap_mask = finite_mask & (ts_out > right_s) & (ts_out < left_s)

        left_half_mask = gap_mask & (ts_out > center)
        right_half_mask = gap_mask & (ts_out < center)

        ts_out[left_half_mask] = left_s
        ts_out[right_half_mask] = right_s

        # If a value is exactly equal to center, choose one side consistently.
        ts_out[gap_mask & (ts_out == center)] = right_s

    return ts_out


def modal_hdi_kde(data,p,*,bounds=None,trim_quantiles=(0.005, 0.995),fit_on="trimmed",bw_method=None,gridsize=4096):
    """
    KDE modal mass interval with robust handling of long tails / extreme values.

    Parameters
    ----------
    data : array-like
        1D data.
    p : float
        Desired probability mass around the mode, between 0 and 1.
        p=0 returns only the mode.
        p=1 returns the whole accepted support.
    bounds : tuple or None
        Explicit bounds `(lower, upper)` of the region you want to consider.
        If None, bounds are computed from `trim_quantiles`.
    trim_quantiles : tuple
        Quantiles used to define the accepted support when `bounds=None`.
        Example: (0.005, 0.995) ignores the lowest 0.5% and highest 0.5%.
        For a one-sided long upper tail, use e.g. (0.0, 0.99).
    fit_on : {"trimmed", "winsorized", "all"}
        How to fit the KDE:
        - "trimmed": remove points outside bounds before fitting. Recommended.
        - "winsorized": clip points to bounds before fitting.
        - "all": fit KDE to all data, but compute mass only inside bounds.
    bw_method : str, float, callable, optional
        Passed to scipy.stats.gaussian_kde.
    gridsize : int
        Number of grid points used for numerical approximation.

    Returns
    -------
    dict
        Contains the KDE, mode, interval, selected mass, grid, density, mask,
        accepted bounds, and fraction of data ignored.
    """

    x = np.asarray(data, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size < 2:
        raise ValueError("Need at least two finite data points.")

    if not 0 <= p <= 1:
        raise ValueError("p must satisfy 0 <= p <= 1.")

    # Define the accepted region.
    if bounds is None:
        q_low, q_high = trim_quantiles
        if not 0 <= q_low < q_high <= 1:
            raise ValueError("trim_quantiles must satisfy 0 <= low < high <= 1.")
        lo, hi = np.quantile(x, [q_low, q_high])
    else:
        lo, hi = map(float, bounds)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid bounds.")

    ignored_fraction = np.mean((x < lo) | (x > hi))

    # Choose which data are used to fit the KDE.
    if fit_on == "trimmed":
        x_fit = x[(x >= lo) & (x <= hi)]
    elif fit_on == "winsorized":
        x_fit = np.clip(x, lo, hi)
    elif fit_on == "all":
        x_fit = x
    else:
        raise ValueError("fit_on must be 'trimmed', 'winsorized', or 'all'.")

    if x_fit.size < 2:
        raise ValueError("Too few data points remain after trimming.")

    if np.ptp(x_fit) == 0:
        raise ValueError("Remaining data are constant; KDE is not well-defined.")

    kde = gaussian_kde(x_fit, bw_method=bw_method)

    # Evaluate KDE only on the accepted support.
    x_grid = np.linspace(lo, hi, gridsize)
    density_raw = kde(x_grid)

    # Renormalize density on accepted support.
    total_area = np.trapz(density_raw, x_grid)

    if total_area <= 0 or not np.isfinite(total_area):
        raise ValueError("Could not normalize KDE on the accepted support.")

    density = density_raw / total_area

    mode_idx = np.argmax(density)
    mode = x_grid[mode_idx]

    if p == 0:
        mask = np.zeros_like(x_grid, dtype=bool)
        mask[mode_idx] = True

        return {
            "kde": kde,
            "mode": mode,
            "interval": (mode, mode),
            "mass": 0.0,
            "bounds": (lo, hi),
            "ignored_fraction": ignored_fraction,
            "x_grid": x_grid,
            "density": density,
            "mask": mask,
            "density_threshold": density[mode_idx]}

    if p == 1:
        mask = np.ones_like(x_grid, dtype=bool)

        return {
            "kde": kde,
            "mode": mode,
            "interval": (lo, hi),
            "mass": 1.0,
            "bounds": (lo, hi),
            "ignored_fraction": ignored_fraction,
            "x_grid": x_grid,
            "density": density,
            "mask": mask,
            "density_threshold": 0.0,
        }

    def modal_component_at_threshold(threshold):
        """Connected region around the mode where density >= threshold."""
        above = density >= threshold

        left = mode_idx
        while left > 0 and above[left - 1]:
            left -= 1

        right = mode_idx
        while right < len(x_grid) - 1 and above[right + 1]:
            right += 1

        return left, right

    def component_mass(left, right):
        if right <= left:
            return 0.0
        return np.trapz(density[left:right + 1], x_grid[left:right + 1])

    # Binary search for the highest density threshold whose modal component
    # contains probability mass at least p.
    low = 0.0
    high = density[mode_idx]

    for _ in range(60):
        mid = 0.5 * (low + high)
        left, right = modal_component_at_threshold(mid)
        mass_mid = component_mass(left, right)

        if mass_mid >= p:
            low = mid
        else:
            high = mid

    threshold = low
    left, right = modal_component_at_threshold(threshold)

    interval = (x_grid[left], x_grid[right])
    mass = component_mass(left, right)

    mask = np.zeros_like(x_grid, dtype=bool)
    mask[left:right + 1] = True

    return {
        "kde": kde,
        "mode": mode,
        "interval": interval,
        "mass": mass,
        "bounds": (lo, hi),
        "ignored_fraction": ignored_fraction,
        "x_grid": x_grid,
        "density": density,
        "mask": mask,
        "density_threshold": threshold,
    }

def brisk_light(grid, mask, redshift_sign, a, isco, rs, th, ts,interpolation, thetao, left_s, right_s):
    """
    Calculate the black hole image including the time delay due to lensing and geometric effect but with a restriction in the source to the range [left_s, right_s]

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
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]

    ts_reduce = clip_ts_to_interval(ts, left_s,right_s)
    
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)

    brightness[rs>=isco]= gDisk(rs[rs>=isco],a,redshift_sign[rs>=isco],lamb[rs>=isco],eta[rs>=isco])**gfactor*interpolation(np.vstack([ts_reduce[rs>=isco],x_aux[rs>=isco],y_aux[rs>=isco]]).T)
    brightness[rs<isco]= gGas(rs[rs<isco],a,redshift_sign[rs<isco],lamb[rs<isco],eta[rs<isco])**gfactor*interpolation(np.vstack([ts_reduce[rs<isco],x_aux[rs<isco],y_aux[rs<isco]]).T)

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape) 
    I[mask] = brightness
    return(I)