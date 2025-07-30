"""
Radar equivalent snowpack method from Meloche et al (2025) DOI: 10.5194/egusphere-2024-3169
This method use numba for fast computation. The main method to call is radar_equivalent_snowpack.

Input are numpy arrays of the different snowpack parameters
return a numpy array of the simplified snowpack

Electro Mag calculation mosty from SMRT
by Julien Meloche for eccc
"""

import numpy as np
from numba import njit, vectorize


#global Constant
DENSITY_OF_ICE = 916.7
C_SPEED = 299792458.0
FREEZING_POINT = 273.15

def radar_equivalent_snowpack(thickness, temperature, density, ssa, mode = 'active', freq = 17.5e9, n_layers=3):
    """
    Calculate the radar equivalent snowpack properties.

    This function determines the radar-equivalent characteristics of a snowpack
    based on various physical and environmental parameters.

    Parameters
    ----------
    thickness : numpy array
        Total thickness of the snowpack in meters.
    density : numpy array
        Snow density in kg/m³.
    temperature : float or numpy array
        Snow temperature in degrees kelvin (always < 273.15).
    ssa : float or numpy array
        Specific surface area (SSA) of the snow in m²/kg.
    mode : str, optional
        Radar measurement mode, either 'active' or 'passive' (default is 'active').
    freq : float, optional
        Radar frequency in Hz (default is 17.5 GHz for TSMM upper-frequency).
    n_layers : int, optional
        Number of layers to consider in the simplified snowpack (default is 3).


    Returns
    -------
    numpy.ndarray
        2D array with n_layers row (default is 3).
        Columns represent:
        - Column 0: Total thickness  (to access use results[:, 0])
        - Column 1: temperature (to access use results[:, 1])
        - Column 2: density (to access use results[:, 2])
        - Column 3: specific surface area (SSA) (to access use results[:, 3])

    """
    height = np.cumsum(thickness[::-1])[::-1]
    ke = fast_compute_ke(freq, temperature, density, ssa)
    
    if mode == 'active':
        X = np.column_stack((ke, height))
        # kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
        # labels = kmeans.fit_predict(X)
        _, labels = numba_kmeans(X, n_layers, max_iterations=300)
        # Find unique labels

        b, idx = np.unique(labels, return_index=True)
        unique_labels = b[np.argsort(idx)]
        

    elif mode == 'passive':
        #use equal layer for passive
        labels, unique_labels = n_equal_layer(height, n_layers = n_layers)

    else:
        # if mode is invalid
        return ValueError('mode is invalid, enter active or passive')
    

    return process_dataframe_optimized(thickness, temperature, density, ssa, ke, labels, unique_labels)


### Equal layer method for passive
@njit
def n_equal_layer(height, n_layers):
    """
    Divide snow into n equal thickness layers and compute averages for each layer.

    """
    # Calculate normalized heights
    norm_h = height / height[0]
    
    # Preallocate labels array
    labels = np.zeros(len(norm_h), dtype=np.int64)
    
    # Compute bounds
    for i in range(len(norm_h)):
        for j in range(n_layers):
            upper_bound = 1.0 - (j / n_layers)
            lower_bound = 1.0 - ((j+1) / n_layers)
            
            if lower_bound <= norm_h[i] <= upper_bound:
                labels[i] = j
                break
    
    # Find unique labels
    unique_labels = np.unique(labels)
    
    return labels, unique_labels

### Special averaging
@njit
def _process_label_core(label_thickness, label_temperature, label_density, label_ssa, label_ke):
    """
    Numba-optimized core calculations for processing a single label
    """
    thick = np.sum(label_thickness)
    weights = label_thickness * label_ke
    sum_weights = np.sum(weights)
    sum_thickness = np.sum(label_thickness)
    density_avg = np.sum(label_density * label_thickness) / sum_thickness
    ssa_avg = np.sum(label_ssa * weights) / sum_weights
    temperature_avg = np.sum(label_temperature * weights) / sum_weights
            
    return thick, temperature_avg, density_avg, ssa_avg

@njit
def process_dataframe_optimized(thickness, temperature, density, ssa, ke, labels, unique_labels):
    """
    Process snowpack data with explicit 2D NumPy array return
    """
    # Preallocate 2D results array with known type and size
    results = np.zeros((len(unique_labels), 4), dtype=np.float64)
    
    # Process each label
    for i, label in enumerate(unique_labels):
        # Create boolean mask for this label
        mask = (labels == label)
        
        # Extract data for this label using mask
        label_thickness = thickness[mask]
        label_density = density[mask]
        label_ssa = ssa[mask]
        label_temperature = temperature[mask]
        label_ke = ke[mask]
        
        # Call numba-optimized core function
        thick, temperature_avg, density_avg, ssa_avg = _process_label_core(
            label_thickness, label_temperature, label_density, label_ssa, label_ke
        )
        
        # Store results in preallocated array
        results[i, 0] = thick
        results[i, 1] = temperature_avg
        results[i, 2] = density_avg
        results[i, 3] = ssa_avg
    
    return results

### Numba Kmeans
@njit
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

@njit
def kmeans_plus_plus_init(data, k):
    """
    Initialize centroids using K-means++ algorithm
    """
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    
    # Choose the first centroid randomly
    first_idx = np.random.randint(0, n_samples)
    centroids[0] = data[first_idx]
    
    # Choose remaining centroids
    for i in range(1, k):
        # Calculate distances from points to nearest centroids
        min_dists = np.zeros(n_samples)
        
        for idx in range(n_samples):
            # Initialize with distance to first centroid
            min_dist = euclidean_distance(data[idx], centroids[0])
            
            # Check if other centroids are closer
            for j in range(1, i):
                dist = euclidean_distance(data[idx], centroids[j])
                if dist < min_dist:
                    min_dist = dist
            
            min_dists[idx] = min_dist
        
        # Square distances (as per K-means++ algorithm)
        squared_dists = min_dists ** 2
        
        # Create probability distribution
        sum_squared_dists = np.sum(squared_dists)
        if sum_squared_dists > 0:  # Avoid division by zero
            probs = squared_dists / sum_squared_dists
        else:
            # If all points are very close to centroids, choose randomly
            probs = np.ones(n_samples) / n_samples
        
        # Choose next centroid based on weighted probability
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        
        # Find index where random value falls in cumulative distribution
        next_idx = np.searchsorted(cumulative_probs, r)
        if next_idx >= n_samples:  # Ensure valid index
            next_idx = n_samples - 1
            
        centroids[i] = data[next_idx]
    
    return centroids

@njit
def update_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    counts = np.zeros(k)
    for i in range(data.shape[0]):
        centroids[labels[i]] += data[i]
        counts[labels[i]] += 1
    for i in range(k):
        if counts[i] > 0:
            centroids[i] /= counts[i]
    return centroids

@njit
def assign_clusters(data, centroids):
    n_samples = data.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        # Initialize with distance to first centroid
        min_dist = euclidean_distance(data[i], centroids[0])
        min_centroid = 0
        
        # Check remaining centroids
        for j in range(1, k):
            dist = euclidean_distance(data[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
                min_centroid = j
                
        labels[i] = min_centroid
    
    return labels

@njit
def numba_kmeans(data, k, max_iterations=300, tol=1e-4):
    """
    K-means clustering with K-means++ initialization
    """
    # Initialize centroids using K-means++
    centroids = kmeans_plus_plus_init(data, k)
    
    for _ in range(max_iterations):
        # Assign points to clusters
        labels = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, labels, k)
        
        # Check for convergence
        centroid_shift = np.sum((new_centroids - centroids)**2)
        if centroid_shift < tol:
            break
            
        centroids = new_centroids
    
    return centroids, labels

### ke calculation mostly from SMRT
@njit(nopython = True)
def debye_eqn(ssa, density):
    return  4 * (1 - density / DENSITY_OF_ICE) / (ssa * DENSITY_OF_ICE)  

@njit
def water_permittivity_maetzler87(frequency, temperature):
    """Numba optimized version of water permittivity calculation"""
    freqGHz = frequency / 1e9
    theta = 1 - 300.0 / temperature
    e0 = 77.66 - 103.3 * theta
    e1 = 0.0671 * e0
    f1 = 20.2 + 146.4 * theta + 316 * theta**2
    e2 = 3.52 + 7.52 * theta
    f2 = 39.8 * f1
    
    # Complex calculations
    denom1 = 1.0 - 1j * (freqGHz / f1)
    denom2 = 1.0 - 1j * (freqGHz / f2)
    
    Ew = e2 + (e1 - e2) / denom2 + (e0 - e1) / denom1
    return Ew

@njit
def ice_permittivity(frequency, temperature, liquid_water=0.0):
    """Numba optimized version of ice permittivity calculation"""
    freqGHz = frequency / 1e9
    tempC = temperature - FREEZING_POINT
    
    Ereal = 3.1884 + 9.1e-4 * tempC
    
    theta = 300.0 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    
    B1 = 0.0207
    B2 = 1.16e-11
    b = 335.0
    deltabeta = np.exp(-9.963 + 0.0372 * tempC)
    betam = (B1 / temperature) * (np.exp(b / temperature) / ((np.exp(b / temperature) - 1)**2)) + B2 * freqGHz**2
    beta = betam + deltabeta
    
    Eimag = alpha / freqGHz + beta * freqGHz
    
    epsice = Ereal + 1j * Eimag
    
    if liquid_water <= 0.0:
        return epsice
    
    epswater = water_permittivity_maetzler87(frequency, temperature)
    
    ice_water_frac_volume = 1 - liquid_water
    # Maxwell garnett for spheres
    Cplus = epsice + 2 * epswater
    Cminus = (epsice - epswater) * ice_water_frac_volume
    
    Emg = (Cplus + 2 * Cminus) / (Cplus - Cminus) * epswater
    
    return Emg

@njit
def ks_integrand(mu, frac_volume, corr_length, k0, eps, effective_permittivity):
    """Numba optimized integrand calculation"""
    sintheta_2 = np.sqrt((1.0 - mu) / 2.0)
    k_diff = 2.0 * k0 * sintheta_2 * np.abs(np.sqrt(effective_permittivity))
    
    e0 = 1.0
    depolarization_factors = 1.0 / 3.0
    
    # Compute the fourier transform of the autocorrelation function analytically
    corr_func_at_origin = frac_volume * (1.0 - frac_volume)
    X = (k_diff * corr_length)**2
    ft_corr_fn = corr_func_at_origin * 8.0 * np.pi * corr_length**3 / (1.0 + X)**2
    
    # Mean squared field ratio
    apparent_permittivity = effective_permittivity * (1.0 - depolarization_factors) + e0 * depolarization_factors
    y2 = np.abs(apparent_permittivity / (apparent_permittivity + (eps - e0) * depolarization_factors))**2
    iba_coeff = (1.0 / (4.0 * np.pi)) * np.abs(eps - e0)**2 * y2 * (k0)**4
    p11 = (iba_coeff * ft_corr_fn).real * mu**2
    p22 = (iba_coeff * ft_corr_fn).real
    ks_int = (p11 + p22)
    
    return ks_int.real

@vectorize
def fast_compute_ke(frequency, temperature, density, ssa):
    """Numba optimized ke layer computation"""
    corr_length = debye_eqn(ssa, density)
    frac_volume = density/DENSITY_OF_ICE
    e0 = 1.0
    eps = ice_permittivity(frequency, temperature, liquid_water=0.0)
    
    k0 = 2.0 * np.pi * frequency / C_SPEED
    depolarization_factors = 1.0 / 3.0
    effective_permittivity = e0 * (1.0 + frac_volume * (eps - e0) / 
                                  (e0 + (1.0 - frac_volume) * depolarization_factors * (eps - e0)))
    
    # Calculate scattering coefficient: integrate p11+p12 over mu
    k = 3  # number of samples
    mu_points = np.linspace(1.0, -1.0, 2**k + 1)
    
    # Initialize array for integration results
    y = np.zeros(len(mu_points))
    
    # Calculate integrand at each mu point
    for i in range(len(mu_points)):
        y[i] = ks_integrand(mu_points[i], frac_volume, corr_length, k0, eps, effective_permittivity)
    
    # Perform trapezoidal integration
    ks_int = 0.0
    for i in range(len(mu_points)-1):
        ks_int += (y[i] + y[i+1]) * (mu_points[i] - mu_points[i+1]) / 2.0
    
    ks = ks_int / 4.0  # Normalized by (1/4pi) per Ding et al. (2010)
    # Calculate ka
    ka = 2.0 * k0 * np.sqrt(effective_permittivity).imag
    
    return ka + ks

