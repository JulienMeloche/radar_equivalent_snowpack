import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.integrate import trapezoid
import xarray as xr
from datetime import datetime

#for server hpcr
import sys
sys.path.append("/home/jum002/store5/repo/smrt_fork/smrt")

#smrt local import
from smrt.core.globalconstants import DENSITY_OF_ICE, C_SPEED, FREEZING_POINT
from smrt import sensor_list, make_model, make_snowpack, make_interface
from smrt.emmodel import iba
from smrt.substrate.reflector_backscatter import make_reflector
from smrt.utils import dB


"""
Grouping and averaging functions
equal and Kmean
"""

def two_layer(snow_df, method = 'thick'):
    """
    2 equal thickness method
    method param :str that indicate average method
    """
    #get norm height
    snow_df.loc[:,'norm_h'] = snow_df.height/snow_df.thickness.sum()
    #split by third and average
    snow_1 = avg_snow_sum_thick(snow_df[snow_df.norm_h >= 0.5], method = method)
    snow_2 = avg_snow_sum_thick(snow_df[snow_df.norm_h < 0.5], method = method) 

    return pd.DataFrame([df for df in [snow_1, snow_2] if not df.empty])


def three_layer(snow_df, method = 'thick', freq = 17.5e9):
    """
    3 equal thickness method
    method param :str that need indicate average method
    """
    snow_df['ke'] = compute_ke(snow_df, freq = freq)
    #get norm height
    snow_df.loc[:,'norm_h'] = snow_df.height/snow_df.thickness.sum()
    #split by third and average
    snow_1 = avg_snow_sum_thick(snow_df[snow_df.norm_h >= 0.66], method = method)

    #check if empty
    if not snow_df[(snow_df.norm_h <= 0.66) & (snow_df.norm_h >= 0.34)].empty:
        snow_2 = avg_snow_sum_thick(snow_df[(snow_df.norm_h <= 0.66) & (snow_df.norm_h >= 0.34)], method = method) 
    else:
        snow_2 = pd.DataFrame()

    #check if empty
    if not snow_df[snow_df.norm_h < 0.34].empty:
        snow_3 = avg_snow_sum_thick(snow_df[snow_df.norm_h < 0.34], method = method) 
    else: 
        snow_3 = pd.DataFrame()
    return pd.DataFrame([df for df in [snow_1, snow_2, snow_3] if not df.empty])

def two_layer_k(snow_df, method = 'thick', freq = 17.5e9):
    """
    Kmeans 2 cluster method
    method param :str that need indicate average method
    freq: float for frequency of sensor, defaut is TSMM upper Ku
    """
    snow_df['ke'] = compute_ke(snow_df, freq = freq)
    X = np.vstack([snow_df['ke'].values, snow_df['height'].values]).T
    #X = pd.DataFrame({ 'ke' : snow_df.ke,  'height' : snow_df.height})
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    snow_df['label'] = kmeans.fit_predict(X)

    grouped = [avg_snow_sum_thick(group, method=method, freq=freq) for label, group in snow_df.groupby('label', sort=False)]

    return pd.DataFrame(grouped)


def three_layer_k(snow_df, method = 'thick', freq = 17.5e9):
    """
    Kmeans 3 cluster method
    method param :str that need indicate average method
    freq: float for frequency of sensor, defaut is TSMM upper Ku
    """
    snow_df['ke'] = compute_ke(snow_df, freq =freq)
    X = np.vstack([snow_df['ke'].values, snow_df['height'].values]).T
    #X = pd.DataFrame({ 'ke' : snow_df.ke,  'height' : snow_df.height})
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    snow_df['label'] = kmeans.fit_predict(X)

    grouped = [avg_snow_sum_thick(group, method=method, freq=freq) for label, group in snow_df.groupby('label', sort=False)]

    return pd.DataFrame(grouped)


def avg_snow_sum_thick(snow_df, method='thick', freq=17.5e9):
    """
    Vectorized version of avg_snow_sum_thick function
    """
    # Extract arrays once at the beginning
    thickness = snow_df['thickness'].values
    thick = np.sum(thickness)
    
    # Extract all required data arrays upfront
    snoden = snow_df['SNODEN_ML'].values
    ssa = snow_df['ssa'].values
    tsnow = snow_df['TSNOW_ML'].values
    
    # Pre-calculated weights for all methods
    if method == 'thick':
        weights = thickness
    elif method in ('thick-ke', 'thick-ke-density'):
        ke = snow_df['ke'].values
        weights = thickness * ke
    else:
        return np.nan
    
    # Calculate all weighted averages with optimized vectorized operations
    if method in ('thick', 'thick-ke'):
        # For 'thick' and 'thick-ke', all columns use the same weights
        snow_mean = {
            'SNODEN_ML': np.sum(snoden * weights) / np.sum(weights),
            'ssa': np.sum(ssa * weights) / np.sum(weights),
            'TSNOW_ML': np.sum(tsnow * weights) / np.sum(weights)
        }
    elif method == 'thick-ke-density':
        # For 'thick-ke-density', SNODEN_ML uses thickness as weights, others use weights
        sum_weights = np.sum(weights)
        sum_thickness = np.sum(thickness)
        
        snow_mean = {
            'SNODEN_ML': np.sum(snoden * thickness) / sum_thickness,
            'ssa': np.sum(ssa * weights) / sum_weights,
            'TSNOW_ML': np.sum(tsnow * weights) / sum_weights
        }
    
    snow_mean['thickness'] = thick
    return pd.Series(snow_mean)
    

"""
Running and simulation of SMRT functions

"""
#define function to calculate sig and swe for different layer type and method

def get_sig_swe(df, dates, method, layer_type, model = 'iba', freq = 17.5e9):

    if layer_type == 'n':
        #n layer
        snow_n = [build_snow(df.loc[date,:]) for date in dates]
        param = get_other_var(df, dates)
        result = run_simu(snow_n, model = model, freq = freq, diag_method ='shur_forcedtriu')
        sig = result.sigmaVV().values
        swe = [df.loc[date,:].SNOMA_ML.sum() for date in dates]
        return dB(sig), np.array(swe), param
    
    if layer_type == 'one':
        #1 layer
        snow1 =  [build_snow(avg_snow_sum_thick(df.loc[date,:])) for date in dates]
        one_result = run_simu(snow1, model = model, freq = freq)
        sig = one_result.sigmaVV().values

        return dB(sig)

    if layer_type == 'two':
        snow = [build_snow(two_layer(df.loc[date,:], method = method)) for date in dates]

        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq)
        sig = result.sigmaVV().values
        return dB(sig)
    
    if layer_type == 'two_k':
        snow = [build_snow(two_layer_k(df.loc[date,:], method = method)) for date in dates]

        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq)
        sig = result.sigmaVV().values
        return dB(sig)
    
    if layer_type == 'three':
        snow = [build_snow(three_layer(df.loc[date,:], method = method)) for date in dates]

        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq, diag_method ='shur_forcedtriu')
        sig = result.sigmaVV().values
        return dB(sig)
    
    if layer_type == 'three_k':
        snow = [build_snow(three_layer_k(df.loc[date,:], method = method, freq = freq)) for date in dates]

        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq, diag_method ='shur_forcedtriu')
        sig = result.sigmaVV().values
        return dB(sig)


def debye_eqn(ssa, density):
    return 4 * (1 - density / DENSITY_OF_ICE) / (ssa * DENSITY_OF_ICE)  

def bias(reduced, nlayer):
    return reduced - nlayer

def rmse(sim, mes):
    return np.sqrt(np.mean((sim - mes)**2))


def build_snow(snow_df, transparent = False, transparent_nosurf = False):
    #creat SMRT snowpack from pandas dataframe

    t_soil = 265
    sub = make_reflector(temperature=t_soil, specular_reflection=0, 
                              backscattering_coefficient={'VV' : 0, 'HH' : 0})


    #Creating the snowpack to simulate with the substrate
    if isinstance(snow_df.thickness, np.floating):
        thickness = [snow_df.thickness]
    else:
        thickness = snow_df.thickness
    try:
        if transparent:
            sp = make_snowpack(thickness=thickness, 
                            microstructure_model='exponential',
                            density= snow_df.SNODEN_ML,
                            temperature= snow_df.TSNOW_ML,
                            corr_length = 0.75 *debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)),
                            substrate = sub)
            sp.interfaces = [make_interface('transparent') for inter in range(len(sp.interfaces))]

        elif transparent_nosurf:
            sp = make_snowpack(thickness=thickness, 
                            microstructure_model='exponential',
                            density= snow_df.SNODEN_ML,
                            temperature= snow_df.TSNOW_ML,
                            corr_length = 0.75 *debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)),
                            substrate = sub)
            
            sp.interfaces = [make_interface('transparent') for inter in range(len(sp.interfaces))]
            sp.interfaces[0] = make_interface('flat')

        else:  
            sp = make_snowpack(thickness=thickness, 
                            microstructure_model='exponential',
                            density= snow_df.SNODEN_ML,
                            temperature= snow_df.TSNOW_ML,
                            corr_length = 0.75 *debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)),
                            substrate = sub)
        return sp
    except:
        print(snow_df)

def run_simu(sp, model = 'iba', diag_method = 'eig', freq = 17.5e9, passive = False):
    # Run SMRT from SMRT snowpack

    #Modeling theories to use in SMRT
    if model == 'iba':
        model = make_model("iba", "dort", rtsolver_options=dict(error_handling='nan',
                                                                diagonalization_method=diag_method))

    if model == 'iba_inv':
        model = make_model("iba", "dort", emmodel_options=dict(dense_snow_correction="auto"),
                                          rtsolver_options=dict(error_handling='nan', 
                                                                diagonalization_method=diag_method))
    if model == 'symsce':
        model = make_model("symsce_torquato21", "dort", rtsolver_options=dict(error_handling='nan',
                                                                              diagonalization_method=diag_method))
    


    if passive:
        sensor  = sensor_list.passive(freq, 55)
    else:
        sensor  = sensor_list.active(freq, 35)
    result = model.run(sensor, sp, parallel_computation=True)
    return result


def get_other_var(df, dates):

    mean_temp = [df.loc[date,:].TSNOW_ML.mean() for date in dates]
    mean_rho = [df.loc[date,:].SNODEN_ML.mean() for date in dates]
    mean_ssa = [df.loc[date,:].ssa.mean() for date in dates]
    wet = [df.loc[date,:].WSNO.mean() for date in dates]
    tsurf = [df.loc[date,:].TSNO_SURF.mean() for date in dates]
    snowRate = [df.loc[date,:].SNOWRATE.mean() for date in dates]
    rainRate = [df.loc[date,:].RAINRATE.mean() for date in dates]
    
    param = {'mean_temp' : mean_temp, 
             'mean_rho' : mean_rho,
             'mean_ssa' : mean_ssa,
             'wet' : wet,
             'tsurf' :  tsurf,
             'snowRate' : snowRate,
             'rainRate' : rainRate}
    return param

def import_crocus(file_path, str_year_begin):

    mod = xr.open_dataset(file_path)

    df = mod[['SNODEN_ML','SNOMA_ML','TSNOW_ML','SNODOPT_ML','SNODP']].to_dataframe().dropna() 
    # SNODEN_ML: densite des couches
    # SNOMA_ML: SWE des couches
    # TSNOW_ML: T des couches
    # SNODOPT_ML: diametre optique des couches
    # SNODP: hauteur totale du snowpack

    # #filter at 5 seasons of beginin 
    str_begin = str_year_begin + '-10-01'
    oct01 = datetime.strptime(str_begin, '%Y-%m-%d')

    df = df.loc[oct01:,:]


    df['thickness'] = df[['SNODEN_ML','SNOMA_ML']].apply(lambda x : x[1] / x[0], axis = 1) 
    df['ssa'] = df['SNODOPT_ML'].apply(lambda x: 6/( x * 917) if x>0 else 0)
    df['TSNOW_ML'] = df['TSNOW_ML'].apply(lambda x: 273 if x > 273 else x)
    #df['corr_length'] = df[['SNODEN_ML','ssa']].apply(lambda x: debye_eqn(x[1], x[0]), axis = 1)

    #filter out low snowdepth and small snow layers
    df = df[(df.SNODP > 0.10) & (df.thickness > 0.005)]

    dates = df.groupby(level = 'time').mean().index.get_level_values(0)
    #add height to dataframe
    df.loc[:, 'height'] = np.nan
    for date in dates:
        df_temp = df.loc[date,:]
        df.loc[date,'height'] = np.cumsum(df_temp.thickness.values[::-1])[::-1]

    return df, dates

# def compute_ke(snow_df, freq = 17.5e9, return_ks_ka = False):
#     """
#     add ke to the snow dataframe
#     freq : frequency at which ke is calculated
#     """
#     if isinstance(snow_df.thickness, np.floating):
#         thickness = [snow_df.thickness]
#     else:
#         thickness = snow_df.thickness

#     sp = make_snowpack(thickness=thickness, 
#                         microstructure_model='exponential',
#                         density= snow_df.SNODEN_ML,
#                         temperature= snow_df.TSNOW_ML,
#                         corr_length = debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)))
#     #create sensor
#     sensor  = sensor_list.active(freq, 35)
    
#     #get ks from IBA class
#     ks = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ks for layer in sp.layers])
#     ka = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ka for layer in sp.layers])

#     if return_ks_ka:
#         return ks, ka
#     else:
#         ke = ks + ka

#         return ke

def water_permittivity_maetzler87(frequency, temperature):
    """Calculates the complex water dielectric constant depending on the frequency and temperature
     Based on Mätzler, C., & Wegmuller, U. (1987). Dielectric properties of freshwater
     ice at microwave frequencies. *Journal of Physics D: Applied Physics*, 20(12), 1623-1630.

     :param frequency: frequency in Hz
     :param temperature: temperature in K
     :raises Exception: if liquid water > 0 or salinity > 0 (model unsuitable)
     :returns: complex permittivity of pure ice
"""

    freqGHz = frequency / 1e9

    theta = 1 - 300.0 / temperature

    e0 = 77.66 - 103.3 * theta
    e1 = 0.0671 * e0

    f1 = 20.2 + 146.4 * theta + 316 * theta**2
    e2 = 3.52 + 7.52 * theta
    #  % version of Liebe MPM 1993 uses: e2=3.52
    f2 = 39.8 * f1

    Ew = e2 + (e1 - e2) / complex(1, -freqGHz / f2) + (e0 - e1) / complex(1, -freqGHz / f1)

    return Ew

def ice_permittivity(frequency, temperature, liquid_water = 0):
    """ 
    FROM SMRT**** credit to Ghislain Picard
    Calculates the complex ice dielectric constant depending on the frequency and temperature.

    Based on Mätzler, C. (1998). Thermal Microwave Radiation: Applications for Remote Sensing p456-461
    """

    freqGHz = frequency / 1e9

    tempC = temperature - FREEZING_POINT

    Ereal = 3.1884 + 9.1e-4 * tempC

    theta = 300.0 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)

    B1 = 0.0207
    B2 = 1.16e-11
    b = 335.
    deltabeta = np.exp(- 9.963 + 0.0372 * tempC)
    betam = (B1 / temperature) * (np.exp(b / temperature) / ((np.exp(b / temperature) - 1)**2)) + B2 * freqGHz**2
    beta = betam + deltabeta

    Eimag = alpha / freqGHz + beta * freqGHz

    epsice = Ereal + 1j * Eimag

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity_maetzler87(frequency, temperature)

    ice_water_frac_volume = 1 - liquid_water
    #maxwell garnett for spheres
    # frac volume with water, if no water then same as dry snow
    Cplus = epsice + 2 * epswater
    Cminus = (epsice - epswater) * ice_water_frac_volume

    Emg = (Cplus + 2 * Cminus) / (Cplus - Cminus) * epswater

    return Emg

def ks_integrand(mu, frac_volume, corr_length, k0, eps, effective_permittivity):
    """ 
    FROM SMRT**** credit to Ghislain Picard
    Calculates the complex ice dielectric constant depending on the frequency and temperature.

    Based on Mätzler, C. (1998). Improved Born Approximation
    """
    sintheta_2 = np.sqrt((1. - mu) / 2.)  # = np.sin(theta / 2.)
    k_diff = np.asarray(2. * k0 * sintheta_2 * abs(np.sqrt(effective_permittivity)))

    e0 = 1
    depolarization_factors = 1. / 3.

    """compute the fourier transform of the autocorrelation function analytically"""
    corr_func_at_origin =  frac_volume * (1.0 - frac_volume)
    X = (k_diff* corr_length)**2
    ft_corr_fn = corr_func_at_origin * 8 * np.pi * corr_length**3 / (1. + X)**2

    #mean squared field ratio
    e0 = 1
    depolarization_factors = 1. / 3.
    apparent_permittivity = effective_permittivity * (1 - depolarization_factors) + e0 * depolarization_factors
    y2 =   np.absolute(apparent_permittivity / (apparent_permittivity + (eps - e0) * depolarization_factors))**2.
    iba_coeff = (1. / (4. * np.pi)) * np.absolute(eps - e0)**2. * y2 * (k0)**4
    p11 = (iba_coeff * ft_corr_fn).real * mu**2
    p22 = (iba_coeff * ft_corr_fn).real * 1.
    ks_int = (p11 + p22)

    return ks_int.real

def compute_ke_iba(frequency, temperature, density, ssa):
    """ 
    FROM SMRT**** credit to Ghislain Picard
    Calculates the complex ice dielectric constant depending on the frequency and temperature.

    Based on Mätzler, C. (1998). Improved Born Approximation
    """
    corr_length = debye_eqn(ssa, density)
    frac_volume = density/DENSITY_OF_ICE
    e0 = 1.0
    eps = ice_permittivity(frequency, temperature, liquid_water = 0)
    #eps = complex(3.185, 0.005)

    k0 = 2 * np.pi * frequency / C_SPEED
    depolarization_factors = 1. / 3.
    effective_permittivity = np.complex128(e0 * (1 + frac_volume * (eps - e0) / (e0 + (1. - frac_volume) * depolarization_factors * (eps - e0))))
    
    #ks
    #Calculate scattering coefficient: integrate p11+p12 over mu
    k = 3  # number of samples. This should be adaptative depending on the size/wavelength
    #mu = np.linspace(1, -1, 2**k + 1)
    mu = np.expand_dims(np.linspace(1, -1, 2**k + 1), axis =1)
    y = ks_integrand(mu, frac_volume, corr_length, k0, eps, effective_permittivity)
    ks_int = np.trapz(y, -mu, axis =0)  # integrate on mu between -1 and 1
    ks = ks_int / 4.  # Ding et al. (2010), normalised by (1/4pi)

    #ka
    k0 = 2 * np.pi * frequency / C_SPEED
    ka = 2 * k0 * np.sqrt(effective_permittivity).imag

    return ka + ks

def compute_ke_rayleigh(frequency, temperature, density, ssa):
    frac_volume = density / DENSITY_OF_ICE

    e0 = 1
    eps = ice_permittivity(frequency, temperature, liquid_water = 0)
    lmda = C_SPEED / frequency

    radius = 6/(DENSITY_OF_ICE * ssa)

    k0 = 2 * np.pi / lmda

    ks = frac_volume * 2 * abs((eps - e0) / (eps + 2 * e0))**2 * radius**3 * e0**2 * k0**4
    ka = frac_volume * 9 * k0 * eps.imag * abs(e0 / (eps + 2 * e0))**2 + (1 - frac_volume) * e0.imag * k0

    return ks + ka

def compute_ke(df, freq = 17.5e9, method = 'iba'):
    if method == 'iba':
        return compute_ke_iba(freq, df.TSNOW_ML.values, df.SNODEN_ML.values, df.ssa.values)
    if method == 'rayleigh':
        return compute_ke_rayleigh(freq, df.TSNOW_ML.values, df.SNODEN_ML.values, df.ssa.values)
