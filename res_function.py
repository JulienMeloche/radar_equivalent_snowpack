import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import xarray as xr
from datetime import datetime

#for server hpcr
import sys
sys.path.append("/home/jum002/store5/repo/smrt_fork/smrt")

#smrt local import
from smrt.core.globalconstants import DENSITY_OF_ICE
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


def three_layer(snow_df, method = 'thick'):
    """
    3 equal thickness method
    method param :str that need indicate average method
    """
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

def two_layer_k(snow_df, method = 'thick'):
    """
    Kmeans 2 cluster method
    method param :str that need indicate average method
    freq: float for frequency of sensor to calculate ke, defaut is TSMM upper Ku
    """
    X = pd.DataFrame({ 'ke' : compute_ke(snow_df), 'height' : snow_df.height})
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    snow_df['label'] = kmeans.labels_
    
    df = snow_df.groupby('label', sort = False).apply(lambda x: avg_snow_sum_thick(x, method = method))
    return df

def three_layer_k(snow_df, method = 'thick', freq = 17.5e9):
    """
    Kmeans 3 cluster method
    method param :str that need indicate average method
    freq: float for frequency of sensor, defaut is TSMM upper Ku
    """
    X = pd.DataFrame({ 'ke' : compute_ke(snow_df, freq =freq),  'height' : snow_df.height})
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
    snow_df['label'] = kmeans.labels_
    
    df = snow_df.groupby('label', sort = False).apply(lambda x: avg_snow_sum_thick(x, method = method, freq =freq))
    return df



def compute_ke(snow_df, freq = 17.5e9):
    """
    add ke to the snow dataframe
    freq : frequency at which ke is calculated
    """
    if isinstance(snow_df.thickness, np.floating):
        thickness = [snow_df.thickness]
    else:
        thickness = snow_df.thickness

    sp = make_snowpack(thickness=thickness, 
                        microstructure_model='exponential',
                        density= snow_df.SNODEN_ML,
                        temperature= snow_df.TSNOW_ML,
                        corr_length = debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)))
    #create sensor
    sensor  = sensor_list.active(freq, 35)
    
    #get ks from IBA class
    ks = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ks for layer in sp.layers])
    ka = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ka for layer in sp.layers])
    ke = ks + ka
    return ke

def avg_snow_sum_thick(snow_df, method = 'thick', freq = 17.5e9):
    """
    Averaging method
    method param :str that need indicate average method
    """
    thick = snow_df.thickness.sum()
    if method == 'thick':
        snow_mean = snow_df.apply(lambda x: np.average(x, weights = snow_df.thickness.values), axis =0)
        snow_mean['thickness'] = thick
        return snow_mean
    if method == 'thick-ke':
        snow_df['ke'] = compute_ke(snow_df, freq = freq)
        snow_mean = snow_df.apply(lambda x: np.average(x, weights = snow_df.thickness.values * snow_df.ke.values), axis =0)
        snow_mean['thickness'] = thick
        return snow_mean
    if method == 'thick-ke-density':
        snow_df['ke'] = compute_ke(snow_df, freq = freq)
        df_copy = snow_df.copy()
        density_temp = np.average(df_copy.SNODEN_ML, weights = snow_df.thickness.values )
        snow_mean = snow_df.apply(lambda x: np.average(x, weights =  snow_df.thickness.values*snow_df.ke.values, axis =0))
        snow_mean['thickness'] = thick
        snow_mean['SNODEN_ML'] = density_temp
        return snow_mean
    else:
        print('provide a valid method')
        return np.nan
    

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
        swe = [avg_snow_sum_thick(df.loc[date,:]).SNODEN_ML * avg_snow_sum_thick(df.loc[date,:]).thickness for date in dates]
        return dB(sig), swe

    if layer_type == 'two':
        snow = [build_snow(two_layer(df.loc[date,:], method = method)) for date in dates]
        swe = [(two_layer(df.loc[date,:], method = method).SNODEN_ML * two_layer(df.loc[date,:], method = method).thickness).sum() for date in dates]
        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq)
        sig = result.sigmaVV().values
        return dB(sig), np.array(swe)
    
    if layer_type == 'two_k':
        snow = [build_snow(two_layer_k(df.loc[date,:], method = method)) for date in dates]
        swe = [(two_layer_k(df.loc[date,:], method = method).SNODEN_ML * two_layer_k(df.loc[date,:], method = method).thickness).sum() for date in dates]
        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq)
        sig = result.sigmaVV().values
        return dB(sig), np.array(swe)
    
    if layer_type == 'three':
        snow = [build_snow(three_layer(df.loc[date,:], method = method)) for date in dates]
        swe = [(three_layer(df.loc[date,:], method = method).SNODEN_ML * three_layer(df.loc[date,:], method = method).thickness).sum() for date in dates]
        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq, diag_method ='shur_forcedtriu')
        sig = result.sigmaVV().values
        return dB(sig), np.array(swe)
    
    if layer_type == 'three_k':
        snow = [build_snow(three_layer_k(df.loc[date,:], method = method, freq = freq)) for date in dates]
        swe = [(three_layer_k(df.loc[date,:], method = method).SNODEN_ML * three_layer_k(df.loc[date,:], method = method).thickness).sum() for date in dates]  
        #calculate backscatter
        result = run_simu(snow, model = model, freq = freq, diag_method ='shur_forcedtriu')
        sig = result.sigmaVV().values
        return dB(sig), np.array(swe)


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

def run_simu(sp, model = 'iba', diag_method = 'eig', freq = 17.5e9):
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

