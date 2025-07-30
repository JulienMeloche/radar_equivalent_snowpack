# Radar Equivalent Snowpack

Radar Equivalent Snowpack method from **Meloche et al. (2025)**  
DOI: [10.5194/egusphere-2024-3169](https://doi.org/10.5194/egusphere-2024-3169)

This method computes radar-equivalent snowpack properties using **Numba** for fast computation. The main callable function is `radar_equivalent_snowpack` located in the `res_function_numba.py` file.

---

## Overview

The radar equivalent snowpack method estimates simplified radar-equivalent characteristics of a snowpack from various physical and environmental input parameters. The inputs are numpy arrays, and the output is a numpy 2D array representing a simplified snowpack.

The electromagnetic calculations are primarily derived from the **SMRT** model (https://github.com/smrt-model/smrt). This code was developed by Julien Meloche for Environment and Climate Change Canada (ECCC).

See ``paper`` for jupyter notebooks focusing on the results and Figures from Meloche et al. (2025).

---

## Requirements

Python libraries required to run the code:

- `numpy`  
- `numba`

---

## Usage

```python
radar_equivalent_snowpack(
    thickness: np.ndarray,
    temperature: float | np.ndarray,
    density: np.ndarray,
    ssa: float | np.ndarray,
    mode: str = 'active',
    freq: float = 17.5e9,
    n_layers: int = 3
) -> np.ndarray
```

 ## Parameters
```
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
```

## Returns
```
 numpy.ndarray
     2D array with n_layers row (default is 3).
     Columns represent:
     - Column 0: Total thickness  (to access use results[:, 0])
     - Column 1: temperature (to access use results[:, 1])
     - Column 2: density (to access use results[:, 2])
     - Column 3: specific surface area (SSA) (to access use results[:, 3])
```
