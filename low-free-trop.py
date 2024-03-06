import xarray as xr
import numpy as np
import numba
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps

@numba.njit(parallel=True)
def vintp(var, pres, plevs):
    s = var.shape # time, lev, lat, lon
    out = np.zeros((s[0],len(plevs),s[2],s[3]))
    for t in numba.prange(s[0]):
        for j in range(s[2]):
            for i in range(s[3]):
                out[t,:,j,i] = np.interp(plevs, pres[t,:,j,i], var[t,:,j,i])
                out[t,:,j,i] = np.where(plevs > pres[t,-1,j,i], np.nan, out[t,:,j,i])
    return out

def plot_lft(x_data, y_data, case):
    # 700 hPa specific humidity in g/kg
    # precipitation in mm/h

    # calculate density
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=[100,100])
    H = H.T  # Let each row list bins with common y range.

    # plot the density in log10 scale

    cmap = colormaps['jet']
    cmap_list = [colors.to_rgb(cmap(i)) for i in range(cmap.N)]
    cmap_list[0] = (1, 1, 1)  # Set the first color to white
    modified_cmap = colors.LinearSegmentedColormap.from_list('ModifiedColormap', cmap_list, cmap.N)

    plt.imshow(np.log10(H + 1), interpolation='nearest', origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap=modified_cmap, aspect='auto', vmin=0)

    plt.colorbar(label='log10(samples)')
    plt.xlabel('700 hPa specific humidity (g/kg)')
    plt.ylabel('Precipitation (mm/h)')
    plt.title('Density on 700 hPa specific humidity and precipitation plane')
    plt.savefig(f"/data/W.eddie/RCE/{case}/lft_Q-P.png")
    return

if __name__ == '__main__':
    cases = ['QSPSRCE_r10_8x4km_300K', 'QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    numba.set_num_threads(32) # set number of CPUs used by Numba
    for case in cases:
        print('Processing', case)
        # Load data
        files = os.popen(
            'ls /data/W.eddie/SPCAM/'+case+'/atm/hist/'+case+'.cam.h0.0001-{04..12}-??-00000.nc'
        ).read().strip().split('\n')
        ds = xr.open_mfdataset(files, concat_dim='time', combine='nested',
                                data_vars='minimal', coords='minimal',
                                compat='override', parallel=True).sel(
                                    lev=slice(600,800)
                                )
        Q = ds.Q * 1e3 # kg/kg to g/kg
        prec = ds.PRECT * 1000 * 3600 # m/s to mm/h
        pres = (ds.P0*ds.hyam + ds.PS*ds.hybm).transpose('time','lev','lat','lon')
        Q700 = vintp(Q.values, pres.values, np.array([70000.])).squeeze()
        
        prec_flat = prec.values.flatten()
        mask = prec_flat > 0
        Q700_flat = Q700.flatten()
        plot_lft(Q700_flat[mask], prec_flat[mask])