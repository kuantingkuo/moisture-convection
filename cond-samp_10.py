import xarray as xr
import numpy as np
import numba
import os

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

@numba.njit
def cond_samp(axis1, axis2, idx1, idx2, weight):
    out = np.zeros((len(axis2), len(axis1)))
    for i in range(len(idx1)):
        y = idx2[i]
        x = idx1[i]
        out[y,x] += weight[i]
    return out

if __name__ == '__main__':
    # Define the lat index ranges for the 10 regions
    lat_idxs = [[0, 40], [40, 57], [57, 71], [71, 84], [84, 97], [97, -84], [-84, -71], [-71, -57], [-57, -40], [-40, None]]
    Qlow = 10.
    Qupp = 85.
    Plow = 0.
    Pupp = 25.
    Qaxis = np.linspace(Qlow, Qupp, 100)
    Paxis = np.linspace(Plow, Pupp, 100)
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
                                compat='override', parallel=True)
        counts = np.zeros((10, len(Paxis), len(Qaxis)))
        for i, (start, end) in enumerate(lat_idxs):
            print('region', i)
#            Q = ds.Q[:,:,start:end,:] * 1e3 # kg/kg to g/kg
            Q = ds.TMQ.values[:,start:end,:]
            prec = ds.PRECT[:,start:end,:] * 1000 * 3600 # m/s to mm/h
            pres = (ds.P0*ds.hyam + ds.PS[:,start:end,:]*ds.hybm).transpose('time','lev',...)
#            Q700 = vintp(Q.values, pres.values, np.array([70000.])).squeeze()

            weight = np.cos(np.deg2rad(prec.lat))
            weight = np.expand_dims(weight, axis=(0, 2))
            weight = np.broadcast_to(weight, prec.shape)
            Qidx = np.round((Q - Qlow) / (Qupp - Qlow) * 100.).astype(int)
            Pidx = np.round((prec.values - Plow) / (Pupp - Plow) * 100.).astype(int)
            # Perform conditional sampling for each region
            P_temp = Pidx.flatten()
#            mask = P_temp > 0
            Q_temp = Qidx.flatten()
            w_temp = weight.flatten()
            counts[i,:,:] = cond_samp(Qaxis, Paxis, Q_temp, P_temp, w_temp)

        ds_out = xr.Dataset(
            data_vars=dict(
                counts=(['lat_reg', 'prec', 'TMQ'], counts)
            ),
            coords=dict(
                lat_reg=np.arange(10),
                prec=Paxis,
                TMQ=Qaxis
            ),
            attrs=dict(
                description='Counts of precipitation events for each region on the CWV-Prec plane'
            )
        )
        ds_out.TMQ.attrs['units'] = 'mm'
        ds_out.TMQ.attrs['long_name'] = 'column-integrated water vapor'
        ds_out.prec.attrs['units'] = 'mm/h'
        ds_out.prec.attrs['long_name'] = 'Precipitation rate'
        ds_out.lat_reg.attrs['units'] = 'region'
        ds_out.lat_reg.attrs['long_name'] = 'Latitude region by index 40,57,71,84,97,-84,-71,-57,-40'
        ds_out.counts.attrs['units'] = '#'
        ds_out.counts.attrs['long_name'] = 'Latitude weighted counts'
        ds_out.to_netcdf('/data/W.eddie/RCE/cond_samp.TMQ.prec.'+case+'.nc')
