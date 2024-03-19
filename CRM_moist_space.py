import xarray as xr
import numpy as np
import numba
import glob
import pandas as pd
import sys
from time import time

@numba.njit(parallel=True)
def bin_CWV(idx, w, mse, weight):
    out = np.zeros((200, w.shape[1]))
    out_h = out
    counts = np.zeros((200))
    for i in numba.prange(len(idx)):
        j = idx[i]
        out[j,:] = out[j,:] + w[i,:] * weight[i]
        out_h[j,:] = out_h[j,:] + mse[i,:] * weight[i]
        counts[j] = counts[j] + weight[i]
    return out, out_h, counts

def calc_MSE(t, q, z):
    return 1004.64*t + 2.501e6*q + 9.80616*z

def main(case, lat1, lat2):
    print(case, lat1, lat2)
    CWV_axis = np.arange(0, 100, 0.5)
    path = f'/data/W.eddie/SPCAM/{case}/atm/hist/'
    ncfiles = []
    for m in range(4, 13):
        ncfiles += glob.glob(f'{path}{case}.cam.h?.0001-{m:02d}-??-00000.nc')
    ncfiles = sorted(ncfiles)
    prefixes = ['CRM_QV_', 'CRM_T_', 'CRM_W_']
    with xr.open_mfdataset(ncfiles, combine='by_coords', parallel=True,
            data_vars='minimal', coords='minimal', compat='override').sel(
            lon=[0, 90, 180, 270]).isel(
            time=slice(0,None,2), lat=slice(lat1,lat2), lat_90s_to_90n=slice(lat1,lat2), crm_nx=list(range(8))
            ) as ds:
        print(ds)
        pres = (ds.hyam[-1:1:-1] * ds.P0 + ds.hybm[-1:1:-1] * ds.PS).transpose('time', 'lev', ...)
        presi = (ds.hyai[-1:1:-1] * ds.P0 + ds.hybi[-1:1:-1] * ds.PS).transpose('time', 'ilev', ...)
        dp = presi.data[:,:-1,:,:] - presi.data[:,1:,:,:]
        dp_G = xr.DataArray(data=dp, coords=pres.coords, dims=pres.dims, name='dp')
        dp_C = xr.DataArray(
                data=dp,
                coords=dict(time=ds.time, crm_nz=ds.crm_nz, lat=ds.lat, lon=ds.lon),
                dims=['time', 'crm_nz', 'lat', 'lon'], name='dp'
            )
        T_G = ds.T[:,-1:1:-1]
        Q_G = ds.Q[:,-1:1:-1]
        OMEGA = ds.OMEGA[:,-1:1:-1]
        Z3 = ds.Z3[:,-1:1:-1]
        weight = np.cos(np.deg2rad(ds.lat.values))
        weight_G = np.expand_dims(weight, axis=(0, 2))
        weight_G = np.broadcast_to(weight_G, ds.PS.shape)
        new = []
        for prefix in prefixes:
            selected_vars = [var for var in ds.data_vars if var.startswith(prefix)]
            temp = []
            for var in selected_vars:
                temp.append(ds[var].squeeze())
            new.append(xr.concat(temp, pd.Index([0, 90, 180, 270], name='lon')).rename(prefix[4]))

    print('Calculating on GCM scale...')
    CWV_G = (Q_G * dp_G).sum(dim='lev') / 9.80616
    idx_G = (CWV_G/0.5).round().astype(int) # time, lat, lon
    OMEGA = OMEGA.transpose('time', 'lat', 'lon', 'lev')
    idx_G = idx_G.values.flatten()
    W_G = OMEGA.values.reshape((-1, OMEGA.shape[-1])) / -9.80616
    MSE_G = calc_MSE(T_G, Q_G, Z3)
    MSE_G = MSE_G.transpose('time', 'lat', 'lon', 'lev')
    MSE_G = MSE_G.values.reshape((-1, MSE_G.shape[-1]))
    weight_G = weight_G.flatten()
    bin_W_G, bin_MSE_G, counts_W_G = bin_CWV(idx_G, W_G, MSE_G, weight_G)
    counts_W_G[counts_W_G<=0.] = np.nan
    bin_W_G = bin_W_G / counts_W_G[:,None]
    bin_MSE_G = bin_MSE_G / counts_W_G[:,None]
    sf_G = np.nancumsum(bin_W_G, axis=0)
    sf_G[np.isnan(counts_W_G),:] = np.nan
    Tv = T_G * (1 + 0.61 * Q_G)
    Rd = 287.0423113650487
    Rho = (pres / (Rd * Tv))

    print('Calculating on CRM scale...')
    new = xr.merge(
            new, compat='override', combine_attrs='drop'
        ).rename({'lat_90s_to_90n':'lat'})
    CWV_C = (
            new.Q.transpose('time','crm_nz','lat','lon','crm_nx') * dp_C
        ).sum(dim='crm_nz') / 9.80616
    idx_C = (CWV_C/0.5).round().astype(int) # time, lat, lon, crm_nx
    weight_C = np.expand_dims(weight, axis=(0, 2, 3))
    weight_C = np.broadcast_to(weight_C, idx_C.shape)
    weight_C = weight_C.flatten()
    idx_C = idx_C.values.flatten()
    W = new.W.transpose('time', 'lat', 'lon', 'crm_nx', 'crm_nz')
    Rho = Rho.rename({'lev':'crm_nz'}).transpose('time', 'lat', 'lon', 'crm_nz')
    W_C = (Rho * W).values.reshape((-1, W.shape[-1]))
    print(W_C.shape)
    Z = Z3.rename({'lev':'crm_nz'}).transpose('lon', 'time', 'crm_nz', 'lat')
    MSE_C = calc_MSE(new.T, new.Q, Z)
    MSE_C = MSE_C.transpose('time', 'lat', 'lon', 'crm_nx', 'crm_nz')
    MSE_C = MSE_C.values.reshape((-1, MSE_C.shape[-1]))
    bin_W_C, bin_MSE_C, counts_W_C = bin_CWV(idx_C, W_C, MSE_C, weight_C)
    counts_W_C[counts_W_C<=0.] = np.nan
    bin_W_C = bin_W_C / counts_W_C[:,None]
    bin_MSE_C = bin_MSE_C / counts_W_C[:,None]
    sf_C = np.nancumsum(bin_W_C, axis=0)
    sf_C[np.isnan(counts_W_C),:] = np.nan

    print('Output...')
    outds = xr.Dataset(
        data_vars=dict(
            sf_G=(['lev', 'CWV'], sf_G.transpose()),
            sf_C=(['lev', 'CWV'], sf_C.transpose()),
            bin_W_G=(['lev', 'CWV'], bin_W_G.transpose()),
            bin_W_C=(['lev', 'CWV'], bin_W_C.transpose()),
            bin_MSE_G=(['lev', 'CWV'], bin_MSE_G.transpose()),
            bin_MSE_C=(['lev', 'CWV'], bin_MSE_C.transpose()),
            counts_W_G=(['CWV'], counts_W_G),
            counts_W_C=(['CWV'], counts_W_C)
        ),
        coords=dict(
            lev=OMEGA.lev,
            CWV=(['CWV'], CWV_axis)
        ),
        attrs=dict(
            case=case,
            lat_idxs=f'[{lat1}:{lat2}]'
        )
    )
    outds.CWV.attrs = dict(
        long_name='Column Water Vapor',
        units='mm'
    )
    outds.sf_G.attrs = dict(
        long_name='Streamfunction on GCM scale',
    )
    outds.sf_C.attrs = dict(
        long_name='Streamfunction on CRM scale',
    )
    outds.bin_W_G.attrs = dict(
        long_name='Weighted mass flux on GCM scale (-omega/g)',
        units='kg m-2 s-1'
    )
    outds.bin_W_C.attrs = dict(
        long_name='Weighted mass flux on CRM scale (rho*w)',
        units='kg m-2 s-1'
    )
    outds.bin_MSE_G.attrs = dict(
        long_name='Weighted moist static energy on GCM scale',
        units='J kg-1'
    )
    outds.bin_MSE_C.attrs = dict(
        long_name='Weighted moist static energy on CRM scale',
        units='J kg-1'
    )
    outds.counts_W_G.attrs = dict(
        long_name='Number of samples on GCM scale',
    )
    outds.counts_W_C.attrs = dict(
        long_name='Number of samples on CRM scale',
    )
    outds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_{lat1}_{lat2}.nc')
    return

######
if __name__ == "__main__":
    cases = ['QSPSRCE_r10_8x4km_300K', 'QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    # Define the lat index ranges for the 10 regions
    lat_idxs = [[0, 40], [40, 57], [57, 71], [71, 84], [84, 97], [97, -84], [-84, -71], [-71, -57], [-57, -40], [-40, None]]
    numba.set_num_threads(32) # set number of CPUs used by Numba
    for case in cases:
        for i, (lat1, lat2) in enumerate(lat_idxs):
            main(case, lat1, lat2)
