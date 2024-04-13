import xarray as xr
import numpy as np
import numba
import glob
import sys
from npy_append_array import NpyAppendArray

@numba.njit
def sort_CWV(CWV):
    return CWV.argsort(kind='mergesort')

@numba.njit
def bin_CWV(cwv, mse, w, nrank):
    idx = sort_CWV(cwv)
    axis = np.rint(np.linspace(0, idx.max()+1, nrank+1)).astype(np.int_)
    cwv_mean = np.zeros((nrank))
    mse_mean = np.zeros((nrank, mse.shape[0]))
    w_mean = np.zeros((nrank, w.shape[0]))
    for i in range(nrank):
        i1 = axis[i]
        i2 = axis[i+1]
        cwv_mean[i] = cwv[idx[i1:i2]].mean()
        for k in range(mse.shape[0]):
            mse_mean[i,k] = mse[k,idx[i1:i2]].mean()
            w_mean[i,k] = w[k,idx[i1:i2]].mean()
    return cwv_mean, mse_mean.T, w_mean.T

def calc_sf(bin_W):
    sf = np.nancumsum(bin_W, axis=1)
    return sf

@numba.njit
def calc_rho(T, Q, pres):
    return pres / (287.0423113650487 * T * (1 + 0.608 * Q))

@numba.njit
def calc_massflux_C(w, rho):
    return rho * w

@numba.njit
def calc_MSE(t, q, z):
    return 1004.64*t + 2.501e6*q + 9.80616*z

@numba.njit
def calc_CWV(Q, dp):
    return (Q * dp).sum(axis=1) / 9.80616

def read_data(h0):
    h1 = h0.replace('cam.h0', 'cam.h1')
    print(h1)
    TCfile = h0.replace('cam.h0.', '')
    TCfile = TCfile[:-12]+'.TC_dist.nc'
    ds0 = xr.open_dataset(h0).sel(lon=0).transpose('lat','lev','ilev','time',...)
    ds1 = xr.open_dataset(h1).squeeze().isel(time=slice(0,None,2)).rename({'lat_90s_to_90n':'lat'}).transpose('lat','crm_nz','time','crm_nx',...)
    TCnc = xr.open_dataset(TCfile).sel(time=ds0.time.values, lon=0)
    tc = TCnc.TC.transpose('lat','time').values
    lev = ds0.lev[-1:1:-1]
    lev.attrs['positive'] = 'up'
    pres = (ds0.hyam[-1:1:-1] * ds0.P0 + ds0.hybm[-1:1:-1] * ds0.PS).transpose('lat','lev', ...).values
    presi = (ds0.hyai[-1:1:-1] * ds0.P0 + ds0.hybi[-1:1:-1] * ds0.PS).transpose('lat','ilev', ...).values
    dp = (presi[:,:-1] - presi[:,1:])
    T_G = ds0.T[:,-1:1:-1].values
    Q_G = ds0.Q[:,-1:1:-1].values
    Z3 = ds0.Z3[:,-1:1:-1].values

    Rho = calc_rho(T_G, Q_G, pres) # lat, lev, time

    print('CRM')
    new_ds = []
    prefixes = ['CRM_QV_', 'CRM_T_', 'CRM_W_']
    for prefix in prefixes:
        selected_vars = [var for var in ds1.data_vars if var.startswith(prefix)]
        new_ds.append(ds1[selected_vars[0]].squeeze(drop=True).rename(prefix[4]))
    print('done select variables, merge dataset')
    new_ds = xr.merge(new_ds, compat='override', combine_attrs='drop')
    print('load data...', end=',', flush=True)
    print('Q', end=',', flush=True)
    Q_C = new_ds.Q.values
    print('T', end=',', flush=True)
    T_C = new_ds.T.values
    print('W')
    w_C = new_ds.W.values
    print('calculate CWV')
    CWV_C = calc_CWV(Q_C, dp[:,:,:,None]) # lat, time, crm_nx
    print('calculate MSE')
    MSE_C = calc_MSE(T_C, Q_C, Z3[:,:,:,None]) # lat, lev, time, crm_nx
    print('calculate mass flux')
    W_C = calc_massflux_C(w_C, Rho[:,:,:,None]) # lat, lev, time, crm_nx
    print('reshape data')
    tc = np.expand_dims(tc, axis=2)
    tc = np.broadcast_to(tc, CWV_C.shape)
    tc = tc.reshape((tc.shape[0], -1))
    CWV_C = CWV_C.reshape((CWV_C.shape[0], -1))
    MSE_C = MSE_C.reshape((MSE_C.shape[0], MSE_C.shape[1], -1))
    W_C = W_C.reshape((W_C.shape[0], W_C.shape[1], -1))

    ds0.close()
    ds1.close()
    TCnc.close()
    return CWV_C, MSE_C, W_C, tc

def output(CWV, sf, W, MSE, ref):
    print('Output...')
    lat = ref.lat
    lev = ref.lev[-1:1:-1]
    lev.attrs['positive'] = 'up'
    rank = np.arange(1, 101)
    outds = xr.Dataset(
        data_vars=dict(
            CWV=(['lat', 'rank'], CWV),
            sf=(['lat', 'lev', 'rank'], sf),
            W=(['lat', 'lev', 'rank'], W),
            MSE=(['lat','lev', 'rank'], MSE),
        ),
        coords=dict(
            lat=lat,
            lev=lev,
            rank=(['rank'], rank)
        )
    )
    outds.CWV.attrs = dict(
        long_name='Column Water Vapor at the rank center',
        units='mm'
    )
    outds.sf.attrs = dict(
        long_name='Mass Streamfunction',
    )
    outds.W.attrs = dict(
        long_name='mass flux',
        units='kg m-2 s-1'
    )
    outds.MSE.attrs = dict(
        long_name='Moist static energy',
        units='J kg-1'
    )
    return outds

######
if __name__ == "__main__":
    cases = ['QSPSRCE_r10_8x4km_300K','QSPSRCE_r10_32x4km_300K','QSPSRCE_r10_256x4km_300K']
    nrank = 100
#    numba.set_num_threads(32) # set number of CPUs used by Numba
    for case in cases:
        global first
        first = True
        path = f'/data/W.eddie/SPCAM/{case}/atm/hist/'
        h0s = []
        for m in range(4, 13):
            h0s += glob.glob(f'{path}{case}.cam.h0.0001-{m:02d}-??-00000.nc')
        h0s = sorted(h0s)
        CWV_C_file = NpyAppendArray('CWV_C_temp', delete_if_exists=True)
        MSE_C_file = NpyAppendArray('MSE_C_temp', delete_if_exists=True)
        W_C_file = NpyAppendArray('W_C_temp', delete_if_exists=True)
        TC_file = NpyAppendArray('TC_temp', delete_if_exists=True)
        for h0 in h0s:
            CWV_C_temp, MSE_C_temp, W_C_temp, tc = read_data(h0)
            print('write temp files')
            CWV_C_app = CWV_C_temp.transpose((1,0)).copy()
            MSE_C_app = MSE_C_temp.transpose((2,1,0)).copy()
            W_C_app = W_C_temp.transpose((2,1,0)).copy()
            TC_app = tc.transpose((1,0)).copy()
            CWV_C_file.append(CWV_C_app)
            MSE_C_file.append(MSE_C_app)
            W_C_file.append(W_C_app)
            TC_file.append(TC_app)
        CWV_C_file.close()
        MSE_C_file.close()
        W_C_file.close()
        TC_file.close()
        print('read temp files')
        CWV_C = np.load('CWV_C_temp', mmap_mode="r").transpose((1,0))
        MSE_C = np.load('MSE_C_temp', mmap_mode="r").transpose((2,1,0))
        W_C = np.load('W_C_temp', mmap_mode="r").transpose((2,1,0))
        tc = np.load('TC_temp', mmap_mode="r").transpose((1,0))
        print('binning...')
        CWV_TC_MEAN = np.zeros((CWV_C.shape[0], nrank))
        MSE_TC_MEAN = np.zeros((MSE_C.shape[0], MSE_C.shape[1], nrank))
        W_TC_MEAN = np.zeros((W_C.shape[0], W_C.shape[1], nrank))
        sf_TC = np.zeros((W_C.shape[0], W_C.shape[1], nrank))
        CWV_noTC_MEAN = np.zeros((CWV_C.shape[0], nrank))
        MSE_noTC_MEAN = np.zeros((MSE_C.shape[0], MSE_C.shape[1], nrank))
        W_noTC_MEAN = np.zeros((W_C.shape[0], W_C.shape[1], nrank))
        sf_noTC = np.zeros((W_C.shape[0], W_C.shape[1], nrank))
        for j in range(CWV_C.shape[0]):
            CWV = CWV_C[j][tc[j]==1].copy()
            if CWV.size < 100:
                CWV_TC_MEAN[j], MSE_TC_MEAN[j], W_TC_MEAN[j] = np.nan, np.nan, np.nan
                sf_TC[j] = np.nan
            else:
                MSE = MSE_C[j][:,tc[j]==1].copy()
                W = W_C[j][:,tc[j]==1].copy()
                CWV_TC_MEAN[j], MSE_TC_MEAN[j], W_TC_MEAN[j] = bin_CWV(CWV, MSE, W, nrank)
                sf_TC[j] = calc_sf(W_TC_MEAN[j])

            CWV = CWV_C[j][tc[j]==0].copy()
            if CWV.size < 100:
                CWV_noTC_MEAN[j], MSE_noTC_MEAN[j], W_noTC_MEAN[j] = np.nan, np.nan, np.nan
                sf_noTC[j] = np.nan
            else:
                MSE = MSE_C[j][:,tc[j]==0].copy()
                W = W_C[j][:,tc[j]==0].copy()
                CWV_noTC_MEAN[j], MSE_noTC_MEAN[j], W_noTC_MEAN[j] = bin_CWV(CWV, MSE, W, nrank)
                sf_noTC[j] = calc_sf(W_noTC_MEAN[j])
        ref = xr.open_mfdataset(h0s[-1])
        ds = output(CWV_TC_MEAN, sf_TC, W_TC_MEAN, MSE_TC_MEAN, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_CRM_rank_TC_lon0.nc')
        ds.close()
        ds = output(CWV_noTC_MEAN, sf_noTC, W_noTC_MEAN, MSE_noTC_MEAN, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_CRM_rank_noTC_lon0.nc')
        ds.close()