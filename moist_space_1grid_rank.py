import xarray as xr
import numpy as np
import numba
import glob
import sys

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

@numba.njit
def calc_GCM(T_G, Q_G, OMEGA, Z3, dp_G):
    CWV_G = (Q_G * dp_G).sum(axis=1) / 9.80616
    idx_G = np.round_(CWV_G/0.5).astype(np.int_) # time
    OMEGA = OMEGA.transpose((1,0))
    W_G = OMEGA / -9.80616 # rho*w
    MSE_G = calc_MSE(T_G, Q_G, Z3)
    MSE_G = MSE_G.transpose((1,0))
    return idx_G, W_G, MSE_G

@numba.njit
def calc_CRM(T_C, Q_C, W_C, dp_C, Z_C, Rho):
    CWV_C = (Q_C * dp_C).sum(axis=0) / 9.80616 #calc_CWV(Q_C, dp_C)
    idx_C = np.round_(CWV_C/0.5).astype(np.int_) # time, crm_nx
    W_C = (Rho * W_C).reshape((W_C.shape[0], -1))
    MSE_C = calc_MSE(T_C, Q_C, Z_C)
    return idx_C, W_C, MSE_C

def calc_sf(bin_W):
    sf = np.nancumsum(bin_W, axis=0)
    return sf

@numba.njit
def calc_rho(T, Q, pres):
    return pres / (287.0423113650487 * T * (1 + 0.608 * Q))

@numba.njit
def calc_massflux_C(w, rho):
    return rho * w

@numba.njit
def calc_massflux_G(omega):
    return omega / -9.80616

@numba.njit
def calc_MSE(t, q, z):
    return 1004.64*t + 2.501e6*q + 9.80616*z

@numba.njit
def calc_CWV(Q, dp):
    return (Q * dp).sum(axis=1) / 9.80616

def read_data(h0):
    h1 = h0.replace('cam.h0', 'cam.h1')
    print(h1)
    ds0 = xr.open_dataset(h0).sel(lon=0).transpose('lat','lev','ilev','time',...)
    ds1 = xr.open_dataset(h1).squeeze().isel(time=slice(0,None,2)).rename({'lat_90s_to_90n':'lat'}).transpose('lat','crm_nz','time','crm_nx',...)
    lev = ds0.lev[-1:1:-1]
    lev.attrs['positive'] = 'up'
    pres = (ds0.hyam[-1:1:-1] * ds0.P0 + ds0.hybm[-1:1:-1] * ds0.PS).transpose('lat','lev', ...).values
    presi = (ds0.hyai[-1:1:-1] * ds0.P0 + ds0.hybi[-1:1:-1] * ds0.PS).transpose('lat','ilev', ...).values
    dp = (presi[:,:-1] - presi[:,1:])
    T_G = ds0.T[:,-1:1:-1].values
    Q_G = ds0.Q[:,-1:1:-1].values
    OMEGA = ds0.OMEGA[:,-1:1:-1].values
    Z3 = ds0.Z3[:,-1:1:-1].values
    lat = ds0.lat.values
    weight = np.cos(np.deg2rad(lat)) # lat
    CWV_G = calc_CWV(Q_G, dp) # lat, time
    MSE_G = calc_MSE(T_G, Q_G, Z3) # lat, lev, time
    W_G = calc_massflux_G(OMEGA) # lat, lev, time

    Rho = calc_rho(T_G, Q_G, pres) # lat, lev, time

    new_ds = []
    prefixes = ['CRM_QV_', 'CRM_T_', 'CRM_W_']
    for prefix in prefixes:
        selected_vars = [var for var in ds1.data_vars if var.startswith(prefix)]
        new_ds.append(ds1[selected_vars[0]].squeeze(drop=True).rename(prefix[4]))
    new_ds = xr.merge(new_ds, compat='override', combine_attrs='drop')
    Q_C = new_ds.Q.values
    T_C = new_ds.T.values
    w_C = new_ds.W.values
    CWV_C = calc_CWV(Q_C, dp[:,:,:,None]) # lat, time, crm_nx
    MSE_C = calc_MSE(T_C, Q_C, Z3[:,:,:,None]) # lat, lev, time, crm_nx
    W_C = calc_massflux_C(w_C, Rho[:,:,:,None]) # lat, lev, time, crm_nx
    CWV_C = CWV_C.reshape((CWV_C.shape[0], -1))
    MSE_C = MSE_C.reshape((MSE_C.shape[0], MSE_C.shape[1], -1))
    W_C = W_C.reshape((W_C.shape[0], W_C.shape[1], -1))

    ds0.close()
    ds1.close()
    return CWV_G, MSE_G, W_G, CWV_C, MSE_C, W_C

@numba.njit
def sort_CWV(CWV):
    return CWV.argsort(kind='mergesort')

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
    cases = ['QSPSRCE_r10_8x4km_300K', 'QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    nrank = 100
#    numba.set_num_threads(32) # set number of CPUs used by Numba
    for case in cases:
        global first
        first = True
        path = f'/data/W.eddie/SPCAM/{case}/atm/hist/'
        h0s = []
        h1s = []
        for m in range(4, 13):
            h0s += glob.glob(f'{path}{case}.cam.h0.0001-{m:02d}-??-00000.nc')
        h0s = sorted(h0s)
        for h0 in h0s:
            CWV_G_temp, MSE_G_temp, W_G_temp, CWV_C_temp, MSE_C_temp, W_C_temp = read_data(h0)
            if first:
                CWV_G = CWV_G_temp
                MSE_G = MSE_G_temp
                W_G = W_G_temp
                CWV_C = CWV_C_temp
                MSE_C = MSE_C_temp
                W_C = W_C_temp
                first = False
            else:
                CWV_G = np.concatenate((CWV_G, CWV_G_temp), axis=1)
                MSE_G = np.concatenate((MSE_G, MSE_G_temp), axis=2)
                W_G = np.concatenate((W_G, W_G_temp), axis=2)
                CWV_C = np.concatenate((CWV_C, CWV_C_temp), axis=1)
                MSE_C = np.concatenate((MSE_C, MSE_C_temp), axis=2)
                W_C = np.concatenate((W_C, W_C_temp), axis=2)
        CWV_G_MEAN = np.zeros((CWV_G.shape[0], nrank))
        MSE_G_MEAN = np.zeros((MSE_G.shape[0], MSE_G.shape[1], nrank))
        W_G_MEAN = np.zeros((W_G.shape[0], W_G.shape[1], nrank))
        CWV_C_MEAN = np.zeros((CWV_C.shape[0], nrank))
        MSE_C_MEAN = np.zeros((MSE_C.shape[0], MSE_C.shape[1], nrank))
        W_C_MEAN = np.zeros((W_C.shape[0], W_C.shape[1], nrank))
        for j in range(CWV_G.shape[0]):
            CWV_G_MEAN[j], MSE_G_MEAN[j], W_G_MEAN[j] = bin_CWV(CWV_G[j], MSE_G[j], W_G[j], nrank)
            CWV_C_MEAN[j], MSE_C_MEAN[j], W_C_MEAN[j] = bin_CWV(CWV_C[j], MSE_C[j], W_C[j], nrank)
        sf_G = calc_sf(W_G_MEAN)
        sf_C = calc_sf(W_C_MEAN)
        ref = xr.open_mfdataset(h0s[-1])
        ds = output(CWV_G_MEAN, sf_G, W_G_MEAN, MSE_G_MEAN, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_GCM_rank_lon0.nc')
        ds = output(CWV_C_MEAN, sf_C, W_C_MEAN, MSE_C_MEAN, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_CRM_rank_lon0.nc')