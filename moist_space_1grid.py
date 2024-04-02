import xarray as xr
import numpy as np
import numba
import glob
import sys

@numba.njit
def bin_CWV(idx, w, mse, weight, out, out_h, counts):
    s = w.shape # lev, time
    for i in range(s[1]):
        j = idx[i]
        out[:,j] = out[:,j] + w[:,i] * weight[i]
        out_h[:,j] = out_h[:,j] + mse[:,i] * weight[i]
        counts[j] = counts[j] + weight[i]
    return out, out_h, counts

@numba.njit
def calc_MSE(t, q, z):
    return 1004.64*t + 2.501e6*q + 9.80616*z

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

def calc_sf(bin_W, bin_MSE, counts):
    bin_W = bin_W / counts[:,None,:]
    bin_MSE = bin_MSE / counts[:,None,:]
    sf = np.nancumsum(bin_W, axis=2)
    counts = np.expand_dims(counts, axis=1)
    counts = np.broadcast_to(counts, sf.shape)
    sf[counts<=0.] = np.nan
    return sf, bin_W, bin_MSE

def main(h0):
    CWV_axis = np.arange(0, 100, 0.5)
    print('open dataset and pre-processing')
    h1 = h0.replace('cam.h0', 'cam.h1')
    print(h1)
    ds0 = xr.open_dataset(h0).sel(lon=0)
    ds1 = xr.open_dataset(h1).isel(time=slice(0,None,2))
    lev = ds0.lev[-1:1:-1]
    lev.attrs['positive'] = 'up'
    pres = (ds0.hyam[-1:1:-1] * ds0.P0 + ds0.hybm[-1:1:-1] * ds0.PS).transpose('time', 'lev', ...).values
    presi = (ds0.hyai[-1:1:-1] * ds0.P0 + ds0.hybi[-1:1:-1] * ds0.PS).transpose('time', 'ilev', ...).values
    dp = (presi[:,:-1,:,] - presi[:,1:,:,])
    T_G_all = ds0.T[:,-1:1:-1].values
    Q_G_all = ds0.Q[:,-1:1:-1].values
    OMEGA_all = ds0.OMEGA[:,-1:1:-1].values
    Z3_all = ds0.Z3[:,-1:1:-1].values
    lat = ds0.lat.values
    weight = np.cos(np.deg2rad(lat))
    weight_G_all = np.expand_dims(weight, axis=0)
    weight_G_all = np.broadcast_to(weight_G_all, ds0.PS.shape) # time, lat

    new_ds = []
    prefixes = ['CRM_QV_', 'CRM_T_', 'CRM_W_']
    for prefix in prefixes:
        selected_vars = [var for var in ds1.data_vars if var.startswith(prefix)]
        new_ds.append(ds1[selected_vars[0]].squeeze(drop=True).rename(prefix[4]))
    new_ds = xr.merge(
            new_ds, compat='override', combine_attrs='drop'
        ).rename({'lat_90s_to_90n':'lat'}).transpose('time','crm_nz','lat','crm_nx')
    weight_C_all = np.expand_dims(weight, axis=(0, 2))
    weight_C_all = np.broadcast_to(weight_C_all, new_ds.T[:,0,:,:].shape)

    global first
    global bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C
    if first:
        print('Creating output arrays...') # lat, lev, CWV
        bin_W_G = np.zeros((len(lat), len(lev), len(CWV_axis)))
        bin_W_C = np.zeros((len(lat), len(lev), len(CWV_axis)))
        bin_MSE_G = np.zeros((len(lat), len(lev), len(CWV_axis)))
        bin_MSE_C = np.zeros((len(lat), len(lev), len(CWV_axis)))
        counts_G = np.zeros((len(lat), len(CWV_axis)))
        counts_C = np.zeros((len(lat), len(CWV_axis)))
        first = False

    for j in range(len(lat)):
        print('lat', lat[j])
        T_G = T_G_all[:,:,j]
        Q_G = Q_G_all[:,:,j]
        OMEGA = OMEGA_all[:,:,j]
        Z3 = Z3_all[:,:,j]
        pres_G = pres[:,:,j]
        dp_G = dp[:,:,j]
        weight_G = weight_G_all[:,j]

        print('Calculating on GCM scale...')
        temp_W = np.copy(bin_W_G[j,:,:])
        temp_MSE = np.copy(bin_MSE_G[j,:,:])
        temp_counts = np.copy(counts_G[j,:])
        idx_G, W_G, MSE_G = calc_GCM(T_G, Q_G, OMEGA, Z3, dp_G)
        temp_W, temp_MSE, temp_counts = bin_CWV(
                idx_G, W_G, MSE_G, weight_G,
                temp_W, temp_MSE, temp_counts
            )
        bin_W_G[j,:,:] = temp_W
        bin_MSE_G[j,:,:] = temp_MSE
        counts_G[j,:] = temp_counts

        Tv = T_G * (1 + 0.608 * Q_G)
        Rd = 287.0423113650487
        Rho = (pres_G / (Rd * Tv))
        print('Calculating on CRM scale...')
        new = new_ds.isel(lat=j).squeeze().transpose('crm_nz','time','crm_nx')
        T_C = new.T.values.reshape((new.T.shape[0], -1))
        Q_C = new.Q.values.reshape((new.Q.shape[0], -1))
        W_C = new.W.values.reshape((new.W.shape[0], -1))
        dp_C = dp[:,:,j].transpose((1,0))
        dp_C = np.expand_dims(dp_C, axis=2)
        dp_C = np.broadcast_to(dp_C, new.T.shape)
        dp_C = dp_C.reshape((dp_C.shape[0], -1))
        weight_C = weight_C_all[:,j,:]
        weight_C = weight_C.flatten()
        Z = Z3.transpose((1,0))
        Z_C = np.expand_dims(Z, axis=2)
        Z_C = np.broadcast_to(Z_C, new.T.shape)
        Z_C = Z_C.reshape((Z_C.shape[0], -1))
        temp_W = np.copy(bin_W_C[j,:,:])
        temp_MSE = np.copy(bin_MSE_C[j,:,:])
        temp_counts = np.copy(counts_C[j,:])
        Rho_C = np.broadcast_to(np.expand_dims(Rho.transpose((1,0)), axis=2), new.T.shape)
        Rho_C = Rho_C.reshape((Rho_C.shape[0], -1))
        idx_C, W_C, MSE_C = calc_CRM(T_C, Q_C, W_C, dp_C, Z_C, Rho_C)
        temp_W, temp_MSE, temp_counts = bin_CWV(
                idx_C, W_C, MSE_C, weight_C,
                temp_W, temp_MSE, temp_counts
            )
        bin_W_C[j,:,:] = temp_W
        bin_MSE_C[j,:,:] = temp_MSE
        counts_C[j,:] = temp_counts

    return bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C

def output(sf_G, sf_C, bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C, ref):
    print('Output...')
    lat = ref.lat
    lev = ref.lev[-1:1:-1]
    lev.attrs['positive'] = 'up'
    outds = xr.Dataset(
        data_vars=dict(
            sf_G=(['lat', 'lev', 'CWV'], sf_G),
            sf_C=(['lat', 'lev', 'CWV'], sf_C),
            bin_W_G=(['lat', 'lev', 'CWV'], bin_W_G),
            bin_W_C=(['lat','lev', 'CWV'], bin_W_C),
            bin_MSE_G=(['lat','lev', 'CWV'], bin_MSE_G),
            bin_MSE_C=(['lat','lev', 'CWV'], bin_MSE_C),
            counts_W_G=(['lat','CWV'], counts_G),
            counts_W_C=(['lat','CWV'], counts_C)
        ),
        coords=dict(
            lat=lat,
            lev=lev,
            CWV=(['CWV'], np.arange(0, 100, 0.5))
        ),
        attrs=dict(
            case=case
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
    return outds

######
if __name__ == "__main__":
    cases = ['QSPSRCE_r10_8x4km_300K', 'QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    # Define the lat index ranges for the 10 regions
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
            bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C = main(h0)
        sf_G, bin_W_G, bin_MSE_G = calc_sf(bin_W_G, bin_MSE_G, counts_G)
        sf_C, bin_W_C, bin_MSE_C = calc_sf(bin_W_C, bin_MSE_C, counts_C)
        ref = xr.open_mfdataset(h0s[-1])
        ds = output(sf_G, sf_C, bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_lon0.nc')