import xarray as xr
import numpy as np
import numba
import glob
import sys

@numba.njit(parallel=True)
def bin_CWV(idx, w, mse, weight, out, out_h, counts):
    s = w.shape # lev or crm_nz, cwv_axis
    for i in numba.prange(s[1]):
        j = idx[i]
        out[:,j] = out[:,j] + w[:,i] * weight[i]
        out_h[:,j] = out_h[:,j] + mse[:,i] * weight[i]
        counts[j] = counts[j] + weight[i]
    return out, out_h, counts

def calc_MSE(t, q, z):
    return 1004.64*t + 2.501e6*q + 9.80616*z

def calc_sf(bin_W, bin_MSE, counts):
    bin_W = bin_W / counts[:,None,:]
    bin_MSE = bin_MSE / counts[:,None,:]
    sf = np.nancumsum(bin_W, axis=2)
    counts = np.expand_dims(counts, axis=1)
    counts = np.broadcast_to(counts, sf.shape)
    sf[counts<=0.] = np.nan
    return sf, bin_W, bin_MSE

def main(ncfiles):
    CWV_axis = np.arange(0, 100, 0.5)
    print('open dataset and pre-processing')
    print(ncfiles)
    with xr.open_mfdataset(ncfiles, combine='by_coords',
            data_vars='minimal', coords='minimal', compat='override').sel(
            lon=0).isel(time=slice(0,None,2)) as ds:
        ds.lev.attrs['positive'] = 'up'
        pres = (ds.hyam[-1:1:-1] * ds.P0 + ds.hybm[-1:1:-1] * ds.PS).transpose('time', 'lev', ...)
        presi = (ds.hyai[-1:1:-1] * ds.P0 + ds.hybi[-1:1:-1] * ds.PS).transpose('time', 'ilev', ...)
        dp = (presi.data[:,:-1,:,] - presi.data[:,1:,:,]).compute()
        T_G_all = ds.T[:,-1:1:-1].load()
        Q_G_all = ds.Q[:,-1:1:-1].load()
        OMEGA_all = ds.OMEGA[:,-1:1:-1].load()
        Z3_all = ds.Z3[:,-1:1:-1].load()
        lat = ds.lat.values
        weight = np.cos(np.deg2rad(lat))
        weight_G_all = np.expand_dims(weight, axis=0)
        weight_G_all = np.broadcast_to(weight_G_all, ds.PS.shape)

        new_ds = []
        prefixes = ['CRM_QV_', 'CRM_T_', 'CRM_W_']
        for prefix in prefixes:
            selected_vars = [var for var in ds.data_vars if var.startswith(prefix)]
            temp = []
            for var in selected_vars:
                temp.append(ds[var].squeeze())
            new_ds.append(temp[0].rename(prefix[4]))
        new_ds = xr.merge(
                new_ds, compat='override', combine_attrs='drop'
            ).rename({'lat_90s_to_90n':'lat'}).transpose('time','crm_nz','lat','crm_nx')
        weight_C_all = np.expand_dims(weight, axis=(0, 2))
        weight_C_all = np.broadcast_to(weight_C_all,
                (len(new_ds.time), len(new_ds.lat), len(new_ds.crm_nx))
            )

        dp_G_all = xr.DataArray(data=dp, coords=pres.coords, dims=pres.dims, name='dp')
        dp_C_all = xr.DataArray(
                data=dp,
                coords=dict(time=ds.time, crm_nz=ds.crm_nz, lat=ds.lat),
                dims=['time', 'crm_nz', 'lat'], name='dp'
            )
    global first
    global bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C
    if first:
        print('Creating output arrays...') # lat, lev, CWV
        bin_W_G = np.zeros((len(lat), len(OMEGA_all.lev), len(CWV_axis)))
        bin_W_C = np.zeros((len(lat), len(new_ds.crm_nz), len(CWV_axis)))
        bin_MSE_G = np.zeros((len(lat), len(Q_G_all.lev), len(CWV_axis)))
        bin_MSE_C = np.zeros((len(lat), len(new_ds.crm_nz), len(CWV_axis)))
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
        dp_G = dp_G_all[:,:,j]
        dp_C = dp_C_all[:,:,j]
        weight_G = weight_G_all[:,j]
        weight_C = weight_C_all[:,j]
        new = new_ds.isel(lat=j).squeeze().transpose('time','crm_nz','crm_nx')
        print('Calculating on GCM scale...')
        CWV_G = (Q_G * dp_G).sum(dim='lev') / 9.80616
        idx_G = (CWV_G/0.5).round().astype(int) # time, lat, lon
        OMEGA = OMEGA.transpose('lev',...)
        idx_G = idx_G.values.flatten()
        W_G = OMEGA.values.reshape((OMEGA.shape[0], -1)) / -9.80616
        MSE_G = calc_MSE(T_G, Q_G, Z3)
        MSE_G = MSE_G.transpose('lev',...)
        MSE_G = MSE_G.values.reshape((MSE_G.shape[0], -1))
        weight_G = weight_G.flatten()
        bin_W_G[j], bin_MSE_G[j], counts_G[j] = bin_CWV(
                idx_G, W_G, MSE_G, weight_G,
                bin_W_G[j], bin_MSE_G[j], counts_G[j]
            )

        Tv = T_G * (1 + 0.61 * Q_G)
        Rd = 287.0423113650487
        Rho = (pres_G / (Rd * Tv))
        print('Calculating on CRM scale...')
        CWV_C = (
                new.Q * dp_C
            ).sum(dim='crm_nz') / 9.80616
        idx_C = (CWV_C/0.5).round().astype(int) # time, lat, crm_nx
        idx_C = idx_C.values.flatten()
        weight_C = weight_C.flatten()
        W = new.W.transpose('crm_nz',...)
        Rho = Rho.rename({'lev':'crm_nz'}).transpose('crm_nz',...)
        W_C = (Rho * W).values.reshape((W.shape[0], -1))
        Z = Z3.rename({'lev':'crm_nz'}).transpose('crm_nz',...)
        MSE_C = calc_MSE(new.T, new.Q, Z)
        MSE_C = MSE_C.transpose('crm_nz',...)
        MSE_C = MSE_C.values.reshape((MSE_C.shape[0], -1))
        bin_W_C[j], bin_MSE_C[j], counts_C[j] = bin_CWV(
                idx_C, W_C, MSE_C, weight_C,
                bin_W_C[j], bin_MSE_C[j], counts_C[j]
            )

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
    numba.set_num_threads(32) # set number of CPUs used by Numba
    for case in cases:
        global first
        first = True
        path = f'/data/W.eddie/SPCAM/{case}/atm/hist/'
        h0s = []
        h1s = []
        for m in range(4, 13):
            h0s += glob.glob(f'{path}{case}.cam.h0.0001-{m:02d}-??-00000.nc')
            h1s += glob.glob(f'{path}{case}.cam.h1.0001-{m:02d}-??-00000.nc')
        h0s = sorted(h0s)
        h1s = sorted(h1s)
        for ncfiles in zip(h0s, h1s):
            bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C = main(ncfiles)
        sf_G, bin_W_G, bin_MSE_G = calc_sf(bin_W_G, bin_MSE_G, counts_G)
        sf_C, bin_W_C, bin_MSE_C = calc_sf(bin_W_C, bin_MSE_C, counts_C)
        ref = xr.open_mfdataset(h0s[-1])
        ds = output(sf_G, sf_C, bin_W_G, bin_W_C, bin_MSE_G, bin_MSE_C, counts_G, counts_C, ref)
        ds.to_netcdf(f'/data/W.eddie/RCE/{case}/moisture_space_lon0.nc')
