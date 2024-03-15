import xarray as xr
import numpy as np
import numba

@numba.njit
def vint(vdp):
    return vdp.sum()

@numba.njit(parallel=True)
def streamfuction(v, dp, lat):
    pi = 3.14159265358979323846
    r = 6.37122e6
    g = 9.80616
    vdp = (v * dp).transpose(1,0)
    s = vdp.shape #[time, lat, lon, lev]
    w = 2 * pi * r * np.cos(np.deg2rad(lat))
    sf = np.empty(s)
    for j in numba.prange(s[0]):
        for k in numba.prange(s[1]-1):
            sf[j,k] = w[j] * vint(vdp[j,k:]) * -1
        sf[j,s[1]] = w[j] * vdp[j,s[1]] * -1
    return sf.transpose(1,0)

def preprocess(a, b, p0, ps):
    presi = a*p0 + b*ps
    presi = presi.data
    dp = presi[1:,:] - presi[:-1,:]
    return dp.compute()

def pre(ds):
    return ds[['hyam','hybm','hyai','hybi','P0']]

def main(case):
    print(case, 'open dataset')
    path = '/data/W.eddie/SPCAM/'+case+'/atm/hist/'
    with xr.open_mfdataset(path+case+'.timzonmean.nc') as zonmean:
        V = zonmean.V.squeeze()
        PS = zonmean.PS.squeeze()
        lat = zonmean.lat.values
    with xr.open_mfdataset(path+case+'.cam.h0.0001-01-01-00000.nc',
            preprocess=pre, decode_cf=False) as h0:
        hyam = h0.hyam
        hybm = h0.hybm
        a = h0.hyai
        b = h0.hybi
        p0 = h0.P0
    print('dp')
    dp = preprocess(a, b, p0, PS)
    print('streamfunction')
    sf = streamfuction(V.values, dp, lat)
    sf_xr = xr.DataArray(
            data=sf,
            name='sf',
            dims=['lev','lat'],
            attrs=dict(
                long_name='meridional stream function',
                units='kg/s'
            )
        )
    ds = xr.Dataset(
            data_vars=dict(
                hyam=hyam,
                hybm=hybm,
                P0=p0,
                PS=PS,
                sf=sf_xr
            ),
            coords=dict(
                lev=zonmean.lev,
                lat=zonmean.lat,
            )
        )
    return ds
######
if __name__ == "__main__":
    cases = ['QSPSRCE_r10_8x4km_300K', 'QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    for case in cases:
        ds = main(case)
        print("Output...")
        ds.to_netcdf('/data/W.eddie/SPCAM/'+case+'/atm/hist/'+case+'.timzonmean.streamfunction.nc')
        ds.close()
