import xarray as xr
import numpy as np
import glob

def open_mfdataset(case, n):
    ncfiles = []
    for m in range(4, 13):
        ncfiles += glob.glob(f'/data/W.eddie/SPCAM/{case}/atm/hist/{case}.cam.h{n}.0001-{m:02d}-??-00000.nc')
    # Open all the ncfiles as a single xarray dataset
    ds = xr.open_mfdataset(ncfiles, combine='by_coords', parallel=True, data_vars='minimal', coords='minimal', compat='override')
    qv = [var for var in ds.variables if var.startswith('CRM_QV_')]
    print(ds[qv[0]].squeeze())
    return ds[qv[0]].squeeze().rename('CRM_QV')

def main(case):
    print(case)
    qv = []
    for n in range(1, 5):
        print(f'h{n}')
        qv.append(open_mfdataset(case, n))

    # Calculate the standard deviation along lat_90s_to_90n
    lon = xr.DataArray(
        data=np.array([0, 90, 180, 270]),
        dims='lon',
        attrs=dict(
            long_name='longitude',
            units='degree_east'
        )
    )

    qv = xr.concat(qv, dim=lon)
    print('calculate STD')
    std_dev = qv.std(dim=['lon','time','crm_nx'], ddof=1, keep_attrs=True)
    print('output')
    std_dev.to_netcdf('/data/W.eddie/SPCAM/'+case+'/atm/hist/CRM_QV_std.nc')

if __name__ == '__main__':
    cases = ['QSPSRCE_r10_32x4km_300K', 'QSPSRCE_r10_256x4km_300K']
    for case in cases:
        main(case)
