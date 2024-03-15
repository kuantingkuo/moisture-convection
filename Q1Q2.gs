cases='QSPSRCE_r10_8x4km_300K QSPSRCE_r10_32x4km_300K QSPSRCE_r10_256x4km_300K'
rc=gsfallow('on')
num=count_num(cases)

'open_meridional.gs'
'set lat -60 60'
'set z 2 23'

i=1
while(i<=num)
    case=subwrd(cases,i)
    'c'
    'set grads off'
    'color -levs -0.5 -0.25 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 -kind blue-(1)->white->orange->red'
    'd spdt.'i'*86400'
    'draw ylab Pressure [hPa]'
    'cbar3 [K day`a-1`n]'
    'gxprint /data/W.eddie/RCE/'case'.Q1.png white'

    'c'
    'set grads off'
    'color -levs -1 -0.8 -0.6 -0.4 -0.2 -0.1 0.1 0.4 0.8 1.2 1.6 2 -kind drywet'
    'd spdq.'i'*2.501e6/1004.64*86400'
    'draw ylab Pressure [hPa]'
    'cbar3 [K day`a-1`n]'
    'gxprint /data/W.eddie/RCE/'case'.Q2.png white'
i=i+1
endwhile
