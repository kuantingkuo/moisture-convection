cases='QSPSRCE_r10_8x4km_300K QSPSRCE_r10_32x4km_300K QSPSRCE_r10_256x4km_300K'
labs='D32 D128 D1024'
cols='4 14 2'
rc=gsfallow('on')
num=count_num(cases)
'reinit'
'ini -h'

i=1
while(i<=num)
    case=subwrd(cases,i)
    'open /data/W.eddie/SPCAM/'case'/atm/hist/TC_dist.timmean.ctl'
    'set grads off'
    'set xyrev on'
    'set lat -60 60'
    'set lon 0'
    'set vrange 0 0.07'
    'set xlint 0.01'
    'set ylint 15'
    'set cmark 0'
    'set cthick 8'
    'set ccolor 'subwrd(cols,i)
    'd ave(tc.'i',x=1,x=288)'
    'off'
    i=i+1
endwhile
'legend c 3 'labs' 'cols
'gxprint /data/W.eddie/RCE/TC_frac.png white'
'gxprint /data/W.eddie/RCE/TC_frac.svg white'
