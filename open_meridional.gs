cases='QSPSRCE_r10_8x4km_300K QSPSRCE_r10_32x4km_300K QSPSRCE_r10_256x4km_300K'
rc=gsfallow('on')
num=count_num(cases)
opath='/data/W.eddie/SPCAM/'
'reinit'
'ini -l'

i=1
while(i<=num)
    case=subwrd(cases,i)
    'open 'opath%case'/atm/hist/'case'.timzonmean.ctl'
    i=i+1
endwhile

