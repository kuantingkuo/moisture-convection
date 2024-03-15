cases='QSPSRCE_r10_8x4km_300K QSPSRCE_r10_32x4km_300K QSPSRCE_r10_256x4km_300K'
opath='/data/W.eddie/SPCAM/'
rc=gsfallow('on')
n=count_num(cases)
'reinit'
'ini -l'
'set mproj mollweide'
i=1
while(i<=n)
    case=subwrd(cases,i)
    path=opath%case'/atm/hist/'
    'open 'path%case'.TC_dist.ctl'
    'open 'path'atm.ctl'
    rc=sys('mkdir -p /data/W.eddie/RCE/'case)
    i=i+1
endwhile

'set t 1 last'
tmax=qdims('tmax')
'set lat -25 25'
'off'
t=2
while(t<=tmax)
    'set t 't
    'c'
    'set xlint 60'
    'set ylint 10'
    'mul 1 3 1 1 -yoffset 0.7'
    'set grads off'
    'color -levs 0.1 1 2 3 4 5 6 7 8 9 10 -kind cwb'
    'd prect.6(z=1)*1000*3600'
    'xcbar3 -direction h -yoffset 0.7 -unit [mm h`a-1`n] -caption Precipitation'
    'color 25 55 2 -kind (0,0,0,255)->(255,255,255,0)'
    'set grid on'
    'd tmq.6(z=1)'
    'set grid off'
    'xcbar3 -unit [mm] -caption Column Water Vapor'
    'set gxout contour'
    'set clab off'
    'set ccols 1'
    'set clevs 0.5'
    'set cthick 5'
    'd TC.5'
    frame()
    'mul 1 3 1 2 -yoffset 0.48'
    'set grads off'
    'color -levs 0.1 1 2 3 4 5 6 7 8 9 10 -kind cwb'
    'd prect.4(z=1)*1000*3600'
    'color 25 55 2 -kind (0,0,0,255)->(255,255,255,0)'
    'set grid on'
    'd tmq.4(z=1)'
    'set grid off'
    'set gxout contour'
    'set clab off'
    'set ccols 1'
    'set clevs 0.5'
    'set cthick 5'
    'd TC.3'
    frame()
    'mul 1 3 1 3 -yoffset 0.25'
    'set grads off'
    'color -levs 0.1 1 2 3 4 5 6 7 8 9 10 -kind cwb'
    'd prect.2(z=1)*1000*3600'
    'color 25 55 2 -kind (0,0,0,255)->(255,255,255,0)'
    'set grid on'
    'd tmq.2(z=1)'
    'set grid off'
    'set gxout contour'
    'set clab off'
    'set ccols 1'
    'set clevs 0.5'
    'set cthick 5'
    'd TC.1'
    frame()
    'date'
    'draw title 'result
    tttt=math_format('%04g',t)
    'gxprint /data/W.eddie/RCE/animate/TC_tropics_'tttt'.png white'
    t=t+1
endwhile

function frame()
    'set ccolor 1'
    'set cthick 6'
    'set clevs 0 359.999999999999'
    'd lon'
    'set ccolor 1'
    'set cthick 6'
    'set cstyle 1'
    'set clevs -25 25'
    'd lat'
    return
