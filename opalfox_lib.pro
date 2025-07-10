;+
;LIBRARY:
; opalfox_lib
;PURPOSE:
; OPAL FOX: OPtical ALignment for FOXSI Library 
;           Includes Alignment for LISS
;           
;ROUTINES
; opalfox_addline    - Function to create x and y arrays of points for line segment part of shape
; opalfox_liss       - Function to create shape of LISS base (aperature side)
;
;
; CREATED BY: Orlando Romeo, 02/01/2024
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                                                   ,    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;      XX                    /\                   ,'|    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;    XX  XX ------------ o--'O `.                /  /    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;      XX                 `--.   `-----------._,' ,'     ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                             \              ,---'       ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                              ) )    _,--(  |           ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                             /,^.---'     )/\\          ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                            ((   \\      ((  \\         ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                             \)   \)      \) (/         ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; FUNCTION: opalfox_addline
;           Function to create x and y arrays of points for line segment part of shape based on range of x/y values
;           pres - [Scalar] Pixel resolution per unit
function opalfox_addline,xmin,xmax,ymin,ymax,pres=pres
  ; Set pixel resolution per unit
  if ~keyword_set(pres) then pres =  300
  ; Number of points
  pnum = ceil(pres*sqrt((xmax-xmin)^2.0+(ymax-ymin)^2.0))
  ; Create x and y arrays
  x = dgen(pnum,[xmin,xmax])
  y = dgen(pnum,[ymin,ymax])
  return,[[x],[y]]
end
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; FUNCTION: opalfox_liss
;           Creates shape of LISS base (aperature side) and returns positions of screws/shims
;           rot_ang   - Rotation angle to orient LISS wrt Rocket (degrees)
;           x/yoff    - Horizontal/Vertical Offset in inches from center of aperature
;           plot_liss - If set, plots shape (set to 'ps' to save figure)
;           liss      - Outputs Points of LISS Base Shape
function opalfox_liss,rot_ang=rot_ang,xoff=xoff,yoff=yoff,plot_liss=plot_liss,liss=liss
  ; Set defaults
  if ~keyword_set(rot_ang) then rot_ang = 0.0
  if ~keyword_set(xoff)    then xoff    = 0.0
  if ~keyword_set(yoff)    then yoff    = 0.0
  ; Center of LISS Base
  cntr = [1.0,0.7]
  ; Screw Positions for shimming
  sc1 = [.15,.7]
  sc2 = [1.85,1.225]
  sc3 = [1.85,.175]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; LISS BASE SHAPE
  ; Left side of structure
  lseg = [opalfox_addline(.3,.3,0,.525), opalfox_addline(.3,0,.525,.525),$
          opalfox_addline(0,0,.525,.875),opalfox_addline(0,.3,.875,.875), opalfox_addline(.3,.3,.875,1.4)]
  ; Top side of structure
  tseg = opalfox_addline(.3,2,1.4,1.4)
  ; Right side of structure
  rseg = [opalfox_addline(2,2,1.4,1.05),    opalfox_addline(2,1.7,1.05,1.05),$
          opalfox_addline(1.7,1.7,1.05,.35),opalfox_addline(1.7,2,.35,.35),   opalfox_addline(2,2,.35,0)]
  ; Bottom side of structure
  bseg = opalfox_addline(2,.3,0,0)
  ; Create LISS Structure (subtracted from center)
  liss_pts      = [lseg,tseg,rseg,bseg]
  liss_pts[*,0] = liss_pts[*,0] - cntr[0]
  liss_pts[*,1] = liss_pts[*,1] - cntr[1]
  ; Check for rotation for LISS Shape
  rang = rot_ang*!DPI/180.0
  liss_x = liss_pts[*,0]*cos(rang) - liss_pts[*,1]*sin(rang)
  liss_y = liss_pts[*,0]*sin(rang) + liss_pts[*,1]*cos(rang)
  ; Account for misalignment
  liss = [[liss_x],[liss_y]]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; Compute Final Position of LISS Screws (From center, rotation and x/y offset)
  sc1 = sc1-cntr
  sc2 = sc2-cntr
  sc3 = sc3-cntr
  pt1 = [sc1[0]*cos(rang) - sc1[1]*sin(rang),sc1[0]*sin(rang) + sc1[1]*cos(rang)]; + [xoff,yoff]
  pt2 = [sc2[0]*cos(rang) - sc2[1]*sin(rang),sc2[0]*sin(rang) + sc2[1]*cos(rang)]; + [xoff,yoff]
  pt3 = [sc3[0]*cos(rang) - sc3[1]*sin(rang),sc3[0]*sin(rang) + sc3[1]*cos(rang)]; + [xoff,yoff]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; Plot LISS Base (Facing Aperature)
  if keyword_set(plot_liss) then begin
    ; Symbol Thickness for WIN File
    thck = 1
    ; Check to save figure
    if isa(plot_liss,'string') then begin
      if strmatch(plot_liss,'ps',/fold) then begin
        ps   = plot_liss ;  PS File
        thck = 6         ; Symbol Thickness for PS File
      endif else ps = 0
    endif
    ; Set x/y ranges
    xrange = [min(liss[*,0])-.2,max(liss[*,0])+.2]
    yrange = [min(liss[*,1])-.2,max(liss[*,1])+.2]
    ; Set plot title
    xoff_str = 'X'+vplot_char('off',command='sub',ps=ps)+' = '+string(xoff,format='(f+0.2)')+'in'
    yoff_str = 'Y'+vplot_char('off',command='sub',ps=ps)+' = '+string(yoff,format='(f+0.2)')+'in'
    title = 'LISS ('+xoff_str+', '+yoff_str+')'
    ; Plot Base shape
    vplot,liss[*,0],liss[*,1],lim={isotropic:1,xrange:xrange,yrange:yrange,$
          xtitle:'X'+vplot_char('payload',command='sub')+' (in)',ytitle:'Y'+vplot_char('payload',command='sub')+' (in)',title:title},$
          vps=vps,/grid,save=ps,/delaysave
    ; Plot front aperature
    vplot_sym,'circle',fill=0
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=1.05,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=1.35,syunit='data',sym=8,save=ps,/delaysave
    vplot_sym,'square'
    ; Plot Dead Center
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sym=8,sysize=.25,syunit='data',save=ps,/delaysave
    ; Plot Alignment Crosshairs
    vplot_sym,'crosshairs',thick=thck
    vplot,xoff,yoff,lim={isotropic:1},vps=vps,/overplot,sym=8,sysize=2,shade=254b,save=ps,/delaysave
    vplot_sym,/default
    ; Screw 1
    vplot,pt1[0],pt1[1],vps=vps,/over,syunit='data',sym=8,sysize=.2,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt1[0],pt1[1]-.04,'1',align=0.5
    ; Screw 2
    vplot,pt2[0],pt2[1],vps=vps,/over,syunit='data',sym=8,sysize=.2,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt2[0],pt2[1]-.04,'2',align=0.5
    ; Screw 3
    vplot,pt3[0],pt3[1],vps=vps,/over,syunit='data',sym=8,sysize=.2,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt3[0],pt3[1]-.04,'3',align=0.5
    ; Save figure
    vplot,!values.f_nan,!values.f_nan,vps=vps,/over,save=ps
  endif
  return,[[pt1],[pt2],[pt3]]
end
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; FUNCTION: opalfox_optic
;           Creates shape of XRAY OPTIC and returns positions of screws/shims
;           rot_ang    - Rotation angle to orient optic wrt Rocket (degrees)
;           x/yoff     - Horizontal/Vertical Offset in inches from center of aperature
;           plot_optic - If set, plots shape (set to 'ps' to save figure)
;           optic      - Outputs Points of OPTIC Mounting Surface
function opalfox_optic,rot_ang=rot_ang,xoff=xoff,yoff=yoff,plot_optic=plot_optic,optic=optic
  ; Set defaults
  if ~keyword_set(rot_ang) then rot_ang = 0.0
  if ~keyword_set(xoff)    then xoff    = 0.0
  if ~keyword_set(yoff)    then yoff    = 0.0
  ; Center of OPTIC Base
  cntr = [0.0,0.0]
  ; Screw Positions for shimming
  sc1 = [0.0,2.8]
  sc2 = [cos(30.0*!DPI/180.0),sin(30.0*!DPI/180.0)]*2.8
  sc3 = [cos(-30.0*!DPI/180.0),sin(-30.0*!DPI/180.0)]*2.8
  sc4 = [0.0,-2.8]
  sc5 = [cos(-150.0*!DPI/180.0),sin(-150.0*!DPI/180.0)]*2.8
  sc6 = [cos(150.0*!DPI/180.0),sin(150.0*!DPI/180.0)]*2.8
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; OPTICS BASE SHAPE
  ; Each Side of structure, starting from bottom
  ;seg1 = [opalfox_addline(-.45,.45,-2.95,-2.95),opalfox_addline(.45,2.3333,-2.95,-1.65)]
  ;seg2 = [opalfox_addline(2.3333,2.725,-1.65,-1.025),opalfox_addline(2.725,2.725,-1.025,1.025)]
  ;seg3 = [opalfox_addline(2.725,2.3333,1.025,1.65),opalfox_addline(2.3333,.45,1.65,2.95)]
  ;seg4 = [opalfox_addline(.45,-.45,2.95,2.95),opalfox_addline(-.45,-2.3333,2.95,1.65)]
  ;seg5 = [opalfox_addline(-2.3333,-2.725,1.65,1.025),opalfox_addline(-2.725,-2.725,1.025,-1.025)]
  ;seg6 = [opalfox_addline(-2.725,-2.3333,-1.025,-1.65),opalfox_addline(-2.3333,-.45,-1.65,-2.95)]
  seg1 = [opalfox_addline(-.4488,.4488,-2.8874,-2.8874),opalfox_addline(.4488,2.27624,-2.8874,-1.83242)]
  seg2 = [opalfox_addline(2.27624,2.725,-1.83242,-1.055077),opalfox_addline(2.725,2.725,-1.055077,1.055077)]
  seg3 = [opalfox_addline(2.725,2.27624,1.055077,1.83242),opalfox_addline(2.27624,.4488,1.83242,2.8874)]
  seg4 = [opalfox_addline(.4488,-.4488,2.8874,2.8874),opalfox_addline(-.4488,-2.27624,2.8874,1.83242)]
  seg5 = [opalfox_addline(-2.27624,-2.725,1.83242,1.055077),opalfox_addline(-2.725,-2.725,1.055077,-1.055077)]
  seg6 = [opalfox_addline(-2.725,-2.27624,-1.055077,-1.83242),opalfox_addline(-2.27624,-.4488,-1.83242,-2.8874)]
  
  ; Create OPTICS Structure (subtracted from center)
  opt_pts      = [seg1,seg2,seg3,seg4,seg5,seg6]
  opt_pts[*,0] = opt_pts[*,0] - cntr[0]
  opt_pts[*,1] = opt_pts[*,1] - cntr[1]
  ; Check for rotation for OPTICS Shape
  rang = rot_ang*!DPI/180.0
  opt_x = opt_pts[*,0]*cos(rang) - opt_pts[*,1]*sin(rang)
  opt_y = opt_pts[*,0]*sin(rang) + opt_pts[*,1]*cos(rang)
  ; Account for misalignment
  opt = [[opt_x],[opt_y]]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; Compute Final Position of OPTICS Screws (From center, rotation and x/y offset)
  sc1 = sc1-cntr
  sc2 = sc2-cntr
  sc3 = sc3-cntr
  sc4 = sc4-cntr
  sc5 = sc5-cntr
  sc6 = sc6-cntr
  pt1 = [sc1[0]*cos(rang) - sc1[1]*sin(rang),sc1[0]*sin(rang) + sc1[1]*cos(rang)]; + [xoff,yoff]
  pt2 = [sc2[0]*cos(rang) - sc2[1]*sin(rang),sc2[0]*sin(rang) + sc2[1]*cos(rang)]; + [xoff,yoff]
  pt3 = [sc3[0]*cos(rang) - sc3[1]*sin(rang),sc3[0]*sin(rang) + sc3[1]*cos(rang)]; + [xoff,yoff]
  pt4 = [sc4[0]*cos(rang) - sc4[1]*sin(rang),sc4[0]*sin(rang) + sc4[1]*cos(rang)]; + [xoff,yoff]
  pt5 = [sc5[0]*cos(rang) - sc5[1]*sin(rang),sc5[0]*sin(rang) + sc5[1]*cos(rang)]; + [xoff,yoff]
  pt6 = [sc6[0]*cos(rang) - sc6[1]*sin(rang),sc6[0]*sin(rang) + sc6[1]*cos(rang)]; + [xoff,yoff]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ; Plot XRAY OPTICS
  if keyword_set(plot_optic) then begin
    ; Symbol Thickness for WIN File
    thck = 1
    ; Check to save figure
    if isa(plot_optic,'string') then begin
      if strmatch(plot_optic,'ps',/fold) then begin
        ps   = plot_optic ;  PS File
        thck = 6         ; Symbol Thickness for PS File
      endif else ps = 0
    endif
    ; Set x/y ranges
    xrange = [min(opt[*,0])-.2,max(opt[*,0])+.2]
    yrange = [min(opt[*,1])-.2,max(opt[*,1])+.2]
    ; Set plot title
    xoff_str = 'X'+vplot_char('off',command='sub',ps=ps)+' = '+string(xoff,format='(f+0.2)')+'in'
    yoff_str = 'Y'+vplot_char('off',command='sub',ps=ps)+' = '+string(yoff,format='(f+0.2)')+'in'
    title = 'XRAY OPTICS'; ('+xoff_str+', '+yoff_str+')'
    ; Plot Base shape
    vplot,opt[*,0],opt[*,1],lim={isotropic:1,xrange:xrange,yrange:yrange,$
      xtitle:'X'+vplot_char('payload',command='sub')+' (in)',ytitle:'Y'+vplot_char('payload',command='sub')+' (in)',title:title},$
      vps=vps,/grid,save=ps,/delaysave
    ; Plot front aperature
    vplot_sym,'circle',fill=0
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=5.0,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=4.0,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.95,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.90,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=3.85,syunit='data',sym=8,save=ps,/delaysave
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=1.25,syunit='data',sym=8,save=ps,/delaysave
    vplot_sym,'circle',thick=10
    vplot,0,0,lim={isotropic:1},vps=vps,/overplot,sysize=2.0,syunit='data',sym=8,save=ps,/delaysave
    ; Plot Alignment Crosshairs
    vplot_sym,'crosshairs',thick=thck
    vplot,xoff,yoff,lim={isotropic:1},vps=vps,/overplot,sym=8,sysize=2,shade=254b,save=ps,/delaysave
    vplot_sym,/default
    ; Screw 1
    vplot,pt1[0],pt1[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt1[0],pt1[1]-.25,'1',align=0.5
    ; Screw 2
    vplot,pt2[0],pt2[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt2[0]-.1,pt2[1]-.2,'2',align=0.5
    ; Screw 3
    vplot,pt3[0],pt3[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt3[0]-.15,pt3[1]+.1,'3',align=0.5
    ; Screw 4
    vplot,pt4[0],pt4[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt4[0],pt4[1]+.1,'4',align=0.5
    ; Screw 5
    vplot,pt5[0],pt5[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt5[0]+.15,pt5[1]+.1,'5',align=0.5
    ; Screw 6
    vplot,pt6[0],pt6[1],vps=vps,/over,syunit='data',sym=8,sysize=.15,lim={isotropic:1},shade=43b,save=ps,/delaysave
    xyouts,pt6[0]+.1,pt6[1]-.2,'6',align=0.5
    
    ; Save figure
    vplot,!values.f_nan,!values.f_nan,vps=vps,/over,save=ps
  endif
  return,[[pt1],[pt2],[pt3],[pt4],[pt5],[pt6]]
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
end




