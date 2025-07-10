;+
; PROCEDURE:
;   FOXSI X-RAY OPTICS Alignment Script
; PURPOSE:
;   Align Optics with LoS of FOXSI Payload
; NOTES:
;   See below for a list of inputs directly in script
;
; CREATED BY: Orlando Romeo, 02/01/2024
;-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
; SECTION 0: USER INPUTS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; OPTIC MECHANICAL Properties (inches, degrees)
opt_len   = 5.600                  ; Radius of optic (from center to screws)
rot_ang   = 30.0                      ; Rotation angle to orient optic wrt Rocket (degrees)
; OPTIC ALIGNMENT Properties (degrees)
; Misalignment angle Amount - Usually within 20 arcminutes
mis_ang = 3.03/60.0
; Off Axis Amount (degrees) (from start of unit circle (x=1,y=0), going around CCW (0-360)) - Shim away from shearing
off_ang = 277
; Account for 180 shift from perspective of looking at the payload from the front vs the back
off_ang = (270.0-off_ang)+90.0
; Compute horizontal and vertical offsets in cartesian frame
xoff       = (opt_len*sin(mis_ang*!DPI/180.0))*cos(off_ang*!DPI/180.0)  ; Inches
yoff       = (opt_len*sin(mis_ang*!DPI/180.0))*sin(off_ang*!DPI/180.0)  ; Inches
; Compute horizontal and vertical offsets (degrees)
hoff = asin(xoff/opt_len)/!DPI*180.0
voff = asin(yoff/opt_len)/!DPI*180.0
; SHIM Properties (inches)
shim_thick = 0.002                   ; Shim thickness (in)
shim_rad   = 0.100                   ; Shim Radius
maxsnum    = 5                      ; Max Number of Shims
; SCRIPT Flags
plot_optic = 0                       ; Flag to plot optic
prnt_flg   = 0                       ; Flag set to print all results for every shim combo
mthd_flg   = 1                       ; Flag set method for computing best shim combo (see below)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; SECTION 1: SHIM POSITIONS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Total Misalignment Angle
print,'------------------------------------------------------------------------------------------------------------'
print,string(0,'(I2)')+' SHIMS   -   '+string(hoff,'((f+0.3),"° H_OFF   -  ")')+string(voff,'((f+0.3),"° V_OFF   -  ")')+$
  string(mis_ang,'((f0.3),"° MISALIGNMENT   -    ")')+'METHOD '+trimnum(mthd_flg)
print,'------------------------------------------------------------------------------------------------------------'
; Find screw/shim position
shim_pos = opalfox_optic(rot_ang=rot_ang,xoff=xoff,yoff=yoff,plot_optic=plot_optic,optic=optic)
; Number of Screw/Shim Positions
k = 6
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; SECTION 2: SHIM COMBINATIONS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Iterate over number of shims
for n=1,maxsnum do begin
  ; Initialize best combo given misalignment error
  best_combo = intarr(k)
  best_err   = mis_ang
  best_hoff  = hoff
  best_voff  = voff
  ; Find number of possible combinations by method of stars and bars
  combos = factorial(n+k-1)/(factorial(k-1)*factorial(n))
  ; Output Header Info
  if keyword_set(prnt_flg) then begin
    print,'-------------------------------------------'
    print,'|            '+string(n,'(I2.0)')+' SHIM RESULTS              |'
    print,'|_________________________________________|'
    print,'| '+string(indgen(k)+1,'('+trimnum(k)+'("P",I0.0," "))')+'|    XOFF      YOFF   |  ERROR |'
  endif
  ; Set shifter and space amount based on stars and bars method
  shft  = 0            ; Shift amount for new configuration
  space = intarr(n)    ; Initial space between stars
  space_ind  = 1       ; Keep track of latest space
  space_cntr = 1       ; Counter for new space configuration
  ; Iterate over possible combinations
  for c=0,combos-1 do begin
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; Create array of stars (1) and bars (0)
    starbar = intarr(n+(k-1))
    ; Create indexes for 1 (stars)
    ind = indgen(n)+shft+space
    starbar[ind] = 1
    ; Add padding to determine bins at ends of range
    new_starbar = [0,starbar,0]
    ; Compute cumulative total within each bin
    new_starbar_sum = total(new_starbar,/cumulative)
    new_starbar_sum = new_starbar_sum[where(new_starbar eq 0)]
    ; Find final shim combination
    combo           = new_starbar_sum[1:-1]-new_starbar_sum[0:-2]
    ; Compute shim heights
    shims = combo*shim_thick
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; METHOD 1: Compute Vector Change from 3 Surrounding Shims to form one plane
    if mthd_flg eq 1 then begin
      v = [0,0,0]
      ; Iterate over each main screw
      for ki=0,k-1 do begin
        ; Set screws
        s1 = ki
        s2 = (ki+1) mod k
        s3 = (ki+2) mod k
        ; Find 2 Difference Vectors
        v21 = [shim_pos[*,s2],shims[s2]] - [shim_pos[*,s1],shims[s1]]
        v31 = [shim_pos[*,s3],shims[s3]] - [shim_pos[*,s1],shims[s1]]
        ; Find cross product - normal to plane of 3 points
        crss = crossp(v21,v31)
        if crss[2] lt 0 then crss = -crss  ; Ensure Vector points upward
        ; Find optic length vector
        lcrss = (crss/sqrt(total(crss^2.0)))
        v = v+lcrss
      endfor
      uv = v/sqrt(total(v^2.0))
      ; Compute new offsets from Shim Configuration
      new_hoff = 180/!DPI*asin(uv[0])
      new_voff = 180/!DPI*asin(uv[1])
      ; Compute Error from new offset
      error = sqrt( (new_hoff+hoff)^2.0 + (new_voff+voff)^2.0)
    endif
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; Check best combo so far based on error
    if error lt best_err then begin
      best_err   = error
      best_combo = combo
      best_hoff  = new_hoff
      best_voff  = new_voff
    endif
    ; Output results
    if keyword_set(prnt_flg) then $
      print,string(combo,'("| ",'+trimnum(k)+'(" ",I0," "),"| ")')+string([new_hoff,new_voff],'(2(" ",f+0.3,"°  "),"| ")'),string(error,'((f0.3),"° |")')
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; Re-evaluate shifting and spacing for next configuration (skip after last config)
    if (shft ge ((k-1)-(max(space)))) and (c lt (combos-1)) then begin
      shft  = 0
      ; Ensure space index does not go past first index
      if space_ind ge n then begin
        space_cntr = space_cntr + 1
        if space_cntr gt n then space_cntr = n
        space[0:-space_cntr] = 0
        space_ind = space_cntr-1
      endif
      ; Add space for stars
      space[-space_ind] = space[-space_ind] + 1
      space_ind         = space_ind + 1
    endif else shft = shft+1
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  endfor
  if keyword_set(prnt_flg) then print,'-------------------------------------------'
  ; Output best combo given shim amount
  if total(best_combo) eq 0 then best_err = sqrt( (xoff)^2.0 + (yoff)^2.0)
  print,string(n,'(I2.0)')+' SHIMS   -   '+string(best_hoff+hoff,'((f+0.3),"° H_OFF   -  ")')+string(best_voff+voff,'((f+0.3),"° V_OFF   -  ")')+$
    string(best_err,'((f0.3),"° MISALIGNMENT   -  ")')+string(best_combo,'('+trimnum(k)+'(" ",I3," ")," COMBO")')
endfor
end



