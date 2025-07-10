# OPAL FOXSI
**OP**tical **AL**ignment Python Code for the FOXSI Sounding Rocket Mission
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;                                                   ,    ;;;;
;      XX                    /\                   ,'|    ;;;;
;    XX  XX ------------ o--'O `.                /  /    ;;;;
;      XX                 `--.   `-----------._,' ,'     ;;;;
;                             \              ,---'       ;;;;
;                              ) )    _,--(  |           ;;;;
;                             /,^.---'     )/\\          ;;;;
;                            ((   \\      ((  \\         ;;;;
;                             \)   \)      \) (/         ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

## Image File Directory
The images should be placed in the following directory format:
base directory -> optic team -> "MM-DD-YY" -> "POS#_RUN#.png"

where the optic team folder could be "Heritage", "Marshall", or "Nagoya", and the date is related to the day of the optical alignment test.
The position number (0-6) refers to the specific optic position, and the run number is in order of testing.
The optical positions, as viewed from the back of the payload, are:
   <1>
<0>   <2>
   <6>
<5>   <3>
   <4>

Plesae note that for some optics (Nagoya), a 180 degree flip was needed to match the output of the other optics. 
Always check that the asymmetric angle is in the right direction.

## Image Projection
To fix any projection issues, use the *fixperspective* routine by selected the four corners of the large box surrounding the optic laser image.
You may need to change the brighten keyword to a value ranging from 0 to 300.

## Image Projection
Feel free to change the contrast or brightness of an image using the *alterimage* routine to easily select the ring edges.

## Ring Detection
Follow the prompt when the *fitrings* code runs to detect the rings from the optic image.
Note that one ring includes the inner and outer edge.

## Optic Asymmetry
The possible methods include maxminratio, maxratio, or centerratio.
The final angle asymmetry is outputted in the terminal in degrees, with the difference amount in arcminutes.

## Simulation Comparison
The images from the ray tracing simulations should be in the 'Sims' folder under the optics team folder, corresponding to each optic.
Each image file should be denoted as 'angle_##.png'.

## Shim Finder
IDL code to find positions of shims based on various inputs to align optics.
Goes through all different combinations of shim positions to determine best alignment possible using discrete shim values.
<div style="text-align: center;">
  <img src="images/FOXSI_Optics_ReferenceFig.pdf" alt="FOXSI Optics" width="200"/>
</div>

