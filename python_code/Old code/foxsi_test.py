# FOXSI IV - Optical Alignment:
#            Reads PNG Files of Laser Photos for Alignment
# ----------------------------
# 07/18/2023
# Orlando Romeo
#############################################################
# Import Third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import opalfox
###############################################################################
# Heritage FOXSI 3 Optics
if 0:
    data = np.array([1.1055,1.1363,1.1104,1.1369,1.1713,1.0740,1.2148])
    ang  = np.array([0,.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8])
    p0x0 = np.array([1.0116,1.0265,1.0489,1.0628,1.0868,1.1086,1.1287,1.1508,1.1895,1.2114])
    p2x5 = np.array([1.0092,1.0250,1.0474,1.0666,1.0903,1.1213,1.1442,1.1703,1.1997,1.2152])
    p4x6 = np.array([1.008997,1.02434,1.046569,1.076642,1.09251,1.12271,1.1466107,1.1628,1.1936293,1.2267584])
    p6x1 = np.array([1.0085,1.0281,1.0416,1.0668,1.0870,1.1203,1.1419,1.1647,1.1893,1.2185])
    xmod = [0,3,5,4,6,8,1]
# Flight FOXSI 4 Optics
if 1:
    data = np.array([1.0853,0,0,1.2820,0,0,1.1455])
    ang  = np.array([0,.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8])
    p036x0 = np.array([1.004,1.0361,1.0653,1.1067,1.1377,1.1773,1.2174,1.2552,1.2933,1.3409])
    xmod = [0,3,5,4,6,8,1]
x = ang
y = p036x0
pos = 6
###############################################################################
# Interpolate data
offangle = np.interp(data[pos],y,ang)
###############################################################################
# Plot
plt.figure(figsize=(9,6))
plt.plot(x,y,'r.-',markersize=15)
plt.plot([-1,3],[data[pos],data[pos]],'b--')
plt.plot(offangle,data[pos],'b.',markersize=20)
plt.xlim([-.2,2])
plt.grid()
plt.xlabel('Off Angle (arcmin)')
plt.ylabel('Asymmetry Ratio')
plt.title("Off Axis: P"+str(pos)+ " X"+str(xmod[pos])+ " - {:.3f}\'".format(offangle))





















