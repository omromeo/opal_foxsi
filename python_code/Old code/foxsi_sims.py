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
# Section 0: Initialize Parameters
###############################################################################
pos            = 0
xmod           = 10
angle          = 0.0
proj_flg       = 0                                     # If set to 1, will perform projection check on original image
filetype       = '.png'                                # File Type
file           = "angle_"+str(angle)                   # Original file name
dirf           = "C:/Users/Orlando/Downloads/OPAL_FOXSI/Marshall/Sims/P"+str(pos)+"_X"+str(xmod)+'/'
dirfile        = dirf+file+filetype                    # File Directory 
efile          = "Image_Processing/FOXSI4_Marshall_ellipse_P"+str(pos)+"_X"+str(xmod)+"_A"+str(angle)     # Ellipse File
# -----------------------------------------------------------------------------
newfile        = 'FOXSI_'+file+'_projected'            # New File Name
dirnfile       = dirf+"Image_Processing/"+newfile+filetype # New File Directory
nfile_flg      = os.path.exists(dirnfile)              # Check if newfile exists
###############################################################################
# Section 1: Fix Image Projection
###############################################################################
if proj_flg or not nfile_flg:
    # Read Image File
    image_original = opalfox.readimage(dirfile)
    image_proj = opalfox.fixperspective(image_original, savedir=dirnfile)
else:
    image_proj = opalfox.readimage(dirnfile)
###############################################################################
# Section 2: Fix Image Brightness
###############################################################################
# Find Scaling Factor (Large square is 4 inches in length)
sz   = np.shape(image_proj)
sf = np.array([2.54* 5.0/sz[0], 2.54*5.0/sz[1]])
# Crop grey image with a min brightness set
gimage = np.mean(image_proj, axis=-1)
gimage[np.where(gimage > 200 )] = 0
#opalfox.plotimage(gimage,brighten=0,title="FOXSI: Projected Grid",cmap="gray")
###############################################################################
# Section 3: Bin Image Brightness
###############################################################################
# find median brightness in each bin by r and theta (polar coordinates)
rbin = np.linspace(0, np.sqrt(2), num=501) # Radial bins
tbin = np.linspace(0, 2*np.pi, num=17)     # Theta bins
image_med = opalfox.binimage(gimage,rbin=rbin,tbin=tbin,plot=0)
###############################################################################
# Section 4: Detection of Ring Edges
###############################################################################
# find median brightness in each bin by r and theta (polar coordinates)
rings=2
centers,axes,angles = opalfox.fitrings(gimage,image_bin=image_med,rings=rings,auto=0,rbin=rbin,tbin=tbin,plot=1)
# Save ellipse files
np.savez(dirf+efile,centers=centers,axes=axes,angles=angles)
###############################################################################
# Section 5: Find Asymmetry
###############################################################################
if 0:
    ellipse = np.load(dirf+efile+".npz")
    centers = ellipse['centers']
    axes    = ellipse['axes']
    angles  = ellipse['angles']



###############################################################################
# Old Method of finding asymmetry by max radial distance
if 0:
    # Offset Centers
    new_centers = centers - centers[0,:]
    # Find ellipse points
    ix,iy = opalfox.ellipse(new_centers[0,:],axes[0,:],np.radians(angles[0]),theta=np.linspace(0, 2*np.pi, 10000))
    ox,oy = opalfox.ellipse(new_centers[-1,:],axes[-1,:],np.radians(angles[-1]),theta=np.linspace(0, 2*np.pi, 10000))
    # Find angles
    itheta = np.arctan2(-iy,ix) % (2*np.pi)
    otheta = np.arctan2(-oy,ox) % (2*np.pi)
    # Find radial distances
    ri = np.sqrt(ix**2+iy**2)
    ro = np.sqrt(ox**2+oy**2)
    # Find difference in radial distance
    diffr = np.abs( ri[np.argsort(itheta)] - ro[np.argsort(otheta)] )
    # Find ratio based on max radial distance
    ratio = np.max(diffr)/np.min(diffr)
    theta_sort = np.sort(itheta)
    angle_diff = theta_sort[np.argmax(diffr)]*180/np.pi





# from scipy.signal import convolve,correlate
# from scipy.signal import argrelextrema
# from scipy.ndimage import median_filter,gaussian_filter
# from scipy.signal import find_peaks
# rlen=60
# thi = 9
# s = image_med[:,thi]

# result1       = gaussian_filter(s, sigma=10)
# result        = median_filter(result1, size=rlen)

# plt.figure()
# plt.plot(s)
# plt.plot(result1)
# plt.plot(result)

# localmaxi     = (find_peaks(result,width=int(rlen/2),distance=int(rlen/2)))[0][-1]
# maxval        = np.max(image_med[localmaxi-int(rlen*.2):localmaxi+int(rlen*.2),thi])
# # Find normalized resulting signal based on max value
# result_thi    = result/np.max(result[localmaxi])*maxval
# # Apply gaussian filter to further smooth signal, but retain values
# result_smth = median_filter(image_med[:,thi], size=rlen)
# # Find positive and negative slopes around maxima
# pos_slope = np.gradient(result_smth)
# neg_slope = -1.0*pos_slope
# # Remove small slopes and small values
# pos_slope[np.where((pos_slope < 2) | (result_thi < maxval*0.5))] = 0
# neg_slope[np.where((neg_slope < 2) | (result_thi < maxval*0.5))] = 0
# # Find closest and largest slope to maxima
# pos_pk = find_peaks(pos_slope)[0] - localmaxi
# neg_pk = find_peaks(neg_slope)[0] - localmaxi
# # Assign inner and outer radial distances
# inner_r[thi,ri] = rmid[localmaxi +(pos_pk[np.where(pos_pk < 0)])[-1]]
# outer_r[thi,ri] = rmid[localmaxi +(neg_pk[np.where(neg_pk > 0)])[0]]


# # Set binned stat
# array1 = np.copy(s)
# # Create step function for each row
# zeros_array = np.zeros(int(rlen/2))
# ones_array = np.ones(rlen)
# stepfunc   = np.concatenate((zeros_array,ones_array))


# #array2     = np.repeat(stepfunc[:, np.newaxis],,axis=1)
# # Perform the convolution
# result = convolve(array1, stepfunc, mode='same')
# result = result/np.max(result)*np.max(s)




# # Offset Centers
# new_centers = centers - centers[0,:]

# ix,iy = opalfox.ellipse(new_centers[0,:],axes[0,:],np.radians(angles[0]),theta=np.linspace(0, 2*np.pi, 5000))
# ox,oy = opalfox.ellipse(new_centers[-1,:],axes[-1,:],np.radians(angles[-1]),theta=np.linspace(0, 2*np.pi, 5000))

# itheta = np.arctan2(-iy,ix) % (2*np.pi)
# otheta = np.arctan2(-oy,ox) % (2*np.pi)


# ri = np.sqrt(ix**2+iy**2)
# ro = np.sqrt(ox**2+oy**2)

# diffr = np.abs( ri[np.argsort(itheta)] - ro[np.argsort(otheta)] )

# ratio = np.max(diffr)/np.min(diffr)
# theta_sort = np.sort(itheta)
# angle_diff = theta_sort[np.argmax(diffr)]*180/np.pi


















# new_centers = centers - centers[0,:]

# # Create a grid of points covering both ellipses
# x_grid, y_grid = np.meshgrid(np.linspace(-3,3, 1000), np.linspace(-3,3, 1000))

# # Define the parameters of the two ellipses (center, major axis length, minor axis length, and rotation angle)
# center1 = new_centers[0,:]
# a1, b1 = axes[0,:]
# angle1 = np.radians(angles[0])

# center2 = new_centers[-1,:]
# a2, b2 = axes[-1,:]
# angle2 = np.radians(angles[-1])


# # Transform the grid points by the rotation angles of the ellipses
# rot_x_grid1 = (x_grid - center1[0]) * np.cos(angle1) + (y_grid - center1[1]) * np.sin(angle1)
# rot_y_grid1 = -(x_grid - center1[0]) * np.sin(angle1) + (y_grid - center1[1]) * np.cos(angle1)

# rot_x_grid2 = (x_grid - center2[0]) * np.cos(angle2) + (y_grid - center2[1]) * np.sin(angle2)
# rot_y_grid2 = -(x_grid - center2[0]) * np.sin(angle2) + (y_grid - center2[1]) * np.cos(angle2)

# # Calculate the distances from each grid point to the centers of the ellipses
# dist1 = np.sqrt((rot_x_grid1 / a1)**2 + (rot_y_grid1 / b1)**2)
# dist2 = np.sqrt((rot_x_grid2 / a2)**2 + (rot_y_grid2 / b2)**2)

# # Calculate the difference in distances between the ellipses
# diff_distances = np.abs(dist1 - dist2)

# # Find the point where the difference in distances is the largest
# max_index = np.unravel_index(np.argmax(diff_distances), diff_distances.shape)
# max_point = (x_grid[max_index], y_grid[max_index])
# min_index = np.unravel_index(np.argmin(diff_distances), diff_distances.shape)
# min_point = (x_grid[min_index], y_grid[min_index])

# ratio = np.max(diff_distances)/np.min(diff_distances)

# # Calculate the angle of the asymmetry axis from the center of the ellipses
# asymmetry_angle = np.arctan2(max_point[1] - (center1[1] + center2[1]) / 2, max_point[0] - (center1[0] + center2[0]) / 2)

# # Convert the angle to degrees
# asymmetry_angle_deg = np.degrees(asymmetry_angle)



# asdsad


# [0.0,0.2,0.4,0.]
# [1.0538,1.02882,1.01611,1.00854]






# new_centers = centers - centers[0,:]
# rot_centers = np.copy(new_centers)
# ang_rotation = -1*(np.arctan2(new_centers[-1,1],new_centers[-1,0]) % (2*np.pi))

# rot_centers[:,0] = new_centers[:,0] * np.cos(ang_rotation) - new_centers[:,1] * np.sin(ang_rotation)
# rot_centers[:,1] = new_centers[:,0] * np.sin(ang_rotation) + new_centers[:,1] * np.cos(ang_rotation)


# # Create a color map
# color_map = plt.colormaps.get_cmap('gist_rainbow')
# # Find distance between inner first ring and outer second ring edges
# fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# ax[0].set_title('Original Rings')
# ax[0].set_xlabel("X (cm)")
# ax[0].set_ylabel("Y (cm)")
# ax[0].set_aspect('equal')
# ax[0].invert_yaxis()
# ax[0].grid(color='k',alpha=0.75)
# ax[1].set_title('Rotated Rings')
# ax[1].set_xlabel("X (cm)")
# ax[1].set_ylabel("Y (cm)")
# ax[1].set_aspect('equal')
# ax[1].invert_yaxis()
# ax[1].grid(color='k',alpha=0.75)
# for ri in range(rings):
#     # Get inner and outer indexs
#     i = int(ri*2)
#     o = int(ri*2+1)
#     # Calculate the color index based on the iteration number
#     cindexi = i / (rings*2 - 1)  # Normalize to range [0, 1]
#     cindexo = o / (rings*2 - 1)  # Normalize to range [0, 1]
#     cindex  = ((cindexi+cindexo)/2.0)  # Normalize to range [0, 1]
#     print(cindex)
#     # Compute original ellipse rings
#     xi1,yi1 = opalfox.ellipse(new_centers[i,:],axes[i,:],np.radians(angles[i]))
#     xo1,yo1 = opalfox.ellipse(new_centers[o,:],axes[o,:],np.radians(angles[o]))
#     x1_boundary = np.concatenate([xi1, xo1[::-1]])
#     y1_boundary = np.concatenate([yi1, yo1[::-1]])
#     ax[0].fill(x1_boundary*sf[0], y1_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex))
#     ax[0].plot(new_centers[i,0]*sf[0],new_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
#     ax[0].plot(new_centers[o,0]*sf[0],new_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
#     ax[0].plot(xi1*sf[0],yi1*sf[1],'-',linewidth=4,color=color_map(cindexi))
#     ax[0].plot(xo1*sf[0],yo1*sf[1],'-',linewidth=4,color=color_map(cindexo))
#     # Compute rotated ellipse rings
#     xi2,yi2 = opalfox.ellipse(rot_centers[i,:],axes[i,:],np.radians(angles[i])+ang_rotation)
#     xo2,yo2 = opalfox.ellipse(rot_centers[o,:],axes[o,:],np.radians(angles[o])+ang_rotation)
#     x2_boundary = np.concatenate([xi2, xo2[::-1]])
#     y2_boundary = np.concatenate([yi2, yo2[::-1]])
#     ax[1].fill(x2_boundary*sf[0], y2_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex)) 
#     ax[1].plot(rot_centers[i,0]*sf[0],rot_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
#     ax[1].plot(rot_centers[o,0]*sf[0],rot_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
#     ax[1].plot(xi2*sf[0],yi2*sf[1],'-',linewidth=4,color=color_map(cindexi))
#     ax[1].plot(xo2*sf[0],yo2*sf[1],'-',linewidth=4,color=color_map(cindexo))
    

    



# innermost_edge = opalfox.ellipse(rot_centers[0,:],axes[0,:],np.radians(angles[0])-ang_rotation,theta=0)
# outermost_edge = opalfox.ellipse(rot_centers[-1,:],axes[-1,:],np.radians(angles[-1])-ang_rotation,theta=0)
    

# outermost_edge[1]




#     # x,y = opalfox.ellipse([0,0],[1,2],np.radians(45),theta=np.linspace(0, 2*np.pi, 100))
#     # plt.figure()
#     # plt.plot(x,y)
#     # plt.grid()
#     # plt.gca().set_aspect('equal')
#     # cv2.fitEllipse(np.vstack((x,y), dtype=np.float32).T)



# aa = [1,2]#axes[0,:]*[1,1.5]
# ang = 45
# theta_org = 90
# opalfox.ellipse([0,.0],aa,np.radians(ang),theta=(np.arctan2(np.tan(-np.radians(ang))*aa[0],aa[1])))




# ang=30
# theta = -(np.radians(ang*2) + (np.arctan2(np.tan(-np.radians(ang)),2)))
# 1/2 * np.cos(theta) * np.sin(np.radians(ang)) + 2/2 * np.sin(theta) * np.cos(np.radians(ang))


# new_centers = centers - centers[0,:]

# innermost_x,innermost_y = opalfox.ellipse(new_centers[0,:],axes[0,:],np.radians(angles[0]),theta=np.linspace(0, 2*np.pi, 10000))
# outermost_x,outermost_y = opalfox.ellipse(new_centers[-1,:],axes[-1,:],np.radians(angles[-1]),theta=np.linspace(0, 2*np.pi, 10000))

# innermost_r = np.sqrt(innermost_x**2+innermost_y**2)
# outermost_r = np.sqrt(outermost_x**2+outermost_y**2)
# innermost_t = (np.arctan2(innermost_y,innermost_x) % (2*np.pi))*180/np.pi
# outermost_t = (np.arctan2(outermost_y,outermost_x) % (2*np.pi))*180/np.pi


# t_sort = innermost_t[np.argsort(innermost_t)]
# innermost_r[np.argsort(innermost_t)]



# # Create a grid of points covering both ellipses
# x_grid, y_grid = np.meshgrid(np.linspace(-3,3, 1000), np.linspace(-3,3, 1000))

# # Define the parameters of the two ellipses (center, major axis length, minor axis length, and rotation angle)
# center1 = new_centers[0,:]
# a1, b1 = axes[0,:]
# angle1 = np.radians(angles[0])

# center2 = new_centers[-1,:]
# a2, b2 = axes[-1,:]
# angle2 = np.radians(angles[-1])


# # Transform the grid points by the rotation angles of the ellipses
# rot_x_grid1 = (x_grid - center1[0]) * np.cos(angle1) + (y_grid - center1[1]) * np.sin(angle1)
# rot_y_grid1 = -(x_grid - center1[0]) * np.sin(angle1) + (y_grid - center1[1]) * np.cos(angle1)

# rot_x_grid2 = (x_grid - center2[0]) * np.cos(angle2) + (y_grid - center2[1]) * np.sin(angle2)
# rot_y_grid2 = -(x_grid - center2[0]) * np.sin(angle2) + (y_grid - center2[1]) * np.cos(angle2)

# # Calculate the distances from each grid point to the centers of the ellipses
# dist1 = np.sqrt((rot_x_grid1 / a1)**2 + (rot_y_grid1 / b1)**2)
# dist2 = np.sqrt((rot_x_grid2 / a2)**2 + (rot_y_grid2 / b2)**2)

# # Calculate the difference in distances between the ellipses
# diff_distances = np.abs(dist1 - dist2)

# # Find the point where the difference in distances is the largest
# max_index = np.unravel_index(np.argmax(diff_distances), diff_distances.shape)
# max_point = (x_grid[max_index], y_grid[max_index])

# # Calculate the angle of the asymmetry axis from the center of the ellipses
# asymmetry_angle = np.arctan2(max_point[1] - (center1[1] + center2[1]) / 2, max_point[0] - (center1[0] + center2[0]) / 2)

# # Convert the angle to degrees
# asymmetry_angle_deg = np.degrees(asymmetry_angle)





# # Create a color map
# color_map = plt.colormaps.get_cmap('gist_rainbow')
# # Find distance between inner first ring and outer second ring edges
# fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# ax[0].set_title('Original Rings')
# ax[0].set_xlabel("X (cm)")
# ax[0].set_ylabel("Y (cm)")
# ax[0].set_aspect('equal')
# ax[0].invert_yaxis()
# ax[0].grid(color='k',alpha=0.75)
# ax[1].set_title('Rotated Rings')
# ax[1].set_xlabel("X (cm)")
# ax[1].set_ylabel("Y (cm)")
# ax[1].set_aspect('equal')
# ax[1].invert_yaxis()
# ax[1].grid(color='k',alpha=0.75)
# for ri in range(rings):
#     # Get inner and outer indexs
#     i = int(ri*2)
#     o = int(ri*2+1)
#     # Calculate the color index based on the iteration number
#     cindexi = i / (rings*2 - 1)  # Normalize to range [0, 1]
#     cindexo = o / (rings*2 - 1)  # Normalize to range [0, 1]
#     cindex  = ((cindexi+cindexo)/2.0)  # Normalize to range [0, 1]
#     print(cindex)
#     # Compute original ellipse rings
#     xi1,yi1 = opalfox.ellipse(new_centers[i,:],axes[i,:],np.radians(angles[i]))
#     xo1,yo1 = opalfox.ellipse(new_centers[o,:],axes[o,:],np.radians(angles[o]))
#     x1_boundary = np.concatenate([xi1, xo1[::-1]])
#     y1_boundary = np.concatenate([yi1, yo1[::-1]])
#     ax[0].fill(x1_boundary*sf[0], y1_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex))
#     ax[0].plot(new_centers[i,0]*sf[0],new_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
#     ax[0].plot(new_centers[o,0]*sf[0],new_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
#     ax[0].plot(xi1*sf[0],yi1*sf[1],'-',linewidth=4,color=color_map(cindexi))
#     ax[0].plot(xo1*sf[0],yo1*sf[1],'-',linewidth=4,color=color_map(cindexo))
#     # Compute rotated ellipse rings
#     xi2,yi2 = opalfox.ellipse(rot_centers[i,:],axes[i,:],np.radians(angles[i])+ang_rotation)
#     xo2,yo2 = opalfox.ellipse(rot_centers[o,:],axes[o,:],np.radians(angles[o])+ang_rotation)
#     x2_boundary = np.concatenate([xi2, xo2[::-1]])
#     y2_boundary = np.concatenate([yi2, yo2[::-1]])
#     ax[1].fill(x2_boundary*sf[0], y2_boundary*sf[1], alpha=.5, linewidth=0.0,color=color_map(cindex)) 
#     ax[1].plot(rot_centers[i,0]*sf[0],rot_centers[i,1]*sf[1],'.',markersize=20,color=color_map(cindexi))
#     ax[1].plot(rot_centers[o,0]*sf[0],rot_centers[o,1]*sf[1],'.',markersize=20,color=color_map(cindexo))
#     ax[1].plot(xi2*sf[0],yi2*sf[1],'-',linewidth=4,color=color_map(cindexi))
#     ax[1].plot(xo2*sf[0],yo2*sf[1],'-',linewidth=4,color=color_map(cindexo))
    

    



# innermost_edge = opalfox.ellipse(rot_centers[0,:],axes[0,:],np.radians(angles[0])-ang_rotation,theta=0)
# outermost_edge = opalfox.ellipse(rot_centers[-1,:],axes[-1,:],np.radians(angles[-1])-ang_rotation,theta=0)
    

# outermost_edge[1]












# # %%
# rings = opalfox.selectpoints(image2,n=4,norm=1)

# r_rings = np.sqrt(rings[:,0]**2 + rings[:,1]**2)

# rdiff = (r_rings[1] - r_rings[0])
# rinc = rmid[1] - rmid[0]
# rlen = np.floor(rdiff/rinc).astype(int)



# # Set binned stat
# array1 = np.copy(s)

# minz = []
# # Iterate over theta  to normalize result
# for thi in range(len(tmid)):
#     result = median_filter(array1[:,thi], size=rlen)
#     localmaxi     = (find_peaks(result,width=int(rlen/2),distance=int(rlen/2)))[0][0]
#     maxval        = np.max(array1[localmaxi-int(rlen*.2):localmaxi+int(rlen*.2),thi])
#     result_thi = result/np.max(result[localmaxi])*maxval
    
    
    
    
 
 
#     result_smth = gaussian_filter(result_thi, sigma=3)
#     pos_slope = np.gradient(result_smth)
#     neg_slope = -1.0*pos_slope
    
    
#     pos_slope[np.where((pos_slope < 5) | (result_thi < maxval*0.6))] = 0
#     neg_slope[np.where((neg_slope < 5) | (result_thi < maxval*0.6))] = 0
    
    
#     pos_pk = find_peaks(pos_slope)[0] - localmaxi
#     neg_pk = find_peaks(neg_slope)[0] - localmaxi
    
#     inner_r[thi,1] = rmid[localmaxi +(pos_pk[np.where(pos_pk < 0)])[-1]]
#     outer_r[thi,1] = rmid[localmaxi +(neg_pk[np.where(neg_pk > 0)])[0]]
    
#     """
#     plt.figure()
#     plt.plot(rmid,array1[:,thi])
#     plt.plot(rmid,result_thi)
#     plt.plot(rmid,result_smth)
#     plt.plot(rmid,pos_slope)
#     plt.plot(rmid,neg_slope)
#     """
 
# thh = 15
# inner_x = inner_r[thh,1]*np.cos(tmid[thh]+np.pi/2.0)
# inner_y = inner_r[thh,1]*np.sin(tmid[thh]+np.pi/2.0) 

# outer_x = outer_r[:,1]*np.cos(tmid[:]+np.pi/2.0)
# outer_y = outer_r[:,1]*np.sin(tmid[:]+np.pi/2.0) 

# #x2 = minz*np.sin(tmid)
# #y2 = minz*np.cos(tmid) 


# fig, ax = opalfox.plotimage(image2,cmap='gray')
# ax.axis("on")
# #scatter = ax.scatter([500],[200], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+outer_x*cntr_index[0],cntr_index[1]+outer_y*cntr_index[1], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+inner_x*cntr_index[0],cntr_index[1]+inner_y*cntr_index[1], marker='o', color='red')
# #scatter = ax.scatter(cntr_index[1]+x2*cntr_index[1],cntr_index[0]+y2*cntr_index[0], marker='o', color='red')
# plt.show()



# # Apply Gaussian smoothing to the signal to reduce noise and emphasize edges
# a = gaussian_filter(result_thi, sigma=3)


# import cv2# Convert the signal to grayscale image
# gray_image = ((result_thi - np.nanmin(result_thi)) / (np.nanmax(result_thi) - np.nanmin(result_thi)) * 255).astype(np.uint8)

# # Apply Canny edge detection to the grayscale image
# edges = cv2.Canny(gray_image, threshold1=10, threshold2=255)



# # Create a grid of r, theta coordinates
# theta,r = np.meshgrid(tmid+np.pi,rmid)

# # Plot the values in polar coordinates with a polar projection
# fig, ax = plt.subplots()
# plt.subplot(111, projection='polar')
# plt.pcolormesh(theta, r, s, shading='auto', cmap='viridis')

# plt.scatter(tmid+np.pi,inner_r[:,1], marker='o', color='red')
# plt.scatter(tmid+np.pi,outer_r[:,1], marker='o', color='red')
# plt.colorbar(label='Values')  # Add a colorbar
# plt.title('2D Array Plot in Polar Coordinates')
# plt.gca().set_theta_zero_location('N')  # Set theta zero at the top (North)
# plt.gca().set_theta_direction(-1)  # Set theta increasing clockwise
# plt.tight_layout()
# plt.show()














# # Create step function for each row





# plt.figure()
# plt.plot(array1[:,3])
# plt.plot(smoothed_signal)


# # Compute the cross-correlation
# correlation = correlate(array1[:,3], array2[:,3])

# # Find the time lag that maximizes the correlation
# time_lag = np.argmax(correlation) - (len(array1) - 1)
# plt.figure()
# plt.plot(correlation/np.max(correlation)*255)
# plt.plot(array1[:,3])


# # Iterate over theta  to normalize result
# for thi in range(len(tmid)):
#     localmaxi     = (argrelextrema(result[:,thi], np.greater,order=int(rlen/2),axis=0))[0][1]
#     maxval        = np.max(array1[localmaxi-int(rlen*.2):localmaxi+int(rlen*.2),thi])
#     result_thi = result[:,thi]/np.max(result[localmaxi,thi])*maxval
    
    
#     plt.figure()
#     plt.plot(rmid,array1[:,thi])
#     plt.plot(rmid,result_thi)
    
#     # Find inner and outer ring distances
#     findind = np.where(array1[:,thi] > maxval*.65) - localmaxi
#     findind_pos = np.append(findind[np.where(findind >= 0)],1)
#     findind_neg = np.append((np.abs(findind[np.where(findind <= 0)]))[::-1],1)
#     inner_r[thi,1] = rmid[localmaxi - np.where((np.diff(findind_neg) != 1))[0][0]]
#     outer_r[thi,1] = rmid[localmaxi + np.where((np.diff(findind_pos) != 1))[0][0]]
    
    
#     minz = np.append(minz,rmid[(np.where((array1[:,thi] < 10) & (rmid > 0.6) & (rmid <0.8)))[0][0]])
    
# thh = 3
# inner_x = inner_r[:,1]*np.cos(tmid[:]+np.pi/2.)
# inner_y = -inner_r[:,1]*np.sin(tmid[:]+np.pi/2.)

# outer_x = outer_r[thh,1]*np.cos(tmid[thh]+np.pi/2.)
# outer_y = -outer_r[thh,1]*np.sin(tmid[thh]+np.pi/2.) 

# #x2 = minz*np.sin(tmid)
# #y2 = minz*np.cos(tmid) 


# fig, ax = opalfox.plotimage(image2,cmap='gray')
# ax.axis("on")
# #scatter = ax.scatter([500],[200], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+outer_x*cntr_index[0],cntr_index[1]+outer_y*cntr_index[1], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+inner_x*cntr_index[0],cntr_index[1]+inner_y*cntr_index[1], marker='o', color='red')
# #scatter = ax.scatter(cntr_index[1]+x2*cntr_index[1],cntr_index[0]+y2*cntr_index[0], marker='o', color='red')
# plt.show()

















# #%%

# from scipy.signal import convolve,correlate
# from scipy.signal import argrelextrema

# # Set binned stat
# array1 = np.copy(s)
# # Create step function for each row
# zeros_array = np.zeros(int(rlen/2))
# ones_array = np.ones(rlen)
# stepfunc   = np.concatenate((zeros_array,ones_array, zeros_array))


# array2     = np.repeat(stepfunc[:, np.newaxis],len(tmid),axis=1)
# # Perform the convolution
# result = convolve(array1, array2, mode='same')
# # Create inner and outer distance arrays
# inner_r = np.empty((len(tmid),2))
# outer_r = np.empty((len(tmid),2))
# minz = []


# from scipy.signal import savgol_filter,medfilt


# # Apply the Savitzky-Golay filter for smoothing
# window_length = rlen  # Adjust the window length as needed
# poly_order = 3  # Adjust the polynomial order as needed
# smoothed_signal = savgol_filter(array1[:,3], window_length, poly_order)
# smoothed_signal = medfilt(array1[:,3], kernel_size=window_length)


# plt.figure()
# plt.plot(array1[:,3])
# plt.plot(smoothed_signal)


# # Compute the cross-correlation
# correlation = correlate(array1[:,3], array2[:,3])

# # Find the time lag that maximizes the correlation
# time_lag = np.argmax(correlation) - (len(array1) - 1)
# plt.figure()
# plt.plot(correlation/np.max(correlation)*255)
# plt.plot(array1[:,3])


# # Iterate over theta  to normalize result
# for thi in range(len(tmid)):
#     localmaxi     = (argrelextrema(result[:,thi], np.greater,order=int(rlen/2),axis=0))[0][1]
#     maxval        = np.max(array1[localmaxi-int(rlen*.2):localmaxi+int(rlen*.2),thi])
#     result_thi = result[:,thi]/np.max(result[localmaxi,thi])*maxval
    
    
#     plt.figure()
#     plt.plot(rmid,array1[:,thi])
#     plt.plot(rmid,result_thi)
    
#     # Find inner and outer ring distances
#     findind = np.where(array1[:,thi] > maxval*.65) - localmaxi
#     findind_pos = np.append(findind[np.where(findind >= 0)],1)
#     findind_neg = np.append((np.abs(findind[np.where(findind <= 0)]))[::-1],1)
#     inner_r[thi,1] = rmid[localmaxi - np.where((np.diff(findind_neg) != 1))[0][0]]
#     outer_r[thi,1] = rmid[localmaxi + np.where((np.diff(findind_pos) != 1))[0][0]]
    
    
#     minz = np.append(minz,rmid[(np.where((array1[:,thi] < 10) & (rmid > 0.6) & (rmid <0.8)))[0][0]])
    
# thh = 3
# inner_x = inner_r[:,1]*np.cos(tmid[:]+np.pi/2.)
# inner_y = -inner_r[:,1]*np.sin(tmid[:]+np.pi/2.)

# outer_x = outer_r[thh,1]*np.cos(tmid[thh]+np.pi/2.)
# outer_y = -outer_r[thh,1]*np.sin(tmid[thh]+np.pi/2.) 

# #x2 = minz*np.sin(tmid)
# #y2 = minz*np.cos(tmid) 


# fig, ax = opalfox.plotimage(image2,cmap='gray')
# ax.axis("on")
# #scatter = ax.scatter([500],[200], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+outer_x*cntr_index[0],cntr_index[1]+outer_y*cntr_index[1], marker='o', color='red')
# scatter = ax.scatter(cntr_index[0]+inner_x*cntr_index[0],cntr_index[1]+inner_y*cntr_index[1], marker='o', color='red')
# #scatter = ax.scatter(cntr_index[1]+x2*cntr_index[1],cntr_index[0]+y2*cntr_index[0], marker='o', color='red')
# plt.show()


# fig,ax=plt.subplots()
# scatter = ax.scatter(outer_x,outer_y, marker='o', color='red')
# scatter = ax.scatter(inner_x,inner_y, marker='o', color='red')
# ax.set_xlim([-1,1])
# ax.set_ylim([-1,1])
# plt.show()




















# #%%
# import cv2
# import numpy as np

# # Global variables to store circle properties
# center = None
# radius = None
# drawing = False

# def draw_circle(event, x, y, flags, param):
#     global center, radius, drawing

#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Start drawing the circle
#         center = (x, y)
#         drawing = True

#     elif event == cv2.EVENT_LBUTTONUP:
#         # Finish drawing the circle
#         drawing = False
#         radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
#         cv2.circle(image, center, radius, (255, 0, 0), 2)

# image = image2

# # Create a window to display the image
# cv2.namedWindow('Draw Circle')

# # Set the mouse callback function
# cv2.setMouseCallback('Draw Circle', draw_circle)

# while True:
#     cv2.imshow('Draw Circle', image)

#     # Exit loop when 'q' is pressed
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break




# s[np.where(np.isnan(s))] = 0

# matrix = (np.diff(s,1,axis=0))

# # Find the indices of the last occurrence of 1 in each column
# last_occurrences = (matrix == -1)[::-1].cumsum(axis=0).argmax(axis=0)[::-1]

# rmid[(matrix == -1).cumsum(axis=0).argmax(axis=0)]
# x2 = rmid[(matrix == -1).cumsum(axis=0).argmax(axis=0)]*np.sin(tmid)
# y2 = rmid[(matrix == -1).cumsum(axis=0).argmax(axis=0)]*np.cos(tmid)


# fig, ax = opalfox.plotimage(image2,cmap='gray')
# ax.axis("on")
# #scatter = ax.scatter([500],[200], marker='o', color='red')
# scatter = ax.scatter(cntr_index[1]-x2*cntr_index[1],cntr_index[0]+y2*cntr_index[0], marker='o', color='red')
# plt.show()

# asasd
# z2[np.where(z2 < 20)] = 0
# # Plot the average z values against radial distance
# plt.figure(figsize=(10, 5))

# asdsa



# masked_array = np.ma.array (image_bright, mask=np.isnan(image_bright))
# cmap = plt.colormaps["gray"]
# cmap.set_bad('purple',1.)
# plt.figure(figsize=(24,16))
# plt.imshow(masked_array, interpolation='nearest', cmap=cmap)
# plt.show()
# dfth



# sz = np.shape(masked_array)
# cntr_index = [sz[0]/2.,sz[1]/2.,]

# x = np.linspace(1,-1, num=sz[0])
# y = np.linspace(-1, 1, num=sz[1])
# xx, yy = np.meshgrid(x, y)
# r = np.sqrt(xx**2 + yy**2)


# # Get the 2D indices of elements that satisfy the condition
# row_indices, col_indices = np.where((xx < 0) & (yy < 0))

# # Combine row and column indices into a 2D array of indices
# indices_2d = np.column_stack((row_indices, col_indices))

# plt.figure(figsize=(24,16))
# plt.imshow(image_bright[:,:], cmap='gray')
# plt.title("Original Image")
# plt.axis('on')

# np.max(r[np.where(np.isnan(image_bright))])


# xindex1 = round(cntr_index[0] - np.max(r[np.where(~np.isnan(image_bright))])*sz[0]/2.)
# xindex2 = round(cntr_index[0] + np.max(r[np.where(~np.isnan(image_bright))])*sz[0]/2.)
# yindex1 = round(cntr_index[1] - np.max(r[np.where(~np.isnan(image_bright))])*sz[1]/2.)
# yindex2 = round(cntr_index[1] + np.max(r[np.where(~np.isnan(image_bright))])*sz[1]/2.)

# b2 = image_bright[xindex1:xindex2,yindex1:yindex2]
# masked_array = np.ma.array (b2, mask=np.isnan(b2))
# cmap = plt.colormaps["gray"]
# cmap.set_bad('black',1.)
# plt.figure(figsize=(24,16))
# plt.imshow(masked_array, interpolation='nearest', cmap=cmap)

# plt.figure(figsize=(24,16))
# plt.imshow(b2, cmap='gray')
# plt.title("Original Image")
# plt.axis('on')

# # %%
# outerx = 800 #int(b_sz[0]/2)
# outery = 800 #int(b_sz[0]/2)
# # Find Radial Profiles for each quadrant of brightness
# b_sz = np.shape(brightness)
# quad1 = brightness[:int(b_sz[0]/2),int(b_sz[0]/2):]
# q1 = np.rot90(np.transpose(quad1))
# q1 = q1[0:outerx,0:outery]
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 2)
# plt.imshow(q1, cmap='gray')
# plt.title("QUAD 1")
# plt.axis('on')

# # Assuming you have the x, y, and z arrays for your grid
# x = np.linspace(0, 1, num=outerx)
# y = np.linspace(0, 1, num=outery)
# xx, yy = np.meshgrid(x, y)
# # Calculate the radial distance from the origin (0, 0) for each point
# radial_distance = np.sqrt(xx**2 + yy**2)
# theta_angle     = np.arctan2(yy,xx)
# # Convert theta to range 0 to 2*pi for histogram
# theta_angle     = theta_angle % (2 * np.pi)
# # Find the unique radial distances and their corresponding indices
# unique_radial_distances, indices = np.unique(radial_distance, return_inverse=True)
# # Calculate the average z values for each unique radial distance
# average_z_values = np.bincount(indices, weights=q1.ravel()) / np.bincount(indices)

# # Plot the average z values against radial distance
# plt.figure(figsize=(10, 5))
# plt.plot(unique_radial_distances, average_z_values)
# plt.xlabel('Radial Distance')
# plt.ylabel('Average z Value')
# plt.title('Average z Value vs. Radial Distance')
# plt.show()
