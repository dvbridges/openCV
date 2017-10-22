import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import spline

# open image
mondrian = cv2.imread('Mondrian_grid.jpg',1)

# get image size
height, width, channels = mondrian.shape
# print(width, height, channels)

# # hsv = cv2.cvtColor(mondrian, cv2.COLOR_BGR2HSV)
x_start, x_end = int(width/2)-60, int(width/2)+60, 
y_start, y_end = int(height/2)-60, int(height/2)+60, 

# Create rectangle in centre image
rect = cv2.rectangle(mondrian, (x_start,y_start),(x_end,y_end),(18,247,41),3,8)
# cv2.imshow('mask',rect)

# Show cropped image

cropped_image = mondrian[y_start:y_end,x_start:x_end]
# cv2.imshow('image',cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Get hsv for cropped image
hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
hue =  cv2.calcHist([hsv], [0], None, [12], [0,180]).astype(int)
hue_conversion = hue / hue.sum() # sum = number of pixels - creates percentage of total pixels per bin
print('The number of pixels in the cropped image is 120*120, or {} pixels.'. format(hue.sum()))

# for i in hue:
# 	for j in i:
# 		print(j)

# plt.bar(range(0,12), hue_conversion)
# plt.show()

color = ['b','g','r']
labels = ['h','s','v']
hsv_range = [180,255,255]

fig, ax1 = plt.subplots()

# for i,col in enumerate(color):
# 	histr=cv2.calcHist([hsv], [i], None, [12], [0,hsv_range[i]])
# 	ax1.plot(range(1,13),histr,color=col,label=labels[i], lw = 3)
# 	ax1.legend()
# plt.colorbar(cmap = plt.imshow(mondrian), orientation = 'horizontal', ax=ax1)

# Set x locs
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# Create color map for bars
colors = []
for i in x:
	colors.append(cm.hsv(i/float(13)))
# Smooth line plot
xnew = np.linspace(x.min(),x.max(), 15)
power_smooth = spline(x, hue, xnew )

# Plot hue values
ax1.plot(xnew,power_smooth,color='black',lw = 3, ls = 'dashed')
# ax1.plot(x,hue,color='black',lw = 3, ls = 'dashed')
ax1.bar(range(1,13),hue,color = colors, align = 'center')

# Format plot
csfont ={'fontname': 'serif'}
plt.rcParams["font.family"]='serif'
ax1.set_ylim(0,7000)
ax1.set_xlim(0,13)
ax1.set_xticks(range(1,13))
ax1.set_xlabel('Color Bin', size = 14, **csfont)
ax1.set_ylabel('Number of Pixels', size = 14,**csfont)
# plt.show()

plt.savefig("color_fig.png")