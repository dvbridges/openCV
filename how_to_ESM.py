import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import spline

def imageCapture(filename):
	"""
	Open image with openCV
	"""
	# open image
	mondrian = cv2.imread(filename,1)
	return mondrian

def defineRect(image, writeimage, filename):
	"""
	Define rectangle according to image:
		image: image - openCV image
		writeimage: bool - write image to file
		filename: string - filename for image
	"""

	# get image size
	height, width, channels = image.shape
	# print(width, height, channels)

	# # hsv = cv2.cvtColor(mondrian, cv2.COLOR_BGR2HSV)
	x_start, x_end = int(width/2)-80, int(width/2)+80, 
	y_start, y_end = int(height/2)-80, int(height/2)+80, 

	# Create rectangle in centre image
	rect = cv2.rectangle(image, (x_start,y_start),(x_end,y_end),(18,247,41),3,8)

	# Write image to disk
	if writeimage:
		cv2.imwrite(('rect'+filename), rect)
		# cv2.imshow('mask',rect)
	rectCoords = ((x_start,y_start),(x_end,y_end))

	# Return coordinates
	return rectCoords

def cropImage(image, coords, writeimage, filename):
	"""
	Crop image according to defineRect coordinates:
		coords: 1x2 tuple - x,y,x,y start, end coordinates, respectively
		writeimage: bool - write image to file
		filename: string - filename for image
	"""
	# Crop image
	cropped_image = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]

	# Write image
	if writeimage:
		cv2.imwrite(filename, cropped_image)
	
	# cv2.imshow('image',cropped_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return cropped_image

def getHSV(image, nbins, converted):
	"""
	Get hue, saturation and value from image histogram
		image: image - openCV image
		nbins: int - number of bins for histogram
		converted: bool - whether to scale bins as percentage of total pixels
	"""
	# Get hsv for cropped image
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hue =  cv2.calcHist([hsv], [0], None, [nbins], [0,180]).astype(int)
	hue_conversion = hue / hue.sum() # sum = number of pixels - creates percentage of total pixels per bin
	print('The number of pixels in the cropped image is 120*120, or {} pixels.'.format(hue.sum()))

	if converted:
		return hue_conversion
	return hue

def plotHSV(hist, smooth, smooth_val):
	"""
	Plot histogram
		hist: calcHist histogram
		nbins: int - number of bins for histogram
		converted: bool - whether to scale bins as percentage of total pixels
	"""
	# Create plot
	csfont ={'fontname': 'serif'}
	plt.rcParams["font.family"]='serif'
	plt.rcParams["font.size"]=14
	fig, ax1 = plt.subplots()

	# Set x locs
	x_ax=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
	# Create color map for bars
	colors = []
	for i in x_ax:
		colors.append(cm.hsv(i/float(13)))

	# Plot vals
	# Smooth line plot
	if smooth:
		xnew = np.linspace(x_ax.min(),x_ax.max(), smooth_val)
		power_smooth = spline(x_ax, hist, xnew )
		ax1.plot(xnew,power_smooth,color='black',lw = 3, ls = 'dashed',label = 'Volume')
	else:
		ax1.plot(x_ax,hist,color='black',lw = 3, ls = 'dashed', label = 'Volume')
	ax1.bar(range(1,13),hist,color = colors, align = 'center')

	# Format plot
	ax1.set_ylim(0,(max(hist)+2000))
	ax1.set_xlim(0,13)
	ax1.set_xticks(range(1,13))
	ax1.set_xlabel('Color Bin', **csfont)
	ax1.set_ylabel('Number of Pixels', **csfont)
	ax1.set_title('Hue histogram for {} pixels'.format(hist.sum()), size = 14)
	plt.legend(frameon = False)
	# plt.show()
	plt.savefig("color_fig.png")


def main():
	image = imageCapture('Mondrian_grid.jpg')
	rectCoords = defineRect(image, False, 'Mondrian_grid.jpg')
	cropped_image = cropImage(image, rectCoords, False, 'cropped.jpg')
	hsv = getHSV(cropped_image, 12, False)
	plotHSV(hsv, False, 12)


if __name__ == "__main__":
	main()