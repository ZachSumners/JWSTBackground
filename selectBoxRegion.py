import glob
from astropy.io import fits
from astropy.visualization import (ImageNormalize, MinMaxInterval, LogStretch, PercentileInterval, PowerStretch, SqrtStretch, SinhStretch)
import matplotlib.pyplot as plt
import numpy as np
from os import path
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import matplotlib.patches as patches
from matplotlib import colors
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import pandas as pd

#Access the calibrated datacube.
target = 'UCD-736'
filename = '/home/zach/Desktop/jwst/UCD736/pipeline_output_notebook/jw02567-o037_t036_nirspec_g235h-f170lp_s3d.fits'

hdul = fits.open(filename)
spectrumData = np.array(hdul['SCI'].data)
wavelength = np.linspace(hdul["SCI"].header["WAVSTART"], hdul["SCI"].header["WAVEND"], len(hdul["SCI"].data+1))*(10**6)
dqData = np.array(hdul['DQ'].data)

cubeshape = spectrumData.shape

def calc_mask(image, threshold):
	#Calculate the standard deviation of each image in the cube.
	sigma = np.nanstd(image, axis=(1, 2))
	#Calculate the median of these standard deviations for one unified standard deviation of the entire cube.
	sigmamedian = np.nanmedian(threshold*sigma)
	#Manipulate the shape of the sigma array to make it a 3D cube where all entries of x and y in one image are the same value (sigma of that image). We do this for masking purposes.
	sigma_broadcast = np.broadcast_to(sigma[:, np.newaxis, np.newaxis], image.shape)
	#Return the parts of the image that have pixel values less than a threshold as defined with sigma.
	pixelmask = image < threshold*sigma_broadcast
	maskedImage = image*pixelmask
	return maskedImage, sigmamedian

#CALCULATE THE BACKGROUND
def calcbackground(spectrumData, wavelength, threshold):
	#Wavelength range of interest (in microns)
	wav1 = 2.2
	wav2 = 2.4
	#Count how many pixels are being sampled in the background calculation.
	count = 0
	#Get the indicies of the wavelength range.
	cubeindex_min = (np.abs(wavelength-wav1)).argmin()
	cubeindex_max = (np.abs(wavelength-wav2)).argmin()

	#The mask determines what pixels are used to calculate the background with. True means that pixel is used.
	mask = np.zeros((spectrumData[10].shape), dtype = bool)
	#Iterate through every pixel in the image and see if it satifies the condition.
	for i in range(len(spectrumData[500])):
		for j in range(len(spectrumData[500][i])):
			# (i = 17, j = 21) is the central pixel.
			#The condition is this.
			if ((j < 9 and j > 3) and (i > 3 and i < 17)) or ((j > 32 and j < 38) and (i > 3 and i  < 10)):
				mask[i][j] = True
				count += 1
	#This function returns entries of the original cube that are below a threshold. Use this cube to calculate the background. This function essentially masks bright sources.
	spectrumDataMasked, sigma = calc_mask(spectrumData, threshold)
	#Grab entries of the masked cube that are inside the conditions as defined by the loop above.
	backgroundannulus = spectrumDataMasked[:, mask]

	backgroundspectra = []
	#For every image in the sampled cube, find the background.
	for i in range(len(wavelength)):
		medianvalue = np.nanmedian(backgroundannulus[i])
		backgroundspectra.append(medianvalue)

	#Convert nans to 0 so addition doesn't break later.
	backgroundspectra = np.nan_to_num(backgroundspectra, nan=0)
	
	#Fit a straight line to the background spectra in the wavelengths of interest. Exclude regions with nans (now zeros) because they don't have any signal.
	fittingmask = backgroundspectra[cubeindex_min:cubeindex_max] != 0
	bkgfit = linregress(wavelength[cubeindex_min:cubeindex_max][fittingmask], backgroundspectra[cubeindex_min:cubeindex_max][fittingmask])
	
	df = pd.DataFrame(data = {'Wavelength': wavelength[cubeindex_min:cubeindex_max], 'BackgroundSpectra': backgroundspectra[cubeindex_min:cubeindex_max]})
	df.to_csv(f'/home/zach/Desktop/jwst/BackgroundSpectra/backgroundSpectra_CustomBox_Threshold{int(threshold*10)}.csv')
	
	return bkgfit, backgroundspectra, mask, spectrumDataMasked, sigma, count


#SUBTRACT BACKGROUND

def subtractBackground(wavelength, spectrumData, backgroundspectra, bkgfit):
	#Wavelength range of interest (in microns)
	wav1 = 2.2
	wav2 = 2.4
	#Get the indicies of the wavelength range.
	cubeindex_min = (np.abs(wavelength-wav1)).argmin()
	cubeindex_max = (np.abs(wavelength-wav2)).argmin()

	#Convert nans in datacube to zero so addition doesn't break.
	spectrumDataNonNan = np.nan_to_num(spectrumData, nan=0)
	#Sum the spectra in the specified wavelength range for 2D viewing purposes. (We are summing the "third" dimension/slices into one value).
	TwoDimFlatten = np.sum(spectrumDataNonNan[cubeindex_min:cubeindex_max, :, :], axis=0)

	croppedCube = spectrumData[:, round(cubeshape[1]/2)-6:round(cubeshape[1]/2)+6, round(cubeshape[2]/2)-6:round(cubeshape[2]/2)+6]

	centralpixCroppedCube = np.unravel_index(np.argmax(croppedCube[cubeindex_min+1, :, :]), croppedCube[cubeindex_min+1, :, :].shape)
	centralpix = (centralpixCroppedCube[0] + round(cubeshape[1]/2)-6, centralpixCroppedCube[1] + round(cubeshape[2]/2)-6)
	#centralpix = (15, 15)
	centralpixSpectrum = spectrumData[:, centralpix[0], centralpix[1]]
	
	subtracted = centralpixSpectrum[cubeindex_min:cubeindex_max]-backgroundspectra[cubeindex_min:cubeindex_max]
	
	return centralpixSpectrum, cubeindex_min, cubeindex_max, TwoDimFlatten, subtracted

def plotting(wavelength, cubeindex_min, cubeindex_max, bkgfit, cubeshape, TwoDimFlatten, dqData, mask, spectrumDataMasked, threshold, centralpixSpectrum, backgroundspectra, subtracted, count):
	
	spectrumDataMasked = np.nan_to_num(spectrumDataMasked)
	spectrumDataMasked = np.sum(spectrumDataMasked[cubeindex_min:cubeindex_max, :, :], axis=0)
	
	#Figure initialization.
	gs = GridSpec(2, 4, hspace=0.3, width_ratios = [1, 1, 1, 1.5])
	cmapred = colors.ListedColormap(['red'])
	norm = ImageNormalize(TwoDimFlatten, vmin=0, vmax=60000, stretch=LogStretch())
	norm2 = ImageNormalize(spectrumDataMasked, interval=PercentileInterval(90), stretch=SinhStretch())
	cmap = colors.ListedColormap(['mistyrose'])

	#Plot initializations.
	my_dpi = 100
	fig = plt.figure(figsize=(1469/my_dpi, 871/my_dpi), dpi=my_dpi)
	axs1 = fig.add_subplot(gs[1, :-1])
	axs2 = fig.add_subplot(gs[0, :-1])
	axs3 = fig.add_subplot(gs[0, -1])
	axs4 = fig.add_subplot(gs[1, -1])
	
	#Mark the pixels that have bad quality flags.
	dqFlags = np.sum(dqData[cubeindex_min:cubeindex_max, :, :], axis=0)
	dqFlags = np.ma.masked_where(dqFlags < 512*400, dqFlags, 0)

	#Plot the 1D spectra in the wavelength region of interest.
	axs1.plot(wavelength[cubeindex_min:cubeindex_max], centralpixSpectrum[cubeindex_min:cubeindex_max], label='Original')
	axs1.plot(wavelength[cubeindex_min:cubeindex_max], subtracted, label='Background Subtracted Spectra')

	axs2.plot(wavelength[cubeindex_min:cubeindex_max], bkgfit[0]*wavelength[cubeindex_min:cubeindex_max] + bkgfit[1], label='Background Linear Fit', color='black')
	axs2.plot(wavelength[cubeindex_min:cubeindex_max], backgroundspectra[cubeindex_min:cubeindex_max], label='Background Spectra')
	
	#Plot the original datacube with the 3rd dimension summed up.
	im3 = axs3.imshow(TwoDimFlatten, norm=norm)
	fig.colorbar(im3, ax=axs3)
	
	#Overlay the region sampled for background calculation in red.
	mask = mask.astype(int)
	mask = np.where(mask==0, np.nan, mask)
	normMask = ImageNormalize(mask, vmin=0, vmax=1)
	axs3.imshow(mask, alpha=0.3, cmap=cmapred)
	
	#Plot the bright source masked datacube with the 3rd dimension summed up.
	im4 = axs4.imshow(spectrumDataMasked, norm=norm2)
	fig.colorbar(im4, ax=axs4)
	
	#Overlay the region sampled for background calculation in red.
	axs4.imshow(mask, alpha=0.3, cmap=cmapred)
	
	#Overlay the bad pixels in a light pink.
	axs4.imshow(dqFlags, cmap=cmap)
	axs3.imshow(dqFlags, cmap=cmap)
	
	#Plotting labels, titles and cosmetic changes.
	axs2.set_ylim(-1, 1)
	axs1.set_xlim(2.19, 2.41)
	axs2.set_xlim(2.19, 2.41)
	axs2.set_title(f'R-Squared value: {bkgfit[2]**2}')
	axs1.set_ylabel('Flux')
	axs2.set_ylabel('Flux')
	axs1.set_xlabel('Wavelength (microns)')
	axs1.set_title('Central Pixel Spectrum')
	axs3.invert_yaxis()
	axs4.invert_yaxis()
	axs1.grid()
	axs2.grid()
	axs1.legend()
	axs2.legend()
	text=fig.text(0.5, 0.02, f'Image Size: ({cubeshape[2]}, {cubeshape[1]}), Custom Box, Threshold: {threshold}, Count: {count}', horizontalalignment = 'center')

	fig.suptitle('UCD-736 BACKGROUND SUBTRACTED')
	#plt.show()
	plt.savefig(f'/home/zach/Desktop/jwst/Figures/BoxMedian/CustomBox_Threshold{int(threshold*10)}.png')
	fig.clf()
	plt.close()

threshold_list = np.arange(0.1, 1.1, 0.1)

r_values = []
thresholds = []
sigmas = []
counts = []
typeoperation = []

for threshold in threshold_list:
	print(f'Image Size: ({cubeshape[2]}, {cubeshape[1]}), Box: Custom, Masking Threshold: {threshold}')
		
	bkgfit, backgroundspectra, mask, spectrumDataMasked, sigma, count = calcbackground(spectrumData, wavelength, threshold)
	centralpixSpectrum, cubeindex_min, cubeindex_max, TwoDimFlatten, subtracted = subtractBackground(wavelength, spectrumData, backgroundspectra, bkgfit)
	plotting(wavelength, cubeindex_min, cubeindex_max, bkgfit, cubeshape, TwoDimFlatten, dqData, mask, spectrumDataMasked, threshold, centralpixSpectrum, backgroundspectra, subtracted, count)
	
	r_values.append(bkgfit[2]**2)
	thresholds.append(threshold)
	sigmas.append(sigma)
	counts.append(count)
	typeoperation.append('Median')
				
	
df = pd.DataFrame(data = {'R Values': r_values, 'Thresholds': thresholds, 'Sigma': sigmas, 'Pixels': counts, 'Background Method': typeoperation})
df.to_csv('/home/zach/Desktop/jwst/UCD736/backgroundParamsBoxMedianCustom.csv')
