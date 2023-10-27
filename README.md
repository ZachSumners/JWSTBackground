# JWSTBackground
This project explores the James Webb Space Telescope data reduction pipeline. This code accepts calibrated JWST datacubes and does a secondary background subtraction with custom noise and selection region parameters. Pixels to be included in the background are selected with either a circle or rectangle, with flexible shape and exclusion regions (can remove areas of the circle/box if needed). Bright regions are masked before background calculations, as defined by the "threshold" parameter.

The scripts output plots to compare the background spectra with the central (brightest) pixel. 

Most code in this project is based off these two scripts with slight modifications, so I only upload the above for simplicity.
