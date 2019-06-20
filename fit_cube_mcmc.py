import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
#import seaborn as sns


""" Script written for personal use.

    It creates moment maps as FITS files
    of SOFIA FIFI-LS spectral cubes.
    Fitting is done with a MCMC approach. """
        

def ReadMyCube(filename,info=1):
    
    """ Function to read FITS cube 
    returns array of spectra and header """
    
    print('Reading filename: ',filename)

    hdu_list = fits.open(filename)
    if info == 1:
        hdu_list.info()
    if len(hdu_list) > 1:
        image_data = hdu_list['FLUX'].data
        header = hdu_list['FLUX'].header
    else:
        image_data = hdu_list[0].data
        header = hdu_list[0].header

    print('Size of cube: ', image_data.shape)
   
    return image_data, header


def MakeWaveArray(header):
    
    """ Make wave array from header"""
    
    if header['NAXIS'] != 3:
        raise NameError('File dimension not 3, exiting')
    
    crval3 = header['CRVAL3']
    crpix3 = header['CRPIX3']
    cdelt3 = header['CDELT3']
    nv = header['NAXIS3']
    
    wave_array = np.array([crval3 + (i+1-crpix3) * cdelt3 for i in range(nv)])

    return wave_array


def MyModel(x_array,amp,cent,sig,a0,a1):
    
    model_gauss = amp*np.exp( -1*(x_array-cent)**2/2/sig**2)
    
    model_poly1 = a0 + x_array*a1
    
    return model_gauss+model_poly1


def WriteMomFits(filename,image_data,header):
    
    """ Write intensity and error maps
    as FITS files """
    
    if header['NAXIS'] == 2:
        print("Header already has 2 dimensions")
    elif header['NAXIS'] == 3:
        print("Changing Header from 3 to 2 dimensions")
        header['NAXIS'] = 2
        del header['NAXIS3']
        del header['CTYPE3']
        del header['CUNIT3']
        del header['CRPIX3']
        del header['CRVAL3']
        del header['CDELT3']
    else:
        raise NameError('Header has wrong dimensions')

    fits.writeto(filename, image_data, header, overwrite=True)
    
    return header


###
# MAIN script
###  


### SOURCE parameters
    
vgal = 40. #kms
redshift = vgal/2.998e5

wave_ref = 157.7409 #micron
spec_res = 250 ;280 #kms
spec_res = wave_ref *spec_res/2.998e5 #micron


### READ cube, velocity map; MAKE mask, wave array

sofia_dir = '/Users/dcormier/ITA/SOFIA_NGC6946/fifils_data/'
latest_dir = 'results_cf_19oct2017_atran12.8/'
#filename = sofia_dir+'F0202_FI_IFS_None_RED_WXY_00076-01402.fits'
filename = sofia_dir + latest_dir+'clean_cube_regridded_atran.fits'

cube, header = ReadMyCube(filename,info=1)

mask = cube.copy()
mask[np.isfinite(cube)] = 0
mask[~np.isfinite(cube)] = 1
mask[0:2,:,:] = 1
mask[-2:,:,:] = 1

wave_array = MakeWaveArray(header)
wave_array /= (1.+redshift)
wave_array -= wave_ref

vel_file = sofia_dir + 'ngc6946_vel_co21_15aspix.fits'
hdu_list = fits.open(vel_file)
hd_vel = hdu_list[0].header
map_vel = hdu_list[0].data


####################################
#
# SELECT PART OF THE CUBE FOR TESTING
#
print('Selecting part of the cube for testing!')

cube = cube[:,18,18]
mask = mask[:,18,18]
map_vel = map_vel[18,18]

print('new shape: ',cube.shape)
####################################

if cube.ndim==1:
    nrow, ncol = 1, 1
    nchan = len(cube)
else:
    nrow = len(cube[0,:,0])
    ncol = len(cube[0,0,:])
    nchan = len(cube[:,0,0])


# Initialize data variables to store

master_cube = []
master_wave = []
master_rms = []
master_idx = []
spec_idx = 0


# LOOP over pixels for data storing

for l in range(nrow):
    for m in range(ncol):
            
        ### Prepare data to fit
        
        if cube.ndim==1 :
            #single spectrum
            spectrum = cube
            spec_mask = mask
            vshift = map_vel
        else:
            spectrum = cube[:,l,m]
            spec_mask = mask[:,l,m]
            vshift = map_vel[l,m]
        
        if np.isfinite(vshift) == 0:
            vshift = 0.
        wshift = vshift *wave_ref/2.998e5
        spec_wave = wave_array - wshift

        ibase = (np.abs(spec_wave) > spec_res*1.7) & (spec_mask == 0)
        rms_spec = np.std( spectrum[ibase] )
        outlier = (np.abs(spec_wave) > spec_res*1.7) & (spec_mask == 0) \
            & (np.abs(spec_mask) > 5*rms_spec)
        spectrum[outlier] = np.nan

      #  spectrum[spec_mask == 1] = np.nan

        index = (np.abs(spec_wave) < spec_res*5) & (spec_mask == 0)
        spec_tmp_fit = spectrum[index] -np.median(spectrum[ibase])
        wave_tmp_fit = spec_wave[index]
                                    
        if (np.nansum(spec_tmp_fit) != 0):
            master_cube = np.append(master_cube,spec_tmp_fit)
            master_wave = np.append(master_wave,wave_tmp_fit)
            master_rms = np.append(master_rms,rms_spec*np.ones(len(spec_tmp_fit)))
            master_idx =np.append(master_idx,np.tile(spec_idx,len(spec_tmp_fit)))
            
        spec_idx += 1
        
n_data = len(master_cube)
n_pix = nrow*ncol
master_idx = master_idx.astype(int)


# Define model for MCMC fitting
                        
print('Defining model')
            
basic_model = pm.Model()

with basic_model:
                    
    # Priors for unknown model parameters
    spec_max = np.nanmax(master_cube)
    spec_min = np.nanmin(master_cube)
    wave_max = np.nanmax(master_wave)
    wave_min = np.nanmin(master_wave)
    slope_max = (spec_max-spec_min)/(wave_max-wave_min)
                                
    amp = pm.Uniform('amplitude',lower=0,upper=spec_max*1.2,shape=n_pix)
    cent = pm.Normal('centroid',mu=0,sd=spec_res/2.35,shape=n_pix)
    #cent = pm.Uniform('centroid',lower=-spec_res*0.7,upper=spec_res*0.7,shape=n_pix)
    sig = pm.Uniform('sigma',lower=spec_res*0.9/2.35, upper=spec_res*1.5/2.35,shape=n_pix)
    a0 = pm.Normal('intercept',mu=0,sd=np.median(master_rms),shape=n_pix)
    #a0 = pm.Uniform('intercept',lower=spec_min,upper=spec_max,shape=n_pix)
    a1 = pm.Normal('slope',mu=0,sd=slope_max/10.,shape=n_pix)
    #a1 = pm.Uniform('slope',lower=-slope_max,upper=slope_max,shape=n_pix)

    sigma_corr = pm.Normal('unc_cal',mu=0,sd=0.15)
                        
    # Expected value of outcome
    mu = amp[master_idx]*np.exp(-1*(cent[master_idx]-master_wave)**2/2/sig[master_idx]**2)+\
        a0[master_idx]+a1[master_idx]*master_wave
    mu *= (1.+sigma_corr)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=master_rms, observed=master_cube)

    # Model fitting
    #step = pm.Metropolis()
    n_draws = 2000 # (+500 for tuning) *2 chains
    trace = pm.sample(n_draws)

                                    
# Posterior analysis

print(pm.summary(trace).round(2)) 
pm.traceplot(trace);

#  print(pm.quantiles(trace['amplitude'])) 
#  pm.traceplot(trace, trace.varnames, \
#     priors=[getattr(model[n], 'distribution', None) \
#     for n in trace.varnames], combined=True)
#  tracedf1 = pm.trace_to_dataframe(trace, \
#              varnames=['amplitude', 'centroid', 'sigma'])
#  sns.pairplot(tracedf1);

this_idx = int(nrow*ncol/2)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(master_wave[master_idx ==this_idx], master_cube[master_idx ==this_idx], 'g-')
                
for ii in range(0,n_draws*2,100):
    spec_ii = MyModel(master_wave[master_idx ==this_idx],\
                trace['amplitude'][ii,this_idx],trace['centroid'][ii,this_idx],\
                trace['sigma'][ii,this_idx],trace['intercept'][ii,this_idx],\
                trace['slope'][ii,this_idx])
    ax.plot(master_wave[master_idx ==this_idx], spec_ii, 'C3', alpha=.1)
ax.set(xlabel='Wavelength', ylabel='Flux')
ax.legend(['observed data', 'mcmc'],loc=2)

# Extract best flux and error

fluxes = trace['amplitude']*trace['sigma']*np.sqrt(2.*np.pi) \
            / ((trace['centroid']+wave_ref)**2. *1e13)*29.98
best_flux = np.average(fluxes,axis=0)
best_flux_error = np.std(fluxes,axis=0)

#for this in np.unique(master_idx):
#    print('Best flux and error [%i]: %.2fe-16, %.2fe-16' % \
#          (this, best_flux[this]*1e16, best_flux_error[this]*1e16))
#    print('Flux limits (HPD 95): %.2fe-16, %.2fe-16' % \
#          (pm.hpd(fluxes)[this,0]*1e16, pm.hpd(fluxes)[this,1]*1e16))
#    print('Flux limits (HPD 66=1sig): %.2fe-16, %.2fe-16' % \
#          (pm.hpd(fluxes,alpha=0.34)[this,0]*1e16, pm.hpd(fluxes,alpha=0.34)[this,1]*1e16))

moments_fit = np.zeros((nrow,ncol),dtype=float)
moments_fit_err = np.zeros((nrow,ncol),dtype=float)

for l in range(nrow):
    for m in range(ncol):
        
        moments_fit[l,m] = best_flux[l*ncol+m]
        moments_fit_err[l,m] = best_flux_error[l*ncol+m]


### Update header and write out FITS
        
mywcs = wcs.WCS(header)
pixscale = wcs.utils.proj_plane_pixel_scales(mywcs)
pixscale = np.abs(pixscale[0]) #degree_per_pixel
fact = 1. / (pixscale*np.pi/180.)**2.
moments_fit *= fact
moments_fit_err *= fact

hd_mom = header
hd_mom['EXTNAME'] = 'Fitted flux'
hd_mom['BUNIT'] = 'W/m2/sr'
WriteMomFits('test_file.fits', moments_fit, hd_mom)

hd_mom['EXTNAME'] = 'Fitted error'
WriteMomFits('test_file_err.fits', moments_fit_err, hd_mom)


### END

