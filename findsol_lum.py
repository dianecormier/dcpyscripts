#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import numpy as np
import pandas as pd
from scipy.io.idl import readsav
from astropy.io import ascii
from astropy.table import Table, join
import time


def go_get_obs():
    
    dir_in = '/Users/dcormier/HERSCHEL/DGS/'

    data_init = ascii.read(dir_in+'spectro_files/dgs_all_pacs.ascii',
            format='fixed_width',header_start=None,
            col_starts=(0,11,21,31),
            data_start=0,
            names=('object','dist','metallicity','dummy'),
            )
    
    data_init['metallicity'] = 10**(data_init['metallicity']-(12-3.4962))
    

    data_obs = ascii.read('./data_individual_values_abs_2018.dat',
            delimiter='|',guess=False,
            data_start=0,
            names=('object',\
                   'OIII88','e_OIII88','NIII57','e_NIII57',\
                   'NII122','e_NII122','NeIII15','e_NeIII15',\
                   'NeII12','e_NeII12','SIV10','e_SIV10',\
                   'SIII18','e_SIII18','SIII33','e_SIII33',\
                   'CII157','e_CII157','SiII34','e_SiII34',\
                   'OIV26','e_OIV26','ArII7','e_ArII7',\
                   'ArIII9','e_ArIII9','FeII17','e_FeII17',\
                   'FeII25','e_FeII25','FeIII23','e_FeIII23',\
                   'ArIII21','e_ArIII21','Hua','e_Hua',\
                   'NII205','e_NII205','OI63','e_OI63',\
                   'OI145','e_OI145','H2S0','e_H2S0',\
                   'H2S1','e_H2S1','H2S2','e_H2S2',\
                   'H2S3','e_H2S3','NeV14','e_NeV14',\
                   'NeV24','e_NeV24'),
            )

    #- read in distance_ha, LBOL, LTIR
    data_ha = ascii.read(dir_in+'phot_files/DGS_Halpha_masses.txt',
            format='fixed_width',header_start=None,
            col_starts=(0,15,30,45),
            data_start=2,
            names=('object','dist_ha','LHa','e_LHa'),
            )

    data_lbol = ascii.read(dir_in+'phot_files/DGS_L_BOL_distribution.txt',
            format='fixed_width',header_start=None,
            col_starts=(0,11,27),
            data_start=3,
            names=('object','LBOL','e_LBOL'),
            )

    data_ltir = ascii.read(dir_in+'phot_files/DGS_L_TIR_distribution.txt',
            format='fixed_width',header_start=None,
            col_starts=(0,15,30,45),
            data_start=1,
            names=('object','LTIR','e_LTIR','log_LTIR'),
            )
    
    for i in data_obs['object']:
        new_ind = np.flatnonzero(np.core.defchararray.find(data_lbol['object'],i)!=-1)
        if new_ind.any() : data_lbol['object'][new_ind[0]] = i
        new_ind = np.flatnonzero(np.core.defchararray.find(data_ltir['object'],i)!=-1)
        if new_ind.any() : data_ltir['object'][new_ind[0]] = i
        new_ind = np.flatnonzero(np.core.defchararray.find(data_ha['object'],i)!=-1)
        if new_ind.any() : data_ha['object'][new_ind[0]] = i


  ## read in optical lines (normalized to Hbeta) and abundances
    data_opt = ascii.read('./dgs_fluxes_optical.txt',
            delimiter='|',guess=False,
            data_start=0,
            names=('object','OII3727','e_OII3727','Hbeta','e_Hbeta',\
                   'OIII4959','e_OIII4959','OIII5007','e_OIII5007',\
                   'Ha','e_Ha','NII6584','e_NII6584','SII6716','e_SII6716',\
                   'SII6731','e_SII6731','SIII9069','e_SIII9069',\
                   'SIII9532','e_SIII9532','HeI10829','e_HeI10829'),
            )

    data_ab_obs = ascii.read(dir_in+'DGS_modeling/literature_abundances.txt',
            delimiter='\t',guess=False,
            data_start=0,
            names=('object','O/H','C/H','S/H','N/H',\
                   'Ne/H','Si/H','Fe/H','Ar/H') \
            )
    data_ab_obs['O/H'] -= 12.
    for i in ['C/H','S/H','N/H','Ne/H','Si/H','Fe/H','Ar/H']:
        data_ab_obs[i] += data_ab_obs['O/H']


    
    #- add everytihng together
    master_obs = join(join(join(join(join(join(data_obs, \
                        data_init, join_type='left', keys='object'),\
                        data_ha, join_type='left', keys='object'),\
                        data_ltir, join_type='left', keys='object'),\
                        data_lbol, join_type='left', keys='object'),\
                        data_opt, join_type='left', keys='object'),\
                        data_ab_obs, join_type='left', keys='object')
                
    master_obs['LBOL'] *= 3.846e26/(4*np.pi* (master_obs['dist_ha']*1e6*3.086e16)**2.)
    master_obs['e_LBOL'] *= 3.846e26/(4*np.pi* (master_obs['dist_ha']*1e6*3.086e16)**2.)
    master_obs['LTIR'] *= 3.846e26/(4*np.pi* (master_obs['dist_ha']*1e6*3.086e16)**2.)
    master_obs['e_LTIR'] *= 3.846e26/(4*np.pi* (master_obs['dist_ha']*1e6*3.086e16)**2.)
    master_obs['e_LTIR'] += 0.5*master_obs['LTIR']
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    master_obs['e_LBOL'] = np.nanmax([master_obs['e_LBOL'],abs(master_obs['LBOL']-master_obs['LTIR'])],axis=0)


    master_obs['LHa'] = 10**(master_obs['LHa']-40.)
    master_obs['LHa'] *= 1e-7 /(4*np.pi*1e32*1e-40*(3.0857*master_obs['dist_ha']*1e6)**2.) #erg/s to w/m2
    
    return master_obs




def findsol( \
    galname     = 'no_name', \
    useopt      = 0, \
    usesecond   = 0, \
    usemain     = 0, \
    useltot     = 0, \
    setlown     = 0, \
    usepdr      = 0, \
    useltir     = 0, \
    useh2       = 0, \
    set_model   = 0, \
    singlepdr   = 0, \
    nocii       = 0, \
    noneiii     = 0, \
    nosiv       = 0, \
    no63        = 0, \
    relax       = 0, \
    specific    = 0 \
            ):


    start = time.time()
    np.warnings.filterwarnings('ignore') 
    
#%%%%%%%%%%%%%%%%%%%%
#%%%   Select model grid
#%%%%%%%%%%%%%%%%%%%%

    dir_cloudy = '/Users/dcormier/PDR/Cloudy/c17.00/dcormier/codark/'

    if set_model == 0 : mygrid = 'grid_cont_av5_full/'
    elif set_model == 1 : mygrid = 'grid_cont_av5_full_xrcr/'
    elif set_model == 2 : mygrid = 'grid_inst_av5_sub/'
    elif set_model == 3 : mygrid = 'grid_magphys_av5_sub/'
    elif set_model == 4 : mygrid = 'grid_popstar_av5_sub/'
    elif set_model == 5 : mygrid = 'grid_cont_av5_sub_cstpress/'
    elif set_model == 6 : mygrid = 'grid_cont_av5_sub_cstden/'
    elif set_model == 7 : mygrid = 'grid_cont_av5_sub_lowdgr/'
    elif set_model == 8 : mygrid = 'grid_cont_av5_sub_lowpah/'
    elif set_model == 9 : mygrid = 'grid_cont_av5_sub_vturb0/'
    elif set_model == 10 : mygrid = 'grid_cont_av5_sub_vturb5/'
    else:
        print('No matching model grid... returning')
        return

    dirmodel = dir_cloudy+mygrid

    dir_out = 'plots_individual_abs_' + mygrid
    if (os.path.isdir(dir_out)) == 0: os.mkdir(dir_out)
    
#%%%%%%%%%%%%%%%%%%%%


    this_append = ''
    if (usemain) : this_append += '_HII'
    if (usesecond) : this_append+='_second'
    if (usepdr) : this_append += '_PDR'
    if (useltir) : this_append += '_TIR'
    if (useh2) : this_append += '_H2'
    if (singlepdr) : this_append += '_one'
    if (nocii) : this_append += '_nocii'
    if (noneiii) : this_append += '_noneiii'
    if (nosiv) : this_append += '_nosiv'
    if (no63) : this_append += '_no63'
    if (setlown) : this_append += '_lown'
    if (relax) : this_append += '_relax'
    if (specific) : this_append += '_specific'

    if os.path.isfile(dir_out+'results'+this_append+'.sav2') :
        print('yes')
        this_struct = readsav(dir_out+'results'+this_append+'.sav')
        results = this_struct["results"]
    else :
        results = np.recarray((50,), dtype=[\
                              ('this_object','U10'), ('ndet','i4'), \
                              ('mod_n',np.float32), ('mod_npdr',np.float32), \
                              ('mod_u',np.float32), ('mod_rin',np.float32), \
                              ('mod_age',np.float32), ('mod_g',np.float32), \
                              ('mod_cov',np.float32),('mod_chi',np.float32), \
                              ('mod_chired',np.float32),('mod_inii',np.float32), \
                              ('mod_nii_ratio',np.float32),('mod_cii',np.float32), \
                              ('mod_sil',np.float32),('mod1_n',np.float32), \
                              ('mod1_npdr',np.float32),('mod1_u',np.float32), \
                              ('mod1_rin',np.float32),('mod1_age',np.float32), \
                              ('mod1_g',np.float32),('mod1_cov',np.float32), \
                              ('mod2_n',np.float32),('mod2_npdr',np.float32), \
                              ('mod2_u',np.float32),('mod2_rin',np.float32), \
                              ('mod2_age',np.float32),('mod2_g',np.float32), \
                              ('mod2_cov',np.float32),('mod1_chi',np.float32), \
                              ('mod1_chired',np.float32),('mod1_inii',np.float32),\
                              ('mod1_nii_ratio',np.float32),('mod1_cii',np.float32), \
                              ('mod1_sil',np.float32),('mod1_fc',np.float32),\
                              ('mod1_fs',np.float32),('mod_tfir',np.float32), \
                              ('mod1_tfir',np.float32),('mod2_tfir',np.float32),\
                              ('fcii_neutral_croxall',np.float32,(3)) \
                              ])


    
    
#%%%%%%%%%%%%%%%%%%%%
    #- go get observations and select galaxy!
#%%%%%%%%%%%%%%%%%%%%
    master_obs = go_get_obs()
    print('*****')
    if galname not in master_obs['object'] :
        print('Galaxy observation data not found, returning...')
        print('*****')
        return
    else:
        print('Running galaxy: ' + galname)
        print('*****')
    master_obs = master_obs[master_obs['object'] == galname]







#======================
# Start by reading of model predictions and observed ratios
#======================
    rmod_labels = ['OIII88', \
                'NIII57', \
                'NII122', \
                'NeIII15', \
                'NeII12', \
                'SIV10', \
                'SIII18', \
                'SIII33', \
                'CII157', \
                'SiII34', \
                'OIV26', \
                'ArII7', \
                'ArIII9', \
                'FeII17', \
                'FeII25', \
                'FeIII23', \
                'ArIII21', \
                'LTOT', \
                'Hua', \
                'NII205', \
                'OI63', \
                'OI145', \
                'LTIR', \
                'H2S0', \
                'H2S1', \
                'H2S2', \
                'H2S3', \
                #
                'OII3727', \
                'Hb', \
                'OIII4959', \
                'OIII5007', \
                'Ha', \
                'NII6584', \
                'SII6716', \
                'SII6731', \
                'SIII9069', \
                'SIII9532']


    
    i_to_fit = []
    if (usemain) :
        i_to_fit += ['OIII88','NIII57','NII122','NeII12','SIII18','SIII33',\
                     'ArII7','ArIII9','NII205','Hua']
        if (noneiii == 0) : i_to_fit += ['NeIII15']
        if (nosiv == 0) : i_to_fit += ['SIV10']
    if (usesecond) : i_to_fit += ['FeIII23']
    if (usepdr) :
        i_to_fit += ['OI145']
        if (nocii == 0) : i_to_fit += ['CII157']
        if (no63 == 0) : i_to_fit += ['OI63']
        if (useh2) : i_to_fit += ['H2S0','H2S1','H2S2','H2S3']
        if (usesecond) : i_to_fit += ['SiII34','FeII17','FeII25']
    if (useltir) : i_to_fit += ['LTIR']
    if (specific) : i_to_fit = ['OIII88','NeIII15','CII157','LTIR']
    i_to_check = []
    if (usemain) and (usepdr == 0) : i_to_check += ['CII157']
    if (nocii) and (usepdr) : i_to_check += ['CII157']
    if (usesecond) and (usepdr == 0) : i_to_check += ['SiII34','FeII17','FeII25']
    if (specific) : i_to_check = ['SIV','SIII18','SIII33']
    i_to_fit_opt = []
    if (useopt) : i_to_fit_opt += ['OII3727','OIII4959','OIII5007','Ha',\
                                       'NII6584','SIII9069','SIII9532']
    i_to_check_opt = []
    if (useopt) : i_to_check_opt += ['SII6716','SII6731']
  

    this_struct = readsav(dirmodel+'modelfit.sav',verbose=1)

  # without [0], all are objetcs!
    n = this_struct.modelfit.hden[0]
    npdr = this_struct.modelfit.n_pdr_2[0]
    age = this_struct.modelfit.star[0]
    u = this_struct.modelfit.u_param[0]
    rin = this_struct.modelfit.radius[0]
    g_param = this_struct.modelfit.g0_param[0]
    metal = this_struct.modelfit.ff[0]
    label_arr = this_struct.modelfit.label_arr[0].astype('U13')

    label_arr[label_arr == 'Halpha'] = 'Ha'
    label_arr[label_arr == 'Hbeta'] = 'Hb'
    label_arr[label_arr == 'HIhua'] = 'Hua'
    label_arr[label_arr == 'SII_1'] = 'SII6716'
    label_arr[label_arr == 'SII_2'] = 'SII6731'
    label_arr[label_arr == 'SIII_1'] = 'SIII9069'
    label_arr[label_arr == 'SIII_2'] = 'SIII9532'
    label_arr[label_arr == 'OII_1'] = 'OII3727'
    label_arr[label_arr == 'OII_2'] = 'OII3727'
    label_arr[label_arr == 'OIII_1'] = 'OIII4959'
    label_arr[label_arr == 'OIII_2'] = 'OIII5007'
    label_arr[label_arr == 'NII_1'] = 'NII6584'


    metal_sorted = sorted(metal)
    metal_uniq = np.unique(metal_sorted)

    intens_arr = np.transpose(this_struct.modelfit.intens_arr[0])
    intens_arr_ion = np.transpose(this_struct.modelfit.intens_arr_ion[0])
    intens_arr_if = np.transpose(this_struct.modelfit.intens_arr_if[0])

    sizes = this_struct.modelfit.sizes[0]
    
    tir_lum = this_struct.modelfit.fir_lum[0]
    tot_lum = this_struct.modelfit.tot_lum[0]

    tir_lum = 10**tir_lum
    tot_lum = 10**tot_lum


    nl = len(label_arr)+2 #number of lines+2 for ltir,ltot

    if os.path.isfile(dir_out+'perform'+this_append+'.sav2') :
        print('yes')
        perform = readsav(dir_out+'perform'+this_append+'.sav')
    else :
        perform = np.recarray((50,),dtype=[\
                              ('this_object','U10'), \
                              ('i_fit',np.float32,(nl)), \
                              ('ipred',np.float32,(nl,3)), \
                              ('iperf',np.float32,(nl,2)), \
                              ('iperf_labels','U10',(nl)) \
                              ])

    


#%%%%%%%%%%%%%%%%%%%%
  #-- read in model abundances
#%%%%%%%%%%%%%%%%%%%%
    data_ab_mod = ascii.read('/Users/dcormier/HERSCHEL/DGS/DGS_modeling/grid_abundances.txt',
            delimiter='\t',guess=False,
            data_start=0,
            names=('Zbin','O/H','C/H','N/H','Ne/H',\
                   'S/H','Cl/H','Ar/H','Fe/H','Si/H') \
            )
   ## scale model abundances: to observed if possible, to finer grid otherwise
   ## this should be done for ALL elements...
   ## we should have a function with interpolation for this...
    this_metal = metal_uniq[argmin(abs(master_obs['metallicity']-metal_uniq))]
    i_scale = argmin(abs(master_obs['metallicity']-10**data_ab_mod['Zbin']))
    i_ref = argmin(abs(this_metal-10**data_ab_mod['Zbin']))
    if np.isfinite(master_obs['O/H']):
        scale_o = 10**(master_obs['O/H']-data_ab_mod['O/H'][i_ref])
    else:
        scale_o = 10**(data_ab_mod['O/H'][i_scale]-data_ab_mod['O/H'][i_ref])
    if np.isfinite(master_obs['C/H']):
        scale_c = 10**(master_obs['C/H'][i_scale]-data_ab_mod['C/H'][i_ref])
    else :
        scale_c = 10**(data_ab_mod['C/H'][i_scale]-data_ab_mod['C/H'][i_ref])
    if np.isfinite(master_obs['N/H']):
        scale_n = 10**(master_obs['N/H'][i_scale]-data_ab_mod['N/H'][i_ref])
    else:
        scale_n = 10**(data_ab_mod['N/H'][i_scale]-data_ab_mod['N/H'][i_ref])
    if np.isfinite(master_obs['Ne/H']):
        scale_ne = 10**(master_obs['Ne/H'][i_scale]-data_ab_mod['Ne/H'][i_ref])
    else:
        scale_ne = 10**(data_ab_mod['Ne/H'][i_scale]-data_ab_mod['Ne/H'][i_ref])
    if np.isfinite(master_obs['S/H']):
        scale_s = 10**(master_obs['S/H'][i_scale]-data_ab_mod['S/H'][i_ref])
    else:
        scale_s = 10**(data_ab_mod['S/H'][i_scale]-data_ab_mod['S/H'][i_ref])
    if np.isfinite(master_obs['Ar/H']):
        scale_ar = 10**(master_obs['Ar/H'][i_scale]-data_ab_mod['Ar/H'][i_ref])
    else:
        scale_ar = 10**(data_ab_mod['Ar/H'][i_scale]-data_ab_mod['Ar/H'][i_ref])
    if np.isfinite(master_obs['Fe/H']):
        scale_fe = 10**(master_obs['Fe/H'][i_scale]-data_ab_mod['Fe/H'][i_ref])
    else:
        scale_fe = 10**(data_ab_mod['Fe/H'][i_scale]-data_ab_mod['Fe/H'][i_ref])
    if np.isfinite(master_obs['Si/H']):
        scale_si = 10**(master_obs['Si/H'][i_scale]-data_ab_mod['Si/H'][i_ref])
    else:
        scale_si = 10**(data_ab_mod['Si/H'][i_scale]-data_ab_mod['Si/H'][i_ref])


  ## select grid of given metallicity bin
    n = n[metal == this_metal]
    npdr = npdr[metal == this_metal]
    age = age[metal == this_metal]
    rin = rin[metal == this_metal]
    u = u[metal == this_metal]
    g_param = g_param[metal == this_metal]

    intens_arr = intens_arr[:,metal == this_metal]
    intens_arr_ion =intens_arr_ion[:,metal == this_metal]
    intens_arr_if =intens_arr_if[:,metal == this_metal]
    sizes = sizes[:,metal == this_metal]
    tir_lum = tir_lum[metal == this_metal]
    tot_lum = tot_lum[metal == this_metal]

    metal = metal[metal == this_metal]

    metal_sorted = sorted(metal)
    metal_uniq = np.unique(metal_sorted)
    nmetalbins = len(metal_uniq)

    age_sorted = sorted(age)
    age_uniq = np.unique(age_sorted)
    n_age = age_uniq.size

    age = 10.**age / 1.e6
    nmod = len(age)


  ## scale (intens_arr,intens_arr_ion,intens_arr_if) to abundances
    intens_arr[label_arr == 'CII157',:] *= scale_c
    intens_arr[label_arr == 'SiII34',:] *= scale_si
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['OIII88','OIV26','OIII_1','OIII_2','OII_1',\
                    'OII_2','OI63','OI145']],:] *= scale_o
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['SIV10','SIII18','SIII33','SIII_1','SIII_2',\
                   'SII_1','SII_2']],:] *= scale_s
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['NII122','NII205','NIII57','NII_1']],:] *= scale_n
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['NeII12','NeIII15','NeIII36']],:] *= scale_ne
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['FeII25','FeIII23']],:] *= scale_fe
    intens_arr[[i for i,v in enumerate(label_arr) if v in \
                    ['ArII7','ArIII9','ArIII21']],:] *= scale_ar
    
    intens_arr_ion[label_arr == 'CII157',:] *= scale_c
    intens_arr_ion[label_arr == 'SiII34',:] *= scale_si
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['OIII88','OIV26','OIII_1','OIII_2','OII_1',\
                    'OII_2','OI63','OI145']],:] *= scale_o
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['SIV10','SIII18','SIII33','SIII_1','SIII_2',\
                   'SII_1','SII_2']],:] *= scale_s
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['NII122','NII205','NIII57','NII_1']],:] *= scale_n
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['NeII12','NeIII15','NeIII36']],:] *= scale_ne
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['FeII25','FeIII23']],:] *= scale_fe
    intens_arr_ion[[i for i,v in enumerate(label_arr) if v in \
                    ['ArII7','ArIII9','ArIII21']],:] *= scale_ar
                    
    intens_arr_if[label_arr == 'CII157',:] *= scale_c
    intens_arr_if[label_arr == 'SiII34',:] *= scale_si
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['OIII88','OIV26','OIII_1','OIII_2','OII_1',\
                    'OII_2','OI63','OI145']],:] *= scale_o
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['SIV10','SIII18','SIII33','SIII_1','SIII_2',\
                   'SII_1','SII_2']],:] *= scale_s
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['NII122','NII205','NIII57','NII_1']],:] *= scale_n
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['NeII12','NeIII15','NeIII36']],:] *= scale_ne
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['FeII25','FeIII23']],:] *= scale_fe
    intens_arr_if[[i for i,v in enumerate(label_arr) if v in \
                    ['ArII7','ArIII9','ArIII21']],:] *= scale_ar


  ## extend grid for covering factor
    if (usepdr) :
        n_cov_factor = 6
        cov_factor_array = np.arange(n_cov_factor)/(n_cov_factor-1)
    else :
        n_cov_factor = 1
        cov_factor_array = [1]

    nmod_new = nmod *n_cov_factor #number of models

    n = np.tile(n,n_cov_factor)
    npdr = np.tile(npdr,n_cov_factor)
    age = np.tile(age,n_cov_factor)
    u = np.tile(u,n_cov_factor)
    rin = np.tile(rin,n_cov_factor)
    g_param = np.tile(g_param,n_cov_factor)
    metal = np.tile(metal,n_cov_factor)
    cov_pdr = np.repeat(cov_factor_array,nmod)
    
    sizes = np.tile(sizes,n_cov_factor)
    thickness = sizes[0,:] #transition H+/HI
    thickness_pdr = sizes[1,:] # transition H2/HI - sizes(*,0)
    
    intens_arr_new = np.zeros((len(intens_arr[:,0]),nmod_new),float) * np.nan
    intens_arr_ion_new = np.zeros((len(intens_arr_ion[:,0]),nmod_new),float) * np.nan    
    for tt in range(0,n_cov_factor):
        intens_arr_new[:,tt*nmod:(tt+1)*nmod] = intens_arr_if*(1.-cov_factor_array[tt]) + \
                                                intens_arr*cov_factor_array[tt]
        intens_arr_ion_new[:,tt*nmod:(tt+1)*nmod] = intens_arr_if*(1.-cov_factor_array[tt]) + \
                                                intens_arr_ion*cov_factor_array[tt]
    intens_arr = intens_arr_new
    intens_arr_ion = intens_arr_ion_new

    tir_lum = np.tile(tir_lum,n_cov_factor)
    tot_lum = np.tile(tot_lum,n_cov_factor)

    tot_lum = tot_lum + tir_lum*(cov_pdr-1.)
    tir_lum *= cov_pdr
    

  ## set up variables
    n_sc_factor = 1 #steps in scaling factor
    nmod = int(nmod_new/nmetalbins *n_sc_factor) #number of models
    n_free = 2+(n_age > 1)+(n_sc_factor > 1)+(n_cov_factor > 1)

  ## add TIR, TOT to intens_arr and label_arr
    intens_arr = np.append(intens_arr,np.reshape(tir_lum,(1,len(tir_lum))),axis=0)
    intens_arr = np.append(intens_arr,np.reshape(tot_lum,(1,len(tot_lum))),axis=0)
    intens_arr_ion = np.append(intens_arr_ion,np.reshape(tir_lum*0.,(1,len(tir_lum))),axis=0)
    intens_arr_ion = np.append(intens_arr_ion,np.reshape(tot_lum*0.,(1,len(tot_lum))),axis=0)
    label_arr = np.append(label_arr, ['LTIR','LTOT'],axis=0)
    
    perform['iperf_labels'] = label_arr


  ## scale models to observations
    lines_for_scaling = ['OIII88','NIII57','NII122','NII205',\
                    'NeII12','NeIII15','SIV10','SIII18','SIII33']
    e_lines_for_scaling = ['e_'+ s for s in lines_for_scaling]
        
    med_obs = list()
    snr_obs = list()
    med_mod = np.zeros((nmod,len(lines_for_scaling)), dtype=float) *np.nan
    for i in range(len(lines_for_scaling)) :
        if master_obs[lines_for_scaling[i]] > 0 :
            med_mod[:,i] = intens_arr[label_arr == lines_for_scaling[i]].flatten()
            med_obs.append(master_obs[lines_for_scaling[i]][0])
            snr_obs.append(master_obs[lines_for_scaling[i]][0] / master_obs[e_lines_for_scaling[i]][0])
    
    if med_obs :
        #- weighted average by SNR
        med_scale_obs = sum(med_obs* np.power(snr_obs,2.)) / sum(np.power(snr_obs,2.))
        med_scale_mod = np.nanmean(med_mod,axis=1)
        med_scale_mod[med_scale_mod == 0.] = np.nan
        intens_arr[[i for i,v in enumerate(label_arr)],:] *= med_scale_obs/med_scale_mod
        intens_arr_ion[[i for i,v in enumerate(label_arr)],:] *= med_scale_obs/med_scale_mod
        tir_lum *= med_scale_obs/med_scale_mod
        tot_lum *= med_scale_obs/med_scale_mod


#=======================================
    #- Calculate chi2 in log for single model
#=======================================
    val = np.zeros((nmod,),dtype=float)
    ndet = 0
    
  #--- loop over lines to fit
    for j in i_to_fit : #[i_to_fit,i_to_fit_opt] :
        if master_obs[j][0] > 0 :
            tmp_pred = intens_arr[label_arr == j].flatten()
            tmp_pred[tmp_pred > master_obs[j][0]] = master_obs[j][0]                
            tmp_pred[tmp_pred == 0.] = np.nan
            val += np.power(intens_arr[label_arr == j].flatten() - master_obs[j][0],2) / \
                np.power(tmp_pred*master_obs['e_'+j][0]/master_obs[j][0],2)
            ndet += 1
            print(j, np.nanmin(val),np.nanmax(val))
        if master_obs[j][0] < 0 :
            tmp_pred = intens_arr[label_arr == j].flatten() - np.abs(3*master_obs[j][0])
            tmp_pred[tmp_pred < 0.] = 0.
            val += np.power(tmp_pred,2) / np.power(np.abs(3*master_obs[j][0]),2)  
  #--- add to chi2 for lines to check
    for j in i_to_check :
        if master_obs[j][0] > 0 :
            tmp_pred = intens_arr[label_arr == j].flatten()
            indul = [tmp_pred > (master_obs[j][0]+master_obs['e_'+j][0])]
            val[indul] += np.power(tmp_pred[indul]-master_obs[j][0],2) / \
                        np.power(master_obs[j][0],2)
        if master_obs[j][0] < 0 :
            tmp_pred = intens_arr[label_arr == j].flatten() - np.abs(3*master_obs[j][0])
            indul = [tmp_pred > 0]
            val[indul] += np.power(tmp_pred[indul]-np.abs(3*master_obs[j][0]),2) / \
                        np.power(np.abs(3*master_obs[j][0]),2)
  #--- add to chi2 if LTOT out of [1*LTIR,1*LBOL]
    if (useltot) :
        tmp_pred = intens_arr[label_arr == 'LTOT'].flatten()
        indul = [tmp_pred > (master_obs['LTOT'][0]+master_obs['e_LTOT'][0])]
        val[indul] += np.power(tmp_pred[indul]-master_obs['LTOT'][0],2) / \
                   np.power(master_obs['LTOT'][0],2)
  #--- additional options
    if (setlown) :
        mask = [n != 1.5]
        val[mask] = np.nan



    ndof = max([0,ndet-n_free])
    best = np.nanargmin(val)

    cii_frac = (intens_arr_ion[label_arr == 'CII157']/(master_obs['CII157'][0])).flatten()
    cii_frac[cii_frac > 1] = 1.
    cii_frac[cii_frac < 0] = np.nan
    sil_frac = (intens_arr_ion[label_arr == 'SiII34']/(master_obs['SiII34'][0])).flatten()
    sil_frac[sil_frac > 1] = 1.
    sil_frac[sil_frac < 0] = np.nan

    conf_level_table=[0,1,2.3,3.53,4.72,5.89,7.04,8.18,9.30,\
        10.42,11.54,12.64,13.76,14.84,15.94,17.03,18.11,19.20]
    conf_level = conf_level_table[ndof]


  #-- write results to txt file
    f_results = open(dir_out+galname+this_append+'.txt','w+')
    f_results.write(f'metal bin: {this_metal} \n')
    f_results.write('================================ \n')
    f_results.write(f'Best model: {galname}_best_median_values \n')
    f_results.write(f'Ndet, Nparams = {ndet}, {n_free} \n')
    f_results.write(f'chi2_min   = {val[best]:.4f} \n')
    f_results.write(f'chi2_min_reduced   = {val[best]/ndof:.4f} \n')
    f_results.write(f'n   = {n[best]:.3f} \n')
    f_results.write(f'npdr   = {npdr[best]:.3f} \n')
    f_results.write(f'age = {age[best]:.3f} \n')
    f_results.write(f'U   = {u[best]:.3f} \n')
    f_results.write(f'rin   = {rin[best]:.3f} \n')
    f_results.write(f'G0   = {g_param[best]:.2f} \n')
    f_results.write(f'cov PDR   = {cov_pdr[best]:.2f} \n')
    f_results.write(f'sc_factor   = 1 \n')
    f_results.write(f'cii_frac   = {cii_frac[best]:.5f} \n')


    f_results.write('================================ \n')
    f_results.write('Models within 1-sigma: \n')
    this_cond = val-val[best] <= conf_level
    f_results.write(f'Nb = {sum(this_cond)} \n')
    f_results.write(f'n   = {mean(n[this_cond]):.3f}, {np.nanstd(n[this_cond]):.3f} \n')
    f_results.write(f'npdr   = {mean(npdr[this_cond]):.3f}, {np.nanstd(npdr[this_cond]):.3f} \n')
    f_results.write(f'age = {mean(age[this_cond]):.3f}, {np.nanstd(age[this_cond]):.3f} \n')
    f_results.write(f'U   = {mean(u[this_cond]):.3f}, {np.nanstd(u[this_cond]):.3f} \n')
    f_results.write(f'rin   = {mean(rin[this_cond]):.3f}, {np.nanstd(rin[this_cond]):.3f} \n')
    f_results.write(f'G0   = {mean(g_param[this_cond]):.2f}, {np.nanstd(g_param[this_cond]):.2f} \n')
    f_results.write(f'cov_pdr   = {mean(cov_pdr[this_cond]):.2f}, {np.nanstd(cov_pdr[this_cond]):2f} \n')
    f_results.write(f'sc_factor   = 1, 0 \n')
    f_results.write(f'cii_frac   = {mean(cii_frac[this_cond]):.5f}, {np.nanstd(cii_frac[this_cond]):.5f} \n')
    f_results.write('================================ \n')
    f_results.write('================================ \n')


    print('================================')
    print('Looping over all model combination possible to find mixed model...')


  #==========================
  # Try to find a mixed model
  #==========================
#    ratios = np.zeros((int(1e7),nl),dtype=float)    
#    ratios_ion = np.zeros((int(1e7),nl),dtype=float)    
#    contrib1 = np.zeros((int(1e7),nl),dtype=float)
#    contrib1_ion = np.zeros((int(1e7),nl),dtype=float)
    k = 0

  ##-- loop over all model combinations possible
    imod1todo = np.where(n < 5)
  #-condition on setting density of one model to 30cm-3.
    if (setlown) :
        imod1todo = np.where(n == 1.5)
    imod1todo = np.asarray(imod1todo).flatten()
    for imod1 in imod1todo :
        test1 = np.asarray(np.where(n <= n[imod1])).flatten()
        test2 = np.asarray(np.where(u <= u[imod1])).flatten() # divide by 2 the number of models
        # same PDR cov for the 2 components OR single PDR component
        if (singlepdr) :
            test3 = np.asarray(np.where(cov_pdr == cov_pdr[imod1])).flatten()
        else :
            test3 = np.asarray(np.where(cov_pdr == 0.)).flatten() 
        imod2todo = [v for i,v in enumerate(test1) if v in test2 and v in test3]
        if (relax) : imod2todo = np.where( n <= n[imod1] ) # divide by 2 the number of models
        imod2todo = np.asarray(imod2todo).flatten() 
        for imod2 in imod2todo :
            
                for log_sf in range(10, 100, 10) :

                    scaling_factor = log_sf/1e2 ##10.^float(log_sf)

                    intens_arr_new = intens_arr[:,imod1]*scaling_factor + \
                                    intens_arr[:,imod2]*(1.-scaling_factor)
                    intens_arr_ion_new = intens_arr_ion[:,imod1]*scaling_factor + \
                                    intens_arr_ion[:,imod2]*(1.-scaling_factor)

                    tmp_contrib1 = intens_arr[:,imod1]/intens_arr_new *scaling_factor
                    tmp_contrib1_ion = intens_arr_ion[:,imod1]/intens_arr_ion_new *scaling_factor
                    if k == 0:
                        contrib1 = np.reshape(tmp_contrib1,(1,len(intens_arr_new)))
                        contrib1_ion = np.reshape(tmp_contrib1_ion,(1,len(intens_arr_ion_new)))
                    else :
                        contrib1 = np.append(contrib1,np.reshape(tmp_contrib1,(1,len(intens_arr_new))),axis=0)
                        contrib1_ion = np.append(contrib1_ion,np.reshape(tmp_contrib1_ion,(1,len(intens_arr_ion_new))),axis=0)


                ## scale models to observations
                    med_mod = np.zeros(len(lines_for_scaling), dtype=float) *np.nan
                    for i in range(len(lines_for_scaling)) :
                        if master_obs[lines_for_scaling[i]] > 0 :
                            med_mod[i] = intens_arr_new[label_arr == lines_for_scaling[i]].flatten()

                    if med_obs :
                        #- weighted average by SNR
                        med_scale_mod = np.nanmean(med_mod)
                        if med_scale_mod == 0. : med_scale_mod = np.nan
                        intens_arr_new *= med_scale_obs/med_scale_mod
                        intens_arr_ion_new *= med_scale_obs/med_scale_mod
#

                ## calculate chi2
                    val = 0
                    ndet = 0

                #--- loop over lines to fit                        
                    for j in i_to_fit : #[i_to_fit,i_to_fit_opt] :
                        if master_obs[j][0] > 0 :
                            tmp_pred = np.nanmin([intens_arr_new[label_arr == j], master_obs[j][0]])
                            val += np.power(intens_arr_new[label_arr == j] - master_obs[j][0],2) / \
                                np.power(tmp_pred*master_obs['e_'+j][0]/master_obs[j][0],2)
                            ndet += 1
                        if master_obs[j][0] < 0 :
                            tmp_pred = intens_arr_new[label_arr == j] - np.abs(3*master_obs[j][0])
                            tmp_pred[tmp_pred < 0.] = 0.
                            val += np.power(tmp_pred,2) / np.power(np.abs(3*master_obs[j][0]),2)  
                #--- add to chi2 for lines to check
                    for j in i_to_check :
                        if master_obs[j][0] > 0 :
                            tmp_pred = intens_arr_new[label_arr == j]
                            if tmp_pred > (master_obs[j][0]+master_obs['e_'+j][0]) :
                                val += np.power(tmp_pred-master_obs[j][0],2) / \
                                    np.power(master_obs[j][0],2)
                        if master_obs[j][0] < 0 :
                            tmp_pred = intens_arr_new[label_arr == j] - np.abs(3*master_obs[j][0])
                            if tmp_pred > 0 :
                                val += np.power(tmp_pred-np.abs(3*master_obs[j][0]),2) / \
                                    np.power(np.abs(3*master_obs[j][0]),2)
                #--- add to chi2 if LTOT out of [1*LTIR,1*LBOL]
                    if (useltot) :
                        tmp_pred = intens_arr_new[label_arr == 'LTOT']
                        if tmp_pred > (master_obs['LTOT'][0]+master_obs['e_LTOT'][0]) :
                            val += np.power(tmp_pred-master_obs['LTOT'][0],2) / \
                                np.power(master_obs['LTOT'][0],2)
                        if tmp_pred < (master_obs['LTOT'][0]-master_obs['e_LTOT'][0]) :
                            val += np.power(tmp_pred-master_obs['LTOT'][0],2) / \
                                np.power(tmp_pred,2)
                        

                    ndof = max([0,ndet-(n_free)*2 +(n_cov_factor > 1)])

                    found = 0
                    ns = 0

                    tmp_tab = [imod1, imod2, scaling_factor, thickness[imod1]/thickness[imod2], val, found, ns]
                    if k == 0 :
                        tab = np.reshape(tmp_tab,(1,7))
                        ratios = np.reshape(intens_arr_new,(1,len(intens_arr_new)))
                        ratios_ion = np.reshape(intens_arr_ion_new,(1,len(intens_arr_ion_new)))
                    else :
                        tab = np.append(tab,np.reshape(tmp_tab,(1,7)),axis=0)
                        ratios = np.append(ratios,np.reshape(intens_arr_new,(1,len(intens_arr_new))),axis=0)
                        ratios_ion = np.append(ratios_ion,np.reshape(intens_arr_ion_new,(1,len(intens_arr_ion_new))),axis=0)
                    k += 1

#        print('...',round(float(imod1/max(imod1todo))*1e2), '%')

    print('Total number of models:', k)
    
    tab = tab[0:k-1, :]
    ratios = ratios[0:k-1, :]
    ratios_ion = ratios_ion[0:k-1, :]
    contrib1 = contrib1[0:k-1, :]
    contrib1_ion = contrib1_ion[0:k-1, :]

    ikeep = np.asarray(np.where(np.isfinite(tab[:,4]))).flatten()
    tab = tab[ikeep,:]
    ratios = ratios[ikeep,:]
    ratios_ion = ratios_ion[ikeep,:]
    contrib1 = contrib1[ikeep,:]
    contrib1_ion = contrib1_ion[ikeep,:]

    ind = np.argsort(tab[:, 4])
    tab = tab[ind, :]
    ratios = ratios[ind, :]
    ratios_ion = ratios_ion[ind, :]
    contrib1 = contrib1[ind, :]
    contrib1_ion = contrib1_ion[ind, :]
    best1 = int(tab[0, 0]) #contribution from mod1 for best solution
    best2 = int(tab[0, 1]) #contribution from mod2 for best solution
    scaling_factor_best = tab[0, 2]
    size_ratio_best = tab[0, 3]


    best = 0
    bestval = tab[0,4]
    valred_best = bestval/ndof
    cii_perf = (ratios[:,label_arr == 'CII157']/(master_obs['CII157'][0])).flatten()
    cii_frac = (ratios_ion[:,label_arr == 'CII157']/(master_obs['CII157'][0])).flatten()
    cii_frac[cii_frac > 1] = 1.
    cii_frac[cii_frac < 0] = np.nan
    silii_perf = (ratios[:,label_arr == 'SiII34']/(master_obs['SiII34'][0])).flatten()
    silii_frac = (ratios_ion[:,label_arr == 'SiII34']/(master_obs['SiII34'][0])).flatten()
    silii_frac[silii_frac > 1] = 1.
    silii_frac[silii_frac < 0] = np.nan
#    nii_ratio = (ratios[:,label_arr == 'NII205']/ratios[:,label_arr == 'NII122']).flatten()
#    nii_ratio_obs = master_obs['NII205'][0] / master_obs['NII122'][0]


    conf_level = conf_level_table[ndof]
    double_val = tab[:,4]
    this_cond = double_val-double_val[best] <= conf_level
    

    f_results.write('================================ \n')
    f_results.write('Best mixing model: [mod1, mod2] \n')
    f_results.write(f'minimum chi2 = {bestval:.4f} \n')
    f_results.write(f'chi2_min_reduced   = {valred_best:.4f} \n')
    f_results.write(f'n   = {n[best1]:.3f} , {n[best2]:.3f} \n')
    f_results.write(f'npdr   = {npdr[best1]:.3f} , {npdr[best2]:.3f} \n')
    f_results.write(f'age = {age[best1]:.3f} , {age[best2]:.3f} \n')
    f_results.write(f'U   = {u[best1]:.3f} , {u[best2]:.3f} \n')
    f_results.write(f'rin   = {rin[best1]:.3f} , {rin[best2]:.3f} \n')
    f_results.write(f'G0   = {g_param[best1]:.2f} , {g_param[best2]:.3f} \n')
    f_results.write(f'cov_pdr   = {cov_pdr[best1]:.2f} , {cov_pdr[best2]:.3f} \n')
    f_results.write(f'Scaling factor = {scaling_factor_best:.3f} , {1-scaling_factor_best:.3f} \n')
    f_results.write(f'Relative size mod1/mod2 = {size_ratio_best:.5f} \n')
    f_results.write(f'# Total Contribution to CII and SiII: \n')
    f_results.write(f'{cii_perf[best]:.5f}, {silii_perf[best]:.5f} \n')
    
    f_results.write('================================ \n')
    f_results.write('Models within 1-sigma: \n')
    f_results.write(f'Nb = {sum(this_cond)} \n')
    f_results.write(f'cii_frac   = {mean(cii_frac[this_cond]):.5f}, {np.nanstd(cii_frac[this_cond]):.5f} \n')
    f_results.write('================================ \n')


    f_results.close()
    
    end = time.time()
    print('Time : ', end - start)
   
    
# MAIN BODY
findsol(galname='Haro2',usemain=1,usepdr=1,singlepdr=1,useltir=1)
