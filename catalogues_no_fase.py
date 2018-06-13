import numpy as np
import pandas as pd
from astropy.io.ascii import read as astro_read
from astropy.io import fits

############################################################################
class COSMOS:
    #builder
    def __init__(self,online=True,query_str="SELECT * FROM paudm.cosmos as cosmos "):
	import pandas as pd
        if online:
            #connect and query paudm database
            import sqlalchemy as sqla
            dsn = 'postgresql://readonly:PAUsc1ence@localhost:8892/dm'
            engine = sqla.create_engine(dsn)
            self.tab = pd.read_sql(query_str,engine)
        else:
            #load the table locally
            self.tab = pd.read_csv('../data/COSMOS_paudm_all.csv')
        #instantiate the properties I need
        self.ra = np.array(self.tab['ra'])
        self.dec = np.array(self.tab['dec'])
        self.I_auto = np.array(self.tab['I_auto'])
        self.NbFilt = np.array(self.tab['NbFilt'])
        self.zspec = np.array(self.tab['zspec'])
        self.r50 = np.array(self.tab['r50'])
        self.arcsec_per_pixel = 0.03
        self.r50_arcsec = np.array(self.r50*self.arcsec_per_pixel)
    #useful functions
    def get_tab(self):
        return self.tab
    def get_names(self):
        return self.tab.columns.values
    def hist_rad(self,filename='../plots/hist_rad_cosmos_.png',cut=26.):
        plt.xlabel(r'log $r_{50}$ [arcsec]')
        plt.ylabel('counts')
        rr = self.r50_arcsec[self.get_cut(cut)]
        plt.hist(np.log10(rr),bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.I_auto<=i_mag_cut)

############################################################################
class CFHTLens:
    #builder
    def __init__(self):
        from astropy.io.ascii import read as astro_read
        #load the table locally
        #self.tab = pd.read_table('../data/CFHTLens.tsv')
        #self.tab = pd.read_table('../data/CFHTLens_lotsofcolumns_stars.tsv')
        #self.tab = astro_read('../data/LENS_all_scalelength.csv',format='csv',guess=False,fast_reader=True)#={'chunk_size':100000000})
        #self.tab = pd.read_csv('../data/LENS_all_scalelength.csv')
        #self.tab = pd.read_csv('../data/LENS_all_bulge.csv')
        self.tab = pd.read_csv('../data/LENS_total_light_AB_Kron.csv') # here I deleted some non-used psf columns (but I left the Kron radius)

        #instantiate the properties I need
        self.ra = np.array(self.tab['ALPHA_J2000'])
        self.dec = np.array(self.tab['DELTA_J2000'])
        self.MAG_i = np.array(self.tab['MAG_i'])
        #self.NbFilt = np.array(self.tab['NbFilt'])
        self.Z_B = np.array(self.tab['Z_B'])
        self.FLUX_RADIUS = np.array(self.tab['FLUX_RADIUS'])
        self.arcsec_per_pixel = 0.187
        self.FLUX_RADIUS_arcsec = np.array(self.FLUX_RADIUS*self.arcsec_per_pixel)
        self.KRON_RADIUS = np.array(self.tab['KRON_RADIUS'])
       
        self.fitclass = np.array(self.tab['fitclass'])
        self.CLASS_STAR = np.array(self.tab['CLASS_STAR'])
        '''star_flag: Stars and galaxies are separated using a combination of
          size, i/y-band magnitude and colour information. For i<21, all
          objects with size smaller than the PSF are classified as stars. For
          i>23, all objects are classified as galaxies. In the range 21<i<23,
          a star is defined as size<PSF and chi2_star<2.0*chi2_gal, where the
          chi2's are the best fit chi2's from the galaxy and star libraries
          given by LePhare. NOTE: star_flag is optimized for galaxy studies,
          to keep an almost 100% complete galaxy sample with low (but not
          vanishing) stellar contamination. CLASS_STAR usually gives a cleaner
          star sample, but can lead to serious incompleteness in a galaxy
          sample.'''
        self.SNratio = np.array(self.tab['SNratio'])
        #self.PSF_e1_pix = np.array(self.tab['PSF_e1'])
        #self.PSF_e2_pix = np.array(self.tab['PSF_e2'])
        #self.PSF_eTOT_pix = np.sqrt(self.PSF_e1_pix**2+self.PSF_e2_pix**2)
        #self.PSF_e1 = self.PSF_e1_pix*self.arcsec_per_pixel
        #self.PSF_e2 = self.PSF_e2_pix*self.arcsec_per_pixel
        #self.PSF_eTOT = self.PSF_eTOT_pix*self.arcsec_per_pixel
        
        '''PSF-e1, PSF-e2: means of the PSF model ellipticity components
          measured on each exposure. PSF ellipticities are derived from the
          PSF model at the location of each galaxy and are top-hat weighted
          with radius 8 pixels (1.496 arcsec)'''
        
        #self.PSF_Strehl_ratio = np.array(self.tab['PSF_Strehl_ratio'])
        self.scalelength = np.array(self.tab['scalelength'])
        self.scalelength_arcsec = np.array(self.scalelength*self.arcsec_per_pixel)
        
        self.FWHM_IMAGE_pix = np.array(self.tab['FWHM_IMAGE']) #FWHM assuming a gaussian core [pixels]
        self.FWHM_IMAGE_arcsec = self.FWHM_IMAGE_pix*self.arcsec_per_pixel
        self.FWHM_WORLD_deg = np.array(self.tab['FWHM_WORLD']) #the same but degrees
        self.SEEING_from_FWHM = self.FWHM_IMAGE_arcsec*0.5 # sigma = 0.5 FWHM (for a bidimensional gaussian) 
        self.MASK = np.array(self.tab['MASK'])
        self.bulge_fraction = np.array(self.tab['bulge_fraction'])
        self.A_WORLD = np.array(self.tab['A_WORLD']) # degree
        self.B_WORLD = np.array(self.tab['B_WORLD']) # degree
        self.A_WORLD_arcsec = np.array(self.tab['A_WORLD']*3600.) # arcsec
        self.B_WORLD_arcsec = np.array(self.tab['B_WORLD']*3600.) # arcsec
    
    
    #useful functions    
    def get_tab(self):
        return self.tab
    def get_names(self):
        return self.tab.columns.values
    def hist_rad(self,filename='../plots/hist_rad_CFHTLens_.png',cut=26.):
        plt.xlabel('FLUX_RADIUS [arcsec]')
        plt.ylabel('counts')
        rr = self.FLUX_RADIUS_arcsec[self.get_cut(cut)]
        plt.hist(np.log10(rr),bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.MAG_i<=i_mag_cut)

############################################################################

class CFHTLS_DEEP:
    #builder
    def __init__(self):

        #load the table locally

	with fits.open('../data/CFHTLS_with_seeing.fit') as hdul:
    		self.tab = hdul[1].data

        #self.tab = Table('../data/CFHTLS_with_seeing.fit')
        
        #instantiate the properties I need
        self.ra = np.array(self.tab["RAJ2000"])
        self.dec = np.array(self.tab['DEJ2000'])
        self.imag = np.array(self.tab['imag'])
        #self.NbFilt = np.array(self.tab['NbFilt'])
        #self.zspec = np.array(self.tab['zspec'])
        self.irad = np.array(self.tab['irad'])
        self.arcsec_per_pixel = 0.186
        self.irad_arcsec = np.array(self.irad*self.arcsec_per_pixel)
        
        self.psf_i = np.array(self.tab['imag20'])
    
    #useful functions
    def get_tab(self):
        return self.tab
    #def get_names(self):
    #    return self.tab.getColNames()
    def hist_rad(self,filename='../plots/hist_rad_CFHTLS_.png',cut=26.):
        plt.xlabel(r'log irad [arcsec]')
        plt.ylabel('counts')
        rr = self.irad_arcsec[self.get_cut(cut)]
        plt.hist(np.log10(rr),bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.imag<=i_mag_cut)


############################################################################


class CFHTLS_D2:
    def __init__(self,cfhtls_d):
        select_D2 = np.where((cfhtls_d.ra>149.)&(cfhtls_d.ra<151.)&(cfhtls_d.dec>1.6)&(cfhtls_d.dec<2.8))
        self.ra = cfhtls_d.ra[select_D2]
        self.dec = cfhtls_d.dec[select_D2]
        self.irad = cfhtls_d.irad[select_D2]
        self.imag = cfhtls_d.imag[select_D2]
        self.arcsec_per_pixel = 0.186
        self.irad_arcsec = np.array(self.irad*self.arcsec_per_pixel)
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.imag<=i_mag_cut)


############################################################################


class MATCH_COSMOS_CFHTLS_D2:
    #builder
    def __init__(self):
        import pandas as pd
        #load the table locally
        self.tab = pd.read_csv('../data/match_COSMOS_CFHTLS_.csv')
        #instantiate the properties I need
        self.ra_cosmos = np.array(self.tab['ra_1'])
        self.dec_cosmos = np.array(self.tab['dec_1'])
        self.ra_cfhtls = np.array(self.tab['ra_CFHTLS'])
        self.dec_cfhtls = np.array(self.tab['dec_CFHTLS'])
        self.imag_cfhtls = np.array(self.tab['imag_cfhtls'])
        
        
        
        self.irad_cfhtls = np.array(self.tab['irad_cfhtls_arcsec'])
        self.r50_cosmos = np.array(self.tab['r50_arcsec'])
        
        self.zspec_cosmos = np.array(self.tab['zspec'])
        self.diff_radius = np.array(self.tab['diff_radius'])
        self.separation_match = np.array(self.tab['Separation'])
        
        self.I_auto_cosmos = np.array(self.tab['I_auto'])
        
        
        self.ratio_radii = np.array(self.irad_cfhtls/self.r50_cosmos)
        
        self.sel_good = np.where((np.abs(self.ratio_radii)<=10.)&(self.r50_cosmos>0))
        
        self.ratio_radii_good = self.ratio_radii[self.sel_good]
        
        self.rad_cosmos_good = np.array(self.r50_cosmos)[self.sel_good]
        self.rad_cfhtls_good = np.array(self.irad_cfhtls)[self.sel_good]
        self.mag_cosmos_good = np.array(self.I_auto_cosmos)[self.sel_good]
        self.mag_cfhtls_good = np.array(self.imag_cfhtls)[self.sel_good]
    
    
    
    #useful functions
    def get_tab(self):
        return self.tab
    def get_names(self):
        return self.tab.columns.values
    
    def hist_diff(self,filename='../plots/hist_diff_radii.png',cut=26.):
        plt.xlabel('diff_radius [arcsec]')
        plt.ylabel('counts')
        rr = self.diff_radius[self.get_cut(cut)]
        plt.hist(rr,bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
        return None
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.I_auto_cosmos<=i_mag_cut)

#################################################################################

class CFHTLS_D3:
    def __init__(self,cfhtls_d):
        select_D3 = np.where((cfhtls_d.ra>200.)&(cfhtls_d.ra<250.)&(cfhtls_d.dec>45.)&(cfhtls_d.dec<60.))
        self.ra = cfhtls_d.ra[select_D3]
        self.dec = cfhtls_d.dec[select_D3]
        self.irad = cfhtls_d.irad[select_D3]
        self.imag = cfhtls_d.imag[select_D3]
        self.arcsec_per_pixel = 0.187
        self.irad_arcsec = np.array(self.irad*self.arcsec_per_pixel)
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.imag<=i_mag_cut)


#################################################################################


class LENS_W3:
    def __init__(self,cfhtlens):
        print cfhtlens.ra
        select_W3 = np.where((cfhtlens.ra>200.)&(cfhtlens.ra<250.)&(cfhtlens.dec>45.)&(cfhtlens.dec<60.))
        print select_W3
        self.ra = cfhtlens.ra[select_W3]
        self.dec = cfhtlens.dec[select_W3]
        self.FLUX_RADIUS = cfhtlens.FLUX_RADIUS[select_W3]
        self.MAG_i = cfhtlens.MAG_i[select_W3]
        self.CLASS_STAR = cfhtlens.CLASS_STAR[select_W3]
        self.MASK = cfhtlens.MASK[select_W3]
        self.arcsec_per_pixel = 0.187
        self.FLUX_RADIUS_arcsec = np.array(self.FLUX_RADIUS*self.arcsec_per_pixel)
        self.scalelength = np.array(cfhtlens.scalelength[select_W3])
        self.scalelength_arcsec = np.array(self.scalelength*self.arcsec_per_pixel)
        self.bulge_fraction = np.array(cfhtlens.bulge_fraction[select_W3])
        self.A_WORLD = np.array(cfhtlens.A_WORLD[select_W3]) # degree
        self.B_WORLD = np.array(cfhtlens.B_WORLD[select_W3]) # degree
        self.A_WORLD_arcsec = np.array(cfhtlens.A_WORLD_arcsec[select_W3]) # arcsec
        self.B_WORLD_arcsec = np.array(cfhtlens.B_WORLD_arcsec[select_W3]) # arcsec
        self.fitclass = np.array(cfhtlens.fitclass[select_W3])
        self.KRON_RADIUS = np.array(cfhtlens.KRON_RADIUS[select_W3]) # no unit specified in the paper

        print self.ra
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.MAG_i<=i_mag_cut)


##################################################################################


class MATCH_LS_D3_LENS_W3:
    #builder
    def __init__(self):
        import pandas as pd
        #load the table locally
        self.tab = pd.read_csv('../data/match_D3_W3_topcat.csv')
        #instantiate the properties I need
        self.ra_cfhtls = np.array(self.tab['ra_cfhtls'])
        self.dec_cfhtls = np.array(self.tab['dec_cfhtls'])
        self.imag_cfhtls = np.array(self.tab['imag_cfhtls'])
        self.irad_cfhtls = np.array(self.tab['irad_cfhtls_arcsec'])
        
        self.ra_cfhtlens = np.array(self.tab['ra_cfhtlens'])
        self.dec_cfhtlens = np.array(self.tab['dec_cfhtlens'])
        self.FLUX_RADIUS_cfhtlens = np.array(self.tab['FLUX_RADIUS_cfhtlens_arcsec'])
        self.MAG_i_cfhtlens = np.array(self.tab['MAG_i_cfhtlens'])
        
        self.diff_radius_ls_lens = np.array(self.tab['diff_radius_ls_lens'])
        self.separation_match = np.array(self.tab['Separation'])
        
        
        self.ratio_radii_ls_lens = np.array(self.irad_cfhtls/self.FLUX_RADIUS_cfhtlens)
    
    #self.sel_good = np.where((np.abs(self.ratio_radii)<=10.)&(self.r50_cosmos>0))
    
    #self.ratio_radii_good = self.ratio_radii[self.sel_good]
    
    #self.rad_cosmos_good = np.array(self.r50_cosmos)[self.sel_good]
    #self.rad_cfhtls_good = np.array(self.irad_cfhtls)[self.sel_good]
    #self.mag_cosmos_good = np.array(self.I_auto_cosmos)[self.sel_good]
    #self.mag_cfhtls_good = np.array(self.imag_cfhtls)[self.sel_good]
    
    
    
    #useful functions
    def get_tab(self):
        return self.tab
    def get_names(self):
        return self.tab.columns.values
    
    def hist_diff(self,filename='../plots/hist_diff_radii.png',cut=26.):
        plt.xlabel('diff_radius [arcsec]')
        plt.ylabel('counts')
        rr = self.diff_radius[self.get_cut(cut)]
        plt.hist(rr,bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
        return None
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.I_auto_cosmos<=i_mag_cut)



#################################################################################

class CFHTLS_W3:
    def __init__(self):
        import pandas as pd
        self.tab = pd.read_csv('../data/CFHTLS_W3.csv')
        #instantiate the properties I need
        self.ra = np.array(self.tab['RAJ2000'])
        self.dec = np.array(self.tab['DEJ2000'])
        self.irad = np.array(self.tab['irad'])
        self.imag = np.array(self.tab['imag'])
        self.arcsec_per_pixel = 0.187
        self.irad_arcsec = np.array(self.irad*self.arcsec_per_pixel)
        # flag for the i/y filter (0=i,1=y)
        self.ifl = np.array(self.tab['ifl'])
    
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.imag<=i_mag_cut)


################################################################################


class MATCH_LS_W3_LENS_W3:
    #builder
    def __init__(self):
        import pandas as pd
        
        self.tab = pd.read_csv('../data/match_w3_w3_ls_lens.csv')
        #CFHTLS
        self.ra_cfhtls = np.array(self.tab['RAJ2000'])
        self.dec_cfhtls = np.array(self.tab['DEJ2000'])
        self.imag_cfhtls = np.array(self.tab['imag'])
        self.irad_cfhtls = np.array(self.tab['irad'])*0.187 #arcseconds
        
        #LENS
        self.ra_cfhtlens = np.array(self.tab['ra'])
        self.dec_cfhtlens = np.array(self.tab['dec'])
        self.FLUX_RADIUS_cfhtlens = np.array(self.tab['FLUX_RADIUS[arcsec]'])
        self.scalelength_cfhtlens = np.array(self.tab['scalelength[arcsec]'])
        self.MAG_i_cfhtlens = np.array(self.tab['MAG_i'])
        
        #ALL
        self.separation_match = np.array(self.tab['Separation'])
        
        
        self.diff_radius_ls_lens_fluxradius = self.irad_cfhtls - self.FLUX_RADIUS_cfhtlens
        self.square_diff_radius_ls_lens_fluxradius = self.irad_cfhtls**2 - self.FLUX_RADIUS_cfhtlens**2
        
        self.diff_radius_ls_lens_scalelength = self.irad_cfhtls - self.scalelength_cfhtlens
        
        self.square_diff_radius_ls_lens_scalelength = self.irad_cfhtls**2 - self.scalelength_cfhtlens**2
        
        
        
        self.ratio_radii_ls_lens = np.array(self.irad_cfhtls/self.FLUX_RADIUS_cfhtlens)
    
    #self.sel_good = np.where((np.abs(self.ratio_radii)<=10.)&(self.r50_cosmos>0))
    
    #self.ratio_radii_good = self.ratio_radii[self.sel_good]
    
    #self.rad_cosmos_good = np.array(self.r50_cosmos)[self.sel_good]
    #self.rad_cfhtls_good = np.array(self.irad_cfhtls)[self.sel_good]
    #self.mag_cosmos_good = np.array(self.I_auto_cosmos)[self.sel_good]
    #self.mag_cfhtls_good = np.array(self.imag_cfhtls)[self.sel_good]
    
    
    
    #useful functions
    def get_tab(self):
        return self.tab
    def get_names(self):
        return self.tab.columns.values
    
    def hist_diff(self,filename='../plots/hist_diff_radii.png',cut=26.):
        plt.xlabel('diff_radius [arcsec]')
        plt.ylabel('counts')
        rr = self.diff_radius[self.get_cut(cut)]
        plt.hist(rr,bins=30,range=(-1,1),histtype='step',lw=3)
        plt.savefig(filename)
        plt.show()
        return None
    def get_cut(self,i_mag_cut=22.5):
        return np.where(self.I_auto_cosmos<=i_mag_cut)


################################################################################

def median_in_bins_err(x,y,x1,x2,nbins,p1=0.25,p2=0.75):
    medians_x = np.zeros(nbins)
    medians_y = np.zeros(nbins)
    err_y_low = np.zeros(nbins)
    err_y_high = np.zeros(nbins)
    array = np.array(x)
    array = np.array(y)
    width = (float(x2)-float(x1))/nbins
    for i in range(int(nbins)):
        x_low = x1 + i*width
        x_high = x1+ (i+1)*width
        #print x_low, x_high
        sel = np.where((x>x_low)&(x<x_high))
        temp_ord = np.copy(y[sel])
        temp_ord = np.sort(temp_ord)
        medians_y[i] = temp_ord[int(len(temp_ord)*0.50 + 0.5)]
        err_y_low[i]  = np.abs(temp_ord[int(len(temp_ord)*p1 + 0.5)]-medians_y[i])
        err_y_high[i]  = np.abs(temp_ord[int(len(temp_ord)*p2 + 0.5)]-medians_y[i])
        #err_y[i] = [err_y_low,err_y_high]
        medians_x[i] = np.median(x[sel])
    return np.array(medians_x),np.array(medians_y),err_y_low,err_y_high






