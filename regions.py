
try:
    import unzip_requirements
except ImportError:
    pass

import astropy.units as u
import astropy.coordinates as coord
import json, logging, time, os, decimal, requests
import numpy as np 
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from statistics import median

logger = logging.getLogger("handler_logger")
logger.setLevel(logging.DEBUG)

###################################
######### Helper Functions ########
###################################

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def _get_response(status_code, body):
    if not isinstance(body, str):
        body = json.dumps(body)
    return {
        "statusCode": status_code, 
        'headers': {
            # Required for CORS support to work
            'Access-Control-Allow-Origin': '*',
            # Required for cookies, authorization headers with HTTPS
            'Access-Control-Allow-Credentials': 'true',
        },
        "body": body}

def _get_body(event):
    try:
        return json.loads(event.get("body", ""))
    except:
        logger.debug("event body could not be JSON decoded.")
        return {}


###################################
###### Cache the Fits Files #######
###################################

fits_cache = {}

def _get_fits_header(event):

    body = _get_body(event)
    site = body.get('site')
    base_filename = body.get('base_filename')
    EX01orEX13 = body.get('fitstype')

    file_id = f"{base_filename}-{EX01orEX13}"
    if file_id not in fits_cache:
        fitsFile = _get_fits(site, base_filename, EX01orEX13)
        return fitsFile[0].header

    return fits_cache[file_id][0].header


def _get_fits(site, base_filename, EX01orEX13):
    """
    EX01orEX13 should have a value in ["01", "13"]
    """
    file_id = f"{base_filename}-{EX01orEX13}"
    print(f"Fits cache keys: {fits_cache.keys()}")

    # Only download the file if it's not cached already
    if file_id not in fits_cache:
        print("Fits file not cached; downloading file.")

        # Get the url for the file
        api_url = f"https://api.photonranch.org/fits{EX01orEX13}_url/{site}/{base_filename}"
        print(f"API url: {api_url}")
        file_url = requests.get(api_url).json()
        print(f"File url: {file_url}")

        # Load the file from the url
        with fits.open(file_url) as f:
            fits_cache[file_id] = f
            # check that it is a legit file
            print("data shape: ")
            print(f[0].data.shape)
    
    else:
        print("Fits file already cached. No download needed.")

    return fits_cache[file_id]


###################################
###$$$### Lambda Handler ##########
###################################


import sep
from photutils import centroid_com, centroid_1dg, centroid_2dg
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.modeling import models, fitting
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.visualization import simple_norm
from matplotlib.patches import Ellipse




gs = gridspec.GridSpec(1,2, height_ratios=[0])
matplotlib.rcParams['figure.figsize'] = [18, 8]

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    #print((data).astype(np.int))
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 


def sepAnalysis(data, event, showPlots=False):

    mean, std = np.mean(data), np.std(data)

    data = data.astype(float)

    # measure a spatially varying background on the image
    bkg = sep.Background(data, bw=32, bh=32, fw=3, fh=3)

    # evaluate background as 2-d array, same size as original image
    bkg_image = bkg.back()

    # evaluate the background noise as 2-d array, same size as original image
    bkg_rms = bkg.rms()
    
    # subtract the background
    data_sub = data - bkg

    # Find objects
    objects = sep.extract(data_sub, thresh=10, minarea=9, err=bkg.globalrms, deblend_cont=0.005, deblend_nthresh=32)

    # Sort according to flux, largest to smallest.
    objects = np.sort(objects, order=['flux'])[::-1]

    num_objects = len(objects)
    print(f"number of objects: {num_objects}")

    

    
    if showPlots:

        # plot background-subtracted image
        norm = simple_norm(data_sub, 'sqrt', percent=99.99)
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub), np.std(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                   norm=norm)

        # plot an ellipse for each object
        for i in range(15):
            try:
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=6*objects['a'][i],
                            height=6*objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
            except:
                continue 
        

    # Get the star profiles for each object
    profiles = [] 
    for idx,obj in enumerate(objects):
        
        profile = _get_star_profile(obj, data_sub)
        
        # Only keep the objects with a good gaussian fit.
        if profile['r2'] > 0.99:
            profiles.append(profile) 
            if showPlots:
                _plot_profile(profile)
        
    if len(profiles) == 0:
        return {
            "num_good_stars": 0,
        }
    
    print("Number of good profiles: ", len(profiles))
    
    fits_header = _get_fits_header(event)

    brightest_star = profiles[0]
    print("brightest star: ", _make_json_ready_profile(brightest_star,fits_header))
    
    # The list of profiles is already sorted by flux, big to small.
    median_star = profiles[int(len(profiles)/2)]
    print("median star: ", _make_json_ready_profile(median_star, fits_header))
    
    return_body = {
        "median_star": _make_json_ready_profile(median_star, fits_header),
        "brightest_star": _make_json_ready_profile(brightest_star, fits_header),
        "num_good_stars": len(profiles),
    }
    return return_body
    

def _make_json_ready_profile(star_profile, fits_header):
    obj = star_profile['sep_object']
    profile = {
        "pixscale": fits_header.get('pixscale'),
        "naxis1": fits_header.get('naxis1'),
        "naxis2": fits_header.get('naxis2'),
        "x": obj['x'],
        "y": obj['y'],
        "a": obj['a'],
        "b": obj['b'],
        "theta": obj['theta'],
        "peak": obj['peak'],
        "flux": obj['flux'],
        "gaussian_mean": float(star_profile['fitted_model'].mean.value),
        "gaussian_amplitude": float(star_profile['fitted_model'].amplitude.value),
        "gaussian_stddev": float(star_profile['fitted_model'].stddev.value),
        "gaussian_fwhm": float(star_profile['fitted_model'].fwhm),
        "radial_profile": np.round(star_profile['rad_profile'],4),
        #"star_cutout": star_profile['star_cutout'],
        "r2": star_profile['r2'],
    }
    return profile
    
def _plot_profile(star_profile):
    
    x = np.linspace(0,25,25)
    
    fig, ax = plt.subplots(1,2)
    cutplot = ax[0].imshow(star_profile['star_cutout'], cmap="plasma")
    
    ax[1].plot(x, star_profile['rad_profile'][:25])
    ax[1].plot(x, star_profile['fitted_model'](x))

    print(f"fwhm: ",star_profile['fwhm'])        
    print(f"r2: {star_profile['r2']}")
    print("")
    plt.show()        
    print("---------------------------------------")
    
    
def _get_star_profile(sep_object, data_sub):
    obj = sep_object 
    
    position = (obj['x'], obj['y'])
    size = (50,50)
    cutout = Cutout2D(data_sub, position, size)

    rad_profile = radial_profile(cutout.data, (25,25))
    #rad_profile /= max(rad_profile)
    
    # Fit a gaussian profile
    x = np.linspace(0,25,25)
    fitter = fitting.LevMarLSQFitter()
    model = models.Gaussian1D() 
    # Should the gaussian amplitude have a fixed peak of 1, since our data is normalized?
    # model = models.Gaussian1D(amplitude=1, fixed={'amplitude': True}) 
    fitted_model = fitter(model, x, rad_profile[:25])
    
    # Coefficient of Determination to measure gaussian fit quality
    # see https://stackoverflow.com/questions/29003241/how-to-quantitatively-measure-goodness-of-fit-in-scipy
    y = rad_profile[:25]
    y_fit = fitted_model(x)
    # residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)
    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)
    
    profile = {
        "fwhm": fitted_model.fwhm,
        "fitted_model": fitted_model,
        "r2": r2,
        "flux": obj['flux'],
        "sep_object": obj,
        "rad_profile": rad_profile,
        "star_cutout": cutout.data,
    }
    return profile
    
def _get_region(event):
    body = _get_body(event)

    site = body.get("site")
    base_filename = body.get("base_filename")
    EXversion = body.get("fitstype", "13") # we want the full sized fits file
    print(f"site: {site}")
    print(f"base_filename: {base_filename}")

    fitsfile = _get_fits(site, base_filename, EXversion)
    header = fitsfile[0].header
    data = fitsfile[0].data

    # Region selected by the user. Default to entire image.
    region_x0 = body.get("region_x0", 0)
    region_x1 = body.get("region_x1", 1)
    region_y0 = body.get("region_y0", 0)
    region_y1 = body.get("region_y1", 1)
    x0 = int(data.shape[0] * region_x0)
    x1 = int(data.shape[0] * region_x1)
    y0 = int(data.shape[1] * region_y0)
    y1 = int(data.shape[1] * region_y1)

    # Swap indices if the region was selected "backwards". 
    if x0 > x1:
        swap = x0
        x0 = x1
        x1 = swap
    if y0 > y1:
        swap = y0
        y0 = y1
        y1 = swap

    print(f"region_x0: {region_x0}")
    print(f"region_x1: {region_x1}")
    print(f"region_y0: {region_y0}")
    print(f"region_y1: {region_y1}")
    print(f"x0: {x0}")
    print(f"x1: {x1}")
    print(f"y0: {y0}")
    print(f"y1: {y1}")

    data_region = data[y0:y1, x0:x1]

    # Useful for the client (displaying stuff), so include it in the return.
    relative_coordinates = {
        "x0": min(region_x0, region_x1),
        "x1": max(region_x0, region_x1),
        "y0": min(region_y0, region_y1),
        "y1": max(region_y0, region_y1),
    }
    return (data_region, relative_coordinates)


def getStarProfiles(event, context):

    data_region, region_coords = _get_region(event)
    
    return_data = sepAnalysis(data_region, event, False)
    return_data['region_coords'] = region_coords
    return_data = json.dumps(return_data, cls=NumpyEncoder)
    return _get_response(200, return_data)

def getRegionStats(event, context):

    data_region, _ = _get_region(event)

    #import matplotlib.pyplot as plt
    #plt.imshow(data_region) 
    #plt.savefig("region_preview")

    mean, median, std = sigma_clipped_stats(data_region, sigma=3.0)
    print((mean, median, std))
    return_data = json.dumps({
        "mean": mean,
        "median": median,
        "std": std,
        "min": data_region.min(),
        "max": data_region.max(),
    }, cls=NumpyEncoder)
    return _get_response(200, return_data)




if __name__=="__main__":

    site = "wmd"
    base_filename = "wmd-kf01-20200218-00001139"
    base_filename = "wmd-gf03-20191124-00001228"
    #base_filename = "wmd-gf01-20190924-00008661"
    #base_filename = "wmd-kf01-20200214-00000017"
    #base_filename = "wmd-gf03-20191124-00001208"

    fake_event = {
        "body": json.dumps({
            "site": site,
            "base_filename": base_filename,
            "fitstype": "01",
            
            "region_x0": 0.27168611582788393, 
            "region_x1": 0.5399038982622204,
            "region_y0": 0.5110011199826583,
            "region_y1": 0.8300877921890242,

        })
    }
    fake_event2 = {
        "body": json.dumps({
            "site": site,
            "base_filename": base_filename,
            "region_x1": 0.6891950162267663, 
            "region_x0": 0.7327689898554343,
            "region_y1": 0.6332750834033088,
            "region_y0": 0.698636043846311,
            "fitstype": "01"
        })
    }
    print(json.loads(fake_event.get('body','')))

    getStarProfiles(fake_event, '')

    response = getRegionStats(fake_event, '')
    print(response)
    #response2 = getRegionStats(fake_event2, '')
    #print(response2)
