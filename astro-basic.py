# astro.py

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

logger = logging.getLogger("handler_logger")
logger.setLevel(logging.DEBUG)

###################################
######### Helper Functions ########
###################################

# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

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
        api_url = f"https://api.photonranch.org/api/fits{EX01orEX13}_url/{base_filename}"
        print(f"API url: {api_url}")
        file_url = requests.get(api_url).text
        print(f"File url: {file_url}")

        # Load the file from the url
        with fits.open(fileURL) as f:
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

def getRegionStats(event, context):

    body = _get_body(event)

    site = body.get("site")
    base_filename = body.get("base_filename")
    EXversion = "01" # we want the full sized fits file

    fitsfile = _get_fits(site, base_filename, EXversion)
    header = fitsfile[0].header
    data = fitsfile[0].data

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print((mean, median, std))
    return_data = {
        "mean": mean,
        "median": median,
        "std": std
    }
    return _get_response(200, return_data)

def hello(event,context):
   icrs = coord.ICRS(ra=258.58356362*u.deg, dec=14.55255619*u.deg, radial_velocity=-16.1*u.km/u.s)
   print("icrs:")
   print(icrs)
   return icrs

 
