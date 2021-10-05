#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Fri Oct 01 2021 at 11:12:51 AM
# ==========================================================
# __copyright__ = Copyright (c) 2021, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

import h5py
from RR.cfg import DATA_FOLDER


def read_data(catch_id=2034):
    """The function reads the data for a catchment

    Xd has 5 dimensions in the order
    Precipitation 
    Mean temp
    Solar Radiation
    Max temp
    Min temp

    Args:
        catch_id (int, optional): [description]. Defaults to 2034.

    Returns:
        [type]: [description]
    """

    fname = DATA_FOLDER + f'/data_'+str(catch_id) + '.h5'

    with h5py.File(fname, 'r') as f:
        Xd = f['Xd'][:,:].T
        Y = f['Y'][0,:]
        elp = f['elevation_percentile'][:,0]
        flp = f['flow_length_percentile'][:,0]
        slp = f['slope_percentile'][:,0]
        area = f['catchment_area'][()] # m2

    return Xd, Y, elp, flp, slp, area