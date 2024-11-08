"""
The aim here is to create several versions of an input image with different
orientations

    Mo Shams <MShamsCBR@gmail.com>
    May 07, 2023
"""

import os
import numpy as np
from PIL import Image

# set the save image path
save_path = os.path.join("", "FBA")

category_base = ['image1', 'image2']

# set the tilt range (deg)
min_tilt = 0
max_tilt = 10
step_tilt = 0.1

for icat in category_base:
    source_path = os.path.join("", "FBA", f"{icat}.png")
    im = Image.open(source_path)

    mags = np.arange(min_tilt, max_tilt, step_tilt)
    for imag, mag in enumerate(mags):
        im_output = im.rotate(mag)
        if imag == 0:
            im_output.save(os.path.join(save_path,
                                        f'image{icat}_tilt{imag}.png'))
        else:
            im_output.save(os.path.join(save_path,
                                        f'image{icat}_tilt{imag}_CCW.png'))

    mags = np.arange(min_tilt, max_tilt, step_tilt)
    for imag, mag in enumerate(mags):
        im_output = im.rotate(mag)
        if imag == 0:
            im_output.save(os.path.join(save_path,
                                        f'image{icat}_tilt{imag}.png'))
        else:
            im_output.save(os.path.join(save_path,
                                        f'image{icat}_tilt{imag}_CW.png'))
