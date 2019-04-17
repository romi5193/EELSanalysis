%matplotlib qt4
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from hyperspy.hspy import *

#import
im = hs.load('EELS Spectrum Image 1.dm3')
im.plot()
im.spikes_removal_tool()
im.remove_background()

#isolate Carbon_k-edge
s_C=im.isig[250.0:340.0].deepcopy()
s_C.plot()

#determine # of components

s_C.decomposition(False, algorithm='svd')
s_C.plot_explained_variance_ratio()
s_C.get_decomposition_model(components=5).plot()
s_C.plot_decomposition_results()

s_C.blind_source_separation(number_of_components=6)
s_C.plot_bss_results()
s_C.reverse_bss_component(2)
s_C.reverse_bss_component(1)
s_C.plot_bss_results()

#plot and save final results
utils.plot.plot_images([i for i in s_C.get_bss_loadings()], cmap='cubehelix',
                      per_row=2,
                      colorbar='multi',
                      axes_decor=None,
                      label='auto',
                      scalebar=[0],
                      scalebar_color='black')

utils.plot.plot_spectra(s_C.get_bss_factors(),
                       style='cascade',legend=None,legend_loc=0)
s_C.save("EELS_SI4_CKedge_SVD5.hdf5")
plt.close('all')