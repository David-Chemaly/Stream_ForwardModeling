import dynesty
import corner
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_corner_ndim12(path_save):

    params = np.loadtxt(path_save+'params.txt')

    with open(path_save+'dict_result.pkl', 'rb') as f:
        dict_model = pickle.load(f)
    
    # Create a corner plot with the true values
    figure = corner.corner(dict_model['samps'], 
                            labels = [r'logM$_{halo}$', r'R$_s$', r'$q$', r'dir$_x$', r'dir$_y$', r'dir$_z$', 
                                        r'logm$_{sat}$', r'r$_s$',
                                        r'x$_0$', r'y$_0$', r'z$_0$', r'v$_x$', r'v$_y$', r'v$_z$', 
                                        'time'],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, 
                            title_kwargs={"fontsize": 12},
                            truths=params, 
                            truth_color='red')
    # Show the plot
    plt.show()

