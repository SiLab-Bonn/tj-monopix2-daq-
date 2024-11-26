# start with taskset -c 0  python threshold_plot.py -p path_to_files

import argparse
import os
from datetime import datetime
import matplotlib.backends.backend_pdf as pdf

import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from tqdm import tqdm
from typing import List, Tuple
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import csv


# Parse the path to the folder as an argument
parser = argparse.ArgumentParser(
    description='Combine HistOcc arrays from multiple h5 files')
parser.add_argument('-p', '--path', type=str,
                    help='Path to the folder containing h5 files')
parser.add_argument('-f', '--file_name', type=str,
                    help='Ph5 filename')
parser.add_argument('-fit_HV', '--fit_range_HV', type=int,
                    default=0,
                    help='where the fit range starts in bins')
parser.add_argument('-fit_HVCASC', '--fit_range_HVCASC', type=int,
                    default=0,
                    help='where the fit range starts in bins')
parser.add_argument('-fit_NF', '--fit_range_NF', type=int,
                    default=0,
                    help='where the fit range starts in bins')
parser.add_argument('-fit_NFCASC', '--fit_range_NFCASC', type=int,
                    default=0,
                    help='where the fit range starts in bins')
parser.add_argument('-plot', '--plot_range', type=int,
                    default=0,
                    help='where the plot range starts in bins')
parser.add_argument('-center', '--center', type=int,
                    default=0,
                    help='givees center for threshold fit, if needed')
args = parser.parse_args()

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[0, :] = white
newcmp = ListedColormap(newcolors)

file_name= args.file_name

def is_in_range(val, lower, upper):
    return lower <= val < upper




def get_scan_config(file_name, folder_path):
    with tb.open_file(os.path.join(folder_path, file_name), 'r') as f:
        # Get the HistOcc array
        print(os.path.join(folder_path, file_name))
        start_column = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"start_column"')['value']
        start_column = int(start_column[0].decode("utf-8"))
        stop_column = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"stop_column"')['value']
        stop_column = int(stop_column[0].decode("utf-8"))
        start_row = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"start_row"')['value']
        start_row = int(start_row[0].decode("utf-8"))
        stop_row = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"stop_row"')['value']
        stop_row = int(stop_row[0].decode("utf-8"))
        n_injections = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"n_injections"')['value']
        n_injections = int(n_injections[0].decode("utf-8"))
        v_low_start = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"VCAL_LOW_start"')['value']
        v_low_start = int(v_low_start[0].decode("utf-8"))
        v_low_stop = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"VCAL_LOW_stop"')['value']
        v_low_stop = int(v_low_stop[0].decode("utf-8"))
        v_high = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"VCAL_HIGH"')['value']
        v_high = int(v_high[0].decode("utf-8"))
        v_step = f.root.configuration_in.scan.scan_config.read_where(
            'attribute == b"VCAL_LOW_step"')['value']
        v_step = int(v_step[0].decode("utf-8"))
    return [start_column, stop_column, start_row, stop_row, n_injections,v_low_start,v_low_stop,v_step,v_high]


def get_combined_hist_occ(file_name, delta_v, folder_path):
        with tb.open_file(os.path.join(folder_path, file_name), 'r') as f:
            hist_occ = f.root.HistOcc[:]
            hist_occ = np.asarray(hist_occ)
        return hist_occ


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y


def s_curve_fit(i, j):
    try:
        ydata = combined_hist_occ[i, j]
        if all(item == 0 for item in ydata):
            s_curve_x0[i, j] = 0
        else:
            p0 = [max(ydata)-min(ydata), np.median(xdata), 1, min(ydata)]
            popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
            s_curve_x0[i, j] = popt[1]
            plt.plot(xdata, ydata)
            # plt.show()

    except:
        s_curve_x0[i, j] = 0


def run_s_curve_fit(args):
    i, j = args
    s_curve_fit(i, j)


# Define a Gaussian function to fit to the histogram
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def get_results(file_name, folder_path, col_start, col_stop, row_start, row_stop):
    results = np.zeros(shape=(max(delta_v)+1, 128), dtype='int')
    h5file = tb.open_file(os.path.join(folder_path, file_name),
                              mode="r", title='configuration_in')
    HistToT = h5file.root.HistTot
    arr_ToT = np.asarray(HistToT)

    #arr_ToT_sum = np.zeros(shape=(128), dtype='int')
    for i,v in enumerate(delta_v):
        arr_ToT_sum = np.sum(arr_ToT[col_start:col_stop,row_start:row_stop,i,:], axis=(0, 1))
        for j in range(128):
            results[v, j] = arr_ToT_sum[j]
    h5file.close()
    
    return results
def get_HistOcc(file_name, folder_path):
    h5file = tb.open_file(os.path.join(folder_path, file_name),
                              mode="r", title='configuration_in')
    HistOcc = h5file.root.HistOcc
    arr_Occ = np.asarray(HistOcc)
    h5file.close()
    
    return arr_Occ

def get_HistToT(file_name, folder_path):
    h5file = tb.open_file(os.path.join(folder_path, file_name),
                              mode="r", title='configuration_in')
    HistToT = h5file.root.HistTot
    arr_ToT = np.asarray(HistToT).T
    h5file.close()
    
    return arr_ToT


def get_NoiseMap(file_name, folder_path):
    h5file = tb.open_file(os.path.join(folder_path, file_name),
                              mode="r", title='configuration_in')
    NoiseMap = h5file.root.NoiseMap
    arr_noise = np.asarray(NoiseMap).T
    h5file.close()
    return arr_noise

def get_thresh(file_name, folder_path):
    h5file = tb.open_file(os.path.join(folder_path, file_name),
                              mode="r", title='configuration_in')
    ThresholdMap = h5file.root.ThresholdMap
    s_curve_x0 = np.asarray(ThresholdMap)
    h5file.close()

    return s_curve_x0    


def multiply_by_10(x, pos):
    return int(x * 10.1)

def clean_data(results):
    """
    Fit a Gaussian to each column of the results array and get the mean and std.

    Args:
    results: 2D numpy array of shape (n, m) where n is the number of rows and m is the number of columns.
    x: 1D numpy array of shape (m,) representing the x-axis values.

    Returns:
    Tuple of three 1D numpy arrays: (means, stds, rows).
    """
    # Create the x array
    x = np.arange(0, 128)
    means, stds, rows = [], [], []
    # Iterate over columns and fit with gauss function
    for i, row in enumerate(results):
        # Initial guesses for parameters
        p0 = [row.max(), np.argmax(row), 5]

        # Fit with curve_fit
        try:
            popt, pcov = curve_fit(gauss, x, row, p0=p0)
            amplitude, mean, std = popt
            if mean != 0:
                rows.append(i)
                means.append(mean)
                stds.append(std)
        except:
            # Fit failed, skip this column
            continue
        #except RuntimeWarning:
        #    continue
    nan_indices = np.isnan(means)

    # Remove NaN values and their corresponding elements from both lists
    means = [x for i, x in enumerate(means) if not nan_indices[i]]
    stds = [x for i, x in enumerate(stds) if not nan_indices[i]]
    rows = [x for i, x in enumerate(rows) if not nan_indices[i]]
    return np.array(means), np.array(stds), np.array(rows)

# Define the fit function


def func(x, a, b, c, t):
    return a*x + b - (c/(x-t))

def get_label(start_col):
    if start_col < 224:
        return "DC coupled"
    elif start_col < 448:
        return "DC coupled Casc"
    elif start_col < 480:
        return "AC coupled Casc"
    else:
        return "AC coupled"
    
def get_label_region(region):
    if region[1] == 224:
        return "DC coupled"
    elif region[1] == 448:
        return "DC coupled Casc"
    elif region[1] == 480:
        return "AC coupled Casc"
    else:
        return "AC coupled"


if __name__ == '__main__':
    start_col, stop_col, start_row, stop_row, n_inj,v_low_start,v_low_stop,v_step,v_high = get_scan_config(
        file_name, args.path)
    delta_v = np.array(range(v_high-v_low_start,v_high-v_low_stop,v_step*-1))
    #print(delta_v)
    # Get tot_cal data
        # Define four different regions to analyze
    region1 = (0,224, start_row, stop_row)
    region2 = (224,448, start_row, stop_row)  # Modify these values
    region3 = (448,448+32, start_row, stop_row)  # Modify these values
    region4 = (448+32, 512, start_row, stop_row)  # Modify these values

    # Create a list of regions to analyze
    tot_cal_args = [region1, region2, region3, region4]

    # Create a list to store the results for each region
    tot_cal_results = []

    #tot_cal_args = [(start_col, stop_col, start_row, stop_row)]
    tot_cal_array = []
    for a in tot_cal_args:
        tot_cal_array.append(get_results(file_name, args.path, *a))

    combined_hist_occ = get_combined_hist_occ(
        file_name, delta_v, args.path)

    # Initialize variables
    s_curve_x0 = get_thresh(file_name, args.path)
    s_curve_x0 = np.multiply(s_curve_x0, 10.1)
    s_curve_x0 = s_curve_x0.T
    xdata = np.array(range(np.min(delta_v),np.max(delta_v)+1))
    #print(xdata)
    xdata = xdata* 10.1
    #print(xdata)
    popt_cal_NF = [0,0,0,0]
    popt_cal_NF_CASC = [0,0,0,0]
    popt_cal_HV_CASC  = [0,0,0,0]
    popt_cal_HV  = [0,0,0,0]

    # Create a PDF file
    with pdf.PdfPages(os.path.join(args.path, file_name + '_Calibration_curve.pdf')) as pdf_file:
        # plot 0: tot cal
        for i, tot_cal in enumerate(tot_cal_array):
            #print(i)
            if i ==0:
                fit_range = args.fit_range_NF
                section_name = 'NF'
            elif i ==1:
                fit_range = args.fit_range_NFCASC
                section_name = 'NF CASC'
            elif i ==2:
                fit_range = args.fit_range_HVCASC
                section_name = 'HV CASC'
            elif i ==3:
                fit_range = args.fit_range_HV
                section_name = 'HV'
            
            try:
                #print(tot_cal)
                folder_name = os.path.basename(os.path.normpath(args.path))
                means, stds, rows = clean_data(tot_cal)

                fig = plt.figure(figsize=(8, 6))
                x = rows  # Use rows for x-axis dimension
                #print(x)
                x = x* 10.1
                x_plot = np.linspace(x[args.plot_range],2500,1000)

                y = np.arange(tot_cal.shape[1])  # Use tot_cal.shape[1] for y-axis dimension

                # Create the ScalarMappable object
                cmap = newcmp

                # Plot the colormesh
                mesh = plt.pcolormesh(x, y, tot_cal.T[:,-len(x):], cmap=cmap, norm=colors.LogNorm())

                colorbar = plt.colorbar(mesh)
                colorbar.ax.tick_params(labelsize=12)
                colorbar.set_label('# of pixel', fontsize=14)

                # Set up x-axis tick labels
                ax = plt.gca()
                #ax.xaxis.set_major_formatter(ticker.FuncFormatter(multiply_by_10))

                # Fit the function to the data
                p0 = (0.08, 8, 300, 2.4)
                param_bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

                #print(x,means)
                try:
                    popt_cal, pcov_cal = curve_fit(func, x[fit_range:], means[fit_range:],
                                        sigma=stds[fit_range:], p0=p0, bounds=param_bounds)
                    plt.plot(x_plot, func(x_plot, *popt_cal), color='k', label='fit')
                    plt.text(10, 40, '$f(x)=a\cdot x + b - (c/(x-t))$',
                            color='k', fontsize=14)
                    func_params = ['a', 'b', 'c', 't']
                    for j, param in enumerate(popt_cal):
                        plt.text(
                            20, 35 - j *
                            5, f'{func_params[j]}={param:.3f} ± {np.sqrt(pcov_cal[j, j]):.3f}',
                            color='k', fontsize=14)
                    # Add a shaded region to indicate the fit range
                    plt.axvspan(x[fit_range], x[-1],
                                facecolor='#2ca02c', alpha=0.3)
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print('no fit')

                #plt.scatter(1616 / 10.1, 33, color='red', s=100, label='Fe${55}$-source')
                x = 1640
                y = 23.5
                error = 0.5

                #plt.errorbar(x, y, yerr=error, fmt='o', color='red', label='Fe$^{55}$')
                                # Plot with blue dashed line
                plt.plot(x_plot, func(x_plot, popt_cal[0] + 0.002, popt_cal[1], popt_cal[2], popt_cal[3]), color='blue', linestyle='--', label='+$2\sigma_a$')

                # Plot with magenta dashed line
                plt.plot(x_plot, func(x_plot, popt_cal[0] - 0.002, popt_cal[1], popt_cal[2], popt_cal[3]), color='magenta', linestyle='-.', label='-$2\sigma_a$')

                
                plt.xlabel('Injected Charge [$e^-$]', fontsize=14)
                plt.ylabel('ToT  [25ns]', fontsize=14)
                #plt.title(section_name)
                plt.ylim(0, 50)
                plt.xlim(0, 250*10.1)
                plt.gca().set_aspect('auto', adjustable='box')

                plt.tick_params(axis='both', which='both', labelsize=12)

                plt.legend(fontsize=14)
                plt.tight_layout()

                pdf_file.savefig() 
                if i ==0:
                    popt_cal_NF = popt_cal
                elif i ==1:
                    popt_cal_NF_CASC = popt_cal
                elif i ==2:
                    popt_cal_HV_CASC = popt_cal
                elif i ==3:
                    popt_cal_HV = popt_cal
            except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print('no clibration')

            #plt.show()
            # Save the plot as a PDF and add it to the PDF file
        #    plt.savefig(os.path.join(
        #        args.path, f'{folder_name} {"NF_casc.png" if i == 1 else "NF.png"}'), format='png')
        #    plt.savefig(os.path.join(
        #        args.path, f'{folder_name} {"NF_casc.pdf" if i == 1 else "NF.pdf"}'), format='pdf')
        #    pdf_file.savefig()  # Add this plot to the PDF file
        #print('THIS IS FINE')


        # Plot 1: S-Curve Plot
        print('plot s-curves')
        region1 = [0,224, start_row, stop_row]
        region2 = [224,448, start_row, stop_row]
        region3 = [448,448+32, start_row, stop_row]
        region4 = [448+32, 512, start_row, stop_row]
        regions = [region1, region2, region3, region4]
        #print(regions)
        for r,region in enumerate(regions):
            #print(r,region)
            try:
                fig, ax = plt.subplots()
                # Get the maximum value in the matrix
                matrix = get_HistOcc(file_name, args.path)[region[0]:region[1],region[2]:region[3]]
                max_value = np.max(matrix)

                # Calculate the occupancy values based on the maximum value
                occupancy_values = list(range(int(max_value) + 1))

                # Get the number of steps (length of entries per pixel)
                num_steps = matrix.shape[2]

                # Initialize an empty array to store the counts
                counts = np.zeros((num_steps, len(occupancy_values)))

                # Iterate over each injection step
                for step in range(num_steps):
                    # Count the number of pixels with each occupancy value for the current step
                    for i, occupancy in enumerate(occupancy_values):
                        counts[step, i] = np.sum(matrix[:, :, step] == occupancy)

                
                # Scale the x-axis values by 10.1
                x_values = delta_v * 10.1
                #print(x_values)
                # Limit the x-axis range up to 700
                x_values = x_values[x_values <= 700]

                # Filter the counts matrix based on the x-axis range
                counts = counts[:len(x_values)]
                #print(counts)
                fig, ax = plt.subplots()
                norm = colors.LogNorm(vmin=1, vmax=np.max(counts))
                occupancy_values = [x / 100 for x in occupancy_values]
                image = ax.pcolormesh(x_values, occupancy_values, counts.T, cmap=newcmp, norm=norm)

                # Create a custom tick formatter for displaying tick labels as '10^x'
                class LogFormatterSciNotation(ticker.LogFormatterSciNotation):
                    def __call__(self, x, pos=None):
                        return r'$10^{{{}}}$'.format(int(np.log10(x)))

                # Create the colorbar with logarithmic scale and custom tick formatter
                cbar = fig.colorbar(image, ax=ax, format=LogFormatterSciNotation(), ticks=ticker.LogLocator(base=10.0))
                cbar.set_label('# of pixels', fontsize=14)

                # Set the axis labels and title with larger font sizes
                ax.set_xlabel('Injected Charge [$e^-$]', fontsize=16)
                ax.set_ylabel('Occupancy', fontsize=16)

                #plt.title('S-Curve Plot' +str(region), fontsize=16)
                plt.title('S-Curve Plot', fontsize=16)
                plt.tight_layout()

                # Increase the font size of tick labels
                ax.tick_params(axis='both', which='both', labelsize=14)

                # Increase the font size of the colorbar tick labels
                cbar.ax.tick_params(labelsize=14)
                # Adjust the bottom margin
                plt.subplots_adjust(bottom=0.15)
                pdf_file.savefig()
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print('no clibration')

        #plt.show()
        plt.close('all')
        # Plot 2: Threshold map
        # Create the figure and axes
        fig, ax = plt.subplots()

        # Plot the image
        image = ax.imshow(s_curve_x0[:, :])

        # Set the color bar limits
        cbar = fig.colorbar(image, ax=ax, ticks=np.arange(0, 501, 100))
        cbar.set_label('Threshold [e-]', fontsize=14)
        cbar.ax.set_yticklabels(np.arange(0, 501, 100), fontsize=12)

        # Set the axis labels
        ax.set_xlabel('Column', fontsize=14)
        ax.set_ylabel('Row', fontsize=14)
        plt.tight_layout()

        # Increase the font size of tick labels
        ax.tick_params(axis='both', which='both', labelsize=12)

        pdf_file.savefig() 
        plt.close('all')

        # Plot 3: Threshold distribution
        for r,region in enumerate(regions):
            #print(r,region)
            fig = plt.figure()
            mean_nf, std_nf, mean_nf_casc, std_nf_casc = None, None, None, None
            for i, data in enumerate([s_curve_x0[region[2]:region[3],region[0]:region[1]],]):
                hist, bins = np.histogram(data, bins=50, range=(150,450))
                #print(i)
                x_fit = (bins[1:-1] + bins[2:]) / 2
                x_plot = (bins[1:-1] + bins[2:]) / 2
                label = get_label_region(region)
                try:
                    max_value = np.max(hist)
                    if args.center ==0 :
                        center = np.mean(data)
                    else: 
                        center = args.center
                    #print(center)
                    width = 5
                    #print(max_value, center)
                    p0 = [max_value, center, width]
                    popt, pcov = curve_fit(gauss, x_fit, hist[1:], p0=p0)
                    plt.plot(x_fit, gauss(x_fit, *popt), 'r-', label=label+ ' fit')
                    mean, std = popt[1], popt[2]
                    #print(mean,std)

                    plt.bar(x_plot, hist[1:], width=bins[1] - bins[0], label=label)
                except:
                    continue

            text_box_text =  f"µ = {popt[1]:.0f}$e^-$\n$\sigma$ = {popt[2]:.0f}$e^-$"
            if text_box_text:
                fig.text(0.7, 0.65, text_box_text, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5))

            # Set the axis labels and title
            plt.legend(fontsize=14)
            plt.xlabel('threshold [$e^-$]', fontsize=16)
            plt.ylabel('# pixel', fontsize=16)
            #plt.ylim(0, 400)
            #plt.xlim(0, 500)
            # Increase the font size of tick labels
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            # plt.savefig(os.path.join(args.path, 'Threshold_distribution.pdf'), format='pdf')
            pdf_file.savefig()  # Add this plot to the PDF file
            # plt.show()
            if r ==0:
                threshold_NF, threshold_NF_std =  popt[1], popt[2]
            elif r ==1:
                threshold_NF_CASC,threshold_NF_CASC_std =   popt[1], popt[2]
            elif r ==2:
                threshold_HV_CASC,threshold_HV_CASC_std =   popt[1], popt[2]
            elif r ==3:
                threshold_HV, threshold_HV_std =   popt[1], popt[2]

        plt.close('all')

        # Plot 4 Noise: Threshold map
        # Create the figure and axes
        fig, ax = plt.subplots()

        # Plot the image
        noise_map=get_NoiseMap(file_name, args.path)
        noise_map = np.multiply(noise_map, 10.1) 
        #
        image = ax.imshow(noise_map,vmin=0, vmax=20,)

        # Set the color bar limits
        cbar = fig.colorbar(image, ax=ax)

        cbar.set_label('Noise [e-]', fontsize=14)
        #cbar.ax.set_yticklabels(np.arange(0, 501, 100), fontsize=12)

        # Set the axis labels
        ax.set_xlabel('Column', fontsize=14)
        ax.set_ylabel('Row', fontsize=14)
        plt.tight_layout()

        # Increase the font size of tick labels
        #ax.tick_params(axis='both', which='both', labelsize=12)

        pdf_file.savefig() 
        plt.close('all')

        # Plot 5: Noise distribution
        print('plot noise distributions')
        mean_nf_noise, mean_nf_casc_noise, mean_hv_casc_noise, mean_hv_noise= 0,0,0,0
        std_nf_noise, std_nf_casc_noise, std_hv_casc_noise, std_hv_noise= 0,0,0,0
        for r,region in enumerate(regions):
            #print(r,region)
            fig = plt.figure()
            noise_map=get_NoiseMap(file_name, args.path)
            noise_map = np.multiply(noise_map, 10.1) 
            for i, data in enumerate([noise_map[region[2]:region[3],region[0]:region[1]],]):
                hist_noise, bins_noise = np.histogram(data, bins=50,range=(0,50))
                x_fit_noise = (bins_noise[1:-1] + bins_noise[2:]) / 2
                #print(x_fit_noise)

                x_plot_noise = (bins_noise[1:-1] + bins_noise[2:]) / 2
                plt.bar(x_plot_noise, hist_noise[1:], width=bins_noise[1] - bins_noise[0], label=label)
                print(r)
                try:
                    max_value_noise = np.max(hist)
                    center_noise = np.mean(data)
                    #print(center_noise)
                    width_noise = 10
                    p0_noise = [max_value_noise, center_noise, width_noise]
                    print('p0_noise',p0_noise)
                    #p0_noise = [1000, 5, 2]
                    #print('p0_noise',p0_noise)
                    popt_noise, pcov_noise = curve_fit(gauss, x_fit_noise, hist_noise[1:], p0=p0_noise)
                    label=get_label_region(region)
                    plt.plot(x_fit_noise, gauss(x_fit_noise, *popt_noise), 'r-', label=label+ ' fit')
                    mean_noise, std_noise = popt_noise[1], popt_noise[2]
                    

                    if mean_noise < 1000:  # Exclude plotting if mean is too large
                        if r == 0:
                            mean_nf_noise, std_nf_noise = mean_noise, std_noise
                        elif r ==1:
                            mean_nf_casc_noise, std_nf_casc_noise = mean_noise, std_noise
                        elif r ==2:
                            mean_hv_casc_noise, std_hv_casc_noise = mean_noise, std_noise
                        elif r ==3:
                            mean_hv_noise, std_hv_noise = mean_noise, std_noise
                            text_box_text = ''
                    if r==0:
                        text_box_text = f"µ = {mean_nf_noise:.0f}\n$\sigma$ = {abs(std_nf_noise):.0f}"
                    if r==1:
                        text_box_text = f"µ = {mean_nf_casc_noise:.0f}$e^-$\n$\sigma$ = {abs(std_nf_casc_noise):.0f}$e^-$"
                    if r==2:
                        text_box_text = f"µ = {mean_hv_casc_noise:.0f}$e^-$\n$\sigma$ = {abs(std_hv_casc_noise):.0f}$e^-$"
                    if r==3:
                        text_box_text = f"µ = {mean_hv_noise:.0f}$e^-$\n$\sigma$ = {abs(std_hv_noise):.0f}$e^-$"
                    # Plot the text box if there is content to display
                    fig.text(0.7, 0.65, text_box_text, fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5))

                except Exception as e:
                    print(f"Error occurred: {str(e)}")

            

            # Set the axis labels and title
            plt.legend(fontsize=16)
            #plt.xlim(0,100)
            plt.xlabel('Noises [$e^-$]', fontsize=16)
            plt.ylabel('# pixel', fontsize=16)
            # Increase the font size of tick labels
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            pdf_file.savefig()  # Add this plot to the PDF file


    #print(mean)
    ### NF

    v1 = [popt_cal_NF[0],popt_cal_NF[1],popt_cal_NF[2],popt_cal_NF[3],threshold_NF]
    print('v1',v1)
    rows, cols = (512, 224)
    arr1 = [[[*v1]]*cols]*rows

    v2 = [popt_cal_NF_CASC[0],popt_cal_NF_CASC[1],popt_cal_NF_CASC[2],popt_cal_NF_CASC[3],threshold_NF_CASC]
    print('v2',v2)
    rows, cols = (512, 224)
    arr2 = [[[*v2]]*cols]*rows

    v3 = [popt_cal_HV_CASC[0],popt_cal_HV_CASC[1],popt_cal_HV_CASC[2],popt_cal_HV_CASC[3],threshold_HV_CASC]
    print('v3',v3)
    rows, cols = (512, 32)
    arr3 = [[[*v3]]*cols]*rows

    v4 = [popt_cal_HV[0],popt_cal_HV[1],popt_cal_HV[2],popt_cal_HV[3],threshold_HV]
    print('v4',v4)
    rows, cols = (512, 32)
    arr4 = [[[*v4]]*cols]*rows


    arr=np.concatenate((arr1, arr2, arr3, arr4), axis=1)

    lable_one = np.array(range(0,512))
    lable_two = np.array([0, 1, 2, 3, 4])

    cols = pd.MultiIndex.from_product([lable_one, lable_two])

    df=pd.DataFrame(arr.reshape(512, -1), columns=cols)
    df.to_hdf(os.path.join(args.path, 'charge_calib.h5'),'df',mode='w',format='table')

    print(f"Saved")
    #print(df)


# Assume you have threshold values, errors, noise, and noise errors as lists
threshold_values = [threshold_NF, threshold_NF_CASC, threshold_HV_CASC, threshold_HV]
errors = [threshold_NF_std, threshold_NF_CASC_std, threshold_HV_CASC_std, threshold_HV_std]
noise = [mean_nf_noise, mean_nf_casc_noise, mean_hv_casc_noise, mean_hv_noise]
print('Noise', noise)
noise_errors = [std_nf_noise, std_nf_casc_noise, std_hv_casc_noise, std_hv_noise]

# Combine the lists into a list of tuples, replacing None with 0
data = [(0 if t is None else t, 0 if e is None else e, 0 if n is None else n, 0 if ne is None else ne)
        for t, e, n, ne in zip(threshold_values, errors, noise, noise_errors)]

# Specify the CSV file path
csv_file_path = os.path.join(args.path, 'threshold_and_noise.csv')

# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(['Threshold', 'Error', 'Noise', 'Noise Error'])

    # Write data
    csv_writer.writerows(data)

print(f'Data has been written to {csv_file_path}')
