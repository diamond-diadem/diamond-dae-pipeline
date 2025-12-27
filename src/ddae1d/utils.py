import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import torch
    

def despike(spectrum, max_width=5, local_window=20, height_ratio=4):
    """
    Remove spikes from a 1D spectrum using adaptive thresholding.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array representing the spectrum to be despiked.
    max_width : int, optional
        Maximum allowed width (in samples) for a spike to be considered (default: 5).
    local_window : int, optional
        Number of points to use on each side of a peak to estimate local noise statistics (default: 20).
    height_ratio : float, optional
        Minimum ratio of peak height to local standard deviation to consider a spike (default: 5).

    Returns
    -------
    cleaned_spectrum : np.ndarray
        The despiked spectrum with spikes replaced by linear interpolation.
    spikes : np.ndarray
        Array of same shape as spectrum, containing the removed spike values (zero elsewhere).
    """
    indices = np.arange(len(spectrum))
    # Find all peaks in the spectrum
    peaks, _ = find_peaks(spectrum)
    # Measure the width of each peak at half maximum
    results_half = peak_widths(spectrum, peaks, rel_height=0.5)
    widths = results_half[0]
    left_ips = results_half[2].astype(int)
    right_ips = results_half[3].astype(int)
    selected_peaks = []
    left_bounds = []
    right_bounds = []
    # Loop through each detected peak and apply selection criteria
    for peak, width, left, right in zip(peaks, widths, left_ips, right_ips):
        if width > max_width:
            continue  # Skip peaks that are too wide to be spikes
        local_left = max(0, peak - local_window)
        local_right = min(len(spectrum), peak + local_window)
        # Exclude the peak region itself from local statistics
        local_region = np.concatenate((
            spectrum[local_left:left],
            spectrum[right:local_right]
        ))
        if len(local_region) < 5:
            continue  # Not enough points to estimate local statistics
        local_std = np.std(local_region)
        local_mean = np.mean(local_region)
        if local_std == 0:
            continue  # Avoid division by zero
        peak_height = spectrum[peak] - local_mean
        if peak_height / local_std < height_ratio:
            continue  # Peak is not high enough above local noise
        # Only consider positive spikes
        if peak_height <= 0:
            continue  # Forbid negative spikes
        # This peak is considered a spike
        selected_peaks.append(peak)
        left_bounds.append(left)
        right_bounds.append(right)
    # Create a mask for non-spike regions
    mask = np.ones_like(spectrum, dtype=bool)
    for left, right in zip(left_bounds, right_bounds):
        mask[left:right+1] = False  # Mark spike regions as False
    # Interpolate over spike regions
    f_interp = interp1d(indices[mask], spectrum[mask], kind='linear', fill_value='extrapolate')
    cleaned_spectrum = spectrum.copy()
    cleaned_spectrum[~mask] = f_interp(indices[~mask])
    # Store the removed spikes, prevent negative values
    spikes = np.zeros_like(spectrum)
    spikes_raw = spectrum[~mask] - cleaned_spectrum[~mask]
    spikes[~mask] = np.maximum(spikes_raw, 0)
    # Ensure cleaned_spectrum does not exceed original spectrum where spikes were removed
    cleaned_spectrum[~mask] = spectrum[~mask] - spikes[~mask]
    return cleaned_spectrum, spikes

def despike_iterative(spectrum, max_width=5, local_window=20, height_ratio=4, max_iter=5, verbose=False):
    """
    Iteratively despike the data until no more spikes are detected.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array representing the spectrum to be despiked.
    max_width : int, optional
        Maximum allowed width (in samples) for a spike to be considered (default: 5).
    local_window : int, optional
        Number of points to use on each side of a peak to estimate local noise statistics (default: 20).
    height_ratio : float, optional
        Minimum ratio of peak height to local standard deviation to consider a spike (default: 4).
    max_iter : int, optional
        Maximum number of iterations (default: 5).
    verbose : bool, optional
        If True, print the number of iterations performed (default: False).

    Returns
    -------
    spectrum_despiked : np.ndarray
        The despiked spectrum.
    spikes : np.ndarray
        Array of same shape as spectrum, containing the removed spike values (zero elsewhere).
    """
    spectrum_despiked = spectrum.copy()
    spikes = np.zeros_like(spectrum)
    
    for iter in range(max_iter):
        spectrum_despiked, new_spikes = despike(
            spectrum_despiked, 
            max_width=max_width, 
            local_window=local_window, 
            height_ratio=height_ratio
        )
        if np.all(new_spikes == 0):
            break
        spikes += new_spikes

    if verbose:
        if iter == 0:
            print("No despiking needed.")
        elif iter == 1:
            print(f"Despiked in 1 iteration.")
        else:
            print(f"Despiked in {iter} iterations.")
    
    return spectrum_despiked, spikes

def compute_raman_shift(laser_wavelength, wavelengths):
    """Compute Raman shift (in cm^-1) from a laser wavelength and an array of wavelengths.
    
    Parameters
    ----------
    laser_wavelength : float
        Excitation laser wavelength in nanometers.
    wavelengths : array_like
        Measured/scattered wavelengths in nanometers.
    
    Returns
    -------
    numpy.ndarray
        Raman shift values in cm^-1.
    """
    wavelengths = np.asarray(wavelengths)
    return 1e7 * (1/laser_wavelength - 1/wavelengths)
