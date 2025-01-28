import numpy as np
import tables as tb
from pathlib import Path

#
# Finds the lates noise occupancy scan in a given folder and prints the disabled pixels to terminal
# also these are saved in a yaml file
#

def find_latest_file(path: str, index: str):
    """Find latest file that includes a given subset of strings called index in directory.

    Args:
        path (str): Path to directory. For same directory as python script use for e.q. './target_dir'.
        index (str): Find specific characters in filename.

    Returns:
        path: Path to file in target Director. Use str(find_latest_path(.)) to obtain path as string.
    """
    p = Path(path)
    return max(
        [x for x in p.iterdir() if x.is_file() and index in str(x)],
        key=lambda item: item.stat().st_ctime,
    )

folder_path = ''
filepath_in = find_latest_file(folder_path, 'noise_occupancy_scan_interpreted.h5')

print('Using File: %s' %filepath_in)

with tb.open_file(filepath_in, "r") as in_file:
    pixel_mask = in_file.root.configuration_out.chip.use_pixel[:]
 
print('--- Disabled Pixels ---')   
disabled_pixels = np.array(np.where(pixel_mask==False))
print(disabled_pixels)

with open(folder_path + '/masked_pixels.yaml', 'w') as file:
    file.write('rows , cols\n')
    for i in range(np.shape(disabled_pixels)[1]):
        file.write(str(disabled_pixels[:,i]))