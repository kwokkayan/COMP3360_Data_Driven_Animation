
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from utils import *

'''
Your implement here: Implement more interpolation algorithms on motion interpolation task
To-Do:
Complete the apply_interpolation function with two interpolation algorithms
* Students are encouraged to use existing libraries for this, like SciPy
* Not necessarily Bezier, such as nearest, nearest-up, slinear, quadratic which can be chosen
* You can add parameters with your own design
'''
def apply_interpolation(data, keyframes, interpolation_method, is_rotation=True):
    data_filled = data.copy()
    if interpolation_method == 'linear':
        start_idx = keyframes[0]
        for end_idx in keyframes[1:]:
            start_data, end_data = data[start_idx], data[end_idx]
            in_fills = end_idx - start_idx
            base_incrementation = (end_data - start_data)/(in_fills)
            data_filled[start_idx:end_idx] = np.arange(0, in_fills)[:, np.newaxis]*base_incrementation[np.newaxis, :].repeat(in_fills, axis=0) + data_filled[start_idx]
            # A slower way but easier to understand:
            # for times in range(0, end_idx - start_idx):
            #     data_filled[start_idx + times] = start_data + base_incrementation*times
            start_idx = end_idx
    elif interpolation_method == 'akima':
        # four points tgt
        # start_idx = keyframes[0]
        # for end_idx in keyframes[1:]:
        #     times = np.asarray([0, end_idx - start_idx])
        #     start_data = data[start_idx].reshape(data[start_idx].shape[0], -1)
        #     end_data = data[end_idx].reshape(data[end_idx].shape[0], -1)
        #     splines = [interp1d(times, [start_data[i][0], end_data[i][0]], "quadratic") for i in range(len(start_data))]
        #     for times in range(1, end_idx - start_idx):
        #         result = [s(times) for s in splines]
        #         data_filled[start_idx + times] = result
        #     start_idx = end_idx
        for start_idx in range(0, len(keyframes), 4):
            times = [keyframes[i] for i in range(start_idx + 4)]
            points = [data[i].reshape(data[i].shape[0], -1) for i in times]
            splines = []
            for i in range(len(points[0])):
                x = [j[i][0] for j in points]
                splines.append(interp1d(times, x, "quadratic"))
            for t in range(1, times[-1] - times[0]):
                result = [s(t) for s in splines]
                data_filled[times[0] + t] = result
    elif interpolation_method == 'cubicspline': # cubic spline
        start_idx = keyframes[0]
        for end_idx in keyframes[1:]:
            times = [0, end_idx - start_idx]
            pos = [data[start_idx], data[end_idx]]
            spline = CubicSpline(times, pos)
            for times in range(1, end_idx - start_idx):
                data_filled[start_idx + times] = spline(times)
            start_idx = end_idx
    else:
        raise NotImplementedError('No support interpolation way %s' % interpolation_method)
    return data_filled


OFFSET = 124
METHOD = 'akima'
bvh_file_path = './data/motion_walking.bvh'
rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path)
rotations_fake, positions_fake = np.zeros_like(rotations.qs), np.zeros_like(positions)

keyframes = np.arange(1, rotations.shape[0], OFFSET)

rotations_fake[keyframes] = rotations.qs[keyframes]
positions_fake[keyframes] = positions[keyframes]

for joint_index in range(rotations.shape[1]):
    rotations_fake[:, joint_index, :] = apply_interpolation(rotations_fake[:, joint_index], 
                                                            keyframes=keyframes, 
                                                            interpolation_method=METHOD)
    print("finished joint", joint_index)
positions_fake[:, 0, :] = apply_interpolation(positions_fake[:, 0], 
                                                            keyframes=keyframes,
                                                            interpolation_method=METHOD,
                                                            is_rotation=False)

output_file_path = '%s_interpolate_%s_%s.bvh' % (bvh_file_path[:-4], OFFSET, METHOD)
save(output_file_path, Quaternions(rotations_fake).normalized(), positions_fake, offsets, parents, names, frametime)
error_rotation = np.sum((rotations_fake - rotations.qs)**2, axis=-1).mean()
error_position = (positions_fake[:, 0, :] - positions[:, 0, :]).mean()
print("error log: rotation =", error_rotation, "position =", error_position)
subprocess.call('blender -P load_bvhs.py -- -r %s -c %s' % (bvh_file_path, output_file_path), shell=True)
# subprocess.call('blender -P load_bvhs.py -- -r %s -c %s --render' % (bvh_file_path, output_file_path), shell=True)