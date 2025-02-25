from os.path import join, dirname

import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def scene_reconstruction(
    dir_output: str,
    file_camera: str,
    file_known_point_locations: str,
    file_point_pair_distances: str,
    file_alignment_points: str,
    image_filter_path: str,
):
    """
    Reconstructs the XYZ locations of Aruco markers in a scene.

    Parameters
    ----------
    dir_output : str
        The directory where the output files, including point locations and calibration figures, will be saved.
    file_camera : str
        HDF5 file containing camera parameters.
    file_known_point_locations : str
        CSV file with known point locations.
    file_point_pair_distances : str
        CSV file with distances between point pairs.
    file_alignment_points : str
        CSV file with alignment points.
    image_filter_path : str
        Directory containing images of Aruco markers.

    Notes
    -----
    This function performs the following steps:

    1. Loads the camera parameters from an HDF5 file.
    2. Loads known point locations, point pair distances, and alignment points from CSV files.
    3. Initializes the SceneReconstruction object with the camera parameters and known point locations.
    4. Runs the calibration process to determine the marker positions.
    5. Scales the points based on the provided point pair distances.
    6. Aligns the points using the provided alignment points.
    7. Saves the reconstructed point locations to a CSV file.
    8. Saves calibration figures as PNG files in the output directory.

    Examples
    --------
    >>> # Run on sample data
    >>> dir_base = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
    >>> file_kwargs = {
    >>>     "file_camera": join(dir_base, 'camera.h5'),
    >>>     "file_known_point_locations": join(dir_base, 'known_point_locations.csv'),
    >>>     "image_filter_path": join(dir_base, 'aruco_marker_images/*.JPG'),
    >>>     "file_point_pair_distances": join(dir_base, 'point_pair_distances.csv'),
    >>>     "file_alignment_points": join(dir_base, 'alignment_points.csv'),
    >>> }
    >>> dir_output_working = join(dirname(__file__), 'data/output/scene_reconstruction')
    >>> scene_reconstruction(dir_output_working, **file_kwargs)
    """
    # "ChatGPT 4o" assisted with generating this docstring.

    # Define output directory
    ft.create_directories_if_necessary(dir_output)

    # Set up logger
    lt.logger(join(dir_output, 'log.txt'), lt.log.INFO)

    # Load components
    camera = Camera.load_from_hdf(file_camera)
    known_point_locations = np.loadtxt(file_known_point_locations, delimiter=',', skiprows=1)
    point_pair_distances = np.loadtxt(file_point_pair_distances, delimiter=',', skiprows=1)
    alignment_points = np.loadtxt(file_alignment_points, delimiter=',', skiprows=1)

    # Perform marker position calibration
    cal_scene_recon = SceneReconstruction(camera, known_point_locations, image_filter_path)
    cal_scene_recon.make_figures = True
    cal_scene_recon.run_calibration()

    # Scale points
    point_pairs = point_pair_distances[:, :2].astype(int)
    distances = point_pair_distances[:, 2]
    cal_scene_recon.scale_points(point_pairs, distances)

    # Align points
    marker_ids = alignment_points[:, 0].astype(int)
    alignment_values = Vxyz(alignment_points[:, 1:4].T)
    cal_scene_recon.align_points(marker_ids, alignment_values)

    # Save points as CSV
    cal_scene_recon.save_data_as_csv(join(dir_output, 'point_locations.csv'))

    # Save calibrtion figures
    for fig in cal_scene_recon.figures:
        fig.savefig(join(dir_output, fig.get_label() + '.png'))


if __name__ == '__main__':
    # Setup directories to load example data
    dir_base = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
    file_kwargs = {
        "file_camera": join(dir_base, 'camera.h5'),
        "file_known_point_locations": join(dir_base, 'known_point_locations.csv'),
        "image_filter_path": join(dir_base, 'aruco_marker_images/*.JPG'),
        "file_point_pair_distances": join(dir_base, 'point_pair_distances.csv'),
        "file_alignment_points": join(dir_base, 'alignment_points.csv'),
    }
    dir_output_working = join(dirname(__file__), 'data/output/scene_reconstruction')

    # Run
    scene_reconstruction(dir_output_working, **file_kwargs)
