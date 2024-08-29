"""Module for processing and analyzing SOFAST data for an ensemble of mirrors.

This script performs the following steps:
1. Load saved facet ensemble SOFAST collection data from an HDF5 file.
2. Save projected sinusoidal fringe images to PNG format.
3. Save captured sinusoidal fringe images and mask images to PNG format.
4. Process data with SOFAST and save processed data to HDF5.
5. Generate a suite of plots and save image files.

Examples
--------
To run the script, simply execute it as a standalone program:

>>> python example_process_facet_ensemble.py

This will perform the processing steps and save the results to the data/output/single_facet directory
with the following subfolders:
1_images_fringes_projected - The patterns sent to the display during the SOFAST measurement of the optic.
2_images_captured - The captured images of the displayed patterns as seen by the SOFAST camera
3_processed_data - The processed data from SOFAST.
4_processed_output_figures - The output figure suite from a SOFAST characterization.

Notes
-----
- The script assumes that the input data files are located in the specified directories.
"""

import json
from os.path import join, dirname

import imageio.v3 as imageio

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SofastConfiguration import SofastConfiguration
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_process_facet_ensemble():
    """Performs processing of previously collected SOFAST data of facet ensemble.

    1. Load saved facet ensemble Sofast collection data
    2. Save projected sinusoidal fringe images to PNG format
    3. Save captured sinusoidal fringe images and mask images to PNG format
    4. Processes data with SOFAST and save processed data to HDF5
    5. Generate plot suite and save images files
    """
    # General setup
    # =============

    # Define save dir
    dir_save = join(dirname(__file__), 'data/output/facet_ensemble')
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    lt.logger(join(dir_save, 'log.txt'), lt.log.WARN)

    # Define sample data directory
    dir_data_sofast = join(opencsp_code_dir(), 'test/data/sofast_fringe')
    dir_data_common = join(opencsp_code_dir(), 'test/data/sofast_common')

    # Directory Setup
    file_measurement = join(dir_data_sofast, 'data_measurement/measurement_ensemble.h5')
    file_camera = join(dir_data_common, 'camera_sofast_downsampled.h5')
    file_display = join(dir_data_common, 'display_distorted_2d.h5')
    file_orientation = join(dir_data_common, 'spatial_orientation.h5')
    file_calibration = join(dir_data_sofast, 'data_measurement/image_calibration.h5')
    file_facet = join(dir_data_common, 'Facet_lab_6x4.json')
    file_ensemble = join(dir_data_common, 'Ensemble_lab_6x4.json')
    file_params = join(dir_data_sofast, 'data_expected_facet_ensemble/data.h5')

    # 1. Load saved facet ensemble Sofast collection data
    # =================================================
    camera = Camera.load_from_hdf(file_camera)
    display = DisplayShape.load_from_hdf(file_display)
    orientation = SpatialOrientation.load_from_hdf(file_orientation)
    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    data_ensemble = DefinitionEnsemble.load_from_json(file_ensemble)
    data_facets = [DefinitionFacet.load_from_json(file_facet)] * data_ensemble.num_facets
    data_surfaces = [Surface2DParabolic((500.0, 500.0), False, 10)] * data_ensemble.num_facets

    # 2. Save projected sinusoidal fringe images to PNG format
    # ========================================================
    fringes = Fringes(measurement.fringe_periods_x, measurement.fringe_periods_y)
    images = fringes.get_frames(640, 320, 'uint8', [0, 255])
    dir_save_cur = join(dir_save, '1_images_fringes_projected')
    ft.create_directories_if_necessary(dir_save_cur)
    # Save y images
    for idx_image in range(measurement.num_y_ims):
        image = images[..., idx_image]
        imageio.imwrite(join(dir_save_cur, f'y_{idx_image:02d}.png'), image)
    # Save x images
    for idx_image in range(measurement.num_x_ims):
        image = images[..., measurement.num_y_ims + idx_image]
        imageio.imwrite(join(dir_save_cur, f'x_{idx_image:02d}.png'), image)

    # 3. Save captured sinusoidal fringe images and mask images to PNG format
    # =======================================================================
    dir_save_cur = join(dir_save, '2_images_captured')
    ft.create_directories_if_necessary(dir_save_cur)

    # Save mask images
    for idx_image in [0, 1]:
        image = measurement.mask_images[..., idx_image]
        imageio.imwrite(join(dir_save_cur, f'mask_{idx_image:02d}.png'), image)
    # Save y images
    for idx_image in range(measurement.num_y_ims):
        image = measurement.fringe_images_y[..., idx_image]
        imageio.imwrite(join(dir_save_cur, f'y_{idx_image:02d}.png'), image)
    # Save x images
    for idx_image in range(measurement.num_x_ims):
        image = measurement.fringe_images_x[..., idx_image]
        imageio.imwrite(join(dir_save_cur, f'x_{idx_image:02d}.png'), image)

    # 4. Processes data with Sofast and save processed data to HDF5
    # =============================================================
    dir_save_cur = join(dir_save, '3_processed_data')
    ft.create_directories_if_necessary(dir_save_cur)

    # Calibrate measurement
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = ProcessSofastFringe(measurement, orientation, camera, display)

    # Update sofast processing parameters
    sofast.params = sofast.params.load_from_hdf(file_params, 'DataSofastInput/')

    # Process
    sofast.process_optic_multifacet(data_facets, data_ensemble, data_surfaces)

    # Save processed data to HDF5 format
    sofast.save_to_hdf(join(dir_save_cur, 'data_sofast_processed.h5'))

    # Save measurement statistics to JSON
    config = SofastConfiguration()
    config.load_sofast_object(sofast)
    measurement_stats = config.get_measurement_stats()

    # Save measurement stats as JSON
    with open(join(dir_save_cur, 'measurement_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(measurement_stats, f, indent=3)

    # 5. Generate plot suite and save images files
    # ============================================
    dir_save_cur = join(dir_save, '4_processed_output_figures')
    ft.create_directories_if_necessary(dir_save_cur)

    # Get measured and reference optics
    ensemble_measured = sofast.get_optic()
    region = ensemble_measured.facets[0].mirror.region
    facets = [Facet(MirrorParametric.generate_flat(region)) for _ in range(sofast.num_facets)]
    ensemble_reference = FacetEnsemble(facets)
    ensemble_reference.set_facet_positions(data_ensemble.v_facet_locations)

    # Define viewing/illumination geometry
    v_target_center = Vxyz((0, 0, 100))
    v_target_normal = Vxyz((0, 0, -1))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

    # Save optic objects
    plots = StandardPlotOutput()
    plots.optic_measured = ensemble_measured
    plots.optic_reference = ensemble_reference

    # Update visualization parameters
    plots.options_slope_vis.clim = 7
    plots.options_slope_deviation_vis.clim = 1.5
    plots.options_ray_trace_vis.enclosed_energy_max_semi_width = 1
    plots.options_file_output.to_save = True
    plots.options_file_output.number_in_name = False
    plots.options_file_output.output_dir = dir_save_cur

    # Define ray trace parameters
    plots.params_ray_trace.source = source
    plots.params_ray_trace.v_target_center = v_target_center
    plots.params_ray_trace.v_target_normal = v_target_normal

    # Create standard output plots
    plots.plot()


if __name__ == '__main__':
    example_process_facet_ensemble()
