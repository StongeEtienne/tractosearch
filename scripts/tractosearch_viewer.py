#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse

import numpy as np
import lpqtree

from tractosearch.io import load_slines
from tractosearch.resampling import resample_slines_to_array

try:
    import vtk
    import vtk.util.numpy_support as ns
except ImportError:
    print("This script requires VTK")
    exit()


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Streamlines to view')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--color', nargs=3, type=float, default=None,
                   help="Streamlines color [%(default)s]")

    p.add_argument('--mean_distance', type=float, default=4.0,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF)')

    p.add_argument('--min_count', type=int, default=1,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF)')

    p.add_argument('--resample', type=int, default=32,
                   help='Resample the number of mean-points in streamlines, [%(default)s] \n'
                        'A lower number will increase the number of False positive, \n'
                        'where a streamline with distance > mean_distance could be included.')

    p.add_argument('--nb_mpts', type=int, default=4,
                   help='Number of mean-points for the kdtree internal search, [%(default)s] \n'
                        'does not change the precision, only the computation time.')

    p.add_argument('--no_flip', action='store_true',
                   help='Disable the comparison in both streamlines orientation')

    p.add_argument('--cpu', type=int, default=24,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_header = args.in_tractogram
    if args.in_nii:
        input_header = args.in_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    # Load streamlines
    slines = load_slines(args.in_tractogram, input_header)
    # Resample streamlines
    slines_arr = resample_slines_to_array(slines, args.resample, meanpts_resampling=True, out_dtype=np.float32)


    # Generate the L21 k-d tree with LpqTree
    l21_radius = args.mean_distance * args.resample
    nn = lpqtree.KDTree(metric="l21", radius=l21_radius)

    if args.no_flip:
        nn.fit_and_radius_search(slines_arr, slines_arr,
                                 radius=l21_radius, nb_mpts=args.nb_mpts, count_only=True, n_jobs=args.cpu)
    else:
        nn.fit_and_radius_search(np.concatenate([slines_arr, np.flip(slines_arr, axis=1)]), slines_arr,
                                 radius=l21_radius, nb_mpts=args.nb_mpts, count_only=True, n_jobs=args.cpu)
    counts = nn.get_count()

    # filter streamlines with no enough density
    if args.min_count > 1:
        mask = (counts >= args.min_count)
        counts = counts[mask]
        slines_arr = slines_arr[mask]

    # streamlines measure to each vertex
    scalars = np.repeat(counts, args.resample, axis=None).astype(np.float32)

    # Poly data
    polydata = lines_to_vtk_polydata(slines_arr)

    # Set scalars
    vtk_scalars = ns.numpy_to_vtk(scalars, deep=True)
    vtk_scalars.SetNumberOfComponents(1)
    vtk_scalars.SetName("Scalars")
    polydata.GetPointData().SetScalars(vtk_scalars)

    # Colormap
    test = generate_colormap(scale_range=(np.min(scalars), np.max(scalars)/2.0))

    generate_scene(polydata, colormap=test)

def generate_scene(polydata, colormap=None):
    # Scene parameters
    size = (1000, 800)
    light = (0.4, 0.2, 0.1)
    background = (0.0, 0.0, 0.0)
    camera_rot = (0.0, 0.0, 0.0)
    zoom = 1.0
    #display_colormap = "Range"
    png_magnify = 1
    line_width = 1.0
    line_opacity = 0.2
    max_peel = 20
    use_LOD = True

    # vtk create scene :
    poly_mapper = vtk.vtkPolyDataMapper()
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.InterpolateScalarsBeforeMappingOn()
    poly_mapper.StaticOn()


    poly_mapper.SetInputData(polydata)
    poly_mapper.Update()

    if colormap:
        poly_mapper.SetLookupTable(colormap)
        poly_mapper.SetScalarModeToUsePointData()
        poly_mapper.UseLookupTableScalarRangeOn()

    if use_LOD:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(10000)
        actor.GetProperty().SetPointSize(3)
    else:
        actor = vtk.vtkActor()

    actor.SetMapper(poly_mapper)
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetAmbient(light[0])
    actor.GetProperty().SetDiffuse(light[1])
    actor.GetProperty().SetSpecular(light[2])


    # opacity
    if line_opacity < 1.0:
        actor.GetProperty().SetRepresentationToSurface()
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetOpacity(line_opacity)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.ResetCamera()
    renderer.SetBackground(background)

    if line_opacity < 1.0:
        renderer.UseDepthPeelingOn()
        renderer.SetMaximumNumberOfPeels(max_peel)

    if colormap:
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle("test")
        scalar_bar.SetLookupTable(colormap)
        scalar_bar.SetNumberOfLabels(7)
        renderer.AddActor(scalar_bar)

    camera = renderer.GetActiveCamera()
    camera.Roll(camera_rot[0])
    camera.Elevation(camera_rot[1])
    camera.Azimuth(camera_rot[2])
    camera.Zoom(zoom)


    def key_press(obj, event):
        key = obj.GetKeySym()
        if key == 's' or key == 'S':
            print('Saving image...')
            render_large = vtk.vtkRenderLargeImage()
            render_large.SetInput(renderer)
            render_large.SetMagnification(png_magnify)
            render_large.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetInputConnection(render_large.GetOutputPort())
            writer.SetFileName('tractosearch_img.png')
            writer.Write()
            print('Look for tractosearch_img.png in your current directory.')

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetWindowName("Tractosearch_Viewer")
    window.SetSize(size[0], size[1])

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.AddObserver('KeyPressEvent', key_press)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.Initialize()
    window.Render()
    iren.Start()

    # Stop viewer
    window.RemoveRenderer(renderer)
    renderer.SetRenderWindow(None)

def lines_to_vtk_polydata(lines, colors=None):
    # Get the 3d points_array
    points_array = np.vstack(lines).astype(np.float32)

    nb_lines = len(lines)
    nb_points = len(points_array)
    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines], dtype=np.int32)
    current_position = 0
    for i in range(nb_lines):
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    # Set Points to vtk array format
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(points_array, deep=True, array_type=vtk.VTK_FLOAT))

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines_array = ns.numpy_to_vtk(np.asarray(lines_array), deep=True, array_type=vtk.VTK_ID_TYPE)
    vtk_lines.SetCells(nb_lines, vtk_lines_array)

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    if colors is not None:
        vtk_colors = None
        if (colors is True) or (isinstance(colors, str) and colors == "RGB"):
            cols_arr = np.zeros_like(points_array)
            cols_arr[1:] = np.diff(points_array, axis=0)
            cols_arr[0] = cols_arr[1]
            offsets = np.cumsum(points_per_line)
            cols_arr[offsets[:-1]] = cols_arr[offsets[:-1]+1]
            cols_arr[offsets-1] = cols_arr[offsets-2]
            cols_arr = np.abs(cols_arr) / np.sqrt(np.sum(np.square(cols_arr), axis=1, keepdims=True))
            vtk_colors = numpy_to_vtk_colors(255 * cols_arr)
        else:
            try:
                colors = np.asarray(colors)
                if colors.ndim == 1:  # the same colors for all points
                    if len(colors) == 1 or len(colors) == 3 or len(colors) == 4:
                        vtk_colors = numpy_to_vtk_colors(np.tile(255 * colors, (nb_points, 1)))
                    elif len(colors) == nb_lines:
                        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                        vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])
                    elif len(colors) == nb_points:
                        vtk_colors = numpy_to_vtk_colors(255 * colors)
                if colors.ndim == 2:   # map color to each line
                    colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                    vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])

            except ValueError:
                if len(colors) == nb_lines:
                    # assume one color per points in a list of list
                    vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))

        if vtk_colors is not None:
            vtk_colors.SetName("Colors")
            poly_data.GetPointData().SetScalars(vtk_colors)
        else:
            raise NotImplementedError()

    return poly_data


def numpy_to_vtk_colors(colors):
    vtk_colors = ns.numpy_to_vtk(
        np.asarray(colors), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors

def generate_colormap(scale_range=(0.0, 1.0), hue_range=(0.8, 0.0),
                      saturation_range=(1.0, 1.0), value_range=(0.8, 0.8),
                      nan_color=(0.2, 0.2, 0.2, 1.0)):
    """ Generate colormap's lookup table

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the mininum
        and maximum value of your data. Default is (0, 1).
    hue_range : tuple of floats
        HSV values (min 0 and max 1). Default is (0.8, 0).
    saturation_range : tuple of floats
        HSV values (min 0 and max 1). Default is (1, 1).
    value_range : tuple of floats
        HSV value (min 0 and max 1). Default is (0.8, 0.8).

    Returns
    -------
    lookup_table : vtkLookupTable

    """
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)
    lookup_table.SetNanColor(nan_color)
    lookup_table.Build()
    return lookup_table

if __name__ == '__main__':
    main()
