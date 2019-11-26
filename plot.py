import numpy as np
import open3d as o3d
import argparse
import segmentation.envs.utils as utils
from plyfile import PlyData

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default="test.csv", help="The csv file containing the point cloud which should be plotted. Default: test.csv")
    parser.add_argument("--label_file", type=str, default="test.csv", help="Can be used to plot scenes from the ScanNet dataset. The points can be entered in the --file option. Default: test.csv")
    parser.add_argument("--delimiter", type=str, default=";", help="Delimiter sign of the csv file. Default: ;")
    parser.add_argument("--color", type=bool, default=False, help="Plot a point cloud with colour information with features [x y z r g b]. Default: False")
    parser.add_argument("--color_labels", type=bool, default=False, help="If 'True', the labels of the point cloud will be coloured. Default: False")
    parser.add_argument("--curvature", type=bool, default=False, help="Plot a histogram of the curvature values of the point cloud. Moreover the point cloud with the curvature values will be plotted. The points with high curvature will be plotted in red. Default: False")
    parser.add_argument("--curvature_start", type=float, default=0.0, help="Lower bound to create the curvature histogram. Default: 0.0")
    parser.add_argument("--curvature_stop", type=float, default=0.2, help="Upper bound to create the curvature histogram. Default: 0.2")
    parser.add_argument("--curvature_bins", type=int, default=20, help="Number of bins of the curvature histogram. Default: 20")
    parser.add_argument("--coordinate_system", type=int, default=1, help="If '0', the coordinate system will not be plotted. Default: True")

    args = parser.parse_args()
    
    file_ending = args.file[-4:]
    pcd = o3d.geometry.PointCloud()
    all_points = None
    if file_ending == ".ply":
        pcd = o3d.io.read_point_cloud(args.file)
        all_points = np.asarray(pcd.points)
        if args.label_file is not None:
            if args.label_file[-4:] == ".ply":            
                plydata = PlyData.read(args.label_file)
                labels = plydata["vertex"]["label"]
                all_points = np.hstack((all_points, labels[:,None]))
    elif file_ending == ".txt":
        all_points = np.loadtxt(args.file)
        pcd.points = o3d.utility.Vector3dVector(all_points[:,:3])
    else: # .csv
        all_points = np.loadtxt(args.file, delimiter=args.delimiter)
        #all_points = all_points[::100]
        #print(all_points.shape[0])
        pcd.points = o3d.utility.Vector3dVector(all_points[:,:3])
        
    label_col = 3
    if all_points.shape[1] == 4:
        knn_param = o3d.geometry.KDTreeSearchParamKNN(knn=30)
        pcd.estimate_normals(search_param=knn_param)
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        #pcd.normals = o3d.utility.Vector3dVector(normals)
    elif all_points.shape[1] == 7:
        pcd.normals=o3d.utility.Vector3dVector(all_points[:,3:6])
        label_col = 6
    
    if args.curvature:
        scene = utils.get_scene(nr=0, pc=all_points, label_col=label_col, max_K=30)
        normals, curvature = utils.estimate_normals_curvature(pc=all_points[:,:3], nns = scene.nns)
        utils.hist(curvature)
        start=args.curvature_start
        stop=args.curvature_stop
        n_bins = args.curvature_bins
        step = (stop-start)/n_bins
        bins = np.arange(start=start, stop=stop, step=step)
        print("n bins: ", n_bins)
        idxs = np.digitize(curvature, bins=bins)
        colors = utils.generate_heat_colors(max_colors=n_bins+5)
        color = colors[idxs[:,0]]
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd, utils.coordinate_system()])
        
    if args.color:
        all_points[:,3:] /= 255
        #print(all_points[:5])
        pcd.colors = o3d.utility.Vector3dVector(all_points[:,3:])
    else:
        if args.color_labels:
            colors = utils.generate_colors(max_colors=50)
            labels = all_points[:,label_col].astype(int)
            color = colors[labels]
        else:
            color = np.zeros((all_points.shape[0], 3))
        pcd.colors = o3d.utility.Vector3dVector(color)

    if args.coordinate_system == 1:
        o3d.visualization.draw_geometries([pcd, utils.coordinate_system()])
    else:
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
