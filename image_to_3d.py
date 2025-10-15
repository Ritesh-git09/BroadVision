import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import numpy as np
import open3d as o3d

def estimate_depth(image_path):
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    model.to("cpu")
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    depth = np.array(Image.fromarray(depth).resize(image.size, Image.BILINEAR))
    rgb = np.array(image)
    return depth, rgb

def create_point_cloud(depth, rgb):
    h, w = depth.shape
    fx = fy = 1.2 * w
    cx, cy = w / 2, h / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def clean_point_cloud(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def create_mesh(pcd, depth=9):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    vert_rm = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vert_rm)
    return mesh

def save_mesh(mesh, out_path="output_mesh.obj"):
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Mesh saved to {out_path}")

if __name__ == "__main__":
    img_path = "test.jpg" # <-- Change to your real image filename
    mesh_path = "output_mesh.obj"
    depth, rgb = estimate_depth(img_path)
    pcd = create_point_cloud(depth, rgb)
    pcd = clean_point_cloud(pcd)
    mesh = create_mesh(pcd)
    save_mesh(mesh, mesh_path)
    print("Done! Drag 'output_mesh.obj' into Unity Assets.")
