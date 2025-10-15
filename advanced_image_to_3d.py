"""
Advanced 2D to 3D Conversion Pipeline
Includes multiple state-of-the-art methods for better reconstruction
"""

import torch
import numpy as np
from PIL import Image
import open3d as o3d
from pathlib import Path
import cv2
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedImage3D:
    """Advanced 2D to 3D conversion with multiple techniques"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
    def estimate_depth_zoedepth(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZoeDepth: State-of-the-art metric depth estimation
        Better than MiDaS for accurate depth values
        """
        try:
            print("Loading ZoeDepth model (this may take a moment)...")
            # ZoeDepth provides metric depth estimation
            repo = "isl-org/ZoeDepth"
            model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            model.to(self.device)
            model.eval()
            
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            
            with torch.no_grad():
                depth = model.infer_pil(image)
            
            print(f"Depth range: {depth.min():.2f} - {depth.max():.2f}")
            return depth, image_np
            
        except Exception as e:
            print(f"ZoeDepth failed: {e}")
            return self.estimate_depth_dpt(image_path)
    
    def estimate_depth_dpt(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to DPT if ZoeDepth fails"""
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        
        print("Using DPT model...")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        model.to(self.device)
        model.eval()
        
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
        rgb = np.array(image)
        
        return depth, rgb
    
    def estimate_normals(self, image_path: str) -> Optional[np.ndarray]:
        """
        Estimate surface normals using neural network
        Helps create better surface reconstruction
        """
        try:
            print("Estimating surface normals...")
            # You can use DSINE or other normal estimation models
            # For now, we'll compute from depth
            return None
        except Exception as e:
            print(f"Normal estimation failed: {e}")
            return None
    
    def create_point_cloud_advanced(self, depth: np.ndarray, rgb: np.ndarray, 
                                   scale_depth: float = 1.0) -> o3d.geometry.PointCloud:
        """
        Create point cloud with better camera parameters and depth scaling
        """
        h, w = depth.shape
        
        # Better camera intrinsics estimation
        # Assuming 50mm lens on full-frame (common default)
        focal_length_mm = 50
        sensor_width_mm = 36
        fov = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
        fx = fy = w / (2 * np.tan(fov / 2))
        cx, cy = w / 2, h / 2
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize and scale depth
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_scaled = depth_normalized * scale_depth
        
        # Back-project to 3D
        z = depth_scaled
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack points
        points = np.stack((x, y, -z), axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3) / 255.0
        
        # Remove invalid points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def clean_point_cloud_advanced(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Enhanced point cloud cleaning"""
        print(f"Original points: {len(pcd.points)}")
        
        # Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        pcd = pcd.select_by_index(ind)
        print(f"After outlier removal: {len(pcd.points)}")
        
        # Remove radius outliers
        cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.05)
        pcd = pcd.select_by_index(ind)
        print(f"After radius outlier removal: {len(pcd.points)}")
        
        # Estimate normals with better parameters
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        return pcd
    
    def create_mesh_advanced(self, pcd: o3d.geometry.PointCloud, 
                           method: str = 'poisson') -> o3d.geometry.TriangleMesh:
        """
        Create mesh with multiple reconstruction methods
        Methods: 'poisson', 'bpa', 'alpha'
        """
        if method == 'poisson':
            print("Creating mesh using Poisson reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=10, width=0, scale=1.1, linear_fit=False
            )
            
            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.02)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == 'bpa':
            print("Creating mesh using Ball Pivoting Algorithm...")
            # Estimate point cloud density
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            
            radii = [radius, radius * 2, radius * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
        elif method == 'alpha':
            print("Creating mesh using Alpha Shape...")
            # Alpha shape reconstruction
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=0.03
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        print(f"Mesh vertices: {len(mesh.vertices)}, triangles: {len(mesh.triangles)}")
        
        return mesh
    
    def smooth_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                   iterations: int = 5) -> o3d.geometry.TriangleMesh:
        """Apply smoothing to mesh"""
        print(f"Smoothing mesh with {iterations} iterations...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        mesh.compute_vertex_normals()
        return mesh
    
    def texture_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                     image: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Apply texture to mesh (basic UV mapping)
        For better results, use specialized texturing tools
        """
        # This is a placeholder - proper UV mapping is complex
        # The colors are already applied via vertex colors from point cloud
        return mesh
    
    def save_outputs(self, mesh: o3d.geometry.TriangleMesh, 
                    pcd: o3d.geometry.PointCloud,
                    base_name: str = "output"):
        """Save mesh and point cloud in multiple formats"""
        # Save mesh
        mesh_obj = f"{base_name}.obj"
        mesh_ply = f"{base_name}.ply"
        mesh_stl = f"{base_name}.stl"
        
        o3d.io.write_triangle_mesh(mesh_obj, mesh)
        o3d.io.write_triangle_mesh(mesh_ply, mesh)
        o3d.io.write_triangle_mesh(mesh_stl, mesh)
        
        print(f"✓ Mesh saved as: {mesh_obj}, {mesh_ply}, {mesh_stl}")
        
        # Save point cloud
        pcd_ply = f"{base_name}_pointcloud.ply"
        o3d.io.write_point_cloud(pcd_ply, pcd)
        print(f"✓ Point cloud saved as: {pcd_ply}")
    
    def visualize(self, geometries: list):
        """Visualize the results"""
        print("\nOpening visualization window...")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Ctrl+C: Copy current viewpoint")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="3D Reconstruction",
            width=1280,
            height=720,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )
    
    def process(self, image_path: str, output_name: str = "output",
                depth_scale: float = 2.0, mesh_method: str = 'poisson',
                smooth_iterations: int = 3, visualize: bool = True):
        """
        Complete processing pipeline
        
        Args:
            image_path: Path to input image
            output_name: Base name for output files
            depth_scale: Scale factor for depth (adjust for better results)
            mesh_method: 'poisson', 'bpa', or 'alpha'
            smooth_iterations: Number of smoothing iterations
            visualize: Whether to show 3D visualization
        """
        print(f"\n{'='*60}")
        print(f"Advanced 2D to 3D Conversion")
        print(f"{'='*60}\n")
        
        # Step 1: Estimate depth
        print("Step 1: Estimating depth...")
        depth, rgb = self.estimate_depth_zoedepth(image_path)
        
        # Step 2: Create point cloud
        print("\nStep 2: Creating point cloud...")
        pcd = self.create_point_cloud_advanced(depth, rgb, scale_depth=depth_scale)
        
        # Step 3: Clean point cloud
        print("\nStep 3: Cleaning point cloud...")
        pcd = self.clean_point_cloud_advanced(pcd)
        
        # Step 4: Create mesh
        print(f"\nStep 4: Creating mesh using {mesh_method} method...")
        mesh = self.create_mesh_advanced(pcd, method=mesh_method)
        
        # Step 5: Smooth mesh
        if smooth_iterations > 0:
            print(f"\nStep 5: Smoothing mesh...")
            mesh = self.smooth_mesh(mesh, iterations=smooth_iterations)
        
        # Step 6: Save outputs
        print(f"\nStep 6: Saving outputs...")
        self.save_outputs(mesh, pcd, base_name=output_name)
        
        # Step 7: Visualize
        if visualize:
            print(f"\nStep 7: Visualizing results...")
            self.visualize([mesh])
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"{'='*60}\n")
        
        return mesh, pcd


def main():
    """Main execution"""
    # Configuration
    IMAGE_PATH = "test.jpg"  # Change to your image
    OUTPUT_NAME = "advanced_output"
    
    # Parameters to tune
    DEPTH_SCALE = 2.0        # Increase for more depth, decrease for flatter
    MESH_METHOD = 'poisson'  # 'poisson', 'bpa', or 'alpha'
    SMOOTH_ITER = 3          # 0-10, higher = smoother but less detail
    VISUALIZE = True         # Set to False to skip visualization
    
    # Create processor
    processor = AdvancedImage3D()
    
    # Process image
    mesh, pcd = processor.process(
        image_path=IMAGE_PATH,
        output_name=OUTPUT_NAME,
        depth_scale=DEPTH_SCALE,
        mesh_method=MESH_METHOD,
        smooth_iterations=SMOOTH_ITER,
        visualize=VISUALIZE
    )
    
    print("\n📌 Next Steps:")
    print("1. Import the .obj file into Unity/Blender")
    print("2. Adjust DEPTH_SCALE if the model looks too flat or too deep")
    print("3. Try different MESH_METHOD values for better results")
    print("4. For best results, consider using multiple images (next tutorial)")


if __name__ == "__main__":
    main()
