"""
Multi-view 3D Reconstruction
Uses multiple images of the same object for better 3D reconstruction
This is significantly better than single-image depth estimation
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiView3D:
    """
    Multi-view stereo reconstruction for accurate 3D models
    Requires: 2+ images of the same object from different angles
    """
    
    def __init__(self):
        self.images = []
        self.keypoints = []
        self.descriptors = []
        
    def load_images(self, image_paths: List[str]):
        """Load multiple images"""
        print(f"Loading {len(image_paths)} images...")
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load {path}")
                continue
            self.images.append(img)
        print(f"Loaded {len(self.images)} images successfully")
        
    def detect_features(self):
        """Detect SIFT features in all images"""
        print("Detecting features...")
        sift = cv2.SIFT_create(nfeatures=5000)
        
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            print(f"  Image {i+1}: {len(kp)} features")
            
    def match_features(self, idx1: int, idx2: int) -> List[cv2.DMatch]:
        """Match features between two images"""
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(self.descriptors[idx1], self.descriptors[idx2], k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"  Matched {len(good_matches)} features between image {idx1+1} and {idx2+1}")
        return good_matches
    
    def estimate_camera_poses(self, idx1: int, idx2: int, matches: List[cv2.DMatch]) -> Tuple:
        """Estimate relative camera pose using essential matrix"""
        # Get matched points
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        # Camera intrinsics (estimate from image)
        h, w = self.images[idx1].shape[:2]
        focal = w * 1.2
        pp = (w/2, h/2)
        K = np.array([[focal, 0, pp[0]],
                      [0, focal, pp[1]],
                      [0, 0, 1]])
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        
        return K, R, t, pts1, pts2, mask
    
    def triangulate_points(self, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                          pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from two views"""
        # Projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        # Triangulate
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def create_point_cloud(self, points_3d: np.ndarray, img: np.ndarray, 
                          pts: np.ndarray) -> o3d.geometry.PointCloud:
        """Create colored point cloud from 3D points"""
        # Sample colors from image
        colors = []
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                color = img[y, x][::-1] / 255.0  # BGR to RGB
                colors.append(color)
            else:
                colors.append([0.5, 0.5, 0.5])
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd
    
    def reconstruct_from_pair(self, idx1: int = 0, idx2: int = 1) -> o3d.geometry.PointCloud:
        """Reconstruct 3D from a pair of images"""
        print(f"\nReconstructing from images {idx1+1} and {idx2+1}...")
        
        # Match features
        matches = self.match_features(idx1, idx2)
        
        if len(matches) < 50:
            print("Error: Not enough matches found!")
            return None
        
        # Estimate camera poses
        K, R, t, pts1, pts2, mask = self.estimate_camera_poses(idx1, idx2, matches)
        print(f"  Inliers: {mask.sum()}/{len(mask)}")
        
        # Triangulate points
        points_3d = self.triangulate_points(K, R, t, pts1, pts2, mask)
        print(f"  Triangulated {len(points_3d)} points")
        
        # Create point cloud
        pcd = self.create_point_cloud(points_3d, self.images[idx1], pts1[mask.ravel() == 1])
        
        # Clean outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  After cleaning: {len(pcd.points)} points")
        
        return pcd
    
    def process(self, image_paths: List[str], output_name: str = "multiview_output"):
        """Complete multi-view reconstruction pipeline"""
        print(f"\n{'='*60}")
        print("Multi-View 3D Reconstruction")
        print(f"{'='*60}\n")
        
        # Load images
        self.load_images(image_paths)
        
        if len(self.images) < 2:
            print("Error: Need at least 2 images for reconstruction!")
            return None
        
        # Detect features
        self.detect_features()
        
        # Reconstruct from first pair
        pcd = self.reconstruct_from_pair(0, 1)
        
        if pcd is None:
            return None
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        # Create mesh
        print("\nCreating mesh...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        
        # Clean mesh
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()
        
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Save outputs
        mesh_path = f"{output_name}.obj"
        pcd_path = f"{output_name}_pointcloud.ply"
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        o3d.io.write_point_cloud(pcd_path, pcd)
        
        print(f"\n✓ Saved: {mesh_path}")
        print(f"✓ Saved: {pcd_path}")
        
        # Visualize
        print("\nVisualizing...")
        o3d.visualization.draw_geometries([mesh], window_name="Multi-view Reconstruction")
        
        return mesh, pcd


def main():
    """Example usage"""
    # YOU NEED MULTIPLE IMAGES (2+) of the same object from different angles
    image_paths = [
        "view1.jpg",  # Replace with your images
        "view2.jpg",
        # "view3.jpg",  # Add more if available
    ]
    
    reconstructor = MultiView3D()
    mesh, pcd = reconstructor.process(image_paths, output_name="multiview_output")
    
    print("\n📌 Tips for better multi-view reconstruction:")
    print("1. Take 3-10 photos from different angles (15-30° apart)")
    print("2. Keep the object in the center of frame")
    print("3. Use good lighting and avoid motion blur")
    print("4. Include distinctive features/textures on the object")
    print("5. Overlap between views should be ~70%")


if __name__ == "__main__":
    main()
