"""Heuristic based on Voronoi-based topological structure: promote inter-region edges, penalize intra-region."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    diag = np.eye(n, dtype=bool)

    # Recover point coordinates from distance matrix using MDS
    # D is the pairwise Euclidean distance matrix
    def recover_points(dist_mat):
        """Recover 2D coordinates from distance matrix using classical MDS."""
        m = dist_mat.shape[0]
        # Double centering
        J = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * J @ (dist_mat ** 2) @ J
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 eigenvalues (largest)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Ensure non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)
        # Scale by sqrt of eigenvalues
        U = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
        return U

    # Get approximate 2D coordinates
    points = recover_points(D)
    
    # Compute Voronoi-like region assignment using nearest-neighbor clustering
    # Assign each point to a region based on its spatial neighborhood
    def assign_regions(pts, n_regions=8):
        """Assign points to regions using simple grid-based binning."""
        m = len(pts)
        # Normalize to [0,1] for consistent binning
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        norm_pts = (pts - mins) / ranges
        
        # Create a grid of n_regions x n_regions cells
        cell_size = 1.0 / n_regions
        region_ids = np.zeros(m, dtype=int)
        for i in range(m):
            cx = int(norm_pts[i, 0] / cell_size)
            cy = int(norm_pts[i, 1] / cell_size)
            cx = min(cx, n_regions - 1)
            cy = min(cy, n_regions - 1)
            region_ids[i] = cy * n_regions + cx
        return region_ids

    region_ids = assign_regions(points, n_regions=6)
    num_regions = len(np.unique(region_ids))
    
    # Build region adjacency: which regions are "neighbors" (close in space)
    # For each pair of points, determine if they're in the same or different region
    same_region = (region_ids[:, None] == region_ids[None, :])
    
    # Compute a spatial proximity score: how "bridge-like" an edge is
    # Edges connecting distant points across regions get higher scores
    # Edges within the same region get lower scores
    
    # First, build a base inverse-distance heuristic
    safe_D = D.copy()
    safe_D[diag] = 1.0
    base_inv_dist = 1.0 / safe_D
    
    # Normalize inverse distance to [0, 1]
    base_inv_dist_norm = (base_inv_dist - base_inv_dist.min()) / (base_inv_dist.max() - base_inv_dist.min() + 1e-10)
    
    # Region crossing bonus: edges that cross region boundaries get a boost
    # The farther apart in region space, the more "bridge-like"
    region_dist = np.abs(region_ids[:, None].astype(float) - region_ids[None, :].astype(float))
    max_region_dist = region_dist.max() + 1e-10
    region_cross_score = region_dist / max_region_dist
    
    # Combine: promote edges that are both close in distance AND cross regions
    # But also allow some long-range edges if they connect very different regions
    alpha_dist = 1.5
    alpha_region = 1.2
    
    H = (base_inv_dist_norm ** alpha_dist) * (1.0 + alpha_region * region_cross_score)
    
    # Penalize same-region edges more strongly to encourage region traversal
    H[same_region] *= 0.3
    
    # Add a small constant to avoid zeros
    H = H + 0.1
    
    # Ensure diagonal is minimal
    H[diag] = 1e-9
    
    # Ensure no non-finite or negative values
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
