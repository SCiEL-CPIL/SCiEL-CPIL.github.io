import numpy as np
from scipy.spatial import cKDTree

def construct():
    N = 130
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Repulsive Force Dynamics (RFD)
    # This approach maximizes the minimum distance between points,
    # which serves as a strong proxy for minimizing the covering radius.
    
    num_rfd_iters = 500
    learning_rate = 0.05
    
    for it in range(num_rfd_iters):
        # Decay learning rate
        lr = learning_rate * (0.99 ** it)
        
        # Calculate pairwise distances
        # diff[i, j] = p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Force vector on i due to j is (p_i - p_j) / |p_i - p_j|^3
        # This corresponds to a potential V(r) = 1/r
        dists_cubed = dists ** 3
        np.fill_diagonal(dists_cubed, 1.0)
        
        force_vecs = diff / dists_cubed[:, :, np.newaxis]
        
        # Sum forces
        total_forces = np.sum(force_vecs, axis=1)
        
        # Project forces onto tangent plane
        # p_i is normal to tangent plane at p_i
        # F_tangent = F - (F . p_i) p_i
        dot_prod = np.sum(total_forces * pts, axis=1, keepdims=True)
        tangent_forces = total_forces - dot_prod * pts
        
        # Update positions
        pts = pts + lr * tangent_forces
        
        # Normalize back to sphere
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        
    # 3. Covering Radius Refinement
    # RFD maximizes min-distance, but we want min covering radius.
    # We perform local adjustments to reduce the maximum distance to nearest point.
    
    M = 10000
    i_s = np.arange(M) + 0.5
    z_s = 1.0 - 2.0 * i_s / M
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(1.0 - z_s * z_s)
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    num_refine_iters = 50
    
    for _ in range(num_refine_iters):
        tree = cKDTree(pts)
        dists, idx = tree.query(samples, k=1)
        
        # Find the sample point with the maximum distance
        max_idx = np.argmax(dists)
        worst_sample = samples[max_idx]
        nearest_pt_idx = idx[max_idx]
        
        # Move the nearest point towards the worst sample
        # Step size proportional to the distance
        step_size = 0.1 * dists[max_idx]
        
        # Move in tangent direction
        p = pts[nearest_pt_idx]
        # Vector from p to worst_sample
        v = worst_sample - p
        # Project v onto tangent plane at p
        dot = np.dot(v, p)
        v_tan = v - dot * p
        norm_v = np.linalg.norm(v_tan)
        if norm_v > 1e-6:
            v_tan = v_tan / norm_v
            pts[nearest_pt_idx] = p + step_size * v_tan
            # Renormalize
            pts[nearest_pt_idx] /= np.linalg.norm(pts[nearest_pt_idx])
            
    # Final normalization
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
