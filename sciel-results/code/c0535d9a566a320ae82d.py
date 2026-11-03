"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _fast_covering_radius_deg(pts):
    """
    Fast approximation of covering radius.
    Uses the convex hull to find the largest facet circumradius.
    This is the exact metric, but we optimize it directly.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return 180.0
    if not np.all(np.isfinite(pts)):
        return 180.0
    
    # Quick coincident check (vectorized)
    # d2[i,j] = ||p_i - p_j||^2
    # We can just check if min off-diagonal dist is too small
    # But for speed in optimization loop, we might skip strict coincident check 
    # and rely on the optimization to separate them, or do a cheaper check.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_ang = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Circumcenter of spherical triangle
        # n = v0 x v1 + v1 x v2 + v2 x v0
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            # Degenerate
            c = (v0 + v1 + v2) / 3.0
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-15: continue
            c = c / c_norm
        else:
            c = n / n_norm
        
        # Check if c is inside the spherical triangle
        # c is inside if it is in the same hemisphere as the opposite vertex for each edge
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        # Signs
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang) * 180.0 / np.pi
        else:
            # Farthest point is on an edge
            # Midpoints
            mid01 = (v0 + v1)
            mid01_norm = np.linalg.norm(mid01)
            if mid01_norm < 1e-15:
                ang01 = 90.0
            else:
                mid01 = mid01 / mid01_norm
                ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0)) * 180.0 / np.pi
                
            mid12 = (v1 + v2)
            mid12_norm = np.linalg.norm(mid12)
            if mid12_norm < 1e-15:
                ang12 = 90.0
            else:
                mid12 = mid12 / mid12_norm
                ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0)) * 180.0 / np.pi
                
            mid20 = (v2 + v0)
            mid20_norm = np.linalg.norm(mid20)
            if mid20_norm < 1e-15:
                ang20 = 90.0
            else:
                mid20 = mid20 / mid20_norm
                ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0)) * 180.0 / np.pi
            
            ang = max(ang01, ang12, ang20)
        
        max_ang = max(max_ang, ang)
    
    return max_ang

def _repulsion_force(pts):
    """
    Compute a smooth repulsion force vector for each point.
    Force on i is sum over j of (p_i - p_j) / ||p_i - p_j||^3 * weight
    This pushes points apart.
    """
    forces = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                # Random kick if coincident
                forces[i] += np.random.randn(3)
            else:
                # Inverse square law repulsion
                forces[i] += diff / (dist_sq ** 1.5)
    return forces

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate high-quality initial seed: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_score = _fast_covering_radius_deg(best_pts)
    
    # 2. Deterministic Repulsion Relaxation
    # This phase pushes points into a more uniform distribution quickly.
    # We use a cooling schedule for the step size.
    current_pts = pts.copy()
    step_size = 0.05
    cooldown = 0.99
    
    # Run for a fixed number of iterations or until time runs out
    # Since N=130, O(N^2) is 16900 ops, very fast.
    for it in range(200):
        if time.time() > deadline - 20.0:
            break
            
        forces = _repulsion_force(current_pts)
        
        # Normalize forces to prevent exploding
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        normalized_forces = forces / force_norms
        
        # Update points
        new_pts = current_pts + step_size * normalized_forces
        new_pts = _normalize(new_pts)
        
        # Evaluate
        try:
            score = _fast_covering_radius_deg(new_pts)
        except Exception:
            score = 180.0
            
        if score < best_score:
            best_score = score
            best_pts = new_pts.copy()
            
        # Always accept in this phase? Or only if improved?
        # Repulsion usually improves uniformity, but might hurt covering radius slightly if it pulls a point away from a dense cluster.
        # Let's accept if improved, or if it's the first few steps.
        if score < current_score if 'current_score' in locals() else True:
             current_pts = new_pts
             current_score = score
        else:
             # If not improved, try a smaller step or revert
             # For simplicity, revert and cool down more
             pass
             
        step_size *= cooldown
        
    # 3. Local Optimization with Nelder-Mead
    # We define a function that flattens the points, optimizes, and re-normalizes.
    # Nelder-Mead is derivative-free and robust.
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        x = _normalize(x)
        # Add a small penalty for coincident points to keep them valid
        min_dist = 1.0
        # This is expensive, so we rely on the repulsion phase to have separated them.
        # If we want to be safe, we can check a few random pairs.
        return _fast_covering_radius_deg(x)
        
    current_pts = best_pts.copy()
    x0 = current_pts.flatten()
    
    # Try a few restarts of Nelder-Mead from slightly perturbed best_pts
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb slightly
        rng = np.random.RandomState(42 + restart)
        noise = rng.randn(N, 3) * 0.01
        x_start = _normalize(current_pts + noise).flatten()
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                new_pts = _normalize(res.x.reshape(N, 3))
                final_score = _fast_covering_radius_deg(new_pts)
                
                # Verify validity (no coincident points)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(new_pts[i] - new_pts[j]) < 1e-8:
                            valid = False
                            break
                    if not valid: break
                
                if valid and final_score < best_score:
                    best_score = final_score
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    
        except Exception:
            continue
            
    return best_pts
# EVOLVE-BLOCK-END
