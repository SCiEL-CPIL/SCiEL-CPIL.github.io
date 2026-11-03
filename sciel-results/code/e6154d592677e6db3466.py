"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    simplices = hull.simplices
    A = pts[simplices[:, 0]]
    B = pts[simplices[:, 1]]
    C = pts[simplices[:, 2]]
    
    # Vectorized circumcenter calculation
    V = np.cross(A - B, B - C)
    NormV = np.linalg.norm(V, axis=1, keepdims=True)
    NormV[NormV < 1e-12] = 1.0
    U = V / NormV
    
    # Ensure U points to the correct side (dot with A > 0)
    Dots = np.einsum('ij,ij->i', U, A)
    Signs = np.sign(Dots)
    Signs[Signs == 0] = 1
    U_eff = U * Signs[:, np.newaxis]
    Dots_eff = Dots * Signs
    
    Radii = np.arccos(np.clip(Dots_eff, -1, 1))
    
    return np.max(Radii)

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=100, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _optimize_facet_shrink(pts, max_iter=100, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses vectorized operations for speed.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        simplices = hull.simplices
        A = pts[simplices[:, 0]]
        B = pts[simplices[:, 1]]
        C = pts[simplices[:, 2]]
        
        # Vectorized circumcenter
        V = np.cross(A - B, B - C)
        NormV = np.linalg.norm(V, axis=1, keepdims=True)
        NormV[NormV < 1e-12] = 1.0
        U = V / NormV
        
        # Ensure U points to the correct side (dot with A > 0)
        Dots = np.einsum('ij,ij->i', U, A)
        Signs = np.sign(Dots)
        Signs[Signs == 0] = 1
        U_eff = U * Signs[:, np.newaxis]
        Dots_eff = Dots * Signs
        
        Radii = np.arccos(np.clip(Dots_eff, -1, 1))
        
        max_cr = np.max(Radii)
        if max_cr < 1e-6:
            break
            
        # Weights
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        Weights = np.exp(beta * (Radii - max_cr))
        
        # Accumulate gradients/targets
        VecSum = np.zeros((n, 3))
        WeightsSum = np.zeros(n)
        
        np.add.at(VecSum, simplices[:, 0], Weights[:, np.newaxis] * U_eff)
        np.add.at(VecSum, simplices[:, 1], Weights[:, np.newaxis] * U_eff)
        np.add.at(VecSum, simplices[:, 2], Weights[:, np.newaxis] * U_eff)
        
        np.add.at(WeightsSum, simplices[:, 0], Weights)
        np.add.at(WeightsSum, simplices[:, 1], Weights)
        np.add.at(WeightsSum, simplices[:, 2], Weights)
        
        # Update points
        for i in range(n):
            if WeightsSum[i] > 1e-10:
                Target = VecSum[i] / WeightsSum[i]
                NormT = np.linalg.norm(Target)
                if NormT > 1e-10:
                    Target = Target / NormT
                    Vec = Target - pts[i]
                    VecTan = Vec - np.dot(Vec, pts[i]) * pts[i]
                    NormVec = np.linalg.norm(VecTan)
                    if NormVec > 1e-10:
                        pts[i] = pts[i] + step_size * VecTan
                        
        pts = _normalize(pts)
        step_size *= decay
        if step_size < 1e-8:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            noise_scale = 0.1 * (best_cr / 0.3)
            if noise_scale > 0.5:
                noise_scale = 0.5
            if noise_scale < 0.01:
                noise_scale = 0.01
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=50)
        
        # 2. Facet shrink (Vectorized)
        pts = _optimize_facet_shrink(pts, max_iter=200, initial_step=0.5, decay=0.99, beta_start=2.0, beta_end=50.0)
        
        if time.time() > deadline:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
