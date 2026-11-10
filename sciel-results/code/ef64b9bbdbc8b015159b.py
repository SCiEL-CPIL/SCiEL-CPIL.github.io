# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    Uses spherical code approach with repulsion optimization.
    """
    n = 14
    d = 3
    deadline = time.time() + 620  # 10s safety margin
    
    def score(pts):
        """Compute (d_min/d_max)^2"""
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max <= 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def project_to_sphere(pts):
        """Project points onto unit sphere"""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms
    
    def repulsion_force(pts):
        """Compute repulsion force (gradient of sum of inverse distances)"""
        # For each point, compute force from all other points
        # F_i = sum_{j!=i} (r_i - r_j) / |r_i - r_j|^3
        n_pts = len(pts)
        forces = np.zeros_like(pts)
        for i in range(n_pts):
            for j in range(n_pts):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge for coincident points
                    diff = np.random.randn(3) * 0.001
                    dist = np.linalg.norm(diff)
                forces[i] += diff / (dist ** 3)
        return forces
    
    def optimize_on_sphere(pts, max_iters=5000, lr=0.01):
        """Optimize points on unit sphere using repulsion"""
        pts = project_to_sphere(pts).copy()
        best_score = score(pts)
        best_pts = pts.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            forces = repulsion_force(pts)
            # Move points in direction of force, then reproject
            pts = pts + lr * forces
            pts = project_to_sphere(pts)
            
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
            
            # Decay learning rate
            lr *= 0.9999
        
        return best_pts, best_score
    
    def icosahedron_plus():
        """Generate 14 points: 12 icosahedron vertices + 2 more"""
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        ico = project_to_sphere(ico)
        
        # Add 2 points at "gaps" - try opposite of existing vertices
        # For icosahedron, the gaps are at the centers of triangular faces
        # Approximate by adding points in directions between vertices
        v1 = ico[0]
        v2 = ico[1]
        v3 = ico[4]
        
        # Try several candidate positions for the 13th and 14th points
        candidates = []
        for i in range(12):
            for j in range(i+1, 12):
                mid = ico[i] + ico[j]
                mid = project_to_sphere(mid)
                candidates.append(mid)
        
        # Also try random directions
        rng = np.random.RandomState(42)
        for _ in range(20):
            candidates.append(project_to_sphere(rng.randn(3)))
        
        # Try all pairs of additional points and pick best
        best_pts = ico.copy()
        best_s = score(ico)
        
        # We need 14 points, so add 2
        # Try greedy: add best single point, then best second point
        for cand1 in candidates[:50]:  # Limit for speed
            pts1 = np.vstack([ico, cand1])
            s1 = score(pts1)
            if s1 > best_s:
                # Now add second point
                for cand2 in candidates[:50]:
                    pts2 = np.vstack([pts1, cand2])
                    s2 = score(pts2)
                    if s2 > best_s:
                        best_s = s2
                        best_pts = pts2.copy()
        
        return best_pts
    
    # Strategy 1: Icosahedron-based
    best_overall = None
    best_score_overall = -1
    
    # Start with icosahedron + 2 optimized
    pts1 = icosahedron_plus()
    pts1_opt, s1 = optimize_on_sphere(pts1, max_iters=3000, lr=0.05)
    if s1 > best_score_overall:
        best_score_overall = s1
        best_overall = pts1_opt.copy()
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Strategy 2: Random restarts with repulsion
    rng = np.random.RandomState(123)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initial points on sphere
        pts = rng.randn(n, d)
        pts = project_to_sphere(pts)
        
        # Optimize
        pts_opt, s = optimize_on_sphere(pts, max_iters=2000, lr=0.03)
        if s > best_score_overall:
            best_score_overall = s
            best_overall = pts_opt.copy()
    
    if time.time() > deadline - 5:
        return best_overall
    
    # Strategy 3: Fine-tune the best solution
    if best_overall is not None:
        pts_fine, s_fine = optimize_on_sphere(best_overall, max_iters=5000, lr=0.01)
        if s_fine > best_score_overall:
            best_overall = pts_fine.copy()
            best_score_overall = s_fine
    
    # Final validation
    if best_overall is None:
        # Fallback: random
        best_overall = rng.randn(n, d)
    
    # Ensure finite
    best_overall = np.nan_to_num(best_overall, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return best_overall

# EVOLVE-BLOCK-END
