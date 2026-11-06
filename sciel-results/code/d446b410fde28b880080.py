import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    def score(pts):
        """Compute the radius score."""
        if len(pts) != N:
            return 0.0
        # Check finite
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Distances to boundary
        dists_to_boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
        min_boundary = np.min(dists_to_boundary)
        if min_boundary < -1e-10:
            return 0.0
        # Pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(pts)
        min_pair = np.min(d) / 2.0
        return min(min_pair, min_boundary)
    
    def improve(pts, max_iters=5000, step_scale=1.0):
        """Gradient-free local improvement using steepest ascent on score."""
        current_score = score(pts)
        if current_score <= 0:
            return pts, current_score
        
        best_pts = pts.copy()
        best_score = current_score
        
        # Use Nelder-Mead style but with our custom scoring
        # Actually, let's use a simpler approach: try to increase the minimum constraint
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Find which constraint is binding
            dists_to_boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
            from scipy.spatial.distance import pdist
            d = pdist(pts)
            min_pair = np.min(d) / 2.0
            min_boundary = np.min(dists_to_boundary)
            
            if min_pair < min_boundary:
                # Pairwise distance is binding - push the closest pair apart
                i, j = np.unravel_index(np.argmin(d), (N, N-1))
                if j > i:
                    j -= 1
                # Actually pdist gives upper triangle, let me use cdist
                from scipy.spatial.distance import cdist
                D = cdist(pts, pts)
                np.fill_diagonal(D, np.inf)
                idx = np.unravel_index(np.argmin(D), (N, N))
                i, j = idx
                diff = pts[i] - pts[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    diff = np.random.randn(2)
                    norm = np.linalg.norm(diff)
                dir = diff / norm
                step = 0.001 * step_scale
                pts[i] += dir * step
                pts[j] -= dir * step
            else:
                # Boundary is binding - push the outermost circle inward
                i = np.argmin(dists_to_boundary)
                pt = pts[i]
                norm = np.linalg.norm(pt)
                if norm < 1e-10:
                    continue
                dir = -pt / norm  # toward center
                step = 0.001 * step_scale
                pts[i] += dir * step
            
            new_score = score(pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            else:
                # Revert
                pts = best_pts.copy()
                step_scale *= 0.95
                if step_scale < 1e-6:
                    break
        
        return best_pts, best_score
    
    # --- Initialization: concentric rings ---
    def make_rings_layout():
        """Create initial layout with concentric rings."""
        pts = []
        # Center
        pts.append(np.array([0.0, 0.0]))
        
        # Try different ring configurations
        # For 50 circles: 1 + 6 + 12 + 18 + ... 
        # Ring k has 6*k points (for k>=1), but we can vary
        # Let's try: 1, 6, 12, 18, 13 = 50
        ring_sizes = [6, 12, 18, 13]
        radii = [0.3, 0.55, 0.78, 0.92]
        
        for size, r in zip(ring_sizes, radii):
            angles = np.linspace(0, 2*np.pi, size, endpoint=False)
            for a in angles:
                pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
        
        return np.array(pts[:N])
    
    # Alternative: use a more optimized ring layout
    def make_optimized_rings():
        """Better ring layout with adjusted radii."""
        pts = []
        pts.append(np.array([0.0, 0.0]))
        
        # 49 more circles in rings
        # Ring 1: 6 circles
        # Ring 2: 12 circles  
        # Ring 3: 18 circles
        # Ring 4: 13 circles
        ring_configs = [
            (6, 0.28),
            (12, 0.52),
            (18, 0.74),
            (13, 0.90)
        ]
        
        for size, r in ring_configs:
            angles = np.linspace(0, 2*np.pi, size, endpoint=False)
            # Offset alternate rings for better packing
            for a in angles:
                pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
        
        return np.array(pts[:N])
    
    # Start with a good initial guess
    pts = make_optimized_rings()
    
    # Verify initial is valid
    if score(pts) <= 0:
        # Fallback to simpler layout
        pts = make_rings_layout()
    
    # Multi-restart local optimization
    best_pts = pts.copy()
    best_score = score(pts)
    
    # Try multiple starting points
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            current = best_pts.copy()
        else:
            # Perturb the best solution
            current = best_pts + rng.normal(0, 0.02, (N, 2))
        
        # Run local improvement
        improved, s = improve(current, max_iters=3000, step_scale=1.0)
        
        if s > best_score:
            best_score = s
            best_pts = improved.copy()
        
        # If we're doing well, try more refined optimization
        if best_score > 0.12:
            refined, s2 = improve(best_pts, max_iters=2000, step_scale=0.5)
            if s2 > best_score:
                best_score = s2
                best_pts = refined.copy()
    
    # Final check
    final_score = score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        # Emergency fallback: uniform ring
        pts = make_rings_layout()
        return pts
    
    return best_pts

# EVOLVE-BLOCK-END
