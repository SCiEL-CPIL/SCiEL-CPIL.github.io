import numpy as np
import time
from scipy.spatial import Voronoi

N = 50

def _score(centers):
    """Compute the radius for a given configuration."""
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _hexagonal_init(n):
    """Create a hexagonal close-packed initial configuration."""
    # For N=50, a 7x7 grid with corners removed or 8x6 etc.
    # Let's try to generate a hexagonal lattice and pick the best N points.
    # Lattice spacing s.
    # Rows: y = i * s * sqrt(3)/2
    # Cols: x = j * s + (i % 2) * s / 2
    
    best_pts = None
    best_r = -1
    
    # Try different spacings
    for s in np.linspace(0.1, 0.2, 10):
        pts = []
        # Generate enough points to cover the square
        for i in range(-5, 10):
            for j in range(-5, 10):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                # Center the lattice roughly
                x -= 0.0
                y -= 0.0
                pts.append([x, y])
        pts = np.array(pts)
        
        # Keep only points inside the square [-0.5, 0.5]
        mask = (np.abs(pts[:, 0]) <= 0.5) & (np.abs(pts[:, 1]) <= 0.5)
        pts = pts[mask]
        
        if len(pts) < n:
            continue
            
        # If we have more than n points, we need to select n.
        # A simple heuristic: select points that maximize the minimum distance.
        # This is NP-hard, so we use a greedy approach or just take a subset.
        # For initialization, just taking the first N or a random subset is okay, 
        # but let's try to pick a "spread out" subset.
        
        # Greedy selection
        selected = []
        remaining = pts.copy()
        for _ in range(n):
            if len(remaining) == 0:
                break
            # Pick the point in remaining that has the largest minimum distance to selected
            # If selected is empty, pick the one closest to center? No, pick arbitrary.
            if len(selected) == 0:
                idx = np.random.randint(len(remaining))
            else:
                dists_to_selected = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - np.array(selected)[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists_to_selected)
            
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        if len(selected) == n:
            selected = np.array(selected)
            r = _score(selected)
            if r > best_r:
                best_r = r
                best_pts = selected.copy()
                
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        
    return best_pts

def _repulsion_step(centers, r_target, n_steps=50, lr=0.01):
    """
    Move points to satisfy min distance >= 2*r_target and boundary >= r_target.
    Uses simple repulsion forces.
    """
    centers = centers.copy()
    n = len(centers)
    
    for _ in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Boundary repulsion
        for i in range(n):
            for k in range(2):
                # If point is too close to boundary, push it in
                if centers[i, k] > 0.5 - r_target:
                    forces[i, k] -= (centers[i, k] - (0.5 - r_target))
                elif centers[i, k] < -0.5 + r_target:
                    forces[i, k] += ((-0.5 + r_target) - centers[i, k])
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push
                    forces[i] += np.random.randn(2) * 0.001
                    forces[j] -= np.random.randn(2) * 0.001
                else:
                    min_dist = 2 * r_target
                    if dist < min_dist:
                        # Push apart
                        force_mag = (min_dist - dist) / dist
                        forces[i] += diff * force_mag
                        forces[j] -= diff * force_mag
        
        centers += forces * lr
        # Hard clip to ensure validity for next step? No, let forces handle it, but keep them roughly in bounds
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def _optimize_sa(centers, time_budget=60.0):
    """Simulated Annealing to maximize radius."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    T = 0.005
    T_min = 1e-6
    alpha = 0.9995
    
    while time.time() - start_time < time_budget:
        # Generate move
        idx = np.random.randint(N)
        # Random nudge
        move = np.random.randn(2) * 0.01
        new_centers = current_centers.copy()
        new_centers[idx] += move
        
        # Keep in bounds
        if np.any(np.abs(new_centers) > 0.5):
            continue
            
        new_score = _score(new_centers)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Reheat occasionally
        if np.random.random() < 0.005:
            T = max(T, 0.002)
            
    return best_centers, best_score

def construct():
    """Main entry point."""
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    np.random.seed(42)
    
    # 1. Initialization
    centers = _hexagonal_init(N)
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # 2. Repulsion-based radius expansion
    # Start with a radius slightly smaller than current score
    r_current = best_score * 0.9
    r_max = 0.1 # Upper bound guess
    
    # Try to increase radius using repulsion
    while time.time() - start_time < DEADLINE - 30: # Leave 30s for SA
        r_try = r_current + 0.0005
        if r_try > r_max:
            break
            
        # Attempt to resolve overlaps for this radius
        centers_new = _repulsion_step(centers, r_try, n_steps=100, lr=0.05)
        score_new = _score(centers_new)
        
        if score_new >= r_try - 1e-6:
            # Success! We achieved a valid configuration for r_try
            centers = centers_new
            r_current = r_try
            if score_new > best_score:
                best_score = score_new
                best_centers = centers.copy()
        else:
            # Failed to find a valid config for r_try, try a smaller step
            r_try = r_current + 0.0001
            centers_new = _repulsion_step(centers, r_try, n_steps=100, lr=0.05)
            score_new = _score(centers_new)
            if score_new >= r_try - 1e-6:
                centers = centers_new
                r_current = r_try
                if score_new > best_score:
                    best_score = score_new
                    best_centers = centers.copy()
            else:
                # If even small step fails, break or try SA
                break
                
    # 3. Simulated Annealing Polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        t_sa = min(remaining * 0.7, 30.0)
        centers_sa, score_sa = _optimize_sa(centers, time_budget=t_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_centers = centers_sa.copy()
            
    # 4. Final Local Refinement (optional, if time)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 2:
        # Simple local search: try small moves for each point
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 1:
                break
            improved = False
            for i in range(N):
                for dx in [-0.001, 0.001]:
                    for dy in [-0.001, 0.001]:
                        new_c = best_centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                        s = _score(new_c)
                        if s > best_score:
                            best_centers = new_c
                            best_score = s
                            improved = True
            if not improved:
                break
                
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(c)}")
    print(f"Shape: {c.shape}")
