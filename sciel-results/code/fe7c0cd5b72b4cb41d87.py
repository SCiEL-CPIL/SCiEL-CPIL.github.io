<<<<<<< SEARCH
def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out
=======
def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _simulate_anneal(points, time_limit=30.0, rng=None):
    """Simulated annealing on all 2N coordinates to escape local optima."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    best_pts = points.copy()
    best_score = _calculate_score(points)
    current_pts = points.copy()
    current_score = best_score
    
    T = 0.05  # initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Generate a perturbation: move one or two random points
        n_moves = rng.integers(1, 3)
        new_pts = current_pts.copy()
        for _ in range(n_moves):
            idx = rng.integers(0, N)
            step = rng.normal(0, T * 2, size=2)
            new_pts[idx] += step
        
        # Normalize: recenter and scale to unit hull area
        try:
            hull = ConvexHull(new_pts)
            hull_area = hull.volume
            if hull_area <= 1e-9:
                T *= cooling
                continue
            # Scale to unit area
            scale = 1.0 / np.sqrt(hull_area)
            new_pts = new_pts * scale
            # Recenter
            new_pts -= np.mean(new_pts, axis=0)
        except Exception:
            T *= cooling
            continue
        
        new_score = _calculate_score(new_pts)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            delta = new_score - current_score
            if delta < -5 * T:  # reject if too bad
                pass
            elif rng.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling
    
    return best_pts, best_score
>>>>>>> REPLACE

