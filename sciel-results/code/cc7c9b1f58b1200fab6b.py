import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for safety and return overhead
    
    # 1. Initialization: Hexagonal Lattice
    # We want to fit N points. A hex lattice with spacing s.
    # Area approx N * s^2 * sqrt(3)/2 <= pi * 1^2
    # s approx sqrt(2 * pi / (N * sqrt(3)))
    # Let's just start with a reasonable spacing and adjust.
    # For N=50, best radius is ~0.1258. Diameter ~0.25.
    # Spacing should be slightly larger than diameter to allow packing.
    
    def get_hex_lattice(n_points):
        # Try to find a good spacing
        # Rough estimate: density of hex packing is pi / (2 * sqrt(3)) * (1/r)^2
        # Area of N circles = N * pi * r^2
        # Area of container = pi
        # Max r ~ sqrt(1/N) * sqrt(pi / (pi / (2*sqrt(3)))) = sqrt(2*sqrt(3)/N)
        # For N=50, r ~ sqrt(2*1.732/50) ~ sqrt(0.069) ~ 0.26? No, that's too high.
        # Wait, density of hex is 0.9069.
        # N * pi * r^2 = 0.9069 * pi * 1^2
        # r = sqrt(0.9069 / N) = sqrt(0.9069 / 50) = sqrt(0.0181) ~ 0.134.
        # The optimal is 0.1258. So spacing s = 2*r ~ 0.25.
        
        best_points = None
        best_score = -1
        
        # Try a few spacings around the estimate
        for spacing in [0.24, 0.25, 0.26, 0.27]:
            # Generate points on hex lattice
            # y = j * spacing * sqrt(3)/2
            # x = (i + 0.5 * (j % 2)) * spacing
            max_k = int(1.0 / (spacing * 0.5)) + 2
            points = []
            for j in range(-max_k, max_k + 1):
                for i in range(-max_k, max_k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) < n_points:
                continue
                
            # Select the n_points closest to the origin to keep the shape compact
            radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            indices = np.argsort(radii)
            selected = points[indices[:n_points]]
            
            # Calculate score
            score = calculate_score(selected)
            if score > best_score:
                best_score = score
                best_points = selected.copy()
                
        if best_points is None:
             # Fallback: uniform grid or random if lattice fails (unlikely)
             best_points = np.random.rand(n_points, 2) * 2 - 1
             # Scale to fit
             radii = np.hypot(best_points[:, 0], best_points[:, 1])
             best_points = best_points / np.max(radii) * 0.9
             
        return best_points

    def calculate_score(points):
        """Calculate the minimum radius for a given set of points."""
        # Distance to boundary
        dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
        min_boundary = np.min(dists_boundary)
        
        # Pairwise distances
        min_pair = np.inf
        # N=50, so 1225 pairs. Vectorization is possible but loop is fine for speed if not in tight inner loop?
        # Actually, for optimization, we need gradients, so we'll compute distances in the loop.
        # For initial scoring, a simple loop is fast enough.
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1]) / 2.0
                if d < min_pair:
                    min_pair = d
                    
        return min(min_boundary, min_pair)

    # Initialize
    centers = get_hex_lattice(N)
    current_score = calculate_score(centers)
    best_centers = centers.copy()
    best_score = current_score

    # 2. Optimization: Gradient Ascent
    # We want to maximize the minimum distance.
    # Gradient of min distance is tricky. We use a "force" model.
    # Each point is pushed away from neighbors and boundary if they are "close".
    
    step_size = 0.05
    decay = 0.995
    epsilon = 1e-4 # Tolerance for "active" constraints
    
    while time.time() - start_time < time_limit:
        # Compute gradients
        grads = np.zeros_like(centers)
        
        # Compute all pairwise distances and boundary distances
        # To be efficient, we can vectorize the distance calculation for the gradient step?
        # N=50 is small.
        
        # First, find the current minimum radius to determine which constraints are "active"
        # Or simply sum forces from ALL neighbors, weighted by proximity?
        # A common heuristic: Force = 1 if d < 2r, else 0.
        # But we don't know r exactly, we want to maximize it.
        
        # Better approach: 
        # 1. Find global min distance d_min.
        # 2. For each pair (i,j) with d_ij < d_min + epsilon:
        #      grad_i += (c_i - c_j) / d_ij
        #      grad_j += (c_j - c_i) / d_ij
        # 3. For each i with d_ib < d_min + epsilon:
        #      grad_i += -c_i / |c_i|  (push towards center)
        
        # Calculate d_min
        min_dist = np.inf
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) / 2.0
                if d < min_dist:
                    min_dist = d
        # Boundary
        for i in range(N):
            d = 1.0 - np.hypot(centers[i, 0], centers[i, 1])
            if d < min_dist:
                min_dist = d
        
        # Now apply gradients for active constraints
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.hypot(dx, dy) / 2.0
                if d < min_dist + epsilon:
                    if d > 1e-6:
                        # Unit vector from j to i
                        ux = dx / (2 * d) # Wait, d is half distance. Distance is 2d.
                        # Vector c_i - c_j has length 2d.
                        # Unit vector is (c_i - c_j) / (2d)
                        ux = dx / (2 * d)
                        uy = dy / (2 * d)
                        grads[i, 0] += ux
                        grads[i, 1] += uy
                        grads[j, 0] -= ux
                        grads[j, 1] -= uy
                    else:
                        # If coincident, push apart randomly
                        grads[i, 0] += np.random.randn()
                        grads[i, 1] += np.random.randn()
                        grads[j, 0] -= np.random.randn()
                        grads[j, 1] -= np.random.randn()
                        
        # Boundary
        for i in range(N):
            r = np.hypot(centers[i, 0], centers[i, 1])
            d = 1.0 - r
            if d < min_dist + epsilon:
                if r > 1e-6:
                    # Push towards center: -c_i / r
                    grads[i, 0] -= centers[i, 0] / r
                    grads[i, 1] -= centers[i, 1] / r
                else:
                    # If at center, push out? No, if at center, boundary dist is 1.
                    # If min_dist is small, center point is likely not the bottleneck unless N is very small.
                    pass

        # Update centers
        centers += step_size * grads
        
        # Ensure points stay within a slightly larger box to prevent explosion?
        # The boundary gradient should handle it.
        
        # Calculate new score
        new_score = calculate_score(centers)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            # If score didn't improve, we might be stuck or step too big.
            # Reduce step size and try again? Or just decay step size.
            pass
            
        step_size *= decay
        
        # If step size becomes too small, break
        if step_size < 1e-6:
            break

    # 3. Polish Phase
    # Use remaining time for a local search / fine tuning
    if time.time() - start_time < time_limit - 5:
        polish_centers = best_centers.copy()
        polish_score = best_score
        small_step = 0.01
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ])
        # Normalize directions
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for i in range(N):
                for d in directions:
                    new_c = polish_centers.copy()
                    new_c[i] += small_step * d
                    # Check if still inside container (loose check)
                    if np.hypot(new_c[i, 0], new_c[i, 1]) > 1.0:
                        continue
                    new_s = calculate_score(new_c)
                    if new_s > polish_score + 1e-9:
                        polish_score = new_s
                        polish_centers = new_c
                        improved = True
                        break # Restart direction loop for this point? Or just continue?
                # If we improved, restart the whole pass to ensure convergence
                if improved:
                    break
            # Decay small step
            small_step *= 0.95
            if small_step < 1e-7:
                break
        
        if polish_score > best_score:
            best_score = polish_score
            best_centers = polish_centers

    return best_centers

if __name__ == "__main__":
    # Simple test
    centers = construct()
    print(calculate_score(centers))
