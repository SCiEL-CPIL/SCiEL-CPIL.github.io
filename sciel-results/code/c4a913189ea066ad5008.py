import numpy as np
import time

N = 300

def construct():
    # Strategy: Ring-based packing.
    # Represent the configuration by ring radii and number of circles per ring.
    # Optimize the ring radii and angular offsets (via a simple gradient or random search on radii).
    
    # 1. Determine a good initial ring structure.
    # Approximate number of rings.
    # Area of unit circle is pi.
    # If we have K rings, average area per ring ~ pi/K.
    # Circumference of ring k ~ 2*pi*r_k.
    # Number of circles on ring k ~ 2*pi*r_k / (2*r_circle).
    # This is coupled. Let's just guess K and distribute N circles.
    
    # Heuristic for K:
    # r_circle ~ sqrt(pi/N) ~ 0.1.
    # Diameter ~ 0.2.
    # Max rings ~ 1/0.2 = 5? No, packing is denser.
    # Hexagonal packing has ~3.7 circles per row.
    # Let's try K from 5 to 20 and pick the one that gives the best initial score.
    
    def generate_ring_config(K):
        # Distribute N circles into K rings.
        # More circles in outer rings.
        # r_k ~ (k+0.5)/K
        # n_k proportional to r_k
        weights = np.arange(1, K+1, dtype=float)
        weights = weights / np.sum(weights)
        n_counts = np.floor(weights * N).astype(int)
        
        # Fix rounding errors
        remainder = N - np.sum(n_counts)
        # Add remaining to the largest rings
        indices = np.argsort(weights)[::-1]
        for i in range(remainder):
            n_counts[indices[i % K]] += 1
        
        # Ensure at least 1 in each ring if K is not too large?
        # If K > N, some rings will be 0. We should avoid that.
        # Let's constrain K <= N/2 roughly.
        
        radii = np.zeros(K)
        for k in range(K):
            # Ideal radius for ring k such that circles fit tangentially?
            # Or just space them evenly from 0 to 1?
            # Let's start with radii proportional to index + 0.5
            radii[k] = (k + 0.5) / K
        
        return n_counts, radii

    def build_points_from_rings(n_counts, radii, angles_offsets=None):
        points = []
        for k, n in enumerate(n_counts):
            if n == 0:
                continue
            r = radii[k]
            # Angular spacing
            d_theta = 2 * np.pi / n
            if angles_offsets is None:
                offset = 0.0
            else:
                offset = angles_offsets[k]
            for i in range(n):
                theta = i * d_theta + offset
                points.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(points)

    def calc_score(x):
        if len(x) == 0:
            return -np.inf
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)

    # Find best initial K
    best_initial_points = None
    best_initial_score = -np.inf
    
    # Try K from 5 to 25
    for K in range(5, 26):
        n_counts, radii = generate_ring_config(K)
        if np.sum(n_counts) != N:
            continue
        points = build_points_from_rings(n_counts, radii)
        score = calc_score(points)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_points = points
            best_K = K
            best_n_counts = n_counts
            best_radii = radii

    if best_initial_points is None:
        # Fallback
        return np.random.randn(N, 2) * 0.5

    x = best_initial_points.copy()
    best_x = x.copy()
    best_r = best_initial_score
    
    # Optimization:
    # We can optimize the ring radii and angular offsets.
    # Variables: radii (K values), offsets (K values).
    # Total 2K variables.
    # Use a simple local search / gradient ascent on these parameters.
    # Reconstruct points and calculate score.
    # Since N=300, score calculation is O(N^2) ~ 90k ops. Fast enough.
    
    current_n_counts = best_n_counts
    current_radii = best_radii.copy()
    current_offsets = np.zeros(best_K)
    
    start_time = time.time()
    max_time = 115.0 # Safety margin
    
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Generate neighbors
        improved = False
        
        # 1. Perturb radii
        for _ in range(10):
            new_radii = current_radii.copy()
            idx = np.random.randint(0, best_K)
            new_radii[idx] += np.random.uniform(-step_size, step_size)
            # Clamp radii
            new_radii = np.clip(new_radii, 0.01, 0.99)
            
            points = build_points_from_rings(current_n_counts, new_radii, current_offsets)
            score = calc_score(points)
            
            if score > best_r:
                best_r = score
                best_x = points.copy()
                current_radii = new_radii
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # 2. Perturb offsets
        for _ in range(10):
            new_offsets = current_offsets.copy()
            idx = np.random.randint(0, best_K)
            new_offsets[idx] += np.random.uniform(-step_size, step_size)
            
            points = build_points_from_rings(current_n_counts, current_radii, new_offsets)
            score = calc_score(points)
            
            if score > best_r:
                best_r = score
                best_x = points.copy()
                current_offsets = new_offsets
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # 3. Try changing ring distribution (move one circle from ring i to j)
        if no_improve_count > 20:
            # Try to adjust counts
            new_n_counts = current_n_counts.copy()
            # Move circle from ring with most to ring with least? Or random.
            # Let's try moving a circle from a random ring to another.
            i = np.random.randint(0, best_K)
            j = np.random.randint(0, best_K)
            if i != j and new_n_counts[i] > 1:
                new_n_counts[i] -= 1
                new_n_counts[j] += 1
                
                points = build_points_from_rings(new_n_counts, current_radii, current_offsets)
                score = calc_score(points)
                
                if score > best_r:
                    best_r = score
                    best_x = points.copy()
                    current_n_counts = new_n_counts
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

        # Anneal step size
        if no_improve_count > 50:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 1e-4:
                break

    return best_x

# Run construct to check
if __name__ == "__main__":
    res = construct()
    print(res.shape)
