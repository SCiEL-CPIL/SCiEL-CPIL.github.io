import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin

    # --- Step 1: Hexagonal Lattice Initialization ---
    # Estimate optimal spacing
    # Area of unit disk = pi. N circles area = N * pi * r^2.
    # Hex packing density ~ 0.9069.
    # Effective area per circle ~ 1/N / 0.9069.
    # Linear size ~ sqrt(1/N).
    
    # Generate a hex lattice covering the disk
    points = []
    # Spacing s. We want roughly N points in unit disk.
    # Number of points in hex lattice of radius R with spacing s is approx (pi * R^2) / (s^2 * sqrt(3)/2)
    # Let R=1. N ~ pi / (s^2 * sqrt(3)/2) => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate points in a grid larger than the disk
    # Hex lattice basis: (s, 0) and (s/2, s*sqrt(3)/2)
    # We will iterate over i, j such that the point is within the unit disk
    
    # Determine range for i, j
    # Max coordinate is 1.
    # i * s <= 1 => i <= 1/s
    # j * s * sqrt(3)/2 <= 1 => j <= 2 / (s*sqrt(3))
    
    max_i = int(np.ceil(1.0 / s)) + 2
    max_j = int(np.ceil(2.0 / (s * np.sqrt(3)))) + 2
    
    for j in range(-max_j, max_j + 1):
        y = j * s * np.sqrt(3) / 2
        for i in range(-max_i, max_i + 1):
            x = (i + 0.5 * (j % 2)) * s
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have more points, we need to remove some to get exactly N.
    # If we have fewer, we need to add or adjust spacing.
    # Usually, hex lattice with estimated s gives close to N.
    # Let's try to adjust s to get closer to N.
    
    def generate_hex(s):
        pts = []
        max_i = int(np.ceil(1.0 / s)) + 2
        max_j = int(np.ceil(2.0 / (s * np.sqrt(3)))) + 2
        for j in range(-max_j, max_j + 1):
            y = j * s * np.sqrt(3) / 2
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * s
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        return np.array(pts)
    
    # Binary search for s to get exactly N points? Or just pick the best one.
    # Let's try a range of s values and pick the one with count closest to N.
    # Actually, if count > N, we can randomly remove points from the boundary.
    # If count < N, we increase s? No, smaller s means more points.
    # So if count < N, decrease s. If count > N, increase s.
    
    s_low = s * 0.8
    s_high = s * 1.2
    
    best_pts = None
    best_diff = float('inf')
    
    for _ in range(20):
        s_mid = (s_low + s_high) / 2
        pts = generate_hex(s_mid)
        if len(pts) == 0:
            s_low = s_mid
            continue
        diff = abs(len(pts) - N)
        if diff < best_diff:
            best_diff = diff
            best_pts = pts
            if diff == 0:
                break
        if len(pts) > N:
            s_low = s_mid
        else:
            s_high = s_mid
            
    if best_pts is None or len(best_pts) == 0:
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N)) * 0.95
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # If we have more than N points, keep the N closest to center? Or random?
    # Keeping closest to center is usually better for packing density.
    if len(best_pts) > N:
        dists = np.linalg.norm(best_pts, axis=1)
        indices = np.argsort(dists)
        best_pts = best_pts[indices[:N]]
    elif len(best_pts) < N:
        # This shouldn't happen if binary search worked well, but just in case
        # Add random points? Or just resize?
        # Let's just pad with random points inside
        while len(best_pts) < N:
            p = np.random.randn(2) * 0.5
            if np.linalg.norm(p) < 1.0:
                best_pts = np.vstack([best_pts, p])
                
    positions = best_pts.astype(np.float64)
    
    # --- Step 2: Optimization ---
    
    def compute_radius(pos):
        """Compute the true min radius."""
        # Boundary
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        # Pairs
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        
        return min(min_boundary, min_pair)
    
    best_r = compute_radius(positions)
    best_positions = positions.copy()
    
    # Gradient Ascent on Soft-Min
    # Objective: maximize S = softmin( {d_ij/2}, {1-||p_i||} )
    # softmin(x) = -1/beta * log(sum(exp(-beta * x_k)))
    # grad S = sum( w_k * grad x_k )
    # where w_k = exp(-beta * x_k) / sum(exp(-beta * x_m))
    
    # We will use a fixed beta that is sharp enough.
    # beta = 100 is usually good for double precision.
    
    beta = 50.0
    lr = 0.001
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        pos = positions
        
        # Compute all pairwise distances and boundary distances
        # Diff matrix
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        
        # Pairwise terms: d_ij / 2
        # We only care about i < j
        # Create a list of values and their gradients
        # For efficiency, we can compute the gradient analytically
        
        # Boundary terms: 1 - ||p_i||
        norms = np.linalg.norm(pos, axis=1)
        b_terms = 1.0 - norms
        
        # Collect all constraint values
        # Pairs:
        # i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_vals = dist_mat[i_idx, j_idx] / 2.0
        
        # All values
        # We need to compute weights w_k
        # To avoid overflow in exp(-beta * x), subtract min
        all_vals = np.concatenate([d_vals, b_terms])
        
        # Softmin calculation
        # S = -1/beta * log(sum(exp(-beta * v_k)))
        # grad S = sum( w_k * grad v_k )
        # w_k = exp(-beta * v_k) / sum(exp(-beta * v_m))
        
        # Numerical stability
        min_val = np.min(all_vals)
        # We want to maximize the minimum, so we care about small values.
        # The weights will be dominated by the smallest values.
        
        # Compute logits: -beta * v_k
        logits = -beta * all_vals
        # Shift by min logits to prevent underflow?
        # Actually, since we want the max of the softmin, and softmin approximates min,
        # the gradient points towards increasing the smallest values.
        
        # Let's compute the weights directly.
        # w_k = exp(-beta * v_k) / Z
        # This is equivalent to softmax on -beta * v_k? No.
        # Softmin is -Softmax(-v).
        # Grad Softmin = - Grad Softmax(-v) * (-1) = Grad Softmax(-v) * grad(-v)?
        # Let's stick to the definition:
        # S(v) = (1/beta) * log( sum( exp(-beta * v_k) ) )  <-- Wait, this is LogSumExp, which approximates MAX.
        # We want MIN.
        # Min is approximated by -1/beta * log( sum( exp(-beta * v_k) ) )?
        # If v_k are large, exp is small.
        # Let's check limits.
        # If v_1 << v_2, exp(-beta v_1) >> exp(-beta v_2).
        # Sum ~ exp(-beta v_1).
        # -1/beta * log(exp(-beta v_1)) = -1/beta * (-beta v_1) = v_1.
        # Yes, this approximates the MIN.
        
        # So S = -1/beta * log( sum( exp(-beta * v_k) ) )
        
        # Gradient:
        # dS/dv_k = -1/beta * ( 1 / Sum ) * ( -beta * exp(-beta v_k) )
        # dS/dv_k = ( exp(-beta v_k) / Sum ) = w_k
        
        # So the gradient of the soft-min w.r.t. v_k is just the normalized weight w_k.
        # And the gradient w.r.t. positions is sum( w_k * grad v_k ).
        
        # Compute weights
        # To avoid underflow, we can just compute exp(-beta * (v_k - v_min))
        # Since we only need relative weights, scaling by a constant doesn't matter.
        
        shifted_logits = -beta * (all_vals - min_val)
        exp_logits = np.exp(shifted_logits)
        weights = exp_logits / np.sum(exp_logits)
        
        # Split weights back
        n_pairs = len(d_vals)
        w_pairs = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        # Compute Gradient of Positions
        grad_pos = np.zeros_like(pos)
        
        # Contribution from boundary terms
        # v_i = 1 - ||p_i||
        # grad v_i = -p_i / ||p_i||
        # Note: if ||p_i|| is 0, gradient is undefined, but 0 vector is fine.
        for i in range(N):
            if norms[i] > 1e-10:
                grad_pos[i] += w_bounds[i] * (-pos[i] / norms[i])
        
        # Contribution from pair terms
        # v_{ij} = ||p_i - p_j|| / 2
        # grad_{p_i} v_{ij} = (p_i - p_j) / (2 * ||p_i - p_j||)
        # grad_{p_j} v_{ij} = (p_j - p_i) / (2 * ||p_i - p_j||)
        
        # Vectorized update
        # We have w_pairs for each pair (i, j)
        # diff_vec = pos[i] - pos[j]
        # dist = ||diff_vec||
        # grad_i = diff_vec / (2 * dist)
        # grad_j = -diff_vec / (2 * dist)
        
        # We can accumulate these.
        # For each pair (i, j) with weight w:
        # grad_pos[i] += w * (pos[i] - pos[j]) / (2 * dist_ij)
        # grad_pos[j] += w * (pos[j] - pos[i]) / (2 * dist_ij)
        
        # Using indices
        # diff_vecs = diff[i_idx, j_idx]  # This is (N_pairs, 2)
        # dists = d_vals * 2  # Original distances
        # coeffs = w_pairs / (2 * dists)  # (N_pairs,)
        
        # We need to add coeffs * diff_vecs to grad_pos[i] and subtract from grad_pos[j]
        # We can use np.add.at
        
        diff_vecs = diff[i_idx, j_idx]
        # dists are 2 * d_vals
        # Avoid division by zero
        safe_dists = np.where(d_vals > 1e-10, d_vals, 1e-10)
        coeffs = w_pairs / (2 * safe_dists)
        
        # grad_pos[i] += coeffs * diff_vecs
        # grad_pos[j] -= coeffs * diff_vecs
        
        np.add.at(grad_pos, i_idx, coeffs[:, np.newaxis] * diff_vecs)
        np.add.at(grad_pos, j_idx, -coeffs[:, np.newaxis] * diff_vecs)
        
        # Update positions
        positions = pos + lr * grad_pos
        
        # Projection: Ensure points stay inside the disk
        # This is a hard constraint. If a point goes out, pull it back.
        # The gradient should handle it, but numerical errors might push it out.
        new_norms = np.linalg.norm(positions, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] * (0.9999 / new_norms[mask])
        
        # Evaluate true radius periodically
        if step % 10 == 0:
            current_r = compute_radius(positions)
            if current_r > best_r:
                best_r = current_r
                best_positions = positions.copy()
        
        # Adaptive Learning Rate
        # If we are improving, keep lr. If not, decrease?
        # Or just decay slowly.
        lr = max(0.0001, lr * 0.999)
        
        # If the soft-min value stops changing, we might be stuck.
        # But with gradient ascent on smooth surrogate, it usually works well.
        
    # Final Polish
    # Use a much sharper beta and smaller lr for the last few steps
    beta = 200.0
    lr = 0.0001
    for step in range(200):
        if time.time() > deadline:
            break
            
        pos = best_positions
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        
        norms = np.linalg.norm(pos, axis=1)
        b_terms = 1.0 - norms
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_vals = dist_mat[i_idx, j_idx] / 2.0
        
        all_vals = np.concatenate([d_vals, b_terms])
        min_val = np.min(all_vals)
        
        shifted_logits = -beta * (all_vals - min_val)
        exp_logits = np.exp(shifted_logits)
        weights = exp_logits / np.sum(exp_logits)
        
        n_pairs = len(d_vals)
        w_pairs = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        grad_pos = np.zeros_like(pos)
        
        for i in range(N):
            if norms[i] > 1e-10:
                grad_pos[i] += w_bounds[i] * (-pos[i] / norms[i])
        
        diff_vecs = diff[i_idx, j_idx]
        safe_dists = np.where(d_vals > 1e-10, d_vals, 1e-10)
        coeffs = w_pairs / (2 * safe_dists)
        
        np.add.at(grad_pos, i_idx, coeffs[:, np.newaxis] * diff_vecs)
        np.add.at(grad_pos, j_idx, -coeffs[:, np.newaxis] * diff_vecs)
        
        positions = pos + lr * grad_pos
        
        new_norms = np.linalg.norm(positions, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] * (0.9999 / new_norms[mask])
        
        current_r = compute_radius(positions)
        if current_r > best_r:
            best_r = current_r
            best_positions = positions.copy()
            
    return best_positions

def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    norms = np.linalg.norm(pos, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
    
    # Handle coincident points
    # Simple check for very close points
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    close_mask = dist[i_idx, j_idx] < 1e-8
    if np.any(close_mask):
        close_i = i_idx[close_mask]
        close_j = j_idx[close_mask]
        for i, j in zip(close_i, close_j):
            pos[j] += np.random.randn(2) * 1e-6
            # Re-clamp
            n = np.linalg.norm(pos[j])
            if n > 1.0:
                pos[j] = pos[j] * (0.999 / n)

    return pos

_original_construct = construct
def construct():
    pos = _original_construct()
    return _validate_and_fix(pos)
