"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 200
DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    n = len(pts)
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_boundary = np.minimum(dx, dy).min()
    # Pairwise distances (use chunking to avoid huge memory)
    min_pair = np.inf
    chunk = 200
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        diff = pts[i:i_end, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        # Zero out diagonal
        idx = np.arange(i_end - i)
        dists[idx, idx + i] = np.inf
        min_pair = min(min_pair, dists.min())
    return min(min_pair / 2.0, r_boundary)


def _softmin_logsumexp(x, beta):
    """Soft minimum via log-sum-exp of -x."""
    neg_x = -x
    max_neg = np.max(neg_x)
    return -max_neg - np.log(np.sum(np.exp(neg_x - max_neg))) / beta


def _grad_softmin_neg(pts, beta):
    """
    Gradient of the negative soft-min of pairwise distances and boundary distances.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalent to minimizing -r.
    Use soft-min surrogate: f = -softmin(distances) where distances include
    pairwise distances and boundary distances.
    """
    n = len(pts)
    # Build distance vectors: pairwise + boundary
    # For efficiency, compute gradients analytically
    
    # Boundary terms
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_boundary = np.minimum(dx, dy)
    
    # Pairwise distances
    min_pair = np.inf
    # We'll accumulate gradient contributions
    
    # Use a weighted soft-min approach
    # f = -1/beta * log(sum(exp(-beta * d_i))) for all constraints d_i
    # grad f = sum(exp(-beta*d_i) * grad d_i) / sum(exp(-beta*d_i)) * (-1)
    # Wait: f = -(1/beta) * log(sum exp(-beta * d_i))
    # df/dd_i = (1/beta) * (-beta * exp(-beta*d_i)) / sum(exp(-beta*d_j)) = -exp(-beta*d_i)/sum(...)
    # So grad f = sum_i (-exp(-beta*d_i)/sum_j exp(-beta*d_j)) * grad d_i
    # We want to maximize f, so we go in direction of grad f.
    
    # Collect all distances and their gradients
    # Boundary: d_i = min(dx_i, dy_i) -> use soft min of dx_i, dy_i
    # Pairwise: d_ij = |c_i - c_j|
    
    # For simplicity and speed, use a two-level approach:
    # 1. Soft-min over boundary distances
    # 2. Soft-min over pairwise distances
    # 3. Soft-min over the two results
    
    # Actually, let's just do a direct approach with all constraints
    # Number of constraints: n (boundary) + n*(n-1)/2 (pairwise)
    # For n=200, that's 200 + 19900 = 20100 constraints
    
    # Compute all pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    r_b = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Combine all distances into one vector
    all_dists = np.concatenate([dists.flatten(), r_b])
    
    # Soft-min weights
    neg_beta_d = -beta * all_dists
    max_neg = np.max(neg_beta_d)
    weights = np.exp(neg_beta_d - max_neg)
    weights /= np.sum(weights)
    
    # Gradient of soft-min: 
    # softmin = -1/beta * log(sum exp(-beta*d_i))
    # d(softmin)/d(pts) = sum_i weights_i * (-1) * d(d_i)/d(pts)
    # Wait: let me recompute
    # f = -(1/beta) * log(sum_j exp(-beta * d_j))
    # df/d(pts_k) = -(1/beta) * [sum_j (-beta * exp(-beta*d_j) * dd_j/d(pts_k)) / sum_j exp(-beta*d_j)]
    # = sum_j [exp(-beta*d_j)/sum_j exp(-beta*d_j)] * dd_j/d(pts_k)
    # = sum_j weights_j * dd_j/d(pts_k)
    
    # So grad f = sum_j weights_j * grad d_j
    # We want to maximize f, so move in direction of grad f.
    
    grad = np.zeros_like(pts)
    
    # Pairwise distance gradients
    # d_{ij} = |c_i - c_j|
    # dd_{ij}/dc_i = (c_i - c_j) / d_{ij}
    # dd_{ij}/dc_j = (c_j - c_i) / d_{ij}
    
    # weights for pairwise: first n*(n-1)/2 entries
    w_pair = weights[:n*(n-1)//2].reshape(n, n)
    # Note: dists is n x n with diagonal inf, so weights for diagonal should be ~0
    # But we set diagonal to inf, so exp(-beta*inf) = 0, good.
    
    # For each pair (i,j), contribution to grad[i] += w_pair[i,j] * (pts[i]-pts[j])/dists[i,j]
    # and grad[j] += w_pair[j,i] * (pts[j]-pts[i])/dists[i,j]
    
    # Vectorized:
    # grad_pair[i] = sum_j w_pair[i,j] * (pts[i] - pts[j]) / dists[i,j]
    # = pts[i] * sum_j w_pair[i,j] - sum_j w_pair[i,j] * pts[j] / dists[i,j]
    
    # But dists[i,j] in denominator, so:
    # term[i,j] = w_pair[i,j] / dists[i,j]
    # grad_pair[i] = sum_j term[i,j] * (pts[i] - pts[j])
    # = pts[i] * sum_j term[i,j] - sum_j term[i,j] * pts[j]
    
    # Handle inf in dists: where dists is inf, term should be 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = w_pair / dists
    term[~np.isfinite(term)] = 0
    
    sum_term = term.sum(axis=1)  # shape (n,)
    weighted_pts = term @ pts  # shape (n, 2)
    grad_pair = pts * sum_term[:, None] - weighted_pts
    
    # Boundary gradients
    # d_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # Use soft-min approximation for the min, or just use the active constraint
    # For simplicity, use the active one (hard min) for gradient
    w_b = weights[n*(n-1)//2:]  # shape (n,)
    
    # For each point, d_i = min(dx_i, dy_i)
    # If dx_i < dy_i: dd_i/dx_i = -sign(x_i), dd_i/dy_i = 0
    # If dx_i > dy_i: dd_i/dx_i = 0, dd_i/dy_i = -sign(y_i)
    
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    
    grad_b = np.zeros_like(pts)
    # Where dx <= dy: boundary in x direction
    mask_x = dx <= dy
    grad_b[mask_x, 0] = -w_b[mask_x] * np.sign(pts[mask_x, 0])
    # Where dy < dx: boundary in y direction
    mask_y = ~mask_x
    grad_b[mask_y, 1] = -w_b[mask_y] * np.sign(pts[mask_y, 1])
    
    grad = grad_pair + grad_b
    
    # f = softmin, we want to maximize f
    # So gradient ascent: pts += lr * grad
    return grad


def _initial_hex_grid():
    """Generate initial hexagonal lattice positions for N circles."""
    # Hexagonal packing: rows offset by half spacing
    # Try to fit N circles in a hexagonal arrangement
    # Start with a grid and add hex offset
    
    # Estimate spacing: area per circle ~ 1/sqrt(N) * sqrt(3)/2 for hex
    # For square: spacing ~ 1/sqrt(N)
    # Hex is denser, so we can use slightly smaller spacing
    
    # Let's generate a hex grid and take the first N points
    # Hex grid: row y, col x
    # x = x_spacing * (col + 0.5*(row % 2))
    # y = y_spacing * row
    # y_spacing = x_spacing * sqrt(3)/2
    
    # We want to fit in [-0.5, 0.5]^2
    # Try various spacings
    
    best_pts = None
    best_r = 0
    
    for x_spacing in np.linspace(0.035, 0.045, 20):
        y_spacing = x_spacing * np.sqrt(3) / 2
        # Generate hex grid
        n_rows = int(1.0 / y_spacing) + 2
        n_cols = int(1.0 / x_spacing) + 2
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + y_spacing * (row + 0.5)
            if y > 0.5:
                break
            offset = 0.5 * x_spacing if row % 2 == 1 else 0.0
            for col in range(n_cols):
                x = -0.5 + x_spacing * (col + 0.5) + offset
                if x > 0.5:
                    break
                pts.append((x, y))
        
        if len(pts) >= N:
            pts = np.array(pts[:N])
            # Center the points
            center = pts.mean(axis=0)
            pts -= center
            # Check if all inside
            if np.all(np.abs(pts) < 0.5):
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    # Generate initial hexagonal grid
    pts = _initial_hex_grid()
    
    # Gradient ascent with soft-min surrogate
    beta = 100.0  # Temperature for soft-min (higher = sharper)
    lr = 0.001
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Schedule: decrease beta over time for sharper optimization
    n_steps = 500
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Update beta: start high (smooth), decrease (sharper)
        progress = step / n_steps
        beta = 50.0 + 200.0 * progress  # 50 to 250
        
        # Compute gradient
        grad = _grad_softmin_neg(pts, beta)
        
        # Adaptive learning rate
        lr = 0.002 / (1.0 + step / 100.0)
        
        # Gradient step
        pts_new = pts + lr * grad
        
        # Project back to square if needed
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check improvement
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    # Final polish: try a few more steps with high beta
    if time.time() < deadline - 2.0:
        beta_final = 500.0
        lr_final = 0.0005
        for _ in range(200):
            if time.time() > deadline:
                break
            grad = _grad_softmin_neg(pts, beta_final)
            pts_new = pts + lr_final * grad
            pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
            pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
            r_new = _compute_radius(pts_new)
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_new.copy()
            pts = pts_new
    
    return best_pts
# EVOLVE-BLOCK-END
