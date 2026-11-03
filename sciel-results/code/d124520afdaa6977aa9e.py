import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Vectorized pairwise distance calculation
    # This is O(N^2) but implemented in C via NumPy, so it's fast.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def _smooth_objective_and_grad(x, beta=100.0):
    """
    Objective function for L-BFGS-B.
    We want to maximize the minimum margin (distance to boundary or half distance to other points).
    Let m_i be the margins. We want to maximize min(m_i).
    We approximate min(m_i) with a soft-min: max_m - (1/beta) * log(sum(exp(beta*(max_m - m_i))))
    This function returns the NEGATIVE of this value (to minimize) and the gradient.
    """
    pts = x.reshape(N, 2)
    
    # 1. Boundary margins: 1 - ||p_i||
    norms = np.linalg.norm(pts, axis=1)
    # Avoid division by zero in gradient
    norms_safe = np.where(norms < 1e-10, 1e-10, norms)
    boundary_margins = 1.0 - norms
    
    # 2. Pairwise margins: ||p_i - p_j|| / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pairwise_margins = dists / 2.0
    
    # Combine all margins into a single vector
    # Size: N (boundary) + N*(N-1) (pairwise)
    all_margins = np.concatenate([boundary_margins.ravel(), pairwise_margins.ravel()])
    
    # Soft-min calculation
    max_margin = np.max(all_margins)
    shifted = beta * (max_margin - all_margins)
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp)
    
    # Soft-min value (approximation of min(all_margins))
    soft_min_val = max_margin - (1.0 / beta) * log_sum_exp
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    obj = -soft_min_val
    
    # Gradient calculation
    # d(obj)/dx = - d(soft_min_val)/dx
    # d(soft_min_val)/dx = d(max_margin)/dx - (1/beta) * d(log_sum_exp)/dx
    # Note: max_margin is not differentiable, but in the context of soft-min,
    # the gradient is dominated by the terms with the smallest margins.
    # The gradient of the soft-min is a weighted average of the gradients of the individual margins,
    # where weights are proportional to exp(beta * (max_margin - m_i)).
    
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(x)
    
    # Boundary contributions
    # Margin i: 1 - norm(p_i)
    # d(margin_i)/dx = -p_i / norm(p_i)
    # Contribution to grad: weights[i] * d(soft_min)/dx_i 
    # Actually, d(soft_min)/dx = sum(weights[i] * d(margin_i)/dx) is NOT quite right because max_margin also depends on x.
    # However, for large beta, the soft-min gradient is well-approximated by the gradient of the minimum margin itself.
    # A more rigorous derivation:
    # L = max_m - 1/beta log(sum exp(beta(max_m - m_k)))
    # dL/dx = d(max_m)/dx - sum_k (exp(beta(max_m - m_k)) / sum_j ...) * (beta * (d(max_m)/dx - d(m_k)/dx))
    # This is complex. A simpler and standard approach for "min of functions" is to use the log-sum-exp trick directly on the negative margins.
    # Let's stick to the standard formulation:
    # f(x) = - (1/beta) log(sum exp(-beta * m_k(x)))
    # This is a smooth approximation to -min(m_k).
    # Let's re-derive using this simpler form which avoids max_margin derivative issues.
    
    # Re-calculate using simpler form: f(x) = -1/beta * log(sum(exp(-beta * m_k)))
    # This is equivalent to the previous one up to an additive constant if we shift by max.
    # Let's use the shifted version for numerical stability, but the gradient formula is:
    # grad_f = sum_k (w_k * grad(-m_k)) where w_k = exp(-beta*m_k) / sum(exp(-beta*m_j))
    # Wait, the previous derivation was:
    # soft_min = max_m - 1/beta log(sum(exp(beta(max_m - m_k))))
    # Let u_k = beta * (max_m - m_k).
    # soft_min = max_m - 1/beta log(sum(exp(u_k)))
    # d(soft_min)/dx = d(max_m)/dx - 1/beta * (1/sum(exp(u))) * sum(exp(u_k) * d(u_k)/dx)
    # d(u_k)/dx = -beta * d(m_k)/dx  (since max_m is constant w.r.t x in this local view? No, max_m is one of the m_k)
    # This is tricky.
    
    # Alternative robust approach:
    # Use the gradient of the log-sum-exp of the NEGATIVE margins.
    # Objective: Minimize F(x) = -1/beta * log( sum( exp(-beta * m_k(x)) ) )
    # This F(x) approximates -min(m_k(x)).
    # grad F = -1/beta * 1/S * sum( exp(-beta*m_k) * (-beta * grad m_k) )
    # grad F = 1/S * sum( exp(-beta*m_k) * grad m_k )
    # where S = sum( exp(-beta*m_j) )
    # Let p_k = exp(-beta*m_k) / S. These are the probabilities.
    # grad F = sum( p_k * grad m_k )
    # Since we want to minimize F, and F approx -min(m), minimizing F is equivalent to maximizing min(m).
    # So the direction to move is -grad F = -sum(p_k * grad m_k).
    # Wait, L-BFGS-B minimizes the function passed to it.
    # If we pass F, it will minimize F (which is -min_m). So it will maximize min_m.
    # The gradient of F is sum(p_k * grad m_k).
    # So we return F and grad_F.
    
    # Let's recalculate p_k and grad m_k.
    # m_k are the margins.
    # For boundary i: m_i = 1 - norm(p_i). grad m_i = -p_i / norm(p_i).
    # For pair (i,j): m_ij = dist(i,j)/2. grad m_ij w.r.t p_i = (p_i - p_j) / (2 * dist(i,j)).
    
    # We need to map the weights p_k back to the points.
    # p is a vector of length N + N*(N-1).
    # First N are boundary. Next N*(N-1) are pairs.
    
    # Recalculate exp(-beta * m) for stability
    # We already have exp_shifted = exp(beta * (max_m - m_k)).
    # exp(-beta * m_k) = exp(-beta * max_m) * exp(beta * (max_m - m_k)) = C * exp_shifted.
    # So p_k is proportional to exp_shifted.
    # p_k = exp_shifted / sum(exp_shifted). This is exactly what we calculated as 'weights' above!
    # So the weights 'weights' calculated above are the correct probabilities p_k.
    
    # Now, compute grad F = sum(p_k * grad m_k).
    # grad F is a vector of size 2N.
    
    # 1. Boundary contributions
    # For each i, weight is weights[i].
    # grad m_i = -p_i / norm(p_i)
    # Contribution to grad F: weights[i] * (-p_i / norm(p_i))
    for i in range(N):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[2*i:2*i+2] += w * (-pts[i] / norms[i])
            
    # 2. Pairwise contributions
    # The pairwise margins are stored in a flattened matrix.
    # Index in all_margins for pair (i,j) is N + i*N + j.
    # We need to accumulate gradients for p_i and p_j.
    # grad m_ij w.r.t p_i = (p_i - p_j) / (2 * dist_ij)
    # grad m_ij w.r.t p_j = (p_j - p_i) / (2 * dist_ij) = -grad m_ij w.r.t p_i
    
    # Iterate over pairs. To be efficient, we can vectorize this, but N=50 is small.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            idx = N + i*N + j
            w = weights[idx]
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d > 1e-10:
                vec = pts[i] - pts[j]
                grad_vec = vec / (2.0 * d)
                grad[2*i:2*i+2] += w * grad_vec
                grad[2*j:2*j+2] += w * (-grad_vec)
                
    return obj, grad

def _initial_rings():
    """Generate a good initial guess using ring structure."""
    points = np.zeros((N, 2))
    # Center
    points[0] = [0.0, 0.0]
    idx = 1
    
    # Ring 1: 6 circles
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points[idx] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        idx += 1
        
    # Ring 2: 12 circles
    r2 = 0.5
    offset2 = np.pi / 12
    for k in range(12):
        angle = offset2 + 2 * np.pi * k / 12
        points[idx] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        idx += 1
        
    # Ring 3: 18 circles
    r3 = 0.75
    offset3 = 0.0
    for k in range(18):
        angle = offset3 + 2 * np.pi * k / 18
        points[idx] = [r3 * np.cos(angle), r3 * np.sin(angle)]
        idx += 1
        
    # Ring 4: 13 circles
    r4 = 0.95
    offset4 = 0.0
    for k in range(13):
        angle = offset4 + 2 * np.pi * k / 13
        points[idx] = [r4 * np.cos(angle), r4 * np.sin(angle)]
        idx += 1
        
    return points

def _initial_hexagonal():
    """Generate a hexagonal grid initial guess, clipped to circle."""
    points = []
    # Hexagonal spacing
    # For r ~ 0.125, spacing ~ 0.25
    s = 0.25
    # Generate points in a large hexagonal grid
    # Range -1 to 1
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s * 1.5
            y = j * s * np.sqrt(3) + (i % 2) * s * np.sqrt(3) / 2
            if x**2 + y**2 < 0.9:
                points.append([x, y])
    
    # Sort by distance to center to get a nice packing
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(norms)
    points = points[sorted_idx][:N]
    
    return points

def construct():
    deadline = time.time() + DEADLINE
    best_r = 0.0
    best_points = None
    
    initial_guesses = [
        _initial_rings(),
        _initial_hexagonal()
    ]
    
    # Add a random perturbed ring guess
    p = _initial_rings()
    p += np.random.normal(0, 0.01, p.shape)
    initial_guesses.append(p)
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        start_time = time.time()
        x0 = guess.flatten()
        
        # Use a moderate beta initially, then refine? 
        # L-BFGS-B with a fixed beta=100 is usually sufficient for N=50.
        # We can try increasing beta in later iterations if needed, but let's keep it simple first.
        
        try:
            result = minimize(
                _smooth_objective_and_grad, 
                x0, 
                args=(100.0,), # beta
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                pts = result.x.reshape(N, 2)
                r = _compute_radius(pts)
                
                if r > best_r:
                    best_r = r
                    best_points = pts
                    
        except Exception as e:
            pass
    
    # If we still have time, do a final high-precision polish on the best point
    if best_points is not None and time.time() < deadline - 5:
        # Increase beta for sharper minimum
        x0 = best_points.flatten()
        try:
            result = minimize(
                _smooth_objective_and_grad, 
                x0, 
                args=(500.0,), # Higher beta
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                pts = result.x.reshape(N, 2)
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_points = pts
        except Exception:
            pass

    # Final safety check
    if best_points is None:
        best_points = _initial_rings()
        
    # Ensure strictly inside
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.9999:
            best_points[i] *= 0.9999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = _compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
