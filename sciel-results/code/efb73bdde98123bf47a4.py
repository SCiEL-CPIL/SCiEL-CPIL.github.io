"""Improved packing: gradient ascent on smooth log-sum-exp surrogate with constraint handling."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _score(c):
    """Compute the packing score (largest common radius)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    # Distance to walls
    wall_dists = np.min(np.abs(0.5 - np.abs(c)), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_logsumexp(x, beta):
    """Soft minimum via log-sum-exp: approximates min(x) as beta->inf."""
    x = np.asarray(x, dtype=np.float64)
    # For soft min: -logsumexp(-x*beta)/beta
    neg_x = -x * beta
    m = np.max(neg_x)
    if not np.isfinite(m):
        return 0.0
    s = np.sum(np.exp(neg_x - m))
    return -(m + np.log(s)) / beta

def _objective_and_grad(c, beta):
    """
    Objective: maximize min(pairwise_dist/2, wall_dist)
    Using soft-min surrogate.
    Returns (value, grad).
    """
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances: min over 6 walls
    # For point c_i, wall dist = min(0.5-c_ix, c_ix+0.5, 0.5-c_iy, c_iy+0.5, 0.5-c_iz, c_iz+0.5)
    # = min over k of (0.5 - |c_ik|)
    wall_dists = np.array([0.5 - np.abs(c[:, k]) for k in range(3)])
    # shape: (3, n) -> for each point, 3 wall distances (one per axis)
    # Actually we want for each point the min over all 6 walls, but since symmetric,
    # min(0.5-|c_x|, 0.5-|c_y|, 0.5-|c_z|) is the wall distance for that point.
    # So wall_dists[i] = min_k (0.5 - |c_ik|)
    wall_per_point = np.min(wall_dists, axis=0)  # shape (n,)
    
    # Combine all constraints into one array for soft-min
    # Pairwise: d[i,j]/2 for i<j
    iu = np.triu_indices(n, 1)
    pair_vals = d[iu] / 2.0
    # Wall: wall_per_point[i]
    all_vals = np.concatenate([pair_vals, wall_per_point])
    
    # Soft min
    neg = -all_vals * beta
    m = np.max(neg)
    if not np.isfinite(m):
        return 0.0, np.zeros_like(c)
    exp_neg = np.exp(neg - m)
    s = np.sum(exp_neg)
    if s < 1e-300:
        return 0.0, np.zeros_like(c)
    probs = exp_neg / s
    val = -(m + np.log(s)) / beta
    
    # Gradient of soft min w.r.t. c
    # d(val)/d(c) = -sum_k probs[k] * d(all_vals[k])/d(c)
    grad = np.zeros_like(c)
    
    # Pairwise gradients
    # d(pair_vals[k])/d(c_i) for pair (i,j): d/d(c_i) (|c_i - c_j|/2) = (c_i - c_j)/(2*|c_i-c_j|)
    # We need to accumulate: for each pair (i,j), contribution = -probs[k] * grad_pair
    # Let's build the gradient from pairwise terms
    pair_grad = np.zeros_like(c)
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if d[i, j] < 1e-15:
            continue
        w = -probs[idx]  # negative because we're doing soft min
        # d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d(|c_i - c_j|/2)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
        dir_ij = (c[i] - c[j]) / (2.0 * d[i, j])
        pair_grad[i] += w * dir_ij
        pair_grad[j] += w * (-dir_ij)
    
    # Wall gradients
    # d(wall_per_point[i])/d(c_i_k) = -sign(c_i_k) if that axis is the min
    # But with soft-min over all wall distances per point, we need:
    # For each point i, wall_per_point[i] = min_k (0.5 - |c_ik|)
    # The soft-min is over ALL wall distances AND pairwise distances combined.
    # So we need the gradient of each wall distance term.
    # Wall term for point i, axis k: w_ik = 0.5 - |c_ik|
    # d(w_ik)/d(c_ik) = -sign(c_ik)
    # In the combined array, wall terms start at index len(pair_vals)
    wall_start = len(pair_vals)
    for i in range(n):
        for k in range(3):
            idx = wall_start + i * 3 + k  # Wait, I stored wall_per_point as min over axes per point
            # Actually I need to restructure. Let me redo.
    
    # Let me restructure: build all constraint values explicitly
    # Constraints:
    # 1) For each pair (i,j), i<j: d_ij / 2
    # 2) For each point i and axis k: 0.5 - |c_ik|
    
    # Rebuild properly
    all_vals = np.empty(len(pair_vals) + n * 3)
    all_vals[:len(pair_vals)] = pair_vals
    for i in range(n):
        for k in range(3):
            all_vals[len(pair_vals) + i * 3 + k] = 0.5 - abs(c[i, k])
    
    neg = -all_vals * beta
    m = np.max(neg)
    if not np.isfinite(m):
        return 0.0, np.zeros_like(c)
    exp_neg = np.exp(neg - m)
    s = np.sum(exp_neg)
    if s < 1e-300:
        return 0.0, np.zeros_like(c)
    probs = exp_neg / s
    val = -(m + np.log(s)) / beta
    
    grad = np.zeros_like(c)
    
    # Pairwise part
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if d[i, j] < 1e-15:
            continue
        w = -probs[idx]
        dir_ij = (c[i] - c[j]) / (2.0 * d[i, j])
        grad[i] += w * dir_ij
        grad[j] += w * (-dir_ij)
    
    # Wall part
    for i in range(n):
        for k in range(3):
            idx = len(pair_vals) + i * 3 + k
            w = -probs[idx]
            # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            grad[i, k] += w * (-sgn)
    
    return val, grad

def _project_to_cube(c):
    """Project centers into the cube [-0.5, 0.5]^3."""
    c = np.clip(c, -0.4999, 0.4999)
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initial random positions
    c = np.random.uniform(-0.4, 0.4, size=(N, 3))
    
    best_c = c.copy()
    best_score = _score(best_c)
    
    # Adaptive beta for soft-min
    beta = 50.0
    lr = 0.01
    momentum = 0.9
    vel = np.zeros_like(c)
    
    # Gradient ascent with momentum
    step = 0
    while time.time() < deadline:
        if step % 100 == 0:
            # Increase beta gradually
            beta = min(beta * 1.05, 500.0)
            lr = max(lr * 0.995, 0.001)
        
        val, grad = _objective_and_grad(c, beta)
        
        # Update with momentum
        vel = momentum * vel + lr * grad
        c = c + vel
        c = _project_to_cube(c)
        
        # Check score periodically
        if step % 50 == 0:
            s = _score(c)
            if s > best_score:
                best_score = s
                best_c = c.copy()
        
        step += 1
        
        # If we've made good progress, do a local refinement
        if step % 500 == 0 and best_score > 0.13:
            # Try a few random restarts if time allows
            remaining = deadline - time.time()
            if remaining > 30:
                for _ in range(3):
                    if time.time() > deadline - 5:
                        break
                    c2 = best_c + np.random.normal(0, 0.02, size=(N, 3))
                    c2 = _project_to_cube(c2)
                    s2 = _score(c2)
                    if s2 > best_score:
                        best_score = s2
                        best_c = c2.copy()
    
    # Final polish: try to improve with smaller steps
    remaining = deadline - time.time()
    if remaining > 5:
        c = best_c.copy()
        beta = 200.0
        lr = 0.005
        vel = np.zeros_like(c)
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            val, grad = _objective_and_grad(c, beta)
            vel = 0.9 * vel + lr * grad
            c = c + vel
            c = _project_to_cube(c)
            s = _score(c)
            if s > best_score:
                best_score = s
                best_c = c.copy()
    
    return best_c
# EVOLVE-BLOCK-END
