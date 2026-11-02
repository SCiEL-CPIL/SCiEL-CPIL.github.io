import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure finite and normalized
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    
    # 2. Generate Test Points on Sphere
    # We need a dense set to approximate the max distance.
    # 50,000 points should be sufficient for N=130.
    M = 50000
    # Use a separate spiral or random uniform points. 
    # Random uniform is unbiased.
    Y = np.random.randn(M, 3)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Pre-compute dot products? No, P changes.
    
    # Optimization Parameters
    p_start = 2.0
    p_end = 20.0
    num_steps = 1000
    eta = 0.01 # Initial step size
    
    best_P = P.copy()
    best_score = np.inf
    
    # We will track the "covering radius" approximation using the max of d_k
    # But we minimize sum(d_k^p)
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
            
        # Animate p
        progress = step / float(num_steps)
        p = p_start + (p_end - p_start) * progress
        
        # Compute distances and nearest neighbors
        # Dot product matrix: (N, M)
        # This might be memory intensive: 130 * 50000 * 8 bytes = 52 MB. OK.
        dots = np.dot(P, Y.T) # (N, M)
        
        # Clip to [-1, 1] for numerical stability in arccos
        dots = np.clip(dots, -1.0, 1.0)
        
        # Distances: arccos(dots)
        dists = np.arccos(dots) # (N, M)
        
        # Find min distance for each Y_k
        min_dists = np.min(dists, axis=0) # (M,)
        min_indices = np.argmin(dists, axis=0) # (M,)
        
        # Current objective (soft max approximation)
        # For tracking, let's also compute the true max of min_dists
        current_max_dist = np.max(min_dists)
        if current_max_dist < best_score:
            best_score = current_max_dist
            best_P = P.copy()
            
        # Compute Gradient
        # G_i = sum_{k where min_indices[k]==i} p * min_dists[k]^(p-1) * grad_arccos(P_i, Y_k)
        # grad_arccos(u) = -Y_k / sqrt(1 - (P_i . Y_k)^2)
        
        # Initialize gradients
        G = np.zeros_like(P)
        
        # For each test point k, contribute to gradient of P_{min_indices[k]}
        # We can vectorize this using bincount or sparse operations, but loop is safer for clarity and memory if not careful.
        # Let's try vectorized accumulation.
        
        # We need the specific dot product P_{i_k} . Y_k
        # dots[min_indices, np.arange(M)] gives (M,)
        u_k = dots[min_indices, np.arange(M)]
        
        # Gradient factor for each k
        # d/dP arccos(P.Y) = -Y / sqrt(1 - (P.Y)^2)
        # We need sqrt(1 - u_k^2)
        denom = np.sqrt(np.maximum(1.0 - u_k * u_k, 1e-12))
        
        # Weight for each k: p * min_dists[k]^(p-1)
        weights = p * (min_dists[k] ** (p - 1)) if p > 1 else np.ones(M) # Handle p=1 edge case, though we start at 2
        # Actually, if p=2, weight is 2 * d.
        
        # The vector part is -Y_k / denom_k
        # We need to sum (weight_k * (-Y_k / denom_k)) for all k assigned to i
        
        # Let's create a matrix of weighted vectors
        # vecs = (M, 3)
        vecs = -Y / denom[:, np.newaxis]
        weighted_vecs = vecs * weights[:, np.newaxis]
        
        # Sum weighted_vecs for each i
        # np.bincount is efficient for 1D, but we have 3D.
        # We can use np.add.at
        np.add.at(G, min_indices, weighted_vecs)
        
        # Project Gradient to Tangent Space
        # G_i = G_i - (G_i . P_i) P_i
        proj = np.sum(G * P, axis=1, keepdims=True)
        G = G - proj * P
        
        # Update
        # Simple gradient descent
        # Adaptive step size?
        # Let's use a decaying step size
        eta_current = eta / (1.0 + 0.01 * step)
        
        P = P - eta_current * G
        
        # Renormalize
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
    # Final Polish
    # Run a few steps with very small step size and high p
    p = 50.0
    for _ in range(50):
        if time.time() > deadline:
            break
        dots = np.dot(P, Y.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=0)
        min_indices = np.argmin(dists, axis=0)
        
        u_k = dots[min_indices, np.arange(M)]
        denom = np.sqrt(np.maximum(1.0 - u_k * u_k, 1e-12))
        weights = p * (min_dists ** (p - 1))
        
        vecs = -Y / denom[:, np.newaxis]
        weighted_vecs = vecs * weights[:, np.newaxis]
        
        G = np.zeros_like(P)
        np.add.at(G, min_indices, weighted_vecs)
        
        proj = np.sum(G * P, axis=1, keepdims=True)
        G = G - proj * P
        
        eta_current = 0.001
        P = P - eta_current * G
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
    # Return the best found configuration
    # We should check if best_P is better than the final P
    # Calculate score for both
    def calc_score(pts):
        # Use a slightly denser grid for final evaluation? Or just return the points.
        # The evaluator will calculate the score. We just return the points.
        # But we want to return the points that gave the lowest score we saw.
        return pts
        
    # Heuristic: The "best_P" tracked during the loop might be better than the final P
    # if the optimization diverged or oscillated at the end.
    # Let's compare the max distance on the grid Y for best_P and P.
    
    def max_dist(pts):
        d = np.dot(pts, Y.T)
        d = np.clip(d, -1.0, 1.0)
        dists = np.arccos(d)
        return np.max(np.min(dists, axis=0))
        
    score_best = max_dist(best_P)
    score_final = max_dist(P)
    
    if score_best < score_final:
        return best_P
    else:
        return P
