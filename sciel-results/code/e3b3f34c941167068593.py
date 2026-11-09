import numpy as np
import time

N = 100

def compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    # overlap(s) = sum h[i] * (1 - h[(i+s) % N]) = N/2 - sum h[i]*h[(i+s)%N]
    # We want to minimize max_s overlap(s)
    max_ov = 0.0
    for s in range(1, N):  # s=0 gives overlap = sum h[i](1-h[i]) = N/2 - sum h[i]^2
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    # Also check s=0
    ov0 = np.sum(h * (1.0 - h))
    if ov0 > max_ov:
        max_ov = ov0
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial: uniform
    h = np.full(N, 0.5)
    best_score = compute_score(h)
    best_h = h.copy()
    
    # Try structured initializations
    # 1. Cosine-based
    for k in [1, 2, 3, 5, 7, 10, 13]:
        for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h_test = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N)
            h_test = np.clip(h_test, 0, 1)
            # Adjust sum to N/2
            diff = N / 2 - np.sum(h_test)
            h_test += diff / N
            h_test = np.clip(h_test, 0, 1)
            diff2 = N / 2 - np.sum(h_test)
            h_test += diff2 / N
            score = compute_score(h_test)
            if score < best_score:
                best_score = score
                best_h = h_test.copy()
    
    # 2. Quadratic residue-inspired pattern
    # Use a pseudo-random but structured sequence
    np.random.seed(42)
    for trial in range(20):
        h_test = np.random.uniform(0, 1, N)
        # Make it more structured: use a low-discrepancy sequence
        # Van der Corput / Halton-like
        h_test = 0.5 + 0.4 * np.sin(2 * np.pi * (np.arange(N) / N) * (trial + 1))
        h_test = np.clip(h_test, 0, 1)
        diff = N / 2 - np.sum(h_test)
        h_test += diff / N
        h_test = np.clip(h_test, 0, 1)
        diff2 = N / 2 - np.sum(h_test)
        h_test += diff2 / N
        score = compute_score(h_test)
        if score < best_score:
            best_score = score
            best_h = h_test.copy()
    
    # 3. Coordinate descent refinement
    h = best_h.copy()
    score = best_score
    lr = 0.01
    no_improve = 0
    
    while time.time() < deadline:
        # Compute gradient of max overlap
        # For each shift s, overlap_s = sum h[i](1-h[i+s])
        # d(overlap_s)/d(h[i]) = (1 - h[i+s]) - h[i] (for the term where i is the first index)
        #                      + (-h[i-s]) for the term where i is the second index (shifted)
        # Actually, let's use subgradient: find the shift with max overlap, then take its gradient
        
        max_ov = -1
        max_s = 0
        for s in range(N):
            shifted = np.roll(h, s)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
                max_s = s
        
        # Gradient of overlap at max_s: d/dh[i] sum_j h[j](1-h[(j+max_s)%N])
        # = (1 - h[(i+max_s)%N]) + (-h[(i-max_s)%N])  [the second term from when j = i - max_s]
        # Wait: sum_j h[j](1-h[(j+s)%N])
        # d/dh[i]: term j=i: (1-h[(i+s)%N])
        # term j such that (j+s)%N = i, i.e., j=(i-s)%N: -h[(i-s)%N]
        grad = (1.0 - np.roll(h, max_s)) + (-np.roll(h, -max_s))
        
        # Also need to maintain sum constraint: project gradient
        grad -= np.mean(grad)
        
        # Also maintain [0,1] constraint: use soft projection
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0, 1)
        
        # Re-normalize sum
        diff = N / 2 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        diff2 = N / 2 - np.sum(h_new)
        h_new += diff2 / N
        
        new_score = compute_score(h_new)
        if new_score < score - 1e-10:
            h = h_new.copy()
            score = new_score
            best_h = h.copy()
            best_score = score
            no_improve = 0
            lr = min(lr * 1.1, 0.05)
        else:
            no_improve += 1
            lr = max(lr * 0.95, 0.001)
            if no_improve > 50:
                # Restart with perturbation
                h = best_h + 0.1 * np.random.randn(N)
                h = np.clip(h, 0, 1)
                diff = N / 2 - np.sum(h)
                h += diff / N
                h = np.clip(h, 0, 1)
                diff2 = N / 2 - np.sum(h)
                h += diff2 / N
                no_improve = 0
                lr = 0.01
    
    # Final check
    final_score = compute_score(best_h)
    if final_score < best_score:
        best_score = final_score
    
    return best_h
