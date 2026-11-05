import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)

    # Message Passing for Hard-Core Model (approximation of Max-IS)
    # m[i][j] is message from i to j
    # Initialize messages to 1 for edges, 0 for non-edges
    m = A.copy()
    
    # Iterate
    for _ in range(40):
        if time.time() - start_time > deadline - 5:
            break
        
        # col_sums[i] = sum of messages coming into i
        # m is (n, n). m[k][i] is message from k to i.
        # sum over k is axis=0.
        col_sums = m.sum(axis=0)
        
        # Update m[i][j] for edges (i, j)
        # m[i][j] = 1 / (1 + sum_{k in N(i) \ {j}} m[k][i])
        # sum_{k in N(i)} m[k][i] = col_sums[i]
        # So sum_{k in N(i) \ {j}} m[k][i] = col_sums[i] - m[j][i]
        
        # We need to compute this for all i, j.
        # Denominator for m[i][j] is 1 + col_sums[i] - m[j][i]
        # col_sums[i] is a vector of length n. We need it in rows.
        # m[j][i] is the transpose of m.
        
        denom = 1.0 + col_sums[None, :] - m.T
        new_m = 1.0 / denom
        
        # Apply mask: only update edges
        new_m = new_m * A
        m = new_m

    # Compute beliefs (scores)
    # b_i = 1 / (1 + sum_{j in N(i)} m[j][i])
    # sum_{j in N(i)} m[j][i] is col_sums[i] (from the last iteration)
    # Wait, col_sums was computed before the last update of m.
    # Let's recompute col_sums with the final m.
    final_col_sums = m.sum(axis=0)
    beliefs = 1.0 / (1.0 + final_col_sums)
    
    # Normalize beliefs to [0, 1] or keep as is?
    # Greedy uses descending order. Scale doesn't matter, only order.
    
    # Add tie-breakers
    degrees = A.sum(axis=1)
    deg_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-9)
    
    # Fiedler vector for structural tie-breaking
    try:
        if time.time() - start_time < deadline - 10:
            D = np.diag(degrees)
            L = D - A
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            fiedler_abs = np.abs(fiedler)
            fiedler_norm = (fiedler_abs - fiedler_abs.min()) / (fiedler_abs.max() - fiedler_abs.min() + 1e-9)
        else:
            fiedler_norm = np.zeros(n)
    except Exception:
        fiedler_norm = np.zeros(n)

    # Composite score
    # Primary: Beliefs
    # Secondary: Low degree (penalize high degree)
    # Tertiary: Fiedler (reward boundary)
    
    scores = beliefs - 0.05 * deg_norm + 0.05 * fiedler_norm
    
    return scores.astype(np.float64)
