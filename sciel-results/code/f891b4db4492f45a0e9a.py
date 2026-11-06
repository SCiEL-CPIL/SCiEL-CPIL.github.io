import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 24
    total_candidates = 1 << 23
    
    # Two-pass strategy:
    # Pass 1: Coarse screening with M1=2048 to find top candidates
    # Pass 2: Fine refinement with M2=16384 on top candidates
    
    M1 = 2048  # Coarse grid
    M2 = 16384  # Fine grid for refinement
    
    # Precompute for coarse grid
    angles1 = np.linspace(0, 2 * np.pi, M1, endpoint=False)
    z1 = np.exp(1j * angles1)
    Z1 = np.zeros((N, M1), dtype=complex)
    Z1[0, :] = 1.0
    for k in range(1, N):
        Z1[k, :] = Z1[k-1, :] * z1
    
    base1 = np.sum(Z1[1:N], axis=0)
    Z_flip1 = Z1[1:N]
    AllOnes1 = 1.0 + base1
    
    # Precompute for fine grid
    angles2 = np.linspace(0, 2 * np.pi, M2, endpoint=False)
    z2 = np.exp(1j * angles2)
    Z2 = np.zeros((N, M2), dtype=complex)
    Z2[0, :] = 1.0
    for k in range(1, N):
        Z2[k, :] = Z2[k-1, :] * z2
    
    base2 = np.sum(Z2[1:N], axis=0)
    Z_flip2 = Z2[1:N]
    AllOnes2 = 1.0 + base2
    
    B = 8000  # Batch size for coarse pass
    
    # Pass 1: Coarse screening
    top_candidates = []
    top_scores = []
    
    for start in range(0, total_candidates, B):
        if time.time() > deadline - 40.0:
            break
            
        end = min(start + B, total_candidates)
        B_actual = end - start
        
        idxs = np.arange(start, end, dtype=np.int64)
        
        bits = np.zeros((B_actual, 23), dtype=np.float64)
        for k in range(23):
            bits[:, k] = (idxs >> k) & 1
        
        try:
            contrib = bits @ Z_flip1
            P_vals = AllOnes1[None, :] - 2.0 * contrib
            scores = np.max(np.abs(P_vals), axis=1)
            
            # Keep top 50 from this batch
            local_top = np.argpartition(scores, 50)[:50]
            for li in local_top:
                top_candidates.append(start + int(li))
                top_scores.append(scores[li])
        except Exception:
            pass
    
    # Sort top candidates by score
    if top_candidates:
        sorted_indices = np.argsort(top_scores)
        top_candidates = [top_candidates[i] for i in sorted_indices]
        top_scores = [top_scores[i] for i in sorted_indices]
        
        # Keep only the best 100 for fine refinement
        num_refine = min(100, len(top_candidates))
        refine_candidates = top_candidates[:num_refine]
    else:
        # Fallback if no candidates found
        refine_candidates = [0]
    
    # Pass 2: Fine refinement on top candidates
    best_score = np.inf
    best_idx = refine_candidates[0] if refine_candidates else 0
    
    for idx in refine_candidates:
        if time.time() > deadline - 10.0:
            break
        
        # Extract bits for this candidate
        bits = np.zeros(23, dtype=np.float64)
        for k in range(23):
            bits[k] = (idx >> k) & 1
        
        # Compute P(z) on fine grid
        contrib = bits @ Z_flip2
        P_vals = AllOnes2 - 2.0 * contrib
        
        # Find max |P|
        score = np.max(np.abs(P_vals))
        
        if score < best_score:
            best_score = score
            best_idx = idx
    
    # Reconstruct best coefficients
    best_coeffs = np.ones(N)
    for k in range(23):
        if (best_idx >> k) & 1:
            best_coeffs[k + 1] = -1.0
    
    return [int(c) for c in best_coeffs]
