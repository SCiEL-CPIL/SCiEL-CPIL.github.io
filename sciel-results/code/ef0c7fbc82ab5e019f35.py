import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 24
    total_candidates = 1 << 23
    
    # Two-pass strategy:
    # Pass 1: Coarse screening with M1=4096 to find top candidates
    # Pass 2: Fine refinement with M2=16384 on top candidates
    # Pass 3: Local golden-section refinement on the very best candidates
    
    M1 = 4096  # Coarse grid
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
    
    B = 10000  # Batch size for coarse pass
    
    # Precompute shifts for vectorized bit extraction
    shifts = np.arange(23, dtype=np.int64)

    # Pass 1: Coarse screening
    top_candidates = []
    top_scores = []
    
    for start in range(0, total_candidates, B):
        if time.time() > deadline - 30.0:
            break
            
        end = min(start + B, total_candidates)
        B_actual = end - start
        
        idxs = np.arange(start, end, dtype=np.int64)
        
        # Vectorized bit extraction
        bits = ((idxs[:, None] >> shifts) & 1).astype(np.float64)
        
        try:
            contrib = bits @ Z_flip1
            P_vals = AllOnes1[None, :] - 2.0 * contrib
            scores = np.max(np.abs(P_vals), axis=1)
            
            # Keep top 100 from this batch
            local_top = np.argpartition(scores, 100)[:100]
            for li in local_top:
                top_candidates.append(start + int(li))
                top_scores.append(scores[li])
        except Exception:
            pass
    
    # Sort top candidates by score and keep top 500
    if top_candidates:
        top_candidates = np.array(top_candidates)
        top_scores = np.array(top_scores)
        sort_idx = np.argsort(top_scores)
        top_candidates = top_candidates[sort_idx][:500]
        top_scores = top_scores[sort_idx][:500]
        
        # Pass 2: Fine refinement on top candidates
        best_score = np.inf
        best_idx = 0
        refined_candidates = []
        refined_scores = []
        
        for idx in top_candidates:
            if time.time() > deadline - 20.0:
                break
            
            bits = ((np.array([idx], dtype=np.int64)[:, None] >> shifts) & 1).astype(np.float64)
            
            contrib = bits @ Z_flip2
            P_vals = AllOnes2[None, :] - 2.0 * contrib
            score = np.max(np.abs(P_vals))
            
            refined_candidates.append(int(idx))
            refined_scores.append(score)
            
            if score < best_score:
                best_score = score
                best_idx = int(idx)
        
        # Sort refined candidates
        if refined_candidates:
            refined_candidates = np.array(refined_candidates)
            refined_scores = np.array(refined_scores)
            sort_idx = np.argsort(refined_scores)
            top_refined_indices = refined_candidates[sort_idx][:20]
            
            # Pass 3: Local golden-section refinement on top 20
            # We need to evaluate |P(z)| at arbitrary z
            # P(z) = sum c_k z^k
            # c_k = 1 - 2*b_k where b_k is the bit
            
            def eval_p_abs(z, bits_vec):
                # bits_vec is shape (23,) for c_1..c_23
                # c_0 is always 1
                # P(z) = 1 + sum_{k=1}^{23} c_k z^k
                # c_k = 1 - 2*b_k
                # P(z) = 1 + sum (1-2b_k) z^k = 1 + sum z^k - 2 sum b_k z^k
                # We can precompute sum z^k? No, z changes.
                # Just compute directly.
                z_powers = np.zeros(N, dtype=complex)
                z_powers[0] = 1.0
                for k in range(1, N):
                    z_powers[k] = z_powers[k-1] * z
                coeffs = np.ones(N)
                coeffs[1:] = 1.0 - 2.0 * bits_vec
                return np.abs(np.sum(coeffs * z_powers))
            
            def golden_section_max(func, a, b, tol=1e-9, max_iter=100):
                # Maximize func on [a, b]
                # Golden ratio
                gr = (np.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = func(c)
                fd = func(d)
                for _ in range(max_iter):
                    if b - a < tol:
                        break
                    if fc < fd:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = func(d)
                    else:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = func(c)
                return (a + b) / 2, max(func(a), func(b), func((a+b)/2))
            
            best_final_score = np.inf
            best_final_idx = best_idx
            
            for idx in top_refined_indices:
                if time.time() > deadline - 5.0:
                    break
                
                # Get bits for this idx
                bits_vec = ((np.array([idx], dtype=np.int64) >> shifts) & 1).flatten().astype(np.float64)
                
                # Find peak on fine grid
                bits_mat = bits_vec.reshape(1, 23)
                contrib = bits_mat @ Z_flip2
                P_vals = AllOnes2[None, :] - 2.0 * contrib
                abs_vals = np.abs(P_vals)
                i_max = np.argmax(abs_vals)
                
                # Interval for refinement
                # Angles are in [0, 2pi)
                # i_max corresponds to angles2[i_max]
                # We look in [angles2[i_max-1], angles2[i_max+1]]
                # Handle wrap-around?
                # angles2 is sorted.
                if i_max == 0:
                    a = angles2[-1] - 2*np.pi # effectively -delta
                    b = angles2[1]
                    # Better to work in local coordinates or just use the neighbors
                    # Since it's a circle, let's just use the neighbors in the array
                    # and handle the wrap-around by checking if i_max is 0 or M-1
                    if i_max == 0:
                        left_idx = M2 - 1
                        right_idx = 1
                    else:
                        left_idx = i_max - 1
                        right_idx = i_max + 1
                        if i_max == M2 - 1:
                            right_idx = 0
                            
                    # To simplify, let's just define a function that takes angle
                    # and use the angles from the grid
                    ang_left = angles2[left_idx]
                    ang_right = angles2[right_idx]
                    
                    # Ensure ang_left < ang_right for the search interval?
                    # If wrap-around, ang_left > ang_right.
                    # We can shift angles to make it continuous.
                    # Or just search in [ang_left, ang_right] if no wrap, else handle carefully.
                    # Simplest: search in the local neighborhood.
                    # Let's define the search range as [ang_left, ang_right]
                    # If wrap, ang_left is large, ang_right is small.
                    # We can search in [ang_left - 2*pi, ang_right] or similar.
                    # Let's just use the two neighbors and the center.
                    # Actually, we can just do a local ternary search on the function
                    # defined by angle.
                    
                    def func_angle(theta):
                        z = np.exp(1j * theta)
                        return eval_p_abs(z, bits_vec)
                    
                    # If no wrap
                    if left_idx < right_idx:
                        a, b = angles2[left_idx], angles2[right_idx]
                    else:
                        # Wrap around
                        # Search in [ang_left, ang_left + 2*pi] mapped to [ang_left, ang_right + 2*pi]
                        a = angles2[left_idx]
                        b = angles2[right_idx] + 2 * np.pi
                    
                    # Perform golden section search
                    z_opt, val_opt = golden_section_max(func_angle, a, b, tol=1e-10, max_iter=50)
                    
                    if val_opt < best_final_score:
                        best_final_score = val_opt
                        best_final_idx = int(idx)
            
            best_idx = best_final_idx
    
    # Reconstruct best coefficients
    best_coeffs = np.ones(N)
    for k in range(23):
        if (best_idx >> k) & 1:
            best_coeffs[k + 1] = -1.0
    
    return [int(c) for c in best_coeffs]
