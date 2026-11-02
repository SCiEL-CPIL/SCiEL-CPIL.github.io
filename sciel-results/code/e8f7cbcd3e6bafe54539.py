import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h, exclude_lag0=True):
    S_h = np.sum(h)
    if S_h < 1e-6:
        return np.inf
    S = np.correlate(h, h, mode='full')
    if exclude_lag0:
        max_side = np.max(np.concatenate([S[:N-1], S[N:2*N-1]]))
    else:
        max_side = np.max(S)
    return max_side / (S_h ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # 1. Evaluate Uniform
    s = compute_score(best_h, exclude_lag0=True)
    if s < best_score:
        best_score = s
        best_h = best_h.copy()
        
    # 2. Two-Block Optimization (Discrete Grid + Local Refinement)
    # We model h as two blocks: h1 (width W1, amp A1, center C1) and h2 (width W2, amp A2, center C2)
    # Symmetric around 0 implies C1 = -C2 and W1=W2, A1=A2 is one case, but generally:
    # Due to symmetry of the objective, we can assume the optimal solution is symmetric: h(x) = h(-x).
    # This reduces the degrees of freedom significantly.
    # h(x) is defined on [-0.25, 0.25].
    # Let's search for symmetric two-block structures.
    # Block 1: [center_D - W/2, center_D + W/2]
    # Block 2: [-center_D - W/2, -center_D + W/2]
    # In indices: 
    # Center index is N/2 (0.25 maps to N-1, -0.25 maps to 0). Actually 0 is at N/2.
    # Let's use integer indices for centers and widths.
    
    Ws = np.arange(10, N//2 - 5, 5)
    # D is distance from center (N/2) in indices.
    # Block must not cross center 0? No, it can, but they must be disjoint for the "two block" simple model.
    # If they overlap, it's effectively one block or a more complex shape.
    # Let's enforce disjointness: 2*D > W.
    D_min = int(N/2) // 10 # Small offset
    D_max = int(N/2) - 10
    
    # Amplitudes: Since objective is homogeneous, we can fix A1=1 and vary A2.
    # But symmetry implies A1=A2? Not necessarily.
    # However, if we assume h(x) = h(-x), then the block at D has height A, and at -D has height A.
    # So A1=A2. 
    # What about asymmetric amplitudes? If A1 != A2, the function is not symmetric.
    # Is the optimal solution symmetric? Likely yes due to the domain symmetry and objective symmetry.
    # So let's assume A1 = A2 = A.
    # Then h is determined by W, D, A.
    # But A can be scaled out? 
    # Score = max_k conv_k / (sum h)^2.
    # If we scale h by A, conv scales by A^2, sum scales by A. So score is invariant to A.
    # So for a fixed geometry (W, D), the score is constant regardless of A (as long as A>0).
    # Wait, this is only true if the *shape* of the autoconvolution doesn't change relative peaks.
    # Convolution is quadratic. (A*h_geo) * (A*h_geo) = A^2 (h_geo*h_geo).
    # (Sum A*h_geo)^2 = A^2 (Sum h_geo)^2.
    # So yes, for a fixed set of non-zero indices and their relative values (here 1 vs 1), the score is independent of the absolute scale A.
    # So we can just set A=1.
    
    # Let's iterate W and D.
    found_symmetric_improvement = False
    for W in Ws:
        if time.time() > deadline - 10:
            break
        
        # D must be large enough for disjointness: center1 - W/2 > center2 + W/2?
        # Center1 = N/2 + D, Center2 = N/2 - D.
        # Left edge of right block: N/2 + D - W/2
        # Right edge of left block: N/2 - D + W/2
        # Disjoint if: N/2 + D - W/2 > N/2 - D + W/2  =>  2D > W.
        D_start = int(W/2) + 1
        D_end = int(N/2) - int(W/2) - 1
        
        if D_start >= D_end:
            continue
            
        # Sample D
        # Number of samples depends on remaining time? Let's do a few.
        num_D = 5
        Ds = np.linspace(D_start, D_end, num_D).astype(int)
        
        for D in Ds:
            if time.time() > deadline - 5:
                break
                
            h_test = np.zeros(N)
            c1 = int(N/2 + D)
            c2 = int(N/2 - D)
            
            s1 = c1 - W//2
            e1 = c1 + W//2 + 1
            s2 = c2 - W//2
            e2 = c2 + W//2 + 1
            
            if s1 < 0 or e1 > N or s2 < 0 or e2 > N:
                continue
                
            h_test[s1:e1] = 1.0
            h_test[s2:e2] = 1.0
            
            s = compute_score(h_test, exclude_lag0=True)
            if s < best_score:
                best_score = s
                best_h = h_test.copy()
                found_symmetric_improvement = True
                
    # 3. Local Refinement of Best H
    # If we found a symmetric two-block structure, refine W, D.
    # Also try asymmetric amplitudes? 
    # Let's try refining the discrete indices of the best_h.
    
    if best_score < np.inf:
        # Identify blocks in best_h
        non_zero_idx = np.where(best_h > 1e-6)[0]
        if len(non_zero_idx) > 0:
            diffs = np.diff(non_zero_idx)
            gaps = np.where(diffs > 1)[0]
            
            if len(gaps) == 1:
                split_idx = gaps[0]
                b1_idx = non_zero_idx[:split_idx+1]
                b2_idx = non_zero_idx[split_idx+1:]
                
                b1_s, b1_e = b1_idx[0], b1_idx[-1]
                b2_s, b2_e = b2_idx[0], b2_idx[-1]
                
                W1 = b1_e - b1_s + 1
                W2 = b2_e - b2_s + 1
                A1 = best_h[b1_s]
                A2 = best_h[b2_s]
                
                # Refine positions and widths
                for shift1 in [-2, -1, 0, 1, 2]:
                    for shift2 in [-2, -1, 0, 1, 2]:
                        for dW1 in [-2, -1, 0, 1, 2]:
                            for dW2 in [-2, -1, 0, 1, 2]:
                                if time.time() > deadline - 2:
                                    break
                                
                                nw1 = W1 + dW1
                                nw2 = W2 + dW2
                                if nw1 < 1 or nw2 < 1:
                                    continue
                                    
                                ns1 = b1_s + shift1
                                ne1 = ns1 + nw1 - 1
                                ns2 = b2_s + shift2
                                ne2 = ns2 + nw2 - 1
                                
                                if ns1 < 0 or ne1 >= N or ns2 < 0 or ne2 >= N:
                                    continue
                                    
                                # Check overlap (should be disjoint)
                                if ns1 > ne2 or ns2 > ne1:
                                    h_test = np.zeros(N)
                                    h_test[ns1:ne1+1] = A1
                                    h_test[ns2:ne2+1] = A2
                                    s = compute_score(h_test, exclude_lag0=True)
                                    if s < best_score:
                                        best_score = s
                                        best_h = h_test.copy()
                                        
    # 4. Try a different structure: Three blocks? Or a plateau?
    # Let's try a "plateau" in the middle with zero on the sides.
    # h = 1 on [-W/2, W/2], 0 elsewhere.
    # This is a single block centered at 0.
    # Autoconvolution is a triangle. Max is at 0 (lag 0).
    # But we exclude lag 0.
    # For a single block of width W, the convolution at lag k is (W - |k|) for |k| < W.
    # Max at k=1 is W-1.
    # Sum h = W.
    # Score = (W-1) / W^2 = 1/W - 1/W^2.
    # To minimize, we maximize W.
    # Max W is N (whole interval).
    # Score = (N-1)/N^2 = 1/N - 1/N^2.
    # Uniform score: 
    # h_i = 1/N. Sum = 1.
    # Conv at lag k: sum_{i} h_{i-k} h_i.
    # For uniform, conv_k = (N-|k|)/N^2.
    # Max at k=1 is (N-1)/N^2.
    # So uniform is a single block of width N.
    # Score is 1/N - 1/N^2.
    
    # Can we do better?
    # If we make h zero in the middle?
    # h = 1 on [-N/4, -N/8] and [N/8, N/4].
    # Two blocks, separated by a gap in the middle.
    # This is what we searched for.
    
    # What if we make h zero at the edges?
    # h = 1 on [-N/8, N/8].
    # Single block, width N/2.
    # Score = (N/2 - 1) / (N/2)^2 = (N/2)/ (N^2/4) = 2/N.
    # 2/N is worse than 1/N.
    
    # So spreading out is good.
    # Uniform is the best single block.
    
    # What about two blocks of width N/2, adjacent?
    # That's just one block of width N.
    
    # What about two blocks of width N/2, separated?
    # W = N/2 - 2 (gap of 2).
    # D = N/4 + 1.
    # We searched this.
    
    # Let's try to optimize the gap size and block width more finely.
    # We did a grid search.
    
    # Let's try a "staircase" or "linear ramp"?
    # h_i = i? 
    # This breaks symmetry.
    
    # Let's try to use scipy to optimize the block parameters (W, D) continuously?
    # We can define a function that takes (W, D) and evaluates the score.
    # But the score is non-smooth (max operation).
    
    # Let's stick with the grid search + local refinement.
    # To improve, we can densify the grid for the best W, D region.
    
    # Find best W, D from the grid search? 
    # We didn't store them.
    
    # Let's just do a final local search on the indices of the non-zero values.
    # We did that in step 3.
    
    # One more idea: 
    # The optimal function might not be a step function with constant height 1.
    # But the task says "nonnegative step function".
    # So heights are constant on intervals.
    # We assumed height 1. 
    # Since the score is scale invariant, the absolute height doesn't matter, only the pattern of zeros and non-zeros.
    # So our search over supports (sets of indices) is complete for step functions with 2 levels (0 and A).
    
    # What if the step function has more than 2 levels?
    # E.g., 3 blocks with different heights?
    # Or a block with height 1 and a block with height 0.5?
    # Let's try adding amplitude optimization to the local search.
    
    if best_score < np.inf and not np.allclose(best_h, np.ones(N)/N):
        # Try varying amplitudes of the two blocks
        non_zero_idx = np.where(best_h > 1e-6)[0]
        if len(non_zero_idx) > 0:
            diffs = np.diff(non_zero_idx)
            gaps = np.where(diffs > 1)[0]
            if len(gaps) == 1:
                split_idx = gaps[0]
                b1_idx = non_zero_idx[:split_idx+1]
                b2_idx = non_zero_idx[split_idx+1:]
                
                # Try varying A1, A2
                for A1 in [0.5, 0.8, 1.0, 1.2, 1.5]:
                    for A2 in [0.5, 0.8, 1.0, 1.2, 1.5]:
                        if time.time() > deadline - 1:
                            break
                        h_test = np.zeros(N)
                        h_test[b1_idx] = A1
                        h_test[b2_idx] = A2
                        s = compute_score(h_test, exclude_lag0=True)
                        if s < best_score:
                            best_score = s
                            best_h = h_test.copy()
                            
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_score(np.array(h), exclude_lag0=True))
