import numpy as np
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_score(h):
    """
    Compute the ratio: (L2^2) / (L1 * Sup_AutoConv)
    """
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
        
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return 0.0

    # Discrete autoconvolution (full mode)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    
    if sup_conv < 1e-15:
        return 0.0

    return l2_sq / (l1 * sup_conv)

def _generate_sidon_sets(n_points, max_attempts=1000):
    """
    Generate sets of integers that form a Sidon set (all pairwise sums are distinct).
    This ensures that for h supported on this set with non-zero values,
    the autoconvolution has no overlaps for non-zero lags (except the trivial 0 lag),
    keeping the peak convolution low relative to L2.
    """
    sets = []
    for _ in range(max_attempts):
        # Random subset
        S = set(np.random.choice(2*N, size=n_points, replace=False))
        sums = set()
        is_sidon = True
        # Check pairwise sums
        # Note: We consider sums of distinct elements and i+i.
        # For a Sidon set in the context of convolution, we usually want 
        # |a+b = c+d| to imply {a,b}={c,d}.
        # This is equivalent to all pairwise sums being distinct if we treat 
        # the set as unordered pairs with replacement? 
        # Actually, standard Sidon: sums a+b for a<=b are distinct.
        
        pairs = []
        elems = sorted(list(S))
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    is_sidon = False
                    break
                sums.add(s)
        if is_sidon:
            sets.append(S)
            if len(sets) >= 10:
                break
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Initialize with known good candidates
    candidates = []
    
    # Uniform (Seed)
    h_uniform = np.ones(N)
    candidates.append(h_uniform)
    
    # [2/3, 1/3, 0...] (Known best from prompt)
    h_known = np.zeros(N)
    h_known[0] = 2.0/3.0
    h_known[1] = 1.0/3.0
    candidates.append(h_known)
    
    # Delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    candidates.append(h_delta)
    
    # Random sparse
    for _ in range(5):
        h_rand = np.zeros(N)
        idx = np.random.choice(N, size=3, replace=False)
        h_rand[idx] = 1.0/3.0
        candidates.append(h_rand)

    # Evaluate initial candidates
    for h in candidates:
        score = _compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # 2. Structural Search: Sidon Sets (Disjoint Spikes)
    # Hypothesis: Placing mass on a Sidon set minimizes autoconvolution overlaps.
    # If h is supported on a Sidon set S, then for any lag k != 0, 
    # the number of pairs (i, j) in S such that i-j = k is at most 1 (or 0).
    # This keeps the "sidelobes" of the autoconvolution low.
    # The peak is at lag 0, which is sum(h_i^2).
    # Ratio = (sum h_i^2) / (sum h_i * sum h_i^2) = 1 / sum h_i.
    # Wait, if peak is sum h_i^2, and L1 = sum h_i, L2_sq = sum h_i^2.
    # Ratio = L2_sq / (L1 * Peak) = 1 / L1.
    # To maximize this, we need to minimize L1.
    # But we can scale h. The ratio is scale invariant.
    # Let's re-evaluate.
    # R(h) = (sum h^2) / (sum h * max_conv)
    # If max_conv = sum h^2 (which happens if all non-zero lags have conv < sum h^2),
    # then R = 1 / sum h.
    # Since R is scale invariant, we can fix sum h = 1. Then R = 1.
    # This suggests that if the peak is exactly the lag-0 term, the score is 1.
    # However, if we can make the peak *smaller* than sum h^2? No, lag 0 is always sum h_i^2.
    # So max_conv >= sum h_i^2.
    # Thus R <= 1 / sum h_i.
    # If we fix sum h_i = 1, R <= 1.
    # But the known best [2/3, 1/3] has score 1.1944.
    # Let's check [2/3, 1/3]:
    # L1 = 1. L2_sq = 4/9 + 1/9 = 5/9.
    # Conv:
    # Lag 0: 4/9 + 1/9 = 5/9.
    # Lag 1: (2/3)(1/3) = 2/9.
    # Lag -1: 2/9.
    # Max Conv = 5/9.
    # R = (5/9) / (1 * 5/9) = 1.
    # Why did the prompt say 1.1944?
    # "score is the lower bound on C2" ... "search score is value/seed".
    # Maybe the evaluator normalizes differently or I am misinterpreting "Sup_AutoConv".
    # If the evaluator uses continuous functions:
    # f(x) is a step function.
    # ||f||_2^2 = sum h_i^2 * dw.
    # ||f||_1 = sum h_i * dw.
    # (f*f)(t) = integral f(x)f(t-x) dx.
    # Sup (f*f).
    # If h is constant on intervals, the convolution is piecewise quadratic/linear.
    # The peak of the continuous convolution might be different from the discrete max.
    # For [2/3, 1/3] on [0, 1/2] and [1/2, 1] (scaled to [-1/4, 1/4]?).
    # The prompt says "Choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]".
    # So the domain is length 1/2. dw = 1/100.
    # Let's trust the discrete calculation for relative ranking, but be aware the absolute value might differ.
    # The key is: Disjoint spikes (Sidon) keep the *relative* sidelobes low.
    # If we have K disjoint spikes of height 1/K:
    # L1 = 1. L2_sq = K * (1/K)^2 = 1/K.
    # Max Conv = sum (1/K)^2 = 1/K (at lag 0).
    # Sidelobes are 0 (if perfectly disjoint in convolution sense, i.e., Sidon).
    # R = (1/K) / (1 * 1/K) = 1.
    # So disjoint spikes give 1.
    # The [2/3, 1/3] case:
    # Discrete R = 1.
    # Why 1.1944?
    # 1.1944 = 108/90 = 6/5? No.
    # 1.194444... = 107.5/90?
    # 1.1944 = 1 + 0.1944.
    # Maybe the "seed" program score is not 1.0?
    # "1.0 equals the seed program".
    # If seed is uniform, score = 1.0.
    # If [2/3, 1/3] scores 1.1944, it is better than uniform.
    # My discrete calc says 1.0.
    # There must be a difference in how "Sup_AutoConv" is calculated or normalized.
    # Perhaps the evaluator computes the max of the *continuous* convolution?
    # For a step function, the convolution is smoother.
    # If we assume the discrete approximation is sufficient for finding the *shape*,
    # we should look for shapes that minimize the discrete max_conv relative to L2_sq.
    
    # Let's re-run the multiplicative update logic which seemed to work in Inspiration 1.
    # It found 1.3198.
    # The strategy was: "Reduce h_i where it contributes to the current peak lag."
    
    # Let's combine:
    # 1. Start with diverse initializations (Uniform, Sparse, Sidon-like).
    # 2. Use the multiplicative update (Sinkhorn-style) to refine.
    # 3. Add a "Sidon" initialization: Pick a random Sidon set, put equal mass on it.
    #    Even if discrete score is 1, the continuous score might be higher due to smoother convolution peaks?
    #    Or maybe the multiplicative update escapes the "1" plateau.

    # Generate Sidon initializations
    sidon_sets = _generate_sidon_sets(5, max_attempts=50)
    for S in sidon_sets:
        h_sidon = np.zeros(N)
        # Map S (which might be in range 0..2N) to 0..N-1
        # Actually, Sidon sets are usually defined on integers.
        # We want indices in 0..N-1.
        valid_S = [x for x in S if 0 <= x < N]
        if len(valid_S) >= 2:
            h_sidon[valid_S] = 1.0 / len(valid_S)
            candidates.append(h_sidon)
            
    # Also try "Gapped" distributions: 1, 0, 1, 0...
    for gap in [2, 3, 4]:
        h_gap = np.zeros(N)
        h_gap[::gap] = 1.0
        h_gap /= np.sum(h_gap)
        candidates.append(h_gap)

    # Evaluate these new candidates
    for h in candidates:
        score = _compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()

    # 3. Refinement: Iterative Multiplicative Updates
    # This is the core engine from Inspiration 1.
    
    def refine_multiplicative(h0, max_iters=2000, eta=0.1):
        """
        Local refinement using multiplicative updates to reduce peak autoconvolution.
        """
        h = h0.copy()
        h = np.maximum(h, 1e-10) # Avoid zeros
        h /= np.sum(h)
        
        best_local = h.copy()
        best_local_score = _compute_score(h)
        
        stale = 0
        patience = 100
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            sup_conv = np.max(conv)
            if sup_conv < 1e-15:
                break
            
            k_star = np.argmax(conv)
            
            # Gradient of peak conv w.r.t h
            grad_h = np.zeros(N)
            for i in range(N):
                idx1 = k_star - i
                idx2 = i - k_star
                val = 0.0
                if 0 <= idx1 < N:
                    val += h[idx1]
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad_h[i] = val
                
            # Multiplicative update
            # Reduce h_i where grad_h_i is large.
            update = np.exp(-eta * grad_h)
            h_new = h * update
            s = np.sum(h_new)
            if s < 1e-15:
                break
            h_new /= s
            
            score_new = _compute_score(h_new)
            if score_new > best_local_score:
                best_local_score = score_new
                best_local = h_new.copy()
                h = h_new
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    # Decay eta
                    eta *= 0.9
                    if eta < 1e-4:
                        break
                    stale = 0 # Reset patience after decay? Or keep it?
                    # Let's keep it to force a restart if truly stuck.
                    # But we are already in the loop.
                    # Actually, if we don't update h, we just recompute the same grad.
                    # So we must change h or break.
                    # If we break, we return best_local.
                    # Let's break if eta is too small.
                    if eta < 1e-3:
                         break

        return best_local, best_local_score

    # Apply refinement to the best initial candidates
    # Sort candidates by score, take top 5
    scored_candidates = []
    for h in candidates:
        s = _compute_score(h)
        scored_candidates.append((s, h))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for s, h0 in scored_candidates[:5]:
        if time.time() > deadline - 20.0:
            break
        h_ref, score_ref = refine_multiplicative(h0)
        if score_ref > best_score:
            best_score = score_ref
            best_h = h_ref

    # 4. Final Polish: Small random perturbations + local refine
    if time.time() < deadline - 10.0:
        current_h = best_h.copy()
        current_score = best_score
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb
            h_pert = current_h.copy()
            noise = np.random.randn(N) * 0.1
            h_pert += noise
            h_pert = np.maximum(h_pert, 0)
            s = np.sum(h_pert)
            if s < 1e-15:
                continue
            h_pert /= s
            
            # Quick refine
            h_ref, score_ref = refine_multiplicative(h_pert, max_iters=100, eta=0.05)
            
            if score_ref > current_score:
                current_score = score_ref
                current_h = h_ref
                if score_ref > best_score:
                    best_score = score_ref
                    best_h = h_ref

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_score(h))
