import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # 1. Generate Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    rs_24 = p[:24]

    # 2. Precompute evaluation grid
    M = 1 << 16  # 65536 points for dense sampling
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    k = np.arange(24)
    # Precompute the Vandermonde-like matrix Z[k, j] = exp(i * k * theta[j])
    # This is 24 x 65536, which is manageable.
    Z = np.exp(1j * np.outer(k, theta))
    
    # For fast evaluation, we can use matrix multiplication: P = c @ Z
    # But c is 1D (24,), Z is 24x65536. So P = c @ Z gives 1x65536.
    # Actually, Z is defined as exp(i * k * theta), so Z[k, j] = exp(i k theta_j).
    # P(theta_j) = sum_k c_k exp(i k theta_j) = (c @ Z)[j]

    def max_norm_precise(c):
        """
        Compute sup norm with dense sampling and golden-section refinement of top peaks.
        Returns the max |P(z)|.
        """
        c_arr = np.array(c)
        P = c_arr @ Z
        abs_P = np.abs(P)
        
        # Find top 5 peaks for refinement
        top_indices = np.argsort(abs_P)[-5:][::-1]
        
        best_val = float(np.max(abs_P))
        dtheta = 2 * np.pi / M
        
        def f_single(t):
            # Evaluate P at a single point t
            # P = sum c_k exp(i k t)
            # Using Horner's method for efficiency in scalar evaluation?
            # Or just dot product.
            # c is list, t is scalar.
            # np.exp(1j * k * t) is vector of 24.
            # np.dot(c_arr, np.exp(1j * k * t))
            return np.abs(np.dot(c_arr, np.exp(1j * k * t)))
        
        for idx in top_indices:
            center = theta[idx]
            a = center - dtheta
            b = center + dtheta
            
            # Handle wrap-around
            if a < 0:
                a += 2 * np.pi
            if b > 2 * np.pi:
                b -= 2 * np.pi
            
            # If interval wraps around 0/2pi, split it?
            # For simplicity, if wrap occurs, we might miss the peak if it's exactly at the boundary.
            # But M is large, so peaks are unlikely to be exactly at 0 unless the polynomial has high symmetry.
            # Let's assume no wrap for local refinement, or handle by checking both sides if needed.
            # Given the small window (2pi/65536), wrap is only an issue if center is very close to 0 or 2pi.
            # If a < 0 or b > 2pi, we can just clamp or handle carefully.
            # Actually, if a < 0, the peak might be in [0, b] or [a+2pi, 2pi].
            # Let's just do a simple bounded minimization using scipy if wrap is detected, 
            # or rely on the dense grid being fine enough.
            # For robustness, let's use scipy.optimize.minimize_scalar for the refinement.
            
            def obj(t):
                return -f_single(t)
            
            # If wrap-around is possible, we might need to search in two intervals.
            # But let's try the standard bounded search first.
            # If a < 0 or b > 2pi, we can shift the center or handle it.
            # Let's just clamp for now, as M is large.
            a_c = max(0, a)
            b_c = min(2 * np.pi, b)
            
            if a_c < b_c:
                res = minimize_scalar(obj, bounds=(a_c, b_c), method='bounded', options={'xatol': 1e-11})
                if res.success:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        
        return best_val

    def score(c):
        return max_norm_precise(c) / np.sqrt(24)

    def score_fast(c):
        """Fast score using dense grid only, no refinement."""
        c_arr = np.array(c)
        P = c_arr @ Z
        return float(np.max(np.abs(P))) / np.sqrt(24)

    # 3. Generate candidates
    candidates = []
    
    # Rudin-Shapiro and variants
    candidates.append(rs_24)
    candidates.append(list(reversed(rs_24)))
    candidates.append([-c for c in rs_24])
    candidates.append([-c for c in reversed(rs_24)])
    
    # Rotations and reversals of GRS
    n = 24
    for shift in range(n):
        rotated = rs_24[shift:] + rs_24[:shift]
        candidates.append(rotated)
        candidates.append(list(reversed(rotated)))
        candidates.append([-c for c in rotated])
        candidates.append([-c for c in reversed(rotated)])
        
    # Random candidates
    random.seed(42)
    for _ in range(100):
        candidates.append([random.choice([-1, 1]) for _ in range(n)])
        
    # Periodic patterns
    for period in range(2, 13):
        base = [random.choice([-1, 1]) for _ in range(period)]
        seq = [base[i % period] for i in range(n)]
        candidates.append(seq)
        
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 4. Screening phase
    best_c = None
    best_score = float('inf')
    
    # Quick screen with fast score
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > deadline - 60:
            break
        s = score_fast(c)
        scored_candidates.append((s, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 20 for local search
    top_candidates = [c for s, c in scored_candidates[:20]]
    
    # 5. Local search
    def local_search(c, time_limit=5.0):
        current = list(c)
        current_score = score_fast(current)
        improved = True
        start = time.time()
        
        while improved:
            improved = False
            if time.time() - start > time_limit:
                break
            
            # Single flips
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                old = current[i]
                current[i] = -old
                new_score = score_fast(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
            
            # Double flips
            if time.time() - start > time_limit:
                break
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = score_fast(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
                        
        return current, current_score

    best_c = top_candidates[0]
    best_score = score_fast(best_c)
    
    for c in top_candidates:
        if time.time() > deadline - 30:
            break
        res_c, res_score = local_search(c, time_limit=2.0)
        if res_score < best_score:
            best_score = res_score
            best_c = res_c

    # 6. Random restarts
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            c = [random.choice([-1, 1]) for _ in range(n)]
            res_c, res_score = local_search(c, time_limit=1.0)
            if res_score < best_score:
                best_score = res_score
                best_c = res_c

    # 7. Perturbation from best
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbed = list(best_c)
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, n - 1)
                perturbed[idx] = -perturbed[idx]
            res_c, res_score = local_search(perturbed, time_limit=1.0)
            if res_score < best_score:
                best_score = res_score
                best_c = res_c

    # 8. Final refinement with precise scoring
    if time.time() < deadline - 5:
        # Do a few steps of local search using the precise score to polish
        current = list(best_c)
        current_score = score(current)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old = current[i]
                current[i] = -old
                new_score = score(current)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
                    
        if current_score < best_score:
            best_c = current
            best_score = current_score

    return best_c
