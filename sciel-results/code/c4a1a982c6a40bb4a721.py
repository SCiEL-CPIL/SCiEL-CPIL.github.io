"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Minimize max_s ∫ h(x)(1-h(x+s)) dx over shifts s.
    Structural approach: model h as convex combination of fixed binary patterns,
    optimizing only mixing weights (low-dimensional search).
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    def _overlap_values(h):
        """Compute overlap values for all shifts using FFT for speed."""
        n = len(h)
        # overlap_k = sum_j h[j] * (1 - h[(j+k)%n]) / n
        # = sum_j h[j] / n - sum_j h[j] * h[(j+k)%n] / n
        # = mean(h) - autocorr(h)[k] / n
        # where autocorr is circular autocorrelation
        h_mean = np.mean(h)
        # Circular autocorrelation via FFT
        fft_h = np.fft.fft(h)
        autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real
        overlaps = h_mean - autocorr / n
        return overlaps
    
    def _score(h):
        """Compute the maximum overlap score."""
        overlaps = _overlap_values(h)
        return np.max(overlaps)
    
    def _fix_sum(h):
        """Project h onto the constraint sum(h) = N/2."""
        h = np.clip(h, 0, 1)
        current_sum = np.sum(h)
        target_sum = N / 2
        diff = target_sum - current_sum
        
        if abs(diff) < 1e-10:
            return h
        
        if diff > 0:
            candidates = np.where(h < 1.0)[0]
            if len(candidates) == 0:
                return h
            room = 1.0 - h[candidates]
            total_room = np.sum(room)
            if total_room > 0:
                h[candidates] += diff * room / total_room
        else:
            candidates = np.where(h > 0.0)[0]
            if len(candidates) == 0:
                return h
            h[candidates] += diff * h[candidates] / np.sum(h[candidates])
        
        return np.clip(h, 0, 1)
    
    # Generate diverse binary basis patterns
    def _generate_basis_patterns():
        """Generate a set of diverse binary patterns for the basis."""
        patterns = []
        
        # Pattern 1: Constant 0.5 (balanced)
        patterns.append(np.full(N, 0.5))
        
        # Pattern 2: Alternating
        alt = np.zeros(N)
        alt[1::2] = 1.0
        patterns.append(alt)
        
        # Pattern 3: Quadratic residues mod 5
        qr5 = np.zeros(N)
        for i in range(N):
            if i % 5 in [0, 1]:  # Quadratic residues mod 5
                qr5[i] = 1.0
        patterns.append(qr5)
        
        # Pattern 4: Quadratic residues mod 7
        qr7 = np.zeros(N)
        for i in range(N):
            if i % 7 in [0, 1, 2, 4]:  # Quadratic residues mod 7
                qr7[i] = 1.0
        patterns.append(qr7)
        
        # Pattern 5: Quadratic residues mod 11
        qr11 = np.zeros(N)
        for i in range(N):
            if i % 11 in [0, 1, 3, 4, 5, 9]:  # Quadratic residues mod 11
                qr11[i] = 1.0
        patterns.append(qr11)
        
        # Pattern 6: Random with fixed seed
        np.random.seed(42)
        rand_pat = np.random.uniform(0, 1, N)
        rand_pat = (rand_pat > 0.5).astype(float)
        patterns.append(rand_pat)
        
        # Pattern 7: Block pattern (period 4)
        block4 = np.zeros(N)
        for i in range(N):
            if i % 4 in [0, 1]:
                block4[i] = 1.0
        patterns.append(block4)
        
        # Pattern 8: Block pattern (period 3)
        block3 = np.zeros(N)
        for i in range(N):
            if i % 3 in [0, 1]:
                block3[i] = 1.0
        patterns.append(block3)
        
        # Pattern 9: High contrast alternating with period 2
        hc_alt = np.zeros(N)
        hc_alt[0::2] = 1.0
        patterns.append(hc_alt)
        
        # Pattern 10: Another random
        np.random.seed(123)
        rand_pat2 = np.random.uniform(0, 1, N)
        rand_pat2 = (rand_pat2 > 0.5).astype(float)
        patterns.append(rand_pat2)
        
        return patterns
    
    basis_patterns = _generate_basis_patterns()
    n_patterns = len(basis_patterns)
    
    # Precompute autocorrelations for each basis pattern
    basis_autocorrs = []
    for pat in basis_patterns:
        fft_pat = np.fft.fft(pat)
        ac = np.fft.ifft(fft_pat * np.conj(fft_pat)).real
        basis_autocorrs.append(ac)
    
    basis_means = [np.mean(pat) for pat in basis_patterns]
    
    def _construct_from_weights(weights):
        """Construct h from convex combination of basis patterns."""
        weights = np.clip(weights, 0, 1)
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            return np.full(N, 0.5)
        weights = weights / w_sum
        
        # h = sum_i weights[i] * basis_patterns[i]
        h = np.zeros(N)
        for i in range(n_patterns):
            h += weights[i] * basis_patterns[i]
        
        return h
    
    def _objective_from_weights(weights):
        """Objective: max overlap for h constructed from weights."""
        h = _construct_from_weights(weights)
        h = _fix_sum(h)
        overlaps = _overlap_values(h)
        return np.max(overlaps)
    
    # Strategy 1: Optimize mixing weights using differential evolution
    def _bounds_weights():
        return [(0, 1)] * n_patterns
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try optimizing weights
    if time.time() < deadline - 60.0:
        try:
            result = differential_evolution(
                _objective_from_weights,
                bounds=_bounds_weights(),
                maxiter=200,
                tol=1e-8,
                seed=42,
                polish=True,
                init='latinhypercube',
                popsize=15
            )
            
            if result.fun < best_score:
                h_opt = _construct_from_weights(result.x)
                h_opt = _fix_sum(h_opt)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception as e:
            pass
    
    # Strategy 2: Try pairs of basis patterns (2D optimization)
    if time.time() < deadline - 30.0:
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                if time.time() > deadline - 30.0:
                    break
                
                def _pair_objective(w):
                    """Objective for pair (i, j) with weight w on pattern i."""
                    weights = np.zeros(n_patterns)
                    weights[i] = w
                    weights[j] = 1.0 - w
                    h = _construct_from_weights(weights)
                    h = _fix_sum(h)
                    overlaps = _overlap_values(h)
                    return np.max(overlaps)
                
                # Grid search over w
                w_vals = np.linspace(0, 1, 100)
                best_w = 0.5
                best_pair_score = float('inf')
                for w in w_vals:
                    s = _pair_objective(w)
                    if s < best_pair_score:
                        best_pair_score = s
                        best_w = w
                
                if best_pair_score < best_score:
                    weights = np.zeros(n_patterns)
                    weights[i] = best_w
                    weights[j] = 1.0 - best_w
                    h_opt = _construct_from_weights(weights)
                    h_opt = _fix_sum(h_opt)
                    score = _score(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
    
    # Strategy 3: Local refinement of best solution
    if time.time() < deadline - 10.0:
        # Extract approximate weights from best_h
        # Use least squares to find weights that best approximate best_h
        A = np.column_stack(basis_patterns)
        try:
            weights_ls, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            weights_ls = np.clip(weights_ls, 0, 1)
            if np.sum(weights_ls) > 0:
                weights_ls = weights_ls / np.sum(weights_ls)
            else:
                weights_ls = np.ones(n_patterns) / n_patterns
            
            # Refine with Nelder-Mead
            def _refine_objective(w):
                h = _construct_from_weights(w)
                h = _fix_sum(h)
                overlaps = _overlap_values(h)
                return np.max(overlaps)
            
            result = minimize(
                _refine_objective,
                weights_ls,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.fun < best_score:
                h_opt = _construct_from_weights(result.x)
                h_opt = _fix_sum(h_opt)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: ensure sum is exactly N/2 and values in [0,1]
    final = _fix_sum(best_h)
    final = np.clip(final, 0, 1)
    
    # Verify sum
    if not np.isclose(np.sum(final), N/2, atol=1e-10):
        diff = N/2 - np.sum(final)
        if diff > 0:
            candidates = np.where(final < 1.0)[0]
            room = 1.0 - final[candidates]
            if np.sum(room) > 0:
                final[candidates] += diff * room / np.sum(room)
        else:
            candidates = np.where(final > 0.0)[0]
            if np.sum(final[candidates]) > 0:
                final[candidates] += diff * final[candidates] / np.sum(final[candidates])
        final = np.clip(final, 0, 1)
    
    return final
# EVOLVE-BLOCK-END
