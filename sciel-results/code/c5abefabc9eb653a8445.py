# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580.0  # Safety margin
    
    width = 1.0 / N
    shear = np.arange(N, dtype=np.float64) / N
    
    def compute_1d_union_len(x, c):
        """Compute 1D union length of intervals at height c."""
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        Ls = L[order]
        Rs = R[order]
        total = 0.0
        cur_start = Ls[0]
        cur_end = Rs[0]
        for k in range(1, N):
            if Ls[k] > cur_end:
                total += cur_end - cur_start
                cur_start = Ls[k]
                cur_end = Rs[k]
            else:
                cur_end = max(cur_end, Rs[k])
        total += cur_end - cur_start
        return total
    
    def exact_1d_integral(x):
        """Compute ∫₀¹ u(c)² dc where u(c) is the 1D union length."""
        breakpoints = {0.0, 1.0}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                c1 = (x[j] - x[i] - width) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                c2 = (x[j] - x[i]) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
        
        pts = sorted(breakpoints)
        V = 0.0
        for m in range(len(pts) - 1):
            c0 = pts[m]
            c1 = pts[m+1]
            if c1 - c0 < 1e-12:
                continue
            u0 = compute_1d_union_len(x, c0)
            u1 = compute_1d_union_len(x, c1)
            diff = u1 - u0
            integral_t = u0*u0 + u0*diff + diff*diff/3.0
            V += integral_t * (c1 - c0)
        return V
    
    def keich_positions_1d(n):
        """Keich binary construction for n = 2^k base positions."""
        pos = int(np.log2(n))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(n, dtype=np.float64)
        for idx in range(n):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich
    
    # Generate multiple initial candidates
    candidates = []
    
    # Keich construction
    candidates.append(keich_positions_1d(N))
    
    # Centered shear
    candidates.append(-np.arange(N, dtype=np.float64) / (2.0 * N))
    
    # Compressed centered
    candidates.append(-np.arange(N, dtype=np.float64) / (4.0 * N))
    
    # Expanded centered
    candidates.append(-np.arange(N, dtype=np.float64) / (1.5 * N))
    
    # Linear spacing
    candidates.append(np.linspace(-0.5, 0.5, N))
    
    # Evaluate all candidates
    best_x = candidates[0].copy()
    best_val = exact_1d_integral(best_x)
    
    for cand in candidates[1:]:
        val = exact_1d_integral(cand)
        if val < best_val:
            best_x = cand.copy()
            best_val = val
    
    # Global optimization using differential evolution
    bounds = [(-1.5, 1.5)] * N
    
    if time.time() < deadline - 120:
        try:
            result = differential_evolution(
                exact_1d_integral, bounds, seed=42, maxiter=150, 
                tol=1e-10, popsize=12, polish=False,
                updating='immediate'
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_val:
                best_x = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            result2 = minimize(exact_1d_integral, best_x, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if np.all(np.isfinite(result2.x)) and result2.fun < best_val:
                best_x = result2.copy()
                best_val = result2.fun
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 20:
        try:
            result3 = minimize(exact_1d_integral, best_x, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if np.all(np.isfinite(result3.x)) and result3.fun < best_val:
                best_x = result3.copy()
                best_val = result3.fun
        except Exception:
            pass
    
    # Multi-start local refinement: try perturbing the best solution
    if time.time() < deadline - 10:
        rng_local = np.random.default_rng(999)
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            perturbation = rng_local.normal(0, 0.01, size=N)
            perturbed = best_x + perturbation
            try:
                result4 = minimize(exact_1d_integral, perturbed, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                if np.all(np.isfinite(result4.x)) and result4.fun < best_val:
                    best_x = result4.x.copy()
                    best_val = result4.fun
            except Exception:
                pass
    
    # Construct the 64x2 array using the optimized 1D positions
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]
    
    return positions

# EVOLVE-BLOCK-END
