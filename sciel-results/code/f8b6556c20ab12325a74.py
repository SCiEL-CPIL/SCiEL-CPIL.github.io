import numpy as np
from scipy.special import lpmv, factorial, wigner_3j
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses a spectral tensor formulation for fast and exact evaluation of the L4 norm.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients
    
    # 1. Construct Transformation Matrix T from Real to Complex Basis
    # Y_m_real = sum_m' T[m, m'] Y_m'_complex
    # Indices: 0..20 map to m = -10..10
    def m_to_idx(m):
        return m + 10

    T = np.zeros((n_m, n_m), dtype=complex)
    
    # m = 0
    T[m_to_idx(0), m_to_idx(0)] = 1.0
    
    # m > 0
    for m in range(1, k + 1):
        idx_real = m_to_idx(m)
        idx_comp_pos = m_to_idx(m)
        idx_comp_neg = m_to_idx(-m)
        T[idx_real, idx_comp_pos] = 1.0 / np.sqrt(2.0)
        T[idx_real, idx_comp_neg] = ((-1)**m) / np.sqrt(2.0)
        
    # m < 0
    for m in range(1, k + 1):
        idx_real = m_to_idx(-m)
        idx_comp_pos = m_to_idx(m)
        idx_comp_neg = m_to_idx(-m)
        # Y_-m_real = 1/(i*sqrt(2)) * (Y_m_comp - (-1)^m Y_-m_comp)
        # = -i/sqrt(2) * Y_m_comp + i*(-1)^m/sqrt(2) * Y_-m_comp
        T[idx_real, idx_comp_pos] = -1j / np.sqrt(2.0)
        T[idx_real, idx_comp_neg] = 1j * ((-1)**m) / np.sqrt(2.0)

    # 2. Compute Complex 4-Point Coupling Tensor K_c
    # K_c[m1, m2, m3, m4] = sum_{L,M} <10 m1, 10 m2 | L M> <10 m3, 10 m4 | L M>
    # Non-zero only if m1+m2+m3+m4 = 0
    K_c = np.zeros((n_m, n_m, n_m, n_m), dtype=complex)
    
    # Precompute Clebsch-Gordan coefficients for coupling 10 and 10 to L
    # L can be 0, 2, ..., 20
    cg_coeffs = {}
    for L in range(0, 2 * k + 1, 2):
        for M in range(-L, L + 1):
            for m1 in range(-k, k + 1):
                m2 = M - m1
                if -k <= m2 <= k:
                    # <l m, l' m' | L M>
                    # wigner_3j(l, l', L; m, m', -M)
                    val_3j = wigner_3j(k, k, L, m1, m2, -M)
                    # C = (-1)^(l-m) * sqrt(2L+1) * 3j
                    phase = (-1) ** (k - m1)
                    cg = phase * np.sqrt(2 * L + 1) * val_3j
                    cg_coeffs[(m1, m2, L, M)] = cg

    for m1 in range(-k, k + 1):
        for m2 in range(-k, k + 1):
            for m3 in range(-k, k + 1):
                m4 = -(m1 + m2 + m3)
                if -k <= m4 <= k:
                    # Sum over L, M
                    M = m1 + m2
                    # M must be same for second pair
                    if m3 + m4 != M:
                        continue
                    
                    total = 0.0
                    for L in range(0, 2 * k + 1, 2):
                        if abs(M) > L:
                            continue
                        if (m1, m2, L, M) in cg_coeffs:
                            cg1 = cg_coeffs[(m1, m2, L, M)]
                            cg2 = cg_coeffs[(m3, m4, L, M)]
                            total += cg1 * cg2
                    K_c[m_to_idx(m1), m_to_idx(m2), m_to_idx(m3), m_to_idx(m4)] = total

    # 3. Transform to Real Basis
    # K_r[i, j, k, l] = sum_{m1..m4} T[i, m1] T[j, m2] T[k, m3] T[l, m4] K_c[m1, m2, m3, m4]
    # Use einsum for efficiency
    # T is 21x21, K_c is 21x21x21x21
    # K_r = einsum('im,jn,ko,lp,mnop->ijkl', T, T, T, T, K_c)
    K_r = np.einsum('im,jn,ko,lp,mnop->ijkl', T, T, T, T, K_c)
    K_r = np.real(K_r) # Should be real

    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        
        # F(c) = ||f||_4^4 = sum K_ijkl c_i c_j c_k c_l
        F = np.einsum('ijkl,i,j,k,l->', K_r, c, c, c, c)
        
        if F < 1e-15:
            return 1e10, np.zeros(n_m)
            
        # Gradient of F: G_i = 4 * sum_jkl K_ijkl c_j c_k c_l
        G = 4.0 * np.einsum('ijkl,j,k,l->i', K_r, c, c, c)
        
        # R = F^0.25 / norm_c
        R = F**0.25 / norm_c
        
        # Gradient of R
        grad_R = 0.25 * (F ** -0.75) * (G / norm_c) - (F ** 0.25) * (c / (norm_c ** 3))
        
        return R, grad_R

    # 4. Generate Initial Candidates
    candidates = []
    
    # Pure modes
    for i in range(n_m):
        c = np.zeros(n_m)
        c[i] = 1.0
        candidates.append(c)
        
    # Pairs of modes
    for i in range(n_m):
        for j in range(i + 1, n_m):
            if time.time() > deadline - 60:
                break
            c = np.zeros(n_m)
            c[i] = 1.0
            c[j] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
            
    # Random normalized vectors
    rng = np.random.default_rng(42)
    n_random = 500
    for _ in range(n_random):
        if time.time() > deadline - 30:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 5. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    for c_init in candidates:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            try:
                val_init, _ = objective_and_grad(c_init)
                if val_init < best_val:
                    best_val = val_init
                    best_c = c_init.copy()
            except:
                pass

    # 6. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
