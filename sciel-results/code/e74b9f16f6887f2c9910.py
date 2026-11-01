import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 180.0

    # Precompute factorials and powers for I_n = 2^n n! sqrt(pi)
    # We need n up to 14 (since H12 involves H14 in derivative/second moment)
    # Actually, max degree in P is 12.
    # I_0 uses H0, H4, H8, H12 -> max n=12.
    # I_2 uses x^2 H_k -> H_{k+2} -> max n=14.
    # I_g' uses Q(x) derived from P. P max deg 12. Q max deg 13.
    # Wait, P has even terms only.
    # P = c0 H0 + c4 H4 + c8 H8 + c12 H12.
    # P' has odd terms H1, H3, H7, H11.
    # x P has odd terms H1, H5, H9, H13.
    # Q = P' - x P has odd terms H1, H3, H5, H7, H9, H11, H13.
    # So max m for q_m is 13.
    # I_g' sums q_m^2 2^m m! sqrt(pi). Max m=13.
    
    sqrt_pi = np.sqrt(np.pi)
    max_n = 14
    norms = np.zeros(max_n + 1)
    for n in range(max_n + 1):
        # 2^n n!
        val = 1.0
        for i in range(n):
            val *= 2.0 * (i + 1)
        norms[n] = val * sqrt_pi

    # Degrees present in P
    degrees = [0, 4, 8, 12]
    
    def compute_c12(c0, c4, c8):
        return -(c0 + 3*c4 + 105*c8) / 10395.0

    def objective_and_grad(params):
        c0, c4, c8 = params
        c12 = compute_c12(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        degs = np.array(degrees)
        
        # I_0 = sum c_k^2 norms[k]
        I0 = np.sum(coeffs**2 * norms[degs])
        if I0 < 1e-15:
            return 1e15, np.zeros(3)
        
        # I_2 = sum c_k c_l M_kl
        # M_kl is non-zero if l = k, k+2, k-2
        # M_kk = (k + 0.5) norms[k]
        # M_k,k+2 = 0.25 * norms[k+2]
        # M_k,k-2 = k(k-1) * norms[k-2]
        
        I2 = 0.0
        for i, k in enumerate(degs):
            ci = coeffs[i]
            # term with l=k
            I2 += ci**2 * (k + 0.5) * norms[k]
            # term with l=k+2
            if k + 2 in degs:
                j = degs.index(k + 2)
                cj = coeffs[j]
                I2 += 2 * ci * cj * 0.25 * norms[k+2]
            # term with l=k-2
            if k - 2 in degs and k > 0:
                j = degs.index(k - 2)
                cj = coeffs[j]
                I2 += 2 * ci * cj * k * (k - 1) * norms[k-2]
        
        # I_g'
        # q_m = (m+1) c_{m+1} - 0.5 c_{m-1}
        # m ranges from 1 to 13 (odd)
        # c_m is 0 if m not in degs
        # We can just loop m
        Ig = 0.0
        for m in range(1, 14, 2):
            c_m_plus_1 = 0.0
            if m + 1 in degs:
                c_m_plus_1 = coeffs[degs.index(m + 1)]
            c_m_minus_1 = 0.0
            if m - 1 in degs:
                c_m_minus_1 = coeffs[degs.index(m - 1)]
            
            q_m = (m + 1) * c_m_plus_1 - 0.5 * c_m_minus_1
            if q_m != 0.0:
                Ig += q_m**2 * norms[m]
        
        if I2 < 1e-15 or Ig < 1e-15:
            return 1e15, np.zeros(3)
            
        bound_sq = (I2 / I0) * (Ig / I0)
        bound = np.sqrt(bound_sq)
        
        # Gradient
        # f = 0.5 * log(bound_sq) = 0.5 * (log(I2) + log(Ig) - 2 log(I0))
        # df/dc = 0.5 * (I2'/I2 + Ig'/Ig - 2 I0'/I0)
        
        # I0' = 2 * sum c_k norms[k] * delta_{k, param}
        # I0_c0 = 2 c0 norms[0]
        # I0_c4 = 2 c4 norms[4]
        # I0_c8 = 2 c8 norms[8]
        # Note c12 depends on c0,c4,c8.
        # dc12/dc0 = -1/10395
        # dc12/dc4 = -3/10395
        # dc12/dc8 = -105/10395
        
        dI0_dc0 = 2 * (c0 * norms[0] + c12 * norms[12] * (-1/10395.0))
        dI0_dc4 = 2 * (c4 * norms[4] + c12 * norms[12] * (-3/10395.0))
        dI0_dc8 = 2 * (c8 * norms[8] + c12 * norms[12] * (-105/10395.0))
        
        # I2'
        # I2 = sum_{k,l} c_k c_l M_kl
        # dI2/dc_p = sum_{k,l} (delta_{kp} c_l + c_k delta_{lp}) M_kl = 2 sum_l c_l M_pl
        # We can reuse the structure.
        
        def dI2_dc(param_idx):
            # param_idx 0,1,2 corresponds to c0, c4, c8
            # If param is c0, we need sum_l c_l M_0l
            # M_0l is non-zero for l=0, 2 (but 2 not in degs). So only l=0.
            # M_00 = 0.5 norms[0]
            # So dI2/dc0 = 2 c0 M_00 + 2 c12 M_0,12 (if 12 in degs? No, 12 is not 0+2 or 0-2)
            # Wait, M_kl is symmetric.
            # Let's just compute dI2/dc0, dI2/dc4, dI2/dc8 by differentiating the expression.
            
            pass # Placeholder, will implement explicitly below

        # Explicit derivatives for I2
        # I2 = c0^2 (0.5 n0) + c4^2 (4.5 n4) + c8^2 (8.5 n8) + c12^2 (12.5 n12)
        #    + 2 c0 c4 M_04 (0) + 2 c0 c8 M_08 (0) + 2 c0 c12 M_012 (0)
        #    + 2 c4 c8 M_48 (0.25 n8) + 2 c4 c12 M_412 (0)
        #    + 2 c8 c12 M_812 (0.25 n12)
        #    + 2 c4 c0 M_40 (0) ...
        
        # Let's check non-zero M_kl for k,l in {0,4,8,12}
        # M_00 = 0.5 n0
        # M_44 = 4.5 n4
        # M_88 = 8.5 n8
        # M_1212 = 12.5 n12
        # M_48 = 0.25 n8 (from k=4, l=6? No. k=4, l=4+2=6 not in set. k=8, l=8-2=6 not in set.)
        # Wait. M_kl non-zero if l=k, k+2, k-2.
        # Set {0, 4, 8, 12}.
        # Pairs with diff 2: None.
        # So I2 = c0^2 M_00 + c4^2 M_44 + c8^2 M_88 + c12^2 M_1212.
        # This simplifies things greatly!
        
        M_00 = 0.5 * norms[0]
        M_44 = 4.5 * norms[4]
        M_88 = 8.5 * norms[8]
        M_1212 = 12.5 * norms[12]
        
        I2 = c0**2 * M_00 + c4**2 * M_44 + c8**2 * M_88 + c12**2 * M_1212
        
        dI2_dc0 = 2 * (c0 * M_00 + c12 * M_1212 * (-1/10395.0))
        dI2_dc4 = 2 * (c4 * M_44 + c12 * M_1212 * (-3/10395.0))
        dI2_dc8 = 2 * (c8 * M_88 + c12 * M_1212 * (-105/10395.0))
        
        # I_g'
        # q_m = (m+1) c_{m+1} - 0.5 c_{m-1}
        # Non-zero c's are c0, c4, c8, c12.
        # m=1: q1 = 2 c2 - 0.5 c0 = -0.5 c0
        # m=3: q3 = 4 c4 - 0.5 c2 = 4 c4
        # m=5: q5 = 6 c6 - 0.5 c4 = -0.5 c4
        # m=7: q7 = 8 c8 - 0.5 c6 = 8 c8
        # m=9: q9 = 10 c10 - 0.5 c8 = -0.5 c8
        # m=11: q11 = 12 c12 - 0.5 c10 = 12 c12
        # m=13: q13 = 14 c14 - 0.5 c12 = -0.5 c12
        
        q1 = -0.5 * c0
        q3 = 4.0 * c4
        q5 = -0.5 * c4
        q7 = 8.0 * c8
        q9 = -0.5 * c8
        q11 = 12.0 * c12
        q13 = -0.5 * c12
        
        Ig = q1**2 * norms[1] + q3**2 * norms[3] + q5**2 * norms[5] + \
             q7**2 * norms[7] + q9**2 * norms[9] + q11**2 * norms[11] + q13**2 * norms[13]
             
        dIg_dc0 = 2 * q1 * (-0.5) * norms[1]
        dIg_dc4 = 2 * q3 * 4.0 * norms[3] + 2 * q5 * (-0.5) * norms[5]
        dIg_dc8 = 2 * q7 * 8.0 * norms[7] + 2 * q9 * (-0.5) * norms[9]
        dIg_dc12 = 2 * q11 * 12.0 * norms[11] + 2 * q13 * (-0.5) * norms[13]
        
        # Chain rule for c12
        dIg_dc0 += dIg_dc12 * (-1/10395.0)
        dIg_dc4 += dIg_dc12 * (-3/10395.0)
        dIg_dc8 += dIg_dc12 * (-105/10395.0)
        
        # Gradient of log bound
        grad_log = 0.5 * (dI2_dc0/I2 + dIg_dc0/Ig - 2*dI0_dc0/I0,
                          dI2_dc4/I2 + dIg_dc4/Ig - 2*dI0_dc4/I0,
                          dI2_dc8/I2 + dIg_dc8/Ig - 2*dI0_dc8/I0)
                          
        # Gradient of bound is bound * grad_log
        grad = bound * np.array(grad_log)
        
        return bound, grad

    # Initial guess
    best_coeffs = np.array([-1.13, 0.04, 0.000308])
    best_val, _ = objective_and_grad(best_coeffs)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    starts = [best_coeffs.copy()]
    
    # Local neighborhood
    for _ in range(10):
        starts.append(best_coeffs + rng.normal(0, 0.05, 3))
        
    # Broader search
    for _ in range(10):
        starts.append(rng.uniform(-2, 2, 3))
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        # Use L-BFGS-B with gradient
        try:
            res = minimize(objective_and_grad, start, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return np.array(best_coeffs)
