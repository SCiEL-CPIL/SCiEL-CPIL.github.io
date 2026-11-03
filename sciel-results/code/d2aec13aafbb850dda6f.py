"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Minimizes the maximum absolute circular autoconvolution divided by the squared sum.
    Uses Linear Programming on the power spectrum P = |H|^2.
    """
    n_half = N // 2
    # Variables: P[0], P[1], ..., P[n_half] (P[0] fixed to 1)
    # Plus M (the max autoconv value)
    n_p_vars = n_half + 1
    n_total = n_p_vars + 1
    
    c_obj = np.zeros(n_total)
    c_obj[-1] = 1.0 # Minimize M
    
    A_rows = []
    b_ub_list = []
    
    # Constraints: |C_k| <= M for all k
    # C_k = (1/N) * ( P[0] + 2 * sum_{j=1}^{n_half-1} P[j] cos(2*pi*k*j/N) + P[n_half] cos(2*pi*k*n_half/N) )
    for k in range(N):
        coeffs = np.zeros(n_p_vars)
        coeffs[0] = 1.0 / N
        
        for j in range(1, n_half):
            coeffs[j] = 2.0 * np.cos(2.0 * np.pi * k * j / N) / N
        
        coeffs[n_half] = np.cos(2.0 * np.pi * k * n_half / N) / N
        
        # C_k <= M
        row1 = np.zeros(n_total)
        row1[:n_p_vars] = coeffs
        row1[-1] = -1.0
        A_rows.append(row1)
        b_ub_list.append(0.0)
        
        # -C_k <= M
        row2 = np.zeros(n_total)
        row2[:n_p_vars] = -coeffs
        row2[-1] = -1.0
        A_rows.append(row2)
        b_ub_list.append(0.0)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_ub_list)
    
    # Equality constraints: P[0] = 1
    A_eq = np.zeros((1, n_total))
    A_eq[0, 0] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: P >= 0, M >= 0
    bounds = [(0, None)] * n_p_vars + [(0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            return np.ones(N) / N
            
        P_opt = res.x[:n_p_vars]
        
        # Reconstruct H
        H = np.zeros(N, dtype=complex)
        for j in range(n_half + 1):
            H[j] = np.sqrt(P_opt[j])
            if j != 0 and j != n_half:
                H[N - j] = H[j]
                
        h = np.fft.ifft(H).real
        return h
    except Exception:
        return np.ones(N) / N

# EVOLVE-BLOCK-END
