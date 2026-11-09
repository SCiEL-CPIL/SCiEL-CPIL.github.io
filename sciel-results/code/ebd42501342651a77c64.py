import numpy as np
from scipy.optimize import minimize

def construct():
    # Constants for Hermite polynomials H_n(0)
    # H_n(0) = (-1)^(n/2) * n! / (n/2)!
    def H_n_zero(n):
        return (-1)**(n//2) * (np.math.factorial(n) // np.math.factorial(n//2))
    
    H0_0 = H_n_zero(0)
    H4_0 = H_n_zero(4)
    H8_0 = H_n_zero(8)
    H12_0 = H_n_zero(12)
    
    # Norms of H_n: ||H_n||^2 = sqrt(pi) * 2^n * n!
    def norm_sq(n):
        return np.sqrt(np.pi) * (2**n) * np.math.factorial(n)
    
    N0 = norm_sq(0)
    N4 = norm_sq(4)
    N8 = norm_sq(8)
    N12 = norm_sq(12)
    
    # Objective function
    # c = [c0, c4, c8]
    # c12 is derived from P(0)=0
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
        
        # Calculate N0 = integral P^2 exp(-x^2) dx
        # P = c0 H0 + c4 H4 + c8 H8 + c12 H12
        # Orthogonality: integral H_i H_j = 0 if i!=j
        n0_val = c0**2 * N0 + c4**2 * N4 + c8**2 * N8 + c12**2 * N12
        if n0_val < 1e-15:
            return 1e10
            
        # Calculate N2 = integral x^2 P^2 exp(-x^2) dx
        # x^2 H_j = 0.25 H_{j+2} + (j+0.5) H_j + j(j-1) H_{j-2}
        # We need integral H_i (x^2 H_j) = sqrt(pi) 2^i i! * coeff_of_Hi_in_x2Hj
        
        # Pre-compute contributions for each pair (i, j) where i, j in {0, 4, 8, 12}
        # But since P is linear, P^2 is sum c_i c_j H_i H_j.
        # We can just sum over i, j.
        
        # To be efficient, let's just compute the full sum.
        indices = [0, 4, 8, 12]
        coeffs = [c0, c4, c8, c12]
        
        n2_val = 0.0
        for i_idx, i in enumerate(indices):
            ci = coeffs[i_idx]
            for j_idx, j in enumerate(indices):
                cj = coeffs[j_idx]
                # Term ci * cj * integral H_i x^2 H_j exp(-x^2)
                # integral H_i (x^2 H_j) = norm_sq(i) * (coeff of H_i in x^2 H_j)
                # x^2 H_j has terms H_{j+2}, H_j, H_{j-2}
                
                coeff = 0.0
                if i == j + 2:
                    coeff += 0.25
                elif i == j:
                    coeff += (j + 0.5)
                elif i == j - 2:
                    coeff += j * (j - 1)
                else:
                    coeff = 0.0
                
                if coeff != 0.0:
                    n2_val += ci * cj * norm_sq(i) * coeff
                    
        # Calculate D = integral (P' - x/2 P)^2 exp(-x^2) dx
        # Q = P' - x/2 P
        # Q = sum c_m (7m/4 H_{m-1} - 1/4 H_{m+1})
        
        # Q is a linear combination of H_1, H_3, H_5, H_7, H_9, H_11, H_13
        # Let's build the coefficients for Q
        # q_k for H_k
        q_coeffs = {}
        
        def add_q(k, val):
            if k < 0: return
            q_coeffs[k] = q_coeffs.get(k, 0.0) + val
            
        # From c0 H0: m=0
        # c0 * (0 - 1/4 H1) -> q1 -= 0.25 c0
        add_q(1, -0.25 * c0)
        
        # From c4 H4: m=4
        # c4 * (7*4/4 H3 - 1/4 H5) -> q3 += 7 c4, q5 -= 0.25 c4
        add_q(3, 7.0 * c4)
        add_q(5, -0.25 * c4)
        
        # From c8 H8: m=8
        # c8 * (7*8/4 H7 - 1/4 H9) -> q7 += 14 c8, q9 -= 0.25 c8
        add_q(7, 14.0 * c8)
        add_q(9, -0.25 * c8)
        
        # From c12 H12: m=12
        # c12 * (7*12/4 H11 - 1/4 H13) -> q11 += 21 c12, q13 -= 0.25 c12
        add_q(11, 21.0 * c12)
        add_q(13, -0.25 * c12)
        
        d_val = 0.0
        for k, qk in q_coeffs.items():
            d_val += qk**2 * norm_sq(k)
            
        # Objective: sqrt( (N2/N0) * (D/N0) )
        var_x = n2_val / n0_val
        var_xi = d_val / n0_val
        
        if var_x < 0 or var_xi < 0:
            return 1e10
            
        return np.sqrt(var_x * var_xi)

    # Initial guess
    # Try to minimize. Start with c0=1, c4=0, c8=0
    x0 = np.array([1.0, 0.0, 0.0])
    
    # Use Powell or Nelder-Mead, but with analytical gradient we could use BFGS
    # However, Powell is robust for small dimensions
    res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success or res.fun < objective(x0):
        return np.array(res.x)
    else:
        return x0
