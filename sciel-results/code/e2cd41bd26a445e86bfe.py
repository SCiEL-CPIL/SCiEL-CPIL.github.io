# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import clebsch_gordan, factorial

def rudin_problem10() -> np.ndarray:
    """
    Finds the real coefficients c for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_m c_m Y_m that minimize the ratio ||f||_4 / ||f||_2.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    k = 10
    m_vals = np.arange(-k, k + 1)
    n_m = 2 * k + 1
    
    # Precompute Clebsch-Gordan coefficients for coupling 10 x 10 -> L
    # L must be even (0, 2, ..., 20) for m=0,0,0 to be non-zero
    # We need K[LM, i, j] where i,j index m_vals
    # V_LM = sum_{i,j} c_i c_j K[LM, i, j]
    
    # Store K as a list of (L, M, matrix_21x21) or just a flat array of matrices
    # Since only even L contribute to the 3j(10,10,L,0,0,0) factor?
    # Wait, the 3j symbol with m=0,0,0 is zero if L is odd.
    # But the 3j symbol with m1, m2, M is non-zero for all L, M?
    # The integral is sum over L,M of (3j_0 * 3j_M)^2.
    # 3j_0 is zero for odd L. So only even L contribute.
    
    K_list = []
    LM_indices = []
    
    for L in range(0, 2 * k + 1, 2):
        # 3j symbol for m=0,0,0
        # wigner_3j(10, 10, L, 0, 0, 0)
        # Using relation: 3j(l1,l2,l3,m1,m2,m3) = (-1)^(l1-l2-m3) / sqrt(2l3+1) * CG(l1,m1,l2,m2,l3,-m3)
        # Here m3=0.
        # CG(10, 0, 10, 0, L, 0)
        cg_000 = clebsch_gordan(k, 0, k, 0, L, 0)
        if abs(cg_000) < 1e-15:
            continue
            
        factor_000 = (-1) ** (k - k - 0) / np.sqrt(2 * L + 1) * cg_000
        # (-1)^0 = 1.
        
        for M in range(-L, L + 1):
            # We need 3j(10, 10, L, m1, m2, M)
            # Sum over m1, m2 such that m1+m2+M=0 => m2 = -m1-M
            # If m2 is not in -10..10, it is 0.
            
            mat = np.zeros((n_m, n_m))
            for i, m1 in enumerate(m_vals):
                m2 = -m1 - M
                if m2 < -k or m2 > k:
                    continue
                j = m2 - (-k) # index for m2
                
                # 3j(10, 10, L, m1, m2, M)
                # Relation: 3j = (-1)^(10-10-M) / sqrt(2L+1) * CG(10, m1, 10, m2, L, -M)
                # (-1)^(-M) = (-1)^M
                cg_val = clebsch_gordan(k, m1, k, m2, L, -M)
                val_3j = (-1) ** M / np.sqrt(2 * L + 1) * cg_val
                
                # K coefficient is factor_000 * val_3j
                # Note: The integral formula has sqrt((2l1+1)(2l2+1)(2l3+1)/4pi) * 3j * 3j
                # Wait, the expansion coefficient C_LM was sqrt(441/4pi) * 3j(0) * 3j(M)
                # And integral is sum C_LM C_LM.
                # So K_LM_ij = (sqrt(441/4pi) * factor_000 * val_3j)^2 ?
                # No.
                # Int Y_i Y_j Y_k Y_l = sum_{LM} C^{(ij)}_{LM} C^{(kl)}_{LM}
                # C^{(ij)}_{LM} = sqrt(441/4pi) * 3j(10,10,L,0,0,0) * 3j(10,10,L,m_i,m_j,M)
                # Let A_L = sqrt(441/4pi) * 3j(10,10,L,0,0,0)
                # Then C^{(ij)}_{LM} = A_L * 3j(10,10,L,m_i,m_j,M)
                # Integral = sum_{LM} (A_L * 3j_{ij}) (A_L * 3j_{kl}) = sum_{LM} A_L^2 * 3j_{ij} * 3j_{kl}
                # So K_LM_ij = A_L^2 * 3j(10,10,L,m_i,m_j,M)
                
                # Calculate A_L^2
                # A_L = sqrt(441/4pi) * factor_000
                # A_L^2 = (441/4pi) * factor_000^2
                
                # We can just store the full coefficient for K_LM_ij
                # But we need to sum over LM.
                # Let's store K_LM as a matrix.
                
                mat[i, j] = (441.0 / (4.0 * np.pi)) * (factor_000 ** 2) * (val_3j ** 2)
                # Wait, is it 3j_{ij} * 3j_{kl}?
                # V_LM = sum_{i,j} c_i c_j (A_L * 3j_{ij})
                # J = sum_{LM} V_LM^2
                # So we define K_LM_ij = A_L * 3j(10,10,L,m_i,m_j,M)
                # Then V_LM = sum c_i c_j K_LM_ij
                # J = sum V_LM^2
                # This is correct.
                
                # So we should store K_LM_ij = A_L * 3j_val
                # Let's re-calculate.
                pass
            
            # Re-do the loop to store K_LM_ij = A_L * 3j_val
            A_L = np.sqrt(441.0 / (4.0 * np.pi)) * factor_000
            for i, m1 in enumerate(m_vals):
                m2 = -m1 - M
                if m2 < -k or m2 > k:
                    continue
                j = m2 - (-k)
                cg_val = clebsch_gordan(k, m1, k, m2, L, -M)
                val_3j = (-1) ** M / np.sqrt(2 * L + 1) * cg_val
                mat[i, j] = A_L * val_3j
            
            K_list.append(mat)
            LM_indices.append((L, M))
            
    n_LM = len(K_list)
    K_stack = np.array(K_list) # Shape (n_LM, 21, 21)

    def objective(c):
        # V_LM = sum_{i,j} c_i c_j K_LM[i,j]
        # Vectorize: C_pair = c[:, None] * c[None, :]
        # V = np.einsum('lij,ij->l', K_stack, c[:, None] * c[None, :])
        C_pair = c[:, None] * c[None, :]
        V = np.einsum('lij,ij->l', K_stack, C_pair)
        return np.sum(V ** 2)

    def gradient(c):
        C_pair = c[:, None] * c[None, :]
        V = np.einsum('lij,ij->l', K_stack, C_pair)
        # dV_l/dc_k = sum_j K_l[k,j] c_j + sum_i K_l[i,k] c_i
        # Since K is symmetric, = 2 * sum_j K_l[k,j] c_j
        # W_l_k = 2 * (K_l @ c)_k
        # grad_k = 2 * sum_l V_l * W_l_k
        # W = 2 * np.einsum('lij,j->li', K_stack, c)
        W = 2.0 * np.einsum('lij,j->li', K_stack, c)
        grad = 2.0 * np.einsum('l,li->i', V, W)
        return grad

    def hessian(c):
        C_pair = c[:, None] * c[None, :]
        V = np.einsum('lij,ij->l', K_stack, C_pair)
        W = 2.0 * np.einsum('lij,j->li', K_stack, c) # W[l, k]
        # H = 2 * sum_l ( W_l W_l^T + V_l K_l )
        # Term 1: sum_l W_l[:, None] * W_l[None, :]
        H1 = np.einsum('li, lj -> ij', W, W)
        # Term 2: sum_l V_l K_l[i,j]
        H2 = np.einsum('l, lij -> ij', V, K_stack)
        return 2.0 * (H1 + H2)

    best_c = None
    best_val = np.inf

    rng = np.random.default_rng(42)
    inits = []
    # Add some structured inits
    c0 = np.zeros(21)
    c0[10] = 1.0 # m=0
    inits.append(c0)
    
    c1 = np.zeros(21)
    c1[0] = 1.0 # m=-10
    c1[20] = 1.0 # m=10
    c1 /= np.linalg.norm(c1)
    inits.append(c1)
    
    for _ in range(20):
        cr = rng.standard_normal(21)
        cr /= np.linalg.norm(cr)
        inits.append(cr)

    for c_init in inits:
        try:
            res = minimize(
                objective, 
                c_init, 
                method='trust-constr',
                jac=gradient,
                hess=hessian,
                constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2 * c}],
                options={'maxiter': 500, 'gtol': 1e-12, 'xtol': 1e-12}
            )
            
            if res.success or (hasattr(res, 'fun') and res.fun < best_val):
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
        except Exception:
            pass
            
    if best_c is None:
        best_c = np.ones(21)
        
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c

# EVOLVE-BLOCK-END
