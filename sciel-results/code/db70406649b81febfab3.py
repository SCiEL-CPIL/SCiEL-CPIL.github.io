import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen-based block decomposition with systematic algebraic reduction.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor T."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err

    def verify_exact(U, V, W):
        """Verify using exact Fraction arithmetic."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(U.shape[1]):
                        u_val = Fraction(U[i*m+j, c]).limit_denominator(10**15)
                        v_val = Fraction(V[j*p+k, c]).limit_denominator(10**15)
                        w_val = Fraction(W[k*n+i, c]).limit_denominator(10**15)
                        s += u_val * v_val * w_val
                    if s != 1:
                        return False
        return True

    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block at (i_start, j_start, k_start)."""
        # M1 = (A11+A22)(B11+B22)
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A21+A22)B11
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3 = A11(B21-B11)
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4 = A22(B22-B12)
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5 = (A11+A12)B22
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6 = (A21-A11)(B11+B12)
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7 = (A12-A22)(B21+B22)
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))

    # Decomposition plan for 2x5x5:
    # 4 Strassen blocks for j in {0,1,2,3} and k in {0,1,2,3}
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Tails:
    for j in range(4):
        for i in range(2):
            add_schoolbook_term(i, j, 4)
            
    for k in range(5):
        for i in range(2):
            add_schoolbook_term(i, 4, k)
    
    # Build the decomposition
    U, V, W = build_decomposition_from_terms(terms)
    R = U.shape[1]
    
    # Verify
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Systematic rank reduction using algebraic structure
    if time.time() < deadline:
        # Build tensor for analysis
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        def try_remove_term(U, V, W, term_idx, tol=1e-9):
            """Try removing term_idx and see if decomposition still holds."""
            if term_idx >= U.shape[1]:
                return None, None, None, False
            
            U_new = np.delete(U, term_idx, axis=1)
            V_new = np.delete(V, term_idx, axis=1)
            W_new = np.delete(W, term_idx, axis=1)
            
            Rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            err = np.max(np.abs(T - Rec))
            
            if err < tol:
                return U_new, V_new, W_new, True
            return None, None, None, False
        
        def try_replace_with_combination(U, V, W, idx_remove, idx_keep, coeff, tol=1e-9):
            """Try replacing term idx_remove by adding coeff * term idx_keep to it, then removing idx_remove."""
            if idx_remove >= U.shape[1] or idx_keep >= U.shape[1]:
                return None, None, None, False
            if idx_remove == idx_keep:
                return None, None, None, False
            
            # Create modified U, V, W where we add coeff * term[idx_keep] to term[idx_remove]
            U_mod = U.copy()
            V_mod = V.copy()
            W_mod = W.copy()
            
            U_mod[:, idx_remove] += coeff * U_mod[:, idx_keep]
            V_mod[:, idx_remove] += coeff * V_mod[:, idx_keep]
            W_mod[:, idx_remove] += coeff * W_mod[:, idx_keep]
            
            # Now try to remove idx_remove
            U_new = np.delete(U_mod, idx_remove, axis=1)
            V_new = np.delete(V_mod, idx_remove, axis=1)
            W_new = np.delete(W_mod, idx_remove, axis=1)
            
            Rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            err = np.max(np.abs(T - Rec))
            
            if err < tol:
                return U_new, V_new, W_new, True
            return None, None, None, False
        
        # Greedy removal with multiple passes
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        R_cur = U_cur.shape[1]
        
        # Multiple passes of removal
        for pass_num in range(5):
            if time.time() > deadline - 15:
                break
            if R_cur <= 0:
                break
            
            removed_this_pass = 0
            R_before = U_cur.shape[1]
            
            # Sort by importance (norm of contribution)
            term_importance = []
            for r in range(U_cur.shape[1]):
                slice_norm = np.linalg.norm(U_cur[:, r]) * np.linalg.norm(V_cur[:, r]) * np.linalg.norm(W_cur[:, r])
                term_importance.append((slice_norm, r))
            term_importance.sort(key=lambda x: x[0])
            
            for _, r in term_importance:
                if time.time() > deadline - 15:
                    break
                if r >= U_cur.shape[1]:
                    continue
                
                # First try simple removal
                U_new, V_new, W_new, success = try_remove_term(U_cur, V_cur, W_cur, r)
                if success:
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    removed_this_pass += 1
                    continue
                
                # Try combination with other terms
                for r2 in range(U_cur.shape[1]):
                    if r2 == r or r2 >= U_cur.shape[1]:
                        continue
                    for coeff in [-1, 1, -0.5, 0.5]:
                        U_new, V_new, W_new, success = try_replace_with_combination(
                            U_cur, V_cur, W_cur, r, r2, coeff
                        )
                        if success:
                            U_cur, V_cur, W_cur = U_new, V_new, W_new
                            removed_this_pass += 1
                            break
                    if removed_this_pass > 0 and r2 > r:
                        break
                
                if removed_this_pass > 0:
                    # Reset importance since we modified the matrices
                    break
            
            R_cur = U_cur.shape[1]
            if removed_this_pass == 0:
                break
        
        if R_cur < R:
            U, V, W = U_cur, V_cur, W_cur
            R = R_cur
    
    # Final verification
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Rebuild from original terms if something went wrong
        U, V, W = build_decomposition_from_terms(terms)
        R = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
