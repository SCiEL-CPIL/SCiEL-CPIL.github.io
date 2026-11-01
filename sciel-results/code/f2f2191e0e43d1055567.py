# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] where C[k*n+i] = sum_j A[i,j] * B[j,k]
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Compute actual tensor from decomposition
    T_actual = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = np.abs(T_actual - T_expected)
    max_diff = np.max(diff)
    return max_diff < tol, max_diff

def _tensor_from_factors(U, V, W):
    """Compute tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _try_rank13_alternating(deadline):
    """
    Try to find a rank-13 decomposition using alternating minimization
    initialized from the rank-14 Strassen decomposition with one column removed.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Get rank-14 factors
    r14 = _build_rank14()
    U14 = np.array(r14['U'], dtype=float)
    V14 = np.array(r14['V'], dtype=float)
    W14 = np.array(r14['W'], dtype=float)
    
    best_result = None
    best_err = float('inf')
    
    # Try removing each column from the rank-14 decomposition
    for drop in range(14):
        if time.time() > deadline - 20:
            break
        
        keep = [i for i in range(14) if i != drop]
        U_init = U14[:, keep].copy()
        V_init = V14[:, keep].copy()
        W_init = W14[:, keep].copy()
        
        # Run alternating minimization
        U_cur, V_cur, W_cur = _alternating_minimize(
            U_init, V_init, W_init, T_expected, 
            max_iter=200, tol=1e-12, deadline=deadline - 20
        )
        
        T_actual = _tensor_from_factors(U_cur, V_cur, W_cur)
        err = np.max(np.abs(T_actual - T_expected))
        
        if err < best_err:
            best_err = err
            if err < 1e-8:
                # Round to nearest half-integer and verify
                U_rounded = np.round(U_cur * 2) / 2
                V_rounded = np.round(V_cur * 2) / 2
                W_rounded = np.round(W_cur * 2) / 2
                
                T_check = _tensor_from_factors(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - T_expected)) < 1e-8:
                    result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    valid, _ = _verify_decomposition(result)
                    if valid:
                        return result
                
                # Also try integer rounding
                U_int = np.round(U_cur)
                V_int = np.round(V_cur)
                W_int = np.round(W_cur)
                T_check2 = _tensor_from_factors(U_int, V_int, W_int)
                if np.max(np.abs(T_check2 - T_expected)) < 1e-8:
                    result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    valid, _ = _verify_decomposition(result)
                    if valid:
                        return result
    
    return None

def _alternating_minimize(U, V, W, T, max_iter=200, tol=1e-12, deadline=None):
    """
    Alternating minimization for tensor rank decomposition.
    Minimize ||T - sum_r U[:,r] outer V[:,r] outer W[:,r]||_F
    """
    n, m, p = T.shape
    a_dim, b_dim, c_dim = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each a, solve V^T W * U[:,a] = T[a,:,:].reshape(-1)
        for a in range(a_dim):
            # Build system: sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a,:,:]
            # This is a linear system in U[a,:]
            # M[r, s] = sum_{b,c} V[b,r] * W[c,s] * T[a,b,c]
            # Actually, we want: sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a,:,:]
            # Flatten: sum_r U[a,r] * vec(V[:,r] outer W[:,r]) = vec(T[a,:,:])
            # So: U[a,:] * A = vec(T[a,:,:]) where A[r, (b,c)] = V[b,r]*W[c,s]... 
            # Let me think more carefully.
            
            # We want to minimize ||T[a,:,:] - sum_r U[a,r] * (V[:,r] outer W[:,r])||^2
            # Taking derivative w.r.t U[a,r]: -2 sum_{b,c} T[a,b,c] V[b,r] W[c,r] + 2 U[a,r] sum_{b,c} V[b,r]^2 W[c,r]^2 = 0
            # Wait, that's not quite right. Let me use the standard approach.
            
            # The residual for mode-a slice: R[a] = T[a,:,:] - sum_r U[a,r] * (V[:,r] outer W[:,r])
            # dR[a]/dU[a,r] = -(V[:,r] outer W[:,r])
            # So the normal equation: sum_r (V[:,r] outer W[:,r])^T (V[:,r'] outer W[:,r']) U[a,r'] = vec(T[a,:,:])
            
            # Let's build the Gram matrix G where G[r,r'] = sum_{b,c} V[b,r]*W[c,r] * V[b,r']*W[c,r']
            G = np.zeros((rank, rank))
            rhs = np.zeros(rank)
            for r in range(rank):
                for rp in range(rank):
                    G[r, rp] = np.sum(V[:, r] * W[:, r] * V[:, rp] * W[:, rp])
                rhs[r] = np.sum(T[a, :, :] * V[:, r] * W[:, r])
            
            # Solve G * U[a,:] = rhs
            try:
                U_cur[a, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), rhs)
            except np.linalg.LinAlgError:
                U_cur[a, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), rhs, rcond=None)[0]
        
        # Update V: for each b, solve similarly
        for b in range(b_dim):
            G = np.zeros((rank, rank))
            rhs = np.zeros(rank)
            for r in range(rank):
                for rp in range(rank):
                    G[r, rp] = np.sum(U[:, r] * W[:, r] * U[:, rp] * W[:, rp])
                rhs[r] = np.sum(T[:, b, :] * U[:, r] * W[:, r])
            try:
                V_cur[b, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), rhs)
            except np.linalg.LinAlgError:
                V_cur[b, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), rhs, rcond=None)[0]
        
        # Update W: for each c, solve similarly
        for c in range(c_dim):
            G = np.zeros((rank, rank))
            rhs = np.zeros(rank)
            for r in range(rank):
                for rp in range(rank):
                    G[r, rp] = np.sum(U[:, r] * V[:, r] * U[:, rp] * V[:, rp])
                rhs[r] = np.sum(T[:, :, c] * U[:, r] * V[:, r])
            try:
                W_cur[c, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), rhs)
            except np.linalg.LinAlgError:
                W_cur[c, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), rhs, rcond=None)[0]
        
        # Check convergence
        T_actual = _tensor_from_factors(U_cur, V_cur, W_cur)
        err = np.max(np.abs(T_actual - T))
        if err < tol:
            break
    
    return U_cur, V_cur, W_cur

def _try_rank13_structured(deadline):
    """
    Try a structured rank-13 approach: use Strassen for the left 2x2 block (7 products)
    and a modified scheme for the right 2x2 block that shares one product with the left.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Try various structured rank-13 patterns
    # Pattern: 7 for left block + 6 for right block with one shared product
    
    # The shared product could be (A00+A11) * (B00+B11+B02+B13) type
    # or we can try to merge some of the Strassen products across blocks
    
    # Let's try: use the standard 7 Strassen products for left block,
    # and for the right block, reuse M1 = (A00+A11)(B00+B11) by extending it
    
    # Actually, let's try a different approach: parameterize a rank-13 solution
    # with the constraint that it has the Strassen structure for the left block
    # and 6 additional products for the right block
    
    U = np.zeros((4, 13))
    V = np.zeros((8, 13))
    W = np.zeros((8, 13))
    
    # Left block: standard Strassen (products 0-6)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1
    
    # Right block: 6 products (7-12)
    # N1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1
    
    # N2 = (A10+A11)B02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1
    
    # N3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1
    
    # N4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1
    
    # N5 = (A00+A01)B13
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1
    
    # This gives us 13 products. Check if it works.
    # We're missing: N7 = (A01-A11)(B12+B13) which contributes to W[4,13]
    # And we have an extra product somewhere
    
    # Let's verify
    T_actual = _tensor_from_factors(U, V, W)
    err = np.max(np.abs(T_actual - T_expected))
    
    if err < 1e-8:
        result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        valid, _ = _verify_decomposition(result)
        if valid:
            return result
    
    # If not, try alternating minimization from this initialization
    U_opt, V_opt, W_opt = _alternating_minimize(
        U, V, W, T_expected, max_iter=300, tol=1e-12, deadline=deadline - 20
    )
    T_check = _tensor_from_factors(U_opt, V_opt, W_opt)
    if np.max(np.abs(T_check - T_expected)) < 1e-8:
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        T_final = _tensor_from_factors(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(T_final - T_expected)) < 1e-8:
            result = {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            valid, _ = _verify_decomposition(result)
            if valid:
                return result
    
    return None

def _try_rank12_structured(deadline):
    """
    Try a structured rank-12 approach.
    """
    n, m, p = 2, 2, 4
    target_rank = 12
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Try a Laderman-inspired approach for 2x2 * 2x4
    # Use 6 products for left block (Laderman-style) and 6 for right block
    
    U = np.zeros((4, 12))
    V = np.zeros((8, 12))
    W = np.zeros((8, 12))
    
    # Left block - try to use 6 products
    # This is non-trivial since 2x2*2x2 needs at least 7 (Strassen)
    # So we need to share products between blocks
    
    # Strategy: use 7 products that serve both blocks partially
    # P1 = (A00+A11)(B00+B11+B02+B13)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1; V[2, 0] = 1; V[7, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[4, 0] = 1; W[7, 0] = 1
    
    # P2 = (A10+A11)(B00+B02)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[2, 1] = 1
    W[1, 1] = 1; W[5, 1] = 1
    
    # P3 = A00(B01-B11+B03-B13)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1; V[3, 2] = 1; V[7, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1; W[6, 2] = 1; W[7, 2] = 1
    
    # P4 = A11(B10-B00+B12-B02)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1; V[6, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1; W[4, 3] = 1; W[5, 3] = 1
    
    # P5 = (A00+A01)(B11+B13)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1; V[7, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1; W[4, 4] = -1; W[6, 4] = 1
    
    # P6 = (A10-A00)(B00+B01+B02+B03)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1; W[7, 5] = 1
    
    # P7 = (A01-A11)(B10+B11+B12+B13)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1; V[6, 6] = 1; V[7, 6] = 1
    W[0, 6] = 1; W[4, 6] = 1
    
    # P8-P12: additional products
    # P8 = A00*B00
    U[0, 7] = 1
    V[0, 7] = 1
    W[0, 7] = 1
    
    # P9 = A01*B11
    U[1, 8] = 1
    V[5, 8] = 1
    W[3, 8] = 1
    
    # P10 = A10*B00
    U[2, 9] = 1
    V[0, 9] = 1
    W[1, 9] = 1
    
    # P11 = A11*B11
    U[3, 10] = 1
    V[5, 10] = 1
    W[3, 10] = 1
    
    # P12 = A01*B10
    U[1, 11] = 1
    V[4, 11] = 1
    W[0, 11] = 1
    
    # Verify
    T_actual = _tensor_from_factors(U, V, W)
    err = np.max(np.abs(T_actual - T_expected))
    
    if err < 1e-8:
        result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        valid, _ = _verify_decomposition(result)
        if valid:
            return result
    
    # Try alternating minimization from this initialization
    U_opt, V_opt, W_opt = _alternating_minimize(
        U, V, W, T_expected, max_iter=300, tol=1e-12, deadline=deadline - 20
    )
    T_check = _tensor_from_factors(U_opt, V_opt, W_opt)
    if np.max(np.abs(T_check - T_expected)) < 1e-8:
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        T_final = _tensor_from_factors(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(T_final - T_expected)) < 1e-8:
            result = {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            valid, _ = _verify_decomposition(result)
            if valid:
                return result
    
    return None

def construct():
    deadline = time.time() + 120  # Safety margin
    
    # Try rank-12 first
    result12 = _try_rank12_structured(deadline)
    if result12 is not None:
        return result12
    
    # Try rank-13
    result13 = _try_rank13_structured(deadline)
    if result13 is not None:
        return result13
    
    # Try rank-13 with alternating minimization from rank-14
    result13_alt = _try_rank13_alternating(deadline)
    if result13_alt is not None:
        return result13_alt
    
    # Fall back to rank-14
    return _build_rank14()
# EVOLVE-BLOCK-END
