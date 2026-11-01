import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def build_tensor():
    """Build the 4x16x16 tensor for M_{2,2,8}."""
    T = np.zeros((NM, MP, PN), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Build tensor
    T = build_tensor()
    
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != T[a, b, c]:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            for i in range(4):
                U[i][col] = u[i]
                
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def try_rank_reduction(target_rank, T, time_limit):
    """
    Try to find a decomposition of target_rank using alternating least squares.
    """
    nm, mp, pn = T.shape
    
    # Initialize U, V, W
    # Use random initialization, or seed with Strassen terms if target_rank >= 28
    # For now, random
    U = np.random.randn(nm, target_rank)
    V = np.random.randn(mp, target_rank)
    W = np.random.randn(pn, target_rank)
    
    start_time = time.time()
    best_residual = np.inf
    
    for iteration in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Solve for U
        # T is (nm, mp, pn). We want U (nm, R) such that U @ (V^T * W) approx T
        # Reshape T to (nm, mp*pn)
        T_flat = T.reshape(nm, mp * pn)
        # Construct matrix A_u where A_u[i, r] = sum_{b,c} V[b,r] W[c,r] ? No.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed V, W, this is linear in U.
        # T_flat[a, b*pn+c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Let M_u[r, b*pn+c] = V[b,r] * W[c,r]
        # Then T_flat = U @ M_u^T
        # U = T_flat @ (M_u^T)^+
        
        M_u = np.zeros((target_rank, mp * pn))
        for r in range(target_rank):
            M_u[r, :] = (V[:, r] * W[:, r]).reshape(mp * pn)
        
        # Solve U M_u^T = T_flat
        # U = T_flat @ pinv(M_u^T)
        try:
            U_new = T_flat @ np.linalg.pinv(M_u.T)
        except np.linalg.LinAlgError:
            break
            
        # Solve for V
        # T_flat_v: (mp, nm*pn)
        T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        M_v = np.zeros((target_rank, nm * pn))
        for r in range(target_rank):
            M_v[r, :] = (U[:, r] * W[:, r]).reshape(nm * pn)
            
        try:
            V_new = T_flat_v @ np.linalg.pinv(M_v.T)
        except np.linalg.LinAlgError:
            break
            
        # Solve for W
        # T_flat_w: (pn, nm*mp)
        T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        M_w = np.zeros((target_rank, nm * mp))
        for r in range(target_rank):
            M_w[r, :] = (U[:, r] * V[:, r]).reshape(nm * mp)
            
        try:
            W_new = T_flat_w @ np.linalg.pinv(M_w.T)
        except np.linalg.LinAlgError:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Calculate residual
        # Reconstruct T
        T_recon = np.zeros_like(T)
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        residual = np.linalg.norm(T - T_recon)
        if residual < best_residual:
            best_residual = residual
            U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
        if residual < 1e-6:
            break
            
    if best_residual < 1e-6:
        # Try to round to half-integers
        U_rounded = np.round(U_best * 2) / 2
        V_rounded = np.round(V_best * 2) / 2
        W_rounded = np.round(W_best * 2) / 2
        
        # Convert to lists of Fractions for verification
        U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_rounded]
        V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_rounded]
        W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_rounded]
        
        if verify(U_frac, V_frac, W_frac):
            return U_frac, V_frac, W_frac
            
    return None

def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Try to find lower rank decompositions
    T = build_tensor()
    
    # Try ranks 27, 26, 25, 24
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Allow 10 seconds per rank attempt
        time_for_attempt = min(10, remaining_time - 5)
        
        result = try_rank_reduction(target_rank, T, time_for_attempt)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # 2. Fall back to rank 28
    if time.time() < deadline:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    # 3. Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
