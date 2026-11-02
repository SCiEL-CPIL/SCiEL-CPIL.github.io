"""
Matrix multiplication 4x4x7 tensor decomposition search.

Strategy:
1. Build a structured decomposition using Strassen-style block methods for the 4x4x4 part,
   then extend to handle the 4x7 case.
2. Use a systematic algebraic approach: for each rank-1 term, we need
   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to sum over c to give the product tensor.
   The product tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
3. Key insight: if we set U and V to be sparse with small integer entries, then
   W[k*n+i, c] = 1 / (sum_j U[i*m+j,c] * V[j*p+k,c]) must be a half-integer.
   This means the denominator must be ±1, ±1/2, ±2, etc.
4. We use a greedy/iterative approach: fix U and V columns one at a time, choosing
   them to satisfy as many (i,k) pairs as possible with nice W values.
"""
import time
import numpy as np
from itertools import product, combinations_with_replacement

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            C_computed += wc.T * (uc * vc).sum(axis=1)[:, None] if False else \
                np.einsum('ij,jk,ki->ik', uc, vc, np.eye(p) if False else np.ones((p,n)))
        # Actually let's just do it properly
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc_vals = W[:, c]  # shape (p*n,)
            # C_computed[i,k] += wc_vals[k*n+i] * sum_j uc[i,j]*vc[j,k]
            prod = uc @ vc  # (n,p)
            # wc_vals is indexed as [k*n+i], so wc_vals.reshape(p,n).T gives [i,k]
            wc_mat = wc_vals.reshape(p, n).T  # (n,p)
            C_computed += wc_mat * prod
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _try_strassen_444():
    """
    Build a rank-28 decomposition for 4x4x4 using 2x2 Strassen blocks.
    Then we'll extend this to 4x4x7.
    
    For 4x4 matrices split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    
    Strassen for each 2x2 block product uses 7 multiplications.
    Total: 4 blocks * 7 = 28 multiplications.
    """
    n, m, p = 4, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen formulas for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4 with 2x2 blocks, indices:
    # Block (bi,bj) where bi,bj in {0,1}
    # A block (bi,bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bj,bk) has rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block (bi,bk) has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    
    # Strassen terms for block (bi,bj,bk):
    # M1: U has +1 at (2bi,2bj),(2bi+1,2bj+1); V has +1 at (2bj,2bk),(2bj+1,2bk+1)
    # M2: U has +1 at (2bi+1,2bj),(2bi+1,2bj+1); V has +1 at (2bj,2bk)
    # M3: U has +1 at (2bi,2bj); V has +1 at (2bj,2bk+1), -1 at (2bj+1,2bk+1)
    # M4: U has +1 at (2bi+1,2bj+1); V has +1 at (2bj+1,2bk), -1 at (2bj,2bk)
    # M5: U has +1 at (2bi,2bj),(2bi,2bj+1); V has +1 at (2bj+1,2bk+1)
    # M6: U has +1 at (2bi+1,2bj), -1 at (2bi,2bj); V has +1 at (2bj,2bk),(2bj,2bk+1)
    # M7: U has +1 at (2bi,2bj+1), -1 at (2bi+1,2bj+1); V has +1 at (2bj+1,2bk),(2bj+1,2bk+1)
    
    # C contributions:
    # C11 (rows 2bi,2bi+1; cols 2bk,2bk+1):
    #   (2bi,2bk): M1+M4-M5+M7 -> W: +1,+1,-1,+1
    #   (2bi,2bk+1): M1+M4-M5+M7 -> same
    #   (2bi+1,2bk): M1+M4-M5+M7 -> same  
    #   (2bi+1,2bk+1): M1+M4-M5+M7 -> same
    # C12 (rows 2bi,2bi+1; cols 2(bk+1),2(bk+1)+1):
    #   all: M3+M5 -> W: +1,+1
    # C21 (rows 2(bi+1),2(bi+1)+1; cols 2bk,2bk+1):
    #   all: M2+M4 -> W: +1,+1
    # C22 (rows 2(bi+1),2(bi+1)+1; cols 2(bk+1),2(bk+1)+1):
    #   all: M1-M2+M3+M6 -> W: +1,-1,+1,+1
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # M1
                U[2*bi, 2*bj, col] = 1
                U[2*bi+1, 2*bj+1, col] = 1
                V[2*bj, 2*bk, col] = 1
                V[2*bj+1, 2*bk+1, col] = 1
                # W for C11: +1, C12: 0, C21: 0, C22: +1
                W[2*bk, 2*bi, col] = 1      # C11(0,0)
                W[2*bk+1, 2*bi, col] = 1    # C11(0,1)
                W[2*bk, 2*bi+1, col] = 1    # C11(1,0)
                W[2*bk+1, 2*bi+1, col] = 1  # C11(1,1)
                W[2*bk, 2*bi+2, col] = 1    # C22(0,0)
                W[2*bk+1, 2*bi+2, col] = 1  # C22(0,1)
                W[2*bk, 2*bi+3, col] = 1    # C22(1,0)
                W[2*bk+1, 2*bi+3, col] = 1  # C22(1,1)
                col += 1
                
                # M2
                U[2*bi+1, 2*bj, col] = 1
                U[2*bi+1, 2*bj+1, col] = 1
                V[2*bj, 2*bk, col] = 1
                # W for C21: +1
                W[2*bk, 2*bi+2, col] = 1    # C21(0,0)
                W[2*bk+1, 2*bi+2, col] = 1  # C21(0,1)
                W[2*bk, 2*bi+3, col] = 1    # C21(1,0)
                W[2*bk+1, 2*bi+3, col] = 1  # C21(1,1)
                col += 1
                
                # M3
                U[2*bi, 2*bj, col] = 1
                V[2*bj, 2*bk+1, col] = 1
                V[2*bj+1, 2*bk+1, col] = -1
                # W for C12: +1
                W[2*bk+1, 2*bi, col] = 1    # C12(0,0)
                W[2*bk+2, 2*bi, col] = 1    # C12(0,1)
                W[2*bk+1, 2*bi+1, col] = 1  # C12(1,0)
                W[2*bk+2, 2*bi+1, col] = 1  # C12(1,1)
                col += 1
                
                # M4
                U[2*bi+1, 2*bj+1, col] = 1
                V[2*bj+1, 2*bk, col] = 1
                V[2*bj, 2*bk, col] = -1
                # W for C11: +1, C21: +1
                W[2*bk, 2*bi, col] = 1      # C11(0,0)
                W[2*bk+1, 2*bi, col] = 1    # C11(0,1)
                W[2*bk, 2*bi+1, col] = 1    # C11(1,0)
                W[2*bk+1, 2*bi+1, col] = 1  # C11(1,1)
                W[2*bk, 2*bi+2, col] = 1    # C21(0,0)
                W[2*bk+1, 2*bi+2, col] = 1  # C21(0,1)
                W[2*bk, 2*bi+3, col] = 1    # C21(1,0)
                W[2*bk+1, 2*bi+3, col] = 1  # C21(1,1)
                col += 1
                
                # M5
                U[2*bi, 2*bj, col] = 1
                U[2*bi, 2*bj+1, col] = 1
                V[2*bj+1, 2*bk+1, col] = 1
                # W for C12: +1, C11: -1
                W[2*bk+1, 2*bi, col] = 1    # C12(0,0)
                W[2*bk+2, 2*bi, col] = 1    # C12(0,1)
                W[2*bk+1, 2*bi+1, col] = 1  # C12(1,0)
                W[2*bk+2, 2*bi+1, col] = 1  # C12(1,1)
                W[2*bk, 2*bi, col] = -1     # C11(0,0)
                W[2*bk+1, 2*bi, col] = -1   # C11(0,1)
                W[2*bk, 2*bi+1, col] = -1   # C11(1,0)
                W[2*bk+1, 2*bi+1, col] = -1 # C11(1,1)
                col += 1
                
                # M6
                U[2*bi+1, 2*bj, col] = 1
                U[2*bi, 2*bj, col] = -1
                V[2*bj, 2*bk, col] = 1
                V[2*bj, 2*bk+1, col] = 1
                # W for C22: +1
                W[2*bk+1, 2*bi+2, col] = 1  # C22(0,0)
                W[2*bk+2, 2*bi+2, col] = 1  # C22(0,1)
                W[2*bk+1, 2*bi+3, col] = 1  # C22(1,0)
                W[2*bk+2, 2*bi+3, col] = 1  # C22(1,1)
                col += 1
                
                # M7
                U[2*bi, 2*bj+1, col] = 1
                U[2*bi+1, 2*bj+1, col] = -1
                V[2*bj+1, 2*bk, col] = 1
                V[2*bj+1, 2*bk+1, col] = 1
                # W for C11: +1
                W[2*bk, 2*bi, col] = 1      # C11(0,0)
                W[2*bk+1, 2*bi, col] = 1    # C11(0,1)
                W[2*bk, 2*bi+1, col] = 1    # C11(1,0)
                W[2*bk+1, 2*bi+1, col] = 1  # C11(1,1)
                col += 1
    
    assert col == 28
    return {"U": U, "V": V, "W": W}

def _extend_to_447(strassen_444, n, m, p):
    """
    Extend the 4x4x4 Strassen decomposition to 4x4x7.
    We need to handle columns 4,5,6 of B (0-indexed).
    For these, we use schoolbook: rank 4*4*3 = 48 additional terms.
    Total: 28 + 48 = 76.
    """
    U444 = strassen_444["U"].copy()
    V444 = strassen_444["V"].copy()
    W444 = strassen_444["W"].copy()
    
    rank_444 = 28
    rank_total = rank_444 + 4 * 4 * 3
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Copy 4x4x4 part
    U[:, :rank_444] = U444
    V[:, :rank_444] = V444
    W[:, :rank_444] = W444
    
    # Add schoolbook terms for columns 4,5,6 of B
    col = rank_444
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    assert col == rank_total
    return {"U": U, "V": V, "W": W}

def _try_combined_search(n, m, p, start_time, deadline):
    """
    Try to find a better decomposition than 76 by searching for structured patterns.
    Key idea: for the 4x4x7 case, we can potentially share some Strassen terms
    across the extra columns if we design the decomposition carefully.
    
    Alternative: use a different block structure.
    Split 4x4 A into 2x2 blocks, but handle B differently.
    
    Let's try: use Strassen for the 4x4x4 part (rank 28), and for the 4x4x3 part,
    try to use a rank less than 48.
    
    For 4x4x3: schoolbook is 48. Can we do better?
    Using 2x2 Strassen blocks: split A into 2x2, B (4x3) into... it's not square.
    
    Let's try a direct search for the 4x4x3 part with rank < 48.
    """
    # First, let's verify our 4x4x4 Strassen decomposition
    s444 = _try_strassen_444()
    if _verify_decomposition(s444["U"].tolist(), s444["V"].tolist(), s444["W"].tolist(), 4, 4, 4):
        # Good, now extend to 4x4x7
        result_76 = _extend_to_447(s444, n, m, p)
        if _verify_decomposition(result_76["U"].tolist(), result_76["V"].tolist(), result_76["W"].tolist(), n, m, p):
            return result_76
    return None

def _search_443_part(start_time, deadline):
    """
    Search for a low-rank decomposition of 4x4x3 matrix multiplication.
    Schoolbook: 48.
    Try to find rank < 48.
    """
    n, m, p = 4, 4, 3
    best = None
    best_rank = n * m * p
    
    # Try various target ranks
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 50 if target_rank < 40 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate sparse U and V
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            
            for c in range(target_rank):
                # Random sparse pattern
                n_u = np.random.randint(1, 5)
                n_v = np.random.randint(1, 5)
                u_idx = np.random.choice(n*m, size=n_u, replace=False)
                v_idx = np.random.choice(m*p, size=n_v, replace=False)
                for idx in u_idx:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = np.random.choice([-1, 1])
            
            # Check if W can be computed with nice values
            W = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                uc = U[:, c].reshape(n, m)
                vc = V[:, c].reshape(m, p)
                prod = uc @ vc  # (n,p)
                for i in range(n):
                    for k in range(p):
                        if abs(prod[i,k]) < 1e-12:
                            valid = False
                            break
                        W[k*n+i, c] = 1.0 / prod[i,k]
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W entries are half-integers
            W2 = W * 2
            W2r = np.round(W2)
            if not np.allclose(W2, W2r, atol=1e-9):
                continue
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = (W2r / 2).tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best is not None and best_rank < 40:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based construction (target: rank 76)
    try:
        result = _try_combined_search(n, m, p, start_time, deadline)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    except Exception:
        pass
    
    # Try to improve the 4x4x3 part
    if time.time() < deadline - 20 and best_rank > 60:
        try:
            s444 = _try_strassen_444()
            if _verify_decomposition(s444["U"].tolist(), s444["V"].tolist(), s444["W"].tolist(), 4, 4, 4):
                # Search for better 4x4x3 decomposition
                result_443 = _search_443_part(start_time, deadline - 20)
                if result_443 is not None:
                    r443 = len(result_443["U"][0])
                    if r443 < 48:
                        # Combine: Strassen 4x4x4 (28) + improved 4x4x3 (r443)
                        U444 = s444["U"]
                        V444 = s444["V"]
                        W444 = s444["W"]
                        
                        U = np.zeros((n*m, 28 + r443))
                        V = np.zeros((m*p, 28 + r443))
                        W = np.zeros((p*n, 28 + r443))
                        
                        U[:, :28] = U444
                        V[:, :28] = V444
                        W[:, :28] = W444
                        
                        U443 = np.array(result_443["U"])
                        V443 = np.array(result_443["V"])
                        W443 = np.array(result_443["W"])
                        
                        # 4x4x3 decomposition: U is (4*4, r443), V is (4*3, r443), W is (3*4, r443)
                        # We need to embed V443 into the full V (4*7, r443)
                        # V443 has rows for B[0:4, 4:7] which are rows 4*0+4 to 4*0+6, 4*1+4 to...
                        # Actually B is 4x7, so V should be (4*7, rank)
                        # V443 is (4*3, r443) corresponding to B columns 4,5,6
                        # In the full V, these are at positions j*7+k for j in 0..3, k in 4..6
                        for j in range(4):
                            for k in range(3):
                                V[j*7 + (4+k), 28:28+r443] = V443[j*3+k, :]
                        
                        # W443 is (3*4, r443) corresponding to C columns 4,5,6
                        # In full W, these are at k*4+i for k in 4..6, i in 0..3
                        for k in range(3):
                            for i in range(4):
                                W[(4+k)*4 + i, 28:28+r443] = W443[k*4+i, :]
                        
                        U[:, 28:28+r443] = U443
                        
                        combined = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        if _verify_decomposition(combined["U"], combined["V"], combined["W"], n, m, p):
                            r = 28 + r443
                            if r < best_rank:
                                best = combined
                                best_rank = r
        except Exception:
            pass
    
    return best
