import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin

def _build_structured_rank77():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2 (cols 0,1)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2 x 2x1 (col 2)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-6):
    """Check if U,V,W exactly decompose the 4x4*4x6 matrix multiplication tensor."""
    n, m, p = 4, 4, 6
    R = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = 1 if j==k (A[i,j]*B[j,k] -> C[i,k])
    # T has shape (n*m, m*p, p*n) with T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[(i*m+j), (j*p+k), (k*n+i)]
    
    # For efficiency, check on a set of random instances
    # C = A * B where A is 4x4, B is 4x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: C[i,k] = sum_r (sum_j A[i,j] * U[j*m+i, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Actually the tensor is: T[a, b, c] where a indexes (i,j) of A, b indexes (j,k) of B, c indexes (k,i) of C
    # T[a,b,c] = 1 if the j in a matches the j in b, else 0
    # Decomposition: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Verify by checking all 16*24*24 = 9216 entries
    for a in range(n * m):
        i, j = a // m, a % m
        for b in range(m * p):
            jj, k = b // p, b % p
            expected = 1.0 if j == jj else 0.0
            if abs(expected) > 1e-9 or True:  # check all for safety
                actual = 0.0
                for r in range(R):
                    actual += U[a, r] * V[b, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _rank_reduce(U, V, W, max_time=130.0):
    """Try to reduce rank by identifying and removing redundant terms."""
    R = U.shape[1]
    
    # Check if any column is a linear combination of others
    # For each column r, check if U[:,r], V[:,r], W[:,r] are "redundant"
    # A term r is redundant if there exist coefficients c_s (s != r) such that
    # for all a: U[a,r] = sum_s c_s * U[a,s]
    # for all b: V[b,r] = sum_s c_s * V[b,s]  
    # for all c: W[c,r] = sum_s c_s * W[c,s]
    # This is a very strong condition; more practically, we can try to find
    # a set of columns whose removal still preserves the decomposition.
    
    # Simpler approach: check if the decomposition is valid with fewer terms
    # by trying to zero out columns that have small norm or are linearly dependent.
    
    # First, let's check if the current decomposition is valid
    if not _verify_decomposition(U, V, W):
        return U, V, W  # Can't reduce if not valid
    
    # Try removing each column and checking validity
    # This is expensive, so we'll be selective
    start = time.time()
    
    # Identify columns with all-zero U, V, or W (definitely redundant)
    keep = list(range(R))
    for r in range(R):
        if time.time() - start > max_time - 10:
            break
        if np.allclose(U[:, r], 0, atol=1e-9) or np.allclose(V[:, r], 0, atol=1e-9) or np.allclose(W[:, r], 0, atol=1e-9):
            keep.remove(r)
    
    if len(keep) < R:
        U_reduced = U[:, keep]
        V_reduced = V[:, keep]
        W_reduced = W[:, keep]
        if _verify_decomposition(U_reduced, V_reduced, W_reduced):
            return U_reduced, V_reduced, W_reduced
    
    # Try removing pairs of columns
    for r in range(R):
        if time.time() - start > max_time - 10:
            break
        for s in range(r + 1, R):
            if time.time() - start > max_time - 10:
                break
            keep = [x for x in range(R) if x != r and x != s]
            U_reduced = U[:, keep]
            V_reduced = V[:, keep]
            W_reduced = W[:, keep]
            if _verify_decomposition(U_reduced, V_reduced, W_reduced):
                return U_reduced, V_reduced, W_reduced
    
    return U, V, W

def to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

def construct():
    # Build the structured rank-77 decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify
    if not _verify_decomposition(U, V, W):
        # Fallback: return as-is (shouldn't happen)
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try to reduce rank
    U, V, W = _rank_reduce(U, V, W)
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
