# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
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

def _find_low_rank_decomp_2x3x3(deadline):
    """
    Find a low-rank decomposition for 2x3 * 3x3.
    Returns (U, V, W) for the decomposition.
    """
    n, m, p = 2, 3, 3
    U0, V0, W0 = _rank1_decomposition(n, m, p)
    rank0 = n * m * p # 18
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
    # For schoolbook, T[i, j, k] is 1 if i, j, k correspond to a valid multiplication term, else 0.
    # Actually, for matrix multiplication, T[i, j, k] = 1 if i = i*m+j, j = j*p+k, k = k*n+i?
    # No, the indices are:
    # U index: i*m + j (row of A, col of A)
    # V index: j*p + k (row of B, col of B)
    # W index: k*n + i (row of C, col of C)
    # The tensor T[i_U, i_V, i_W] is 1 if the term (i,j,k) is present.
    # In schoolbook, each term is distinct.
    
    # We want to find a subset of terms of size R such that the tensor can be represented.
    # Let the subset be S.
    # We want to find coefficients c_r for r in S such that
    # sum_{r in S} c_r * U[:,r] V[:,r] W[:,r] = T
    # This is a linear system in c_r.
    
    # Flatten T
    T_flat = np.zeros((n*m) * (m*p) * (p*n))
    for r in range(rank0):
        # Get indices for term r
        i_U = 0
        i_V = 0
        i_W = 0
        # We need to recover i, j, k from r
        # r = i * m * p + j * p + k
        k = r % p
        j = (r // p) % m
        i = r // (m * p)
        
        idx_U = i * m + j
        idx_V = j * p + k
        idx_W = k * n + i
        
        # Flatten index
        idx = (idx_U * (m*p) + idx_V) * (p*n) + idx_W
        T_flat[idx] = 1.0

    # Try ranks from 17 down to 1
    for R in range(rank0 - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Generate combinations of R terms from rank0 terms
        # This might be too many for small R.
        # We only need to try if it's feasible.
        # For R=17, 18 combos.
        # For R=16, 153 combos.
        # For R=15, 816 combos.
        # For R=14, 3060 combos.
        # We will stop if the number of combos is too large or time runs out.
        
        # We'll just iterate and check time.
        found = False
        for combo in combinations(range(rank0), R):
            if time.time() > deadline:
                return None # Time out
            
            # Build matrix A of shape (num_entries, R)
            # A[idx, c] = U[idx_U, r] * V[idx_V, r] * W[idx_W, r]
            # We only need to consider entries where T_flat is non-zero?
            # No, we need to match the whole tensor.
            # But T_flat is sparse (only 18 non-zeros).
            # So we only need to satisfy the equations for the non-zero entries of T.
            # Wait, if we use a subset of terms, the resulting tensor might have non-zero entries where T is 0.
            # So we need to check all entries.
            # However, the tensor space is small.
            
            # Let's just use the non-zero entries of T as constraints.
            # If the solution satisfies the non-zero entries, does it satisfy the zero entries?
            # Not necessarily.
            # So we must check all entries.
            
            # But the tensor has (n*m)*(m*p)*(p*n) = 6*9*6 = 324 entries.
            # R is ~17.
            # We can build a 324 x R matrix.
            
            A = np.zeros((324, R))
            b = np.zeros(324)
            
            for c, r in enumerate(combo):
                k = r % p
                j = (r // p) % m
                i = r // (m * p)
                
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                
                # This term contributes to T[idx_U, idx_V, idx_W]
                idx = (idx_U * (m*p) + idx_V) * (p*n) + idx_W
                A[idx, c] = 1.0
                
            # Wait, this assumes the term is 1.
            # But we are looking for coefficients c_r.
            # So A[idx, c] should be 1 if the term r contributes to idx.
            # And b[idx] should be T_flat[idx].
            
            # Let's re-evaluate.
            # T = sum_{r in S} c_r * T_r
            # where T_r is the rank-1 tensor for term r.
            # T_r[i_U, i_V, i_W] = 1 if i_U = i*m+j, i_V = j*p+k, i_W = k*n+i, else 0.
            # So A[idx, c] = T_r[idx].
            
            # We can just fill A by checking each term.
            # This is O(R * 324).
            
            A = np.zeros((324, R))
            for c, r in enumerate(combo):
                k = r % p
                j = (r // p) % m
                i = r // (m * p)
                
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                
                idx = (idx_U * (m*p) + idx_V) * (p*n) + idx_W
                A[idx, c] = 1.0
            
            b = T_flat
            
            # Solve A c = b
            try:
                coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                if len(residuals) > 0:
                    residual = residuals[0]
                else:
                    residual = np.linalg.norm(A @ coeffs - b)
                
                if residual < 1e-8:
                    # Found a valid decomposition
                    # Construct U, V, W for the subset
                    U_new = [[0] * R for _ in range(n*m)]
                    V_new = [[0] * R for _ in range(m*p)]
                    W_new = [[0] * R for _ in range(p*n)]
                    
                    for c, r in enumerate(combo):
                        k = r % p
                        j = (r // p) % m
                        i = r // (m * p)
                        
                        idx_U = i * m + j
                        idx_V = j * p + k
                        idx_W = k * n + i
                        
                        U_new[idx_U][c] = coeffs[c]
                        V_new[idx_V][c] = 1.0
                        W_new[idx_W][c] = 1.0
                        # Wait, we can distribute the coefficient.
                        # U[idx_U][c] = coeffs[c]
                        # V[idx_V][c] = 1.0
                        # W[idx_W][c] = 1.0
                        # This is valid.
                    
                    return U_new, V_new, W_new
            except Exception:
                pass
        
        # If no subset found for this R, try smaller R
        # But we should probably break if R gets too small and we haven't found anything,
        # as it's unlikely to exist if 17 didn't work?
        # Actually, if 17 doesn't work, 16 might.
    
    return None

def _construct_2x3x9(deadline):
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Uses block decomposition based on 2x3 * 3x3.
    """
    n, m, p = 2, 3, 9
    
    # Try to find a low-rank decomposition for 2x3 * 3x3
    U233, V233, W233 = _find_low_rank_decomp_2x3x3(deadline)
    
    if U233 is not None:
        R233 = len(U233[0])
        # Build 2x3 * 3x9 decomposition
        # B is 3x9, split into three 3x3 blocks B1, B2, B3
        # C = [A B1, A B2, A B3]
        # We use the same decomposition for each block.
        
        rank = 3 * R233
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            # Map block b to columns 3*b to 3*b+2 of B and C
            for r in range(R233):
                col = block * R233 + r
                
                # U is the same for all blocks
                for i in range(n * m):
                    U[i][col] = U233[i][r]
                
                # V maps to the specific block
                for j in range(m):
                    for k in range(3):
                        idx_V_233 = j * 3 + k
                        idx_V_239 = j * 9 + (block * 3 + k)
                        V[idx_V_239][col] = V233[idx_V_233][r]
                        
                # W maps to the specific block
                for i in range(n):
                    for k in range(3):
                        idx_W_233 = k * n + i
                        idx_W_239 = (block * 3 + k) * n + i
                        W[idx_W_239][col] = W233[idx_W_233][r]
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    """
    start = time.time()
    deadline = start + 135
    
    try:
        result = _construct_2x3x9(deadline)
        return result
    except Exception as e:
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
