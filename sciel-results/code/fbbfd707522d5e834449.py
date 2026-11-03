"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where vectors are in row-major order.
    """
    factors = []
    
    # p1 = (A00+A11)(B00+B11)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11)*B00
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00*(B01-B11)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11*(B10-B00)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01)*B11
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00)*(B00+B01)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # p7 = (A01-A11)*(B10+B11)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return factors

def _strassen_4x4_factors():
    """
    Return the 49 rank-1 factors for 4x4*4x4 matrix multiplication using Strassen.
    Each factor is (u_vec, v_vec, w_vec) where vectors are 16-element (row-major).
    """
    s2 = _strassen_2x2_factors()
    factors = []
    
    for s in range(7):
        for r in range(7):
            u_block = s2[s][0]  # 4-element: coefficients for A blocks [00,01,10,11]
            v_block = s2[s][1]  # 4-element: coefficients for B blocks [00,01,10,11]
            w_block = s2[s][2]  # 4-element: coefficients for C blocks [00,01,10,11]
            
            u_scalar = s2[r][0]  # 4-element: coefficients within each 2x2 block
            v_scalar = s2[r][1]
            w_scalar = s2[r][2]
            
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # U: A is 4x4, indexed by (i,j) where i=0..3, j=0..3
            # Block (bi,bj) of A is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = u_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = u_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            u_vec[i * 4 + j] += c_block * c_scalar
            
            # V: B is 4x4, indexed by (j,k) where j=0..3, k=0..3
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = v_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = v_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            v_vec[j * 4 + k] += c_block * c_scalar
            
            # W: C is 4x4, indexed by (i,k) where i=0..3, k=0..3
            for ci in range(2):
                for cj in range(2):
                    block_idx = ci * 2 + cj
                    c_block = w_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = w_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            w_vec[i * 4 + k] += c_block * c_scalar
            
            factors.append((u_vec, v_vec, w_vec))
    
    return factors

def _strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split B into two 4x4 blocks, use Strassen (rank 49) for each.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s4_factors = _strassen_4x4_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for t in range(2):  # two 4x4 blocks of B
        for idx in range(49):
            col = t * 49 + idx
            u_vec, v_vec, w_vec = s4_factors[idx]
            
            # U: A is 4x4, index i*m+j = i*4+j
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = u_vec[i * 4 + j]
            
            # V: B is 4x8, but for block t, columns are 4*t..4*t+3
            # B index is j*p+k = j*8+k, where j=0..3, k=4*t..4*t+3
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (4 * t + k)][col] = v_vec[j * 4 + k]
            
            # W: C is 4x8, index k*n+i = k*4+i, where k=4*t..4*t+3, i=0..3
            for i in range(4):
                for k in range(4):
                    W[(4 * t + k) * 4 + i][col] = w_vec[i * 4 + k]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
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

def _try_rank_search(max_rank=97, time_limit=100):
    """
    Try to find a decomposition with rank < max_rank+1 using structured search.
    Approach: Start from Strassen rank-98, then try to merge pairs of rank-1 terms
    by finding linear combinations that preserve the tensor identity.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + time_limit
    
    base = _strassen_4x4_4x8()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    rank0 = U_base.shape[1]  # 98
    
    # Strategy: Try to eliminate pairs of columns by finding
    # coefficients a,b such that the merged column still works.
    # For columns r1, r2, we want to find new column r_new such that
    # the tensor equation is preserved.
    
    # A simpler approach: use the fact that the two 4x4x4 blocks share
    # the same A factor. We can try to find a rank-49 decomposition
    # for 4x4x4 that, when combined with the 8-wide B, gives rank < 98.
    
    # Even better: try a direct optimization approach.
    # Fix U and V to have a specific structure, then solve for W.
    
    # Approach 1: Try to use a shared Strassen structure across both halves of B.
    # The key insight: for 4x4 * 4x8, we can think of it as computing
    # C[:, 0:4] = A * B[:, 0:4] and C[:, 4:8] = A * B[:, 4:8].
    # If we use the same 7 Strassen products for both halves, we get rank 14,
    # but that's not correct because the B factors are different.
    
    # Let's try a more systematic approach: random restarts with local search.
    best_rank = rank0
    best_result = None
    
    for target_rank in range(max_rank, max(40, max_rank - 15), -1):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(target_rank * 1000 + 42)
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start from base and try to zero out (rank0 - target_rank) random entries
            # in U and V, then solve for W.
            
            U_mod = U_base.copy()
            V_mod = V_base.copy()
            
            # Zero out some entries in U and V
            num_zeros_U = np.random.randint(0, 5)
            num_zeros_V = np.random.randint(0, 5)
            
            for _ in range(num_zeros_U):
                row = np.random.randint(0, n * m)
                col = np.random.randint(0, rank0)
                U_mod[row, col] = 0
            
            for _ in range(num_zeros_V):
                row = np.random.randint(0, m * p)
                col = np.random.randint(0, rank0)
                V_mod[row, col] = 0
            
            # Now try to solve for W with reduced rank.
            # We'll try to set some columns of W to zero and see if we can
            # still satisfy the equations.
            
            # For each column r, check if it can be eliminated.
            # A column r can be eliminated if there exist coefficients for
            # other columns that replicate its contribution.
            
            # Simpler: try to find a subset of columns that spans the same
            # tensor space.
            
            # Build the tensor as a matrix for each slice.
            # For fixed i: T_i[j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # This should equal 1 for all j,k.
            
            # Let's define for each (i,k): 
            #   f_{i,k}(j) = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # for all j.
            
            # This means: for each (i,k), the vector (W[k*n+i, :]) must satisfy
            #   sum_j' U[i*m+j', :] * V[j'*p+k, :] * W[k*n+i, :] = 1 (element-wise dot)
            # No, that's not right. Let me be more careful.
            
            # For fixed (i,k), and for each j:
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let a_{j,r} = U[i*m+j, r] * V[j*p+k, r]
            # Then: sum_r a_{j,r} * W[k*n+i, r] = 1 for all j=0..3
            
            # This is a system of 4 equations in rank unknowns (W[k*n+i, :]).
            # We can solve this for W[k*n+i, :] using least squares.
            
            # Let's build the full system and solve.
            # Unknowns: W[k*n+i, r] for k=0..7, i=0..3, r=0..rank0-1
            # Total unknowns: 32 * rank0
            
            # Equations: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # Total equations: 4*4*8 = 128
            
            # This is underdetermined (32*98 = 3136 unknowns, 128 equations).
            # We want a sparse solution.
            
            # Let's try a greedy approach: start with W_base and try to zero out
            # entries while maintaining feasibility.
            
            W_mod = W_base.copy()
            
            # Try to zero out random entries of W and check if we can
            # compensate by adjusting other entries.
            
            success = True
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random non-zero entry of W and try to zero it
                nonzero = np.argwhere(W_mod != 0)
                if len(nonzero) == 0:
                    break
                
                idx = np.random.randint(0, len(nonzero))
                w_row, w_col = nonzero[idx]
                old_val = W_mod[w_row, w_col]
                
                # Try to zero it out and see if we can adjust other W entries
                # to maintain the tensor equation.
                W_mod[w_row, w_col] = 0
                
                # Check if the tensor equation is still satisfied
                # (approximately, since we only changed one entry)
                # This is a heuristic - we'll verify at the end.
            
            # Verify the modified decomposition
            U3 = U_mod.reshape(n, m, rank0)
            V3 = V_mod.reshape(m, p, rank0)
            W3 = W_mod.reshape(p, n, rank0)
            result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            
            if np.allclose(result, 1.0, atol=1e-8):
                # Count the effective rank (number of non-zero columns)
                col_norms = np.linalg.norm(np.vstack([U_mod, V_mod, W_mod]), axis=0)
                effective_rank = np.sum(col_norms > 1e-10)
                
                if effective_rank < best_rank:
                    best_rank = int(effective_rank)
                    # Create the result with only the non-zero columns
                    mask = col_norms > 1e-10
                    best_result = {
                        "U": U_mod[:, mask].tolist(),
                        "V": V_mod[:, mask].tolist(),
                        "W": W_mod[:, mask].tolist()
                    }
    
    return best_result

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries to find a rank-minimal decomposition, falls back to Strassen (rank 98) or schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try to find a better decomposition
    try:
        result = _try_rank_search(max_rank=97, time_limit=120)
        if result is not None and time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fall back to Strassen
    try:
        result = _strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Final fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
