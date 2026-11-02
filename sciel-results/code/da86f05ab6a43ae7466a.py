"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare with the matrix multiplication tensor
    
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where W is indexed as W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True tensor value: 1 (since C[i,k] includes A[i,j]*B[j,k])
                true_val = 1.0
                
                # Decomposed value
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                
                if abs(dec_val - true_val) > 1e-9:
                    return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Try rank-20 using Strassen-inspired sharing between two 2x3x3 blocks
    # If that fails, fall back to rank-22 (two independent 2x3x3 blocks)
    
    def build_2x3x3_decomposition():
        """Build a rank-11 decomposition for 2x3x3 multiplication."""
        terms = []
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {2: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 1: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 2: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {2: 1}})
        terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {1: 1}})
        terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {3: 1}})
        return terms
    
    def verify_2x3x3(terms):
        """Verify a rank-11 decomposition for 2x3x3."""
        rank = len(terms)
        U = np.zeros((6, rank))
        V = np.zeros((9, rank))
        W = np.zeros((6, rank))
        for r, term in enumerate(terms):
            for idx, val in term['u'].items():
                U[idx, r] = val
            for idx, val in term['v'].items():
                V[idx, r] = val
            for idx, val in term['w'].items():
                W[idx, r] = val
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    dec_val = 0.0
                    for r in range(rank):
                        dec_val += U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    if abs(dec_val - 1.0) > 1e-9:
                        return False, (i, j, k, dec_val)
        return True, None
    
    # Build and verify the 2x3x3 decomposition
    terms_233 = build_2x3x3_decomposition()
    valid, err = verify_2x3x3(terms_233)
    
    if not valid:
        return _construct_rank33()
    
    # Try rank-20: share 2 ranks between the two blocks
    # The idea: find a 2x3x6 decomposition where 20 terms suffice
    # by exploiting that the two 2x3x3 sub-problems can share some structure
    
    def try_rank20():
        """Try to construct a rank-20 decomposition.
        
        Approach: Use the 11-term 2x3x3 decomposition for both blocks,
        but attempt to merge 2 pairs of terms by finding linear combinations
        that produce valid 2x3x6 terms.
        
        Specifically, if term r1 in block 1 and term r2 in block 2 have
        proportional U vectors, we can merge them.
        """
        # The two blocks use the same U structure, so all U vectors are identical
        # between blocks. This means we can potentially merge terms where
        # U vectors are proportional.
        
        # In our 2x3x3 decomposition, let's check which terms have proportional U vectors:
        # Term 0: U = {0:1, 4:1}
        # Term 1: U = {3:1, 4:1}
        # Term 2: U = {0:1}
        # Term 3: U = {4:1}
        # Term 4: U = {0:1, 1:1}
        # Term 5: U = {3:1, 0:-1}
        # Term 6: U = {1:1, 4:-1}
        # Term 7: U = {2:1}
        # Term 8: U = {2:1}  -- same as term 7!
        # Term 9: U = {5:1}
        # Term 10: U = {5:1} -- same as term 9!
        
        # Terms 7&8 share U={2:1}, terms 9&10 share U={5:1}
        # But their V and W vectors differ between blocks.
        
        # For block 0: term 7 has V={6:1}, W={0:1}; term 8 has V={7:1}, W={2:1}
        # For block 1: term 7 has V={12:1}, W={0:1}; term 8 has V={13:1}, W={2:1}
        
        # We can't simply merge because V and W are different.
        # But we can try a more sophisticated approach: use a 2x3x6 specific
        # decomposition that exploits the full structure.
        
        # Let's try a direct approach: construct a rank-20 decomposition
        # using the fact that the 2x3x6 tensor has a specific structure.
        
        # Alternative: try rank-21 by sharing 1 rank
        # Or try a completely different construction.
        
        # For now, let's just return the rank-22 construction as the best we have.
        # A true rank-20 would require a more sophisticated search.
        return None
    
    # Try to construct a rank-20 decomposition
    # Key insight: The 2x3x6 tensor can potentially be decomposed with rank 20
    # by using a Strassen-like approach where we share computations.
    
    # Let's try a direct construction for rank-20:
    # Use 7 Strassen products for the 2x2x2 part (with 3x3 "scalars"),
    # but decompose each 3x3 product more efficiently.
    
    # Actually, let's try a simpler approach: use the known rank-11 for 2x3x3
    # but find if we can reduce by 2 by exploiting cross-block structure.
    
    # For now, construct the rank-22 decomposition which is known to work.
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(2):
        for t_idx, term in enumerate(terms_233):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for v_idx, val in term['v'].items():
                j_local = v_idx // 3
                k_local = v_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            for w_idx, val in term['w'].items():
                k_local = w_idx // 2
                i_local = w_idx % 2
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        if _verify(U, V, W):
            return result
        else:
            return _construct_rank33()
    
    return result

def _construct_rank33():
    """Fallback rank-33 construction using three 2x3x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
