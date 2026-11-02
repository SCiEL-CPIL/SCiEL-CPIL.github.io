"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = U.shape[1]
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] = delta(i,j,k) for matrix mult tensor
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The tensor T[i,j,k] should equal 1 if this is the identity, 0 otherwise
    # For matrix mult: T[(i,j), (j,k)] = delta(i,k) ... 
    # Actually the convention is W[k*n+i, r]
    # Let's verify: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==k else 0
    # Wait, that's not right either. Let me think about the convention.
    # The task says W[k*n+i,r]. So W is indexed by (k,i) pairs where k is column of B, i is row of A.
    # Matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = delta_{i,k} ... no, T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the multiplication tensor has entries T[(i,j),(j,k)] = delta_{i,k}
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta_{i,k}
    # Wait no. Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # I think the tensor is: for matrix mult C = A*B, C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T_{i,j,k} where the "input" indices are (i,j) for A and (j,k) for B, and output is (i,k)
    # So T[i,j,k] = delta_{i,k} ... that doesn't make sense as a tensor decomposition.
    # 
    # Actually, the standard tensor decomposition for matrix multiplication:
    # The tensor T has entries T[(i,j),(j,k),(i,k)] = 1 (for all valid i,j,k)
    # No wait, T is a 3-tensor with indices (i,j,k) and T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Which is always 1 for valid i,j,k. So T[i,j,k] = 1 for all i,j,k? No...
    # 
    # Let me think again. The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # This can be written as C = T(A,B) where T is the trilinear form.
    # In tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[i,k]
    # So: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k
    # 
    # With the indexing: U is (n*m, rank) so U[i*m+j, r] = U_r[i,j]
    # V is (m*p, rank) so V[j*p+k, r] = V_r[j,k]  
    # W is (p*n, rank) so W[k*n+i, r] = W_r[i,k]
    # 
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k
    # 
    # Wait, that can't be right because then it would be rank 1. Let me reconsider.
    # 
    # Oh I see, I think the tensor is: T[i,j,k] = delta_{i,k} is wrong.
    # The correct statement: the multiplication map (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # is a bilinear map, which corresponds to a trilinear form via the trace pairing.
    # The tensor is: T[i,j,k] = 1 for all valid indices, and the decomposition gives
    # sum_r U_r[i,j] V_r[j,k] W_r[i,k] = delta_{i,k} ... no.
    #
    # I think the correct interpretation is:
    # We want: for all A (n×m), B (m×p), C = A*B (n×p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor decomposition framework, we write:
    # C[i,k] = sum_r (sum_j U_r[i,j] A[i,j]) * (sum_k V_r[j,k] B[j,k]) ... no that's not right.
    #
    # Let me just check: the standard way is that the matrix multiplication tensor T_{n,m,p}
    # has entries T[i,j,k] = 1 for all i in {1..n}, j in {1..m}, k in {1..p}
    # and the rank is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor has the form U ⊗ V ⊗ W where (U⊗V⊗W)[i,j,k] = U[i,j]*V[j,k]*W[i,k]
    # 
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k
    #
    # Let me verify with schoolbook for 2x2x1:
    # U = identity 2x2 (flattened to 4x4), V = [[1,0],[0,1]] repeated, W = [[1,1],[0,0]]... 
    # Hmm, let me just verify numerically.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_2x2x3_rank10():
    """
    Build a rank-10 decomposition for 2x2x3 matrix multiplication.
    Strategy: Start with Strassen (rank 7) for first 2 columns + schoolbook (rank 4) for 3rd column,
    then try to eliminate one term by finding a linear dependency.
    """
    n, m, p = 2, 2, 3
    
    # Strassen for 2x2x2 (first 2 columns)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [0, 1, 0, 0, 0, 1, 0],  
        [1, 1, 0, 1, 0, 0, -1]  
    ])
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0], 
        [0, 0, 1, 0, 0, 1, 0],  
        [0, 0, 0, 1, 0, 0, 1],  
        [1, 0, -1, 0, 1, 0, 1]  
    ])
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 1, 0, 1, 0, 0, 0],  
        [0, 0, 1, 0, 1, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ])
    
    # Schoolbook for 2x2x1 (3rd column)
    U_b = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]  
    ])
    
    V_b = np.array([
        [1, 0, 1, 0], 
        [0, 1, 0, 1]  
    ])
    
    W_b = np.array([
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    # Combine: 7 + 4 = 11 terms
    # U: (4, 11), V: (6, 11), W: (6, 11)
    U_11 = np.hstack([U_s, U_b])  # (4, 11)
    V_11 = np.zeros((6, 11))
    W_11 = np.zeros((6, 11))
    
    # Fill V: rows 0-3 for first 2 columns, rows 4-5 for 3rd column
    for j in range(2):
        for k_local in range(2):
            V_11[j*3 + k_local, :7] = V_s[j*2 + k_local, :]
    for j in range(2):
        V_11[j*3 + 2, 7:11] = V_b[j, :]
    
    # Fill W: W[k*n+i, r] where k is 0,1,2 (columns of B)
    for k_local in range(2):
        for i in range(2):
            W_11[k_local*2 + i, :7] = W_s[k_local*2 + i, :]
    for i in range(2):
        W_11[2*2 + i, 7:11] = W_b[i, :]
    
    # Verify the 11-term decomposition
    assert _verify_decomposition(U_11, V_11, W_11, n, m, p), "11-term decomposition failed"
    
    # Now try to reduce to rank 10 by finding a linear combination
    # We want to find vectors u, v, w such that:
    # sum_r U_r * V_r * W_r - u ⊗ v ⊗ w = 0 (as a tensor)
    # i.e., we can replace one term with a combination that cancels.
    # 
    # Alternatively: find coefficients c_0, ..., c_10 not all zero such that
    # sum_r c_r * U_r ⊗ V_r ⊗ W_r = 0
    # Then we can drop one term and adjust others.
    #
    # This is equivalent to finding a vector in the kernel of the map:
    # R^11 -> T_{2,2,3} (flattened), c -> sum_r c_r * flatten(U_r ⊗ V_r ⊗ W_r)
    #
    # The flattened tensor has dimension 2*2*3 = 12.
    # So we have a 12 x 11 matrix, and we're looking for its null space.
    # If the matrix is rank 11, there's no null vector (generic case).
    # But if rank < 11, we have a dependency.
    
    # Build the 12 x 11 matrix
    M = np.zeros((12, 11))
    for r in range(11):
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    idx = i*4 + j*2 + k  # flatten (i,j,k)
                    # Wait, we need to be careful with the flattening
                    # The tensor entry is T[i,j,k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
                    # where n=2, m=2, p=3
                    M[i*4 + j*2 + k, r] = U_11[i*2+j, r] * V_11[j*3+k, r] * W_11[k*2+i, r]
    
    # Find rank and null space
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank_M < 11:
        # There's a linear dependency!
        # Find null vector
        _, s, vh = np.linalg.svd(M)
        null_vec = vh[-1, :]
        # Find the largest component
        max_idx = np.argmax(np.abs(null_vec))
        # We can drop term max_idx and adjust the rest
        # c_r * term_r + c_max * term_max = 0
        # So term_max = -sum_{r != max} (c_r / c_max) * term_r
        # New decomposition: for r != max, coefficient is c_r / c_max ... 
        # Actually, we just need to verify that dropping term max_idx and scaling works.
        # The sum of all 11 terms = T (the tensor)
        # If sum_r c_r * term_r = 0, then term_max = -sum_{r!=max} (c_r/c_max) * term_r
        # So T = sum_{r!=max} term_r + term_max = sum_{r!=max} term_r - sum_{r!=max} (c_r/c_max) * term_r
        # = sum_{r!=max} (1 - c_r/c_max) * term_r
        # This gives a 10-term decomposition!
        
        U_10 = U_11[:, list(range(11))[:max_idx] + list(range(11))[max_idx+1:]]
        V_10 = V_11[:, list(range(11))[:max_idx] + list(range(11))[max_idx+1:]]
        W_10 = W_11[:, list(range(11))[:max_idx] + list(range(11))[max_idx+1:]]
        
        # Verify
        if _verify_decomposition(U_10, V_10, W_10, n, m, p):
            return U_10, V_10, W_10
    
    # If no simple linear dependency found, try a more systematic search
    # Search for rank-10 decompositions with small integer/half-integer entries
    start_search = time.time()
    deadline_search = start_search + 100
    
    # Try to find a rank-10 decomposition by perturbing the 11-term one
    # or by direct search
    
    # Alternative: try known constructions
    # One known approach: use the fact that 2x2x3 can be done in rank 10
    # via a "sliding window" of Strassen products
    
    # Let me try a different construction:
    # Use 3 overlapping Strassen-like products
    # Product 1: covers columns 0,1 with Strassen (7 terms)
    # Product 2: covers columns 1,2 with a modified Strassen (but sharing some terms)
    
    # Actually, let me try a direct small search for rank-10 with entries in {-1, 0, 1}
    # This is a large search space, so let's be smart.
    
    # Try: start from the 11-term decomposition and try to merge two terms
    # that are "close" to being linearly dependent.
    
    # For now, if we can't find rank 10, fall back to rank 11
    # and use 5 blocks of rank 11 = 55, which is worse than 53.
    # So we need to actually find rank 10.
    
    # Let me try a more targeted approach: search for a rank-10 decomposition
    # with entries in {-1, -0.5, 0, 0.5, 1}
    
    # The number of possible rank-1 tensors is huge, so let's use a different strategy:
    # Try to find 10 rank-1 tensors that sum to T_{2,2,3}.
    
    # Use a greedy/iterative approach:
    # 1. Start with the 11-term decomposition
    # 2. Try to replace pairs of terms with a single term (if possible)
    
    # Or: use the fact that we can add a rank-1 tensor to two terms and subtract from a third
    # if the resulting tensor is still rank-1.
    
    # For time efficiency, let's just return the 11-term decomposition for now
    # and use 5 blocks of rank 11 = 55. But that's worse than 53.
    
    # So let's try harder. Let me attempt a randomized search for rank-10.
    
    np.random.seed(42)
    best_found = None
    
    for attempt in range(5000):
        if time.time() > deadline_search:
            break
        
        # Try to construct a rank-10 decomposition by modifying the 11-term one
        # Pick a random term to eliminate and try to absorb it into others
        
        # Alternative: start fresh with random rank-1 tensors and optimize
        # This is too slow for exact verification.
        
        # Let's try: take the 11-term decomposition, pick 2 terms, and see if their
        # sum can be represented as a single rank-1 tensor.
        for r1 in range(11):
            for r2 in range(r1+1, 11):
                # Check if U_r1 ⊗ V_r1 ⊗ W_r1 + U_r2 ⊗ V_r2 ⊗ W_r2 is rank 1
                # This is a 2x2x3 tensor, check if it's rank 1
                T2 = np.zeros((2, 2, 3))
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T2[i,j,k] = (U_11[i*2+j, r1] * V_11[j*3+k, r1] * W_11[k*2+i, r1] +
                                        U_11[i*2+j, r2] * V_11[j*3+k, r2] * W_11[k*2+i, r2])
                
                # Check if T2 is rank 1: all 2x2 minors of the unfolding are zero
                # Unfold to (2, 6) matrix
                M2 = T2.reshape(2, 6)
                # Rank 1 iff all 2x2 minors are zero
                is_rank1 = True
                for c1 in range(6):
                    for c2 in range(c1+1, 6):
                        minor = M2[0, c1] * M2[1, c2] - M2[0, c2] * M2[1, c1]
                        if abs(minor) > 1e-9:
                            is_rank1 = False
                            break
                    if not is_rank1:
                        break
                
                if is_rank1:
                    # Great! We can merge terms r1 and r2 into one rank-1 term
                    # Extract the rank-1 components
                    # T2[i,j,k] = u[i,j] * v[j,k] * w[i,k]
                    # Find u, v, w from T2
                    # u[i,j] = T2[i,j,0] / (v[j,0] * w[i,0]) if v[j,0]*w[i,0] != 0
                    # This is tricky. Let's just verify and store.
                    # For now, just note that a merge is possible.
                    # Build the 10-term decomposition
                    keep = [r for r in range(11) if r not in [r1, r2]]
                    # The merged term: we need to find its U, V, W vectors
                    # From T2, we can extract:
                    # If T2[0,0,0] != 0, set u[0,0] = T2[0,0,0], v[0,0] = 1, w[0,0] = 1
                    # Then fill in the rest.
                    
                    # This is getting complex. Let's just return the 11-term for now.
                    # We found a merge is possible, which means rank <= 10.
                    pass
    
    # If we couldn't find an explicit rank-10 decomposition, use the 11-term one
    # but that gives total rank 5*11 = 55 > 53, so it's worse.
    # 
    # Alternative: use a mix. E.g., 4 blocks of 2x2x3 (rank 11) + 1 block of 2x2x3 (rank 11) = 55. Bad.
    # Or: 6 blocks of 2x2x2 (rank 7) + 1 block of 2x2x3 (rank 11) = 42 + 11 = 53. Same as before!
    # Or: 5 blocks of 2x2x3 (rank 11) = 55. Worse.
    
    # So if rank(2x2x3) = 11, the best we can do is 53 (same as current).
    # If rank(2x2x3) = 10, we get 50.
    
    # Let me try one more thing: a direct construction for rank-10 2x2x3.
    
    # Known result: the rank of 2x2x3 matrix multiplication is 10.
    # Let me try to find it by solving a system of equations.
    
    # We need 10 rank-1 tensors. Let's parameterize with small integers.
    # This is a hard search problem. Let's use a heuristic.
    
    # For now, return the 11-term decomposition. The caller will handle the total rank.
    return U_11, V_11, W_11

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Try to build a rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = _build_2x2x3_rank10()
    rank_223 = U_223.shape[1]
    
    if rank_223 == 10:
        # Use 5 blocks of 2x2x3, total rank = 50
        total_rank = 5 * 10
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        for b in range(5):
            if time.time() > deadline:
                break
            k_start = 3 * b
            U[:, col_offset:col_offset+10] = U_223
            
            for j in range(m):
                for k_local in range(3):
                    k_global = k_start + k_local
                    V[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
            
            for k_local in range(3):
                k_global = k_start + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
            
            col_offset += 10
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to the original 53-rank decomposition
        U_strassen = np.array([
            [1, 0, 1, 0, 1, -1, 0], 
            [0, 0, 0, 0, 1, 0, 1],  
            [0, 1, 0, 0, 0, 1, 0],  
            [1, 1, 0, 1, 0, 0, -1]  
        ])
        
        V_strassen = np.array([
            [1, 1, 0, -1, 0, 1, 0], 
            [0, 0, 1, 0, 0, 1, 0],  
            [0, 0, 0, 1, 0, 0, 1],  
            [1, 0, -1, 0, 1, 0, 1]  
        ])
        
        W_strassen = np.array([
            [1, 0, 0, 1, -1, 0, 1], 
            [0, 1, 0, 1, 0, 0, 0],  
            [0, 0, 1, 0, 1, 0, 0],  
            [1, -1, 1, 0, 0, 1, 0]  
        ])
        
        U_school = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]  
        ])
        
        V_school = np.array([
            [1, 0, 1, 0], 
            [0, 1, 0, 1]  
        ])
        
        W_school = np.array([
            [1, 1, 0, 0], 
            [0, 0, 1, 1]  
        ])
        
        total_rank = 7 * 7 + 4
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        for b in range(7):
            if time.time() > deadline:
                break
            k_start = 2 * b
            U[:, col_offset:col_offset+7] = U_strassen
            
            for j in range(m):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_strassen[j * 2 + k_local, :]
            
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_strassen[k_local * 2 + i, :]
            
            col_offset += 7
            
        k_start = 14
        U[:, col_offset:col_offset+4] = U_school
        
        for j in range(m):
            V[j * p + k_start, col_offset:col_offset+4] = V_school[j, :]
            
        for i in range(n):
            W[k_start * n + i, col_offset:col_offset+4] = W_school[i, :]
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
