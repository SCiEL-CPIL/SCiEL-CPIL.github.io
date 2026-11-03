import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin

    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
        rank = U_mat.shape[1]
        U_m = U_mat.reshape(n, m, rank)
        V_m = V_mat.reshape(m, p, rank)
        W_m = W_mat.reshape(p, n, rank)
        # Compute tensor contraction
        result = np.einsum('ijr,jkr,kir->ijk', U_m, V_m, W_m)
        return np.allclose(result, 1.0, atol=tol)

    def build_strassen():
        """Build the standard rank-28 Strassen decomposition."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 7 steps for 2x2*2x2->2x2
        # Indices: local (row, col) -> flat idx = row*2+col
        strassen_steps = [
            {'u': {(0,0):1, (1,1):1}, 'v': {(0,0):1, (1,1):1}, 'w': {(0,0):1, (1,1):1}},
            {'u': {(1,0):1, (1,1):1}, 'v': {(0,0):1}, 'w': {(1,0):1, (1,1):-1}},
            {'u': {(0,0):1}, 'v': {(0,1):1, (1,1):-1}, 'w': {(0,1):1, (1,1):1}},
            {'u': {(1,1):1}, 'v': {(1,0):1, (0,0):-1}, 'w': {(0,0):1, (1,0):1}},
            {'u': {(0,0):1, (0,1):1}, 'v': {(1,1):1}, 'w': {(0,0):-1, (0,1):1}},
            {'u': {(1,0):1, (0,0):-1}, 'v': {(0,0):1, (0,1):1}, 'w': {(1,1):1}},
            {'u': {(0,1):1, (1,1):-1}, 'v': {(1,0):1, (1,1):1}, 'w': {(0,0):1}}
        ]
        
        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s
                for (lr, lc), coeff in step['u'].items():
                    gi = lr
                    gj = a_col_start + lc
                    U[gi * m + gj, col] += coeff
                for (lr, lc), coeff in step['v'].items():
                    gi = b_row_start + lr
                    gj = b_col_start + lc
                    V[gi * p + gj, col] += coeff
                for (lr, lc), coeff in step['w'].items():
                    gk = c_col_start + lc
                    gi = lr
                    W[gk * n + gi, col] += coeff
        
        # 4 blocks: A cols {0,1} or {2,3}, B rows {0,1} or {2,3}, B cols {0,1} or {2,3}
        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)
        
        return U, V, W

    U, V, W = build_strassen()
    
    if not verify_fast(U, V, W):
        # Fallback: schoolbook rank-32
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 28
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a rank-27 decomposition by structured search
    # Key idea: In the 4-block Strassen, blocks 0 and 2 both use A cols {0,1}
    # and blocks 1 and 3 both use A cols {2,3}. Also blocks 0 and 1 share B rows {0,1},
    # blocks 2 and 3 share B rows {2,3}.
    # 
    # We can try to share the Strassen "sum" terms between blocks that share
    # the same A sub-block or B sub-block.
    #
    # Alternative approach: try all pairs of blocks and see if we can merge
    # one Strassen multiplication by linearity.
    
    # Try: for each pair of blocks, attempt to combine their Strassen decompositions
    # by finding if one product can be shared.
    # 
    # A more systematic approach: try to find a rank-27 decomposition by
    # perturbing the rank-28 and checking if one column becomes redundant.
    
    if time.time() < deadline - 40:
        # Try structured rank-27: merge products between adjacent blocks
        # Blocks: 0=(A01,B01->C01), 1=(A01,B01->C23), 2=(A23,B23->C01), 3=(A23,B23->C23)
        # 
        # Idea: For blocks 0 and 1, they share the same A block (cols 0-1) and same B rows (0-1).
        # The only difference is B cols (0-1 vs 2-3) and C cols (0-1 vs 2-3).
        # In Strassen, M1=(A11+A22)(B11+B22), M5=(A11+A12)B22, etc.
        # If we could combine B11+B22 terms... but B blocks are different.
        #
        # Let's try a different structured approach: 
        # Use the fact that for 2x4*4x4, we can think of it as:
        # C = A_left * B_top + A_right * B_bottom
        # where A_left = A[:,0:2] (2x2), A_right = A[:,2:4] (2x2)
        # B_top = B[0:2,:] (2x4), B_bottom = B[2:4,:] (2x4)
        #
        # C = A_left * B_top + A_right * B_bottom
        #
        # Each term is 2x2 * 2x4 -> 2x4, which can be done with 2 Strassen 2x2*2x2.
        # Total: 4 Strassen decompositions = 28. Same as before.
        #
        # To go lower, we need a fundamentally different approach.
        # Let's try a randomized search for rank-27.
        
        # Approach: Start with rank-28, try to set one column to be a linear
        # combination of others and adjust.
        # 
        # Actually, let's try a more direct approach: search for a rank-27
        # decomposition using the structure of the problem.
        #
        # The tensor T[i,j,k] = 1 for all i,j,k (32 entries).
        # We need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k).
        #
        # Let's try to find if there's a rank-27 solution by checking
        # if the 28 Strassen terms have a linear dependency in the tensor space.
        
        # Compute the "tensor" of each Strassen term
        # term_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Each term_r is a 2x4x4 tensor (32 values)
        # We need sum_r term_r = ones(2,4,4)
        
        # Check if any term is a linear combination of others
        # (in the 32-dim tensor space)
        # If so, we can potentially eliminate it.
        
        tensors = np.zeros((28, 32))
        for r in range(28):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensors[r, i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check rank of the 28x32 matrix
        # If rank < 28, there's a dependency
        rank_tensors = np.linalg.matrix_rank(tensors, tol=1e-9)
        
        if rank_tensors < 28:
            # There's a linear dependency! Try to exploit it.
            # Find null space vector
            _, s, vh = np.linalg.svd(tensors, full_matrices=True)
            null_vec = vh[-1]  # last row of V^H is null space
            
            # If null_vec has exactly one non-zero entry, we can drop that column
            # Otherwise, it's a more complex dependency
            nz = np.nonzero(np.abs(null_vec) > 1e-6)[0]
            if len(nz) == 1:
                drop_col = nz[0]
                # Simply drop this column
                mask = [c for c in range(28) if c != drop_col]
                U27 = U[:, mask]
                V27 = V[:, mask]
                W27 = W[:, mask]
                if verify_fast(U27, V27, W27):
                    best_rank = 27
                    best_U, best_V, best_W = U27, V27, W27
        
        # Try more aggressive: rank-26
        if best_rank > 26 and time.time() < deadline - 30:
            # Try dropping 2 columns
            for c1 in range(28):
                if time.time() > deadline - 25:
                    break
                for c2 in range(c1+1, 28):
                    if time.time() > deadline - 25:
                        break
                    mask = [c for c in range(28) if c != c1 and c != c2]
                    U26 = U[:, mask]
                    V26 = V[:, mask]
                    W26 = W[:, mask]
                    if verify_fast(U26, V26, W26):
                        best_rank = 26
                        best_U, best_V, best_W = U26, V26, W26
                        break
                if best_rank == 26:
                    break
    
    # Try a completely different structured approach for lower rank
    if best_rank > 25 and time.time() < deadline - 20:
        # Try to construct a rank-25 or lower decomposition directly
        # using a different block decomposition strategy.
        #
        # Alternative: use Winograd's approach or other fast algorithms.
        # For now, let's try a greedy approach: build the decomposition
        # one term at a time, always choosing the term that covers the most
        # uncovered tensor entries.
        
        # Actually, for exact half-integer decompositions, let's try
        # a structured search with small coefficients.
        
        # Try rank 27 with a different structure:
        # Use 3 full Strassen blocks (21 products) + 6 additional products
        # to cover the remaining interactions.
        
        # This is getting complex. Let's just return what we have.
        pass
    
    def to_half_integer_list(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-9:
                    new_row.append(rounded)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_out = to_half_integer_list(best_U)
    V_out = to_half_integer_list(best_V)
    W_out = to_half_integer_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
