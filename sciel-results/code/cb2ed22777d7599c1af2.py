"""
Matrix multiplication 2x4x6: Rank-42 Strassen-block construction.
This program constructs a rank-42 exact tensor decomposition for multiplying
a 2x4 matrix by a 4x6 matrix. It uses the Strassen 7-term algorithm for each
2x2x2 block, tiled across the dimensions.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    t0 = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Strassen 7-term pattern for 2x2x2 blocks
    # Local 2x2x2: A has 4 entries (a00,a01,a10,a11), B has 4 (b00,b01,b10,b11), C has 4 (c00,c01,c10,c11)
    # C = A*B, so C(i,k) = sum_j A(i,j)B(j,k)
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00       -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)       -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)       -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11       -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7

    def add_block(i0, j0, k0, offset):
        # A indices: global row i*m + j
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0+1) * m + j0
        a11 = (i0+1) * m + j0 + 1

        # B indices: global row j*p + k
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0+1) * p + k0
        b11 = (j0+1) * p + k0 + 1

        # W indices: W[k*n + i, r]
        w00 = k0 * n + i0
        w01 = (k0+1) * n + i0
        w10 = k0 * n + i0 + 1
        w11 = (k0+1) * n + i0 + 1

        # M1
        U[a00, offset] = 1.0; U[a11, offset] = 1.0
        V[b00, offset] = 1.0; V[b11, offset] = 1.0
        W[w00, offset] = 1.0; W[w11, offset] = 1.0

        # M2
        U[a10, offset+1] = 1.0; U[a11, offset+1] = 1.0
        V[b00, offset+1] = 1.0
        W[w10, offset+1] = 1.0; W[w11, offset+1] = -1.0

        # M3
        U[a00, offset+2] = 1.0
        V[b01, offset+2] = 1.0; V[b11, offset+2] = -1.0
        W[w01, offset+2] = 1.0; W[w11, offset+2] = 1.0

        # M4
        U[a11, offset+3] = 1.0
        V[b10, offset+3] = 1.0; V[b00, offset+3] = -1.0
        W[w00, offset+3] = 1.0; W[w10, offset+3] = 1.0

        # M5
        U[a00, offset+4] = 1.0; U[a01, offset+4] = 1.0
        V[b11, offset+4] = 1.0
        W[w00, offset+4] = -1.0; W[w01, offset+4] = 1.0

        # M6
        U[a10, offset+5] = 1.0; U[a00, offset+5] = -1.0
        V[b00, offset+5] = 1.0; V[b01, offset+5] = 1.0
        W[w11, offset+5] = 1.0

        # M7
        U[a01, offset+6] = 1.0; U[a11, offset+6] = -1.0
        V[b10, offset+6] = 1.0; V[b11, offset+6] = 1.0
        W[w00, offset+6] = 1.0

    # n=2, m=4, p=6
    # 2x2x2 blocks: i_start in {0}, j_start in {0,2}, k_start in {0,2,4}
    # That's 1 * 2 * 3 = 6 blocks, each with 7 terms = 42
    offset = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                add_block(i0, j0, k0, offset)
                offset += 7

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        ones = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ones.add((i*m+j, j*p+k, k*n+i))
        
        # Efficient verification:
        # Compute the tensor T and check against expected
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can do this by iterating over r and accumulating to a 3D array
        T = np.zeros((nm, mp, pn), dtype=float)
        for r in range(r):
            # Outer product of U[:,r], V[:,r], W[:,r]
            # This is equivalent to np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            # But for small sizes, direct loop or np.multiply.outer is fine
            uv = np.outer(U[:, r], V[:, r])
            T += np.multiply.outer(uv, W[:, r])
            
        # Check against expected
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    if abs(T[a,b,c] - 1.0) > tol:
                        return False
                    # Check that other entries are 0?
                    # This is expensive. We can sample or check all.
                    # Given the structure, if the 48 ones are correct and the rank is low,
                    # it's likely correct. But we should check zeros.
        
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    i_c = c % n
                    k_c = c // n
                    j_b = b // p
                    k_b = b % p
                    if not (i_a == i_c and j_a == j_b and k_b == k_c):
                        if abs(T[a,b,c]) > tol:
                            return False
        return True

    # Verify base solution
    if not verify(U, V, W):
        # Fallback: schoolbook rank 48
        rank0 = n * m * p
        U0 = np.zeros((nm, rank0), dtype=float)
        V0 = np.zeros((mp, rank0), dtype=float)
        W0 = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U0[i*m+j, col] = 1.0
                    V0[j*p+k, col] = 1.0
                    W0[k*n+i, col] = 1.0
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Attempt to find a rank-41 decomposition by pruning one term
    # and solving for the new coefficients.
    # The tensor T is fixed. We have 42 terms.
    # If we drop term t, we need to find c_r for r != t such that sum_{r != t} c_r T_r = T.
    # This is a linear system A x = b where A is (48 x 41) and b is the vector of 48 ones (flattened).
    
    # Precompute the tensor contributions of each term as columns in a matrix
    # M is (48, 42) where M[:, t] is the flattened tensor of term t.
    # The target tensor has 1s at specific indices and 0s elsewhere.
    # We only need to satisfy the 48 "1" equations. The "0" equations are implicit in the structure?
    # No, we must ensure the "0" entries remain 0.
    # However, if we find a solution for the "1" entries, we must verify the "0" entries.
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target_indices.append((a, b, c))
                
    num_targets = len(target_indices)
    
    # Build the matrix of term contributions at target indices
    A_matrix = np.zeros((num_targets, rank), dtype=float)
    for t in range(rank):
        for idx, (a, b, c) in enumerate(target_indices):
            A_matrix[idx, t] = U[a, t] * V[b, t] * W[c, t]
            
    # Target vector is all 1s
    b_vector = np.ones(num_targets, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try dropping each term
    for drop_idx in range(rank):
        if time.time() > deadline - 10:
            break
            
        # Select columns except drop_idx
        mask = [i for i in range(rank) if i != drop_idx]
        A_reduced = A_matrix[:, mask]
        
        # Solve A_reduced x = b_vector
        try:
            x, residuals, rank_reduced, s = np.linalg.lstsq(A_reduced, b_vector, rcond=None)
        except:
            continue
            
        # Check residual
        if len(residuals) > 0 and residuals[0] < 1e-8:
            # Verify that the solution produces 0s in the non-target positions
            # And that the coefficients are "nice" (optional, but good for stability)
            # Construct new U, V, W
            new_U = np.zeros((nm, rank-1), dtype=float)
            new_V = np.zeros((mp, rank-1), dtype=float)
            new_W = np.zeros((pn, rank-1), dtype=float)
            
            for new_idx, orig_idx in enumerate(mask):
                new_U[:, new_idx] = U[:, orig_idx] * x[new_idx]
                new_V[:, new_idx] = V[:, orig_idx] * x[new_idx]
                new_W[:, new_idx] = W[:, orig_idx] * x[new_idx]
                
            # Verify full tensor
            if verify(new_U, new_V, new_W):
                best_rank = rank - 1
                best_U, best_V, best_W = new_U, new_V, new_W
                break # Found a rank-41 solution, stop searching for rank-41
                
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
