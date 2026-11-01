"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, use 145s as safety margin
    
    n, m, p = 2, 4, 4
    # A is n x m = 2 x 4
    # B is m x p = 4 x 4
    # C is n x p = 2 x 4
    # A_flat index: i*m + k where i in [0,n), k in [0,m)
    # B_flat index: k*p + j where k in [0,m), j in [0,p)
    # C_flat index: j*n + i where i in [0,n), j in [0,p)  (W[k*n+i, r] convention: W index = j*n + i)
    
    # The tensor T has entries T[a, b, c] where:
    # a = i*m + k (A element at row i, col k)
    # b = k'*p + j (B element at row k', col j)
    # c = j*n + i (C element at row i, col j)
    # T[a, b, c] = 1 if k == k' (i.e., the A col index matches B row index), else 0
    
    # Let's build the full tensor to understand the structure
    # T is of shape (n*m, m*p, p*n) = (8, 16, 8)
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T_full[a_idx, b_idx, c_idx] = 1
    
    # Now let's find a minimal rank decomposition.
    # The tensor rank is the minimum number of rank-1 tensors that sum to T.
    
    # Strategy: Use the structure of the problem.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # This is a bilinear form in A and B for each output element C[i,j].
    
    # For each pair (i,j), we need: C[i,j] = sum_k A[i,k] B[k,j]
    # This can be written as: C[i,j] = A_i . B_j where A_i is the i-th row of A and B_j is the j-th column of B.
    # But A_i has m=4 elements and B_j has m=4 elements, so this is an inner product of 4-dimensional vectors.
    # The rank of a single inner product of two m-dimensional vectors is m (you need m rank-1 terms).
    # But we can share terms across different (i,j) pairs.
    
    # Total output elements: n*p = 2*4 = 8
    # If we did them independently, rank would be 8*4 = 32.
    # With Strassen on 2x2 blocks, we got 28.
    
    # Let me think about this differently.
    # We have A (2x4) and B (4x4).
    # Split A into A1 (2x2) and A2 (2x2), where A1 = A[:, :2] and A2 = A[:, 2:]
    # Split B into B1 (4x2) and B2 (4x2), where B1 = B[:, :2] and B2 = B[:, 2:]
    # Then C = A1 * B1 + A2 * B2, where C is 2x4.
    
    # C1 = A1 * B1 is 2x2, C2 = A2 * B2 is 2x2
    # C = [C1, C2]
    
    # For C1 = A1 * B1: 2x2 * 4x2
    # A1 is 2x2, B1 is 4x2.
    # C1[i,j] = sum_{k=0}^{1} A1[i,k] * B1[k,j] + sum_{k=2}^{3} A2[i,k-2] * B2[k-2,j]
    # Wait, that's not right. Let me re-examine.
    
    # A is 2x4: A[i,k] for i in {0,1}, k in {0,1,2,3}
    # B is 4x4: B[k,j] for k in {0,1,2,3}, j in {0,1,2,3}
    # C is 2x4: C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    # Split by k: C[i,j] = sum_{k=0}^{1} A[i,k] B[k,j] + sum_{k=2}^{3} A[i,k] B[k,j]
    # Let A_left = A[:, :2] (2x2), A_right = A[:, 2:] (2x2)
    # Let B_top = B[:2, :] (2x4), B_bottom = B[2:, :] (2x4)
    # Then C[i,j] = sum_{k=0}^{1} A_left[i,k] B_top[k,j] + sum_{k=0}^{1} A_right[i,k] B_bottom[k,j]
    
    # So C = A_left * B_top + A_right * B_bottom
    # Each term is a 2x2 * 2x4 multiplication.
    
    # For a 2x2 * 2x4 multiplication:
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i in {0,1}, j in {0,1,2,3}
    # This is 8 output elements, each requiring 2 products.
    # Total: 16 products if done naively.
    # But we can use Strassen-like tricks.
    
    # Actually, for 2x2 * 2x4, we can view it as two independent 2x2 * 2x2 multiplications
    # (one for columns 0-1 of B, one for columns 2-3 of B).
    # Each 2x2 * 2x2 can be done in 7 products (Strassen).
    # So 2x2 * 2x4 can be done in 7 + 7 = 14 products.
    
    # Total for C = A_left * B_top + A_right * B_bottom:
    # A_left * B_top: 14 products
    # A_right * B_bottom: 14 products
    # Total: 28 products. This matches the current solution.
    
    # Can we do better? The key question is whether we can share products between
    # A_left * B_top and A_right * B_bottom.
    
    # Let's think about the structure more carefully.
    # A_left = [[a00, a01], [a10, a11]]
    # A_right = [[a02, a03], [a12, a13]]
    # B_top = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # B_bottom = [[b20, b21, b22, b23], [b30, b31, b32, b33]]
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # C[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
    # C[0,3] = a00*b03 + a01*b13 + a02*b23 + a03*b33
    # C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
    # C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
    # C[1,2] = a10*b02 + a11*b12 + a12*b22 + a13*b32
    # C[1,3] = a10*b03 + a11*b13 + a12*b32 + a13*b33  (wait, a12*b32? No, a12*b22 + a13*b32)
    
    # Let me reconsider. Can we find a decomposition with fewer than 28 terms?
    
    # The tensor rank of matrix multiplication M_{2,4,4} is what we're looking for.
    # Known results:
    # - M_{2,2,2} has rank 7 (Strassen)
    # - M_{2,3,3} has rank 11 (Landsberg)
    # - M_{3,3,3} has rank between 19 and 23
    # - M_{2,4,4} - not sure of the exact rank
    
    # Let's try to find a rank-27 decomposition by searching.
    
    # First, let's build the rank-28 Strassen decomposition and verify it.
    rank_28 = 28
    U = np.zeros((n*m, rank_28))  # (8, 28)
    V = np.zeros((m*p, rank_28))  # (16, 28)
    W = np.zeros((p*n, rank_28))  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        # This handles a 2x2 * 2x2 sub-problem
        # A sub-block: rows a_row_base to a_row_base+1, cols a_col_base to a_col_base+1
        # B sub-block: rows b_row_base to b_row_base+1, cols b_col_base to b_col_base+1
        # C sub-block: rows c_row_base to c_row_base+1, cols c_col_base to c_col_base+1
        
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1
    
    # 4 blocks:
    # Block 1: A_left * B_top_left (A rows 0-1, cols 0-1; B rows 0-1, cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # Block 2: A_left * B_top_right (A rows 0-1, cols 0-1; B rows 0-1, cols 2-3)
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    # Block 3: A_right * B_bottom_left (A rows 0-1, cols 2-3; B rows 2-3, cols 0-1)
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    # Block 4: A_right * B_bottom_right (A rows 0-1, cols 2-3; B rows 2-3, cols 2-3)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Verify the rank-28 decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rng = np.random.RandomState(42)
        for _ in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback: return the rank-28 decomposition
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find a rank-27 decomposition.
    # Approach: The 28 rank-1 tensors may be linearly dependent.
    # Build the 28 vectors in R^(8*16*8) = R^1024 and check for dependencies.
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank_28):
        T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Check if T_computed matches T_full
    if not np.allclose(T_computed, T_full, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-27 decomposition by looking for a linear dependency
    # among the 28 rank-1 tensors.
    
    # Build the matrix whose columns are the 28 flattened rank-1 tensors
    vecs = []
    for r in range(rank_28):
        vec = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()
        vecs.append(vec)
    
    mat = np.array(vecs).T  # (1024, 28)
    
    # Find the rank of this matrix
    try:
        s_vals = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
        rank_mat = np.sum(s_vals > 1e-10)
        
        if rank_mat < 28:
            # There is a linear dependency
            # The null space is spanned by the last rows of Vt
            _, s_vals, Vt = np.linalg.svd(mat, full_matrices=False)
            null_vecs = Vt[rank_mat:]  # (28-rank_mat, 28)
            
            # For each null vector, try to eliminate one term
            for nv in null_vecs:
                if time.time() > deadline - 5:
                    break
                
                max_idx = np.argmax(np.abs(nv))
                max_val = nv[max_idx]
                
                if abs(max_val) > 1e-10:
                    # Try to eliminate term max_idx
                    U_sub = np.delete(U, max_idx, axis=1)
                    V_sub = np.delete(V, max_idx, axis=1)
                    W_sub = np.delete(W, max_idx, axis=1)
                    
                    T_sub = np.zeros((n*m, m*p, p*n))
                    for r in range(27):
                        T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                    
                    if np.allclose(T_sub, T_full, atol=1e-10):
                        return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    except Exception:
        pass
    
    # If no rank-27 found by simple elimination, try a more sophisticated search.
    # We can try to find a 27-term decomposition by optimizing over the parameters.
    # This is a non-convex problem, so we'll use a randomized approach.
    
    if time.time() < deadline - 10:
        # Try to find a rank-27 decomposition by searching over possible structures.
        # The key insight is that we can try to merge two of the 28 Strassen terms
        # into one, if they have a compatible structure.
        
        # For each pair of terms (r1, r2), check if their U, V, W vectors have
        # a special relationship that allows merging.
        
        # A simpler approach: try to find a 27-term decomposition by solving
        # a system of equations.
        
        # The tensor T has 1024 entries. A 27-term decomposition has 27*(8+16+8) = 648 parameters.
        # So we have 1024 equations and 648 unknowns, which is overdetermined.
        # But the tensor has a special structure (it's a matrix multiplication tensor),
        # so there may be a solution.
        
        # Let's try a different approach: use the structure of the problem.
        # C[i,j] = sum_k A[i,k] B[k,j]
        # We can write this as:
        # C = A * B
        # where A is 2x4 and B is 4x4.
        
        # The tensor rank of M_{2,4,4} is at most 28 (Strassen).
        # Can we do better?
        
        # Let's try to find a rank-27 decomposition by using the fact that
        # the tensor has a special structure.
        
        # One approach: use the "border rank" or "approximate rank" to guide the search.
        # But we need an exact decomposition.
        
        # Another approach: use the "Coppersmith-Winograd" style recursion, but that's
        # for larger matrices.
        
        # For now, let's try a randomized search for a rank-27 decomposition.
        # We'll try to find 27 rank-1 tensors that sum to T.
        
        # This is a hard problem, so we'll use a heuristic approach.
        # Start with the 28-term Strassen decomposition and try to eliminate one term
        # by adjusting the others.
        
        for elim_r in range(rank_28):
            if time.time() > deadline - 5:
                break
            
            # Try to eliminate term elim_r by adjusting the other 27 terms.
            # The idea is to find small perturbations to the other 27 terms that
            # compensate for the removed term.
            
            # This is equivalent to finding a 27-term decomposition that equals T.
            # We can set up a linear system for the perturbations.
            
            # Let U_r, V_r, W_r be the r-th term.
            # We want: sum_{r != elim_r} (U_r + dU_r) (V_r + dV_r) (W_r + dW_r) = T
            # This is a nonlinear system, so we'll use a first-order approximation:
            # sum_{r != elim_r} (U_r dV_r W_r + U_r V_r dW_r + dU_r V_r W_r) = -U_elim V_elim W_elim
            
            # This is a linear system in the perturbations dU_r, dV_r, dW_r.
            # But the number of unknowns is 27*(8+16+8) = 648, and the number of equations is 1024.
            # So the system is overdetermined, and a solution may not exist.
            
            # Let's try a different approach: use the null space of the 28-term decomposition.
            
            # If the 28 terms are linearly independent, then there is no rank-27 decomposition
            # that is a linear combination of the 28 terms. But there may be a rank-27 decomposition
            # that is not a linear combination of the 28 terms.
            
            # For now, let's just return the rank-28 decomposition.
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
