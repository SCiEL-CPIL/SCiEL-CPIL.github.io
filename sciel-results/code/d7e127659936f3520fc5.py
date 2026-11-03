import numpy as np
import time
from collections import defaultdict

TIME_LIMIT = 120
DEADLINE = time.time() + TIME_LIMIT - 5  # 5s safety margin

def _check_deadline():
    return time.time() > DEADLINE

def construct():
    n, m, p = 3, 4, 8
    
    # 1. Schoolbook baseline: rank = 3*4*8 = 96
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
    
    # 2. Try to find a better decomposition using a greedy rank-reduction search
    # This is a simple heuristic that tries to merge basis vectors.
    # Given time constraints, we only run this for a very limited number of iterations.
    # A full search would be infeasible, but we can try to find a few improvements.
    
    # For this specific case (3,4,8), known best ranks are around 80-90.
    # We start from schoolbook and try to reduce.
    # However, a robust random search is risky. Let's stick to a known good construction
    # if we can encode it, or just return the schoolbook to guarantee correctness.
    
    # Actually, let's try to implement the Block Strassen construction from the seed 
    # but verify it works and then try to optimize the "leftover" part.
    # The seed had rank 88.
    
    # Let's re-implement the seed's logic carefully to ensure it's correct.
    
    rank_seed = 88
    U_s = np.zeros((n * m, rank_seed))
    V_s = np.zeros((m * p, rank_seed))
    W_s = np.zeros((p * n, rank_seed))
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    u_local = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_local = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_local = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [-1, 0, -1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, -1, 0]
    ]
    
    # Block structure:
    # Rows i: 0,1,2. Cols j: 0,1,2,3. Cols k: 0..7
    # Use Strassen for i in {0,1}, j in {0,1} and {2,3}, k in pairs.
    # This covers i=0,1 (2 rows), j=0,1,2,3 (4 cols), k=0..7 (8 cols).
    # Wait, Strassen is for 2x2.
    # We can partition:
    # A (3x4) -> A1 (2x4), A2 (1x4)
    # B (4x8) -> B1 (4x4), B2 (4x4) ? No, B is 4x8.
    # Let's partition B into 4x4 and 4x4.
    # Then A1 * B1 is 2x4. A1 * B2 is 2x4.
    # A2 * B1 is 1x4. A2 * B2 is 1x4.
    
    # Let's use Strassen for A1 (2x2) * B1_left (2x2)? No, B is 4x8.
    
    # The seed used:
    # i in {0,1}, j pairs (0,1), (2,3), k pairs (0,1), (2,3), (4,5), (6,7).
    # This decomposes the 2x4x8 sub-tensor for i=0,1 into 2 * 2 * 4 blocks of 2x2x2.
    # Each 2x2x2 block has rank 7.
    # Total for i=0,1: 2 (j-pairs) * 4 (k-pairs) * 7 = 56.
    # Remaining: i=2 (1x4x8). This is 32 rank-1 terms.
    # Total: 56 + 32 = 88.
    
    # Let's verify this construction is valid.
    # We need to map global indices correctly.
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # i_start is 0. We cover i=0,1.
    for j_start, j_end in j_pairs:
        for k_start, k_end in k_pairs:
            for r in range(7):
                # U: rows i*m+j. i in {0,1}, j in {j_start, j_start+1}
                for li in range(4):
                    li_i = li // 2  # 0 or 1
                    li_j = li % 2   # 0 or 1
                    global_i = 0 + li_i # i_start is 0
                    global_j = j_start + li_j
                    global_idx_u = global_i * m + global_j
                    U_s[global_idx_u, col] = u_local[r][li]
                
                # V: rows j*p+k. j in {j_start, j_start+1}, k in {k_start, k_start+1}
                for lj in range(4):
                    lj_j = lj // 2
                    lj_k = lj % 2
                    global_j = j_start + lj_j
                    global_k = k_start + lj_k
                    global_idx_v = global_j * p + global_k
                    V_s[global_idx_v, col] = v_local[r][lj]
                
                # W: rows k*n+i. k in {k_start, k_start+1}, i in {0,1}
                for lk in range(4):
                    lk_k = lk // 2
                    lk_i = lk % 2
                    global_k = k_start + lk_k
                    global_i = 0 + lk_i
                    global_idx_w = global_k * n + global_i
                    W_s[global_idx_w, col] = w_local[r][lk]
                
                col += 1
    
    # Remaining: i=2. All j, all k.
    for j in range(m):
        for k in range(p):
            global_i = 2
            global_idx_u = global_i * m + j
            global_idx_v = j * p + k
            global_idx_w = k * n + global_i
            
            U_s[global_idx_u, col] = 1
            V_s[global_idx_v, col] = 1
            W_s[global_idx_w, col] = 1
            col += 1
            
    # Verify the seed construction
    # Tensor T[i,j,k] should be delta_{i,k} (no, it's matrix mult)
    # T[i, j, k] = sum_l A[i,l] B[l,k]
    # Decomposition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's do a quick sanity check with random matrices
    A_rand = np.random.rand(n, m)
    B_rand = np.random.rand(m, p)
    C_ref = A_rand @ B_rand
    
    # Construct tensor from factors
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is not quite right. The tensor for matrix mult T[i,j,k] has components T[i,j,k] = 1 if i==k? No.
    # The multiplication tensor T_{n,m,p} is defined such that T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # Wait, the standard tensor is T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C where T(i,j,k) = 1 if the product contributes.
    # For matrix mult, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i,j,k] = 1 if we are multiplying A[i,j] and B[j,k] to get C[i,k].
    # But the indices in the decomposition are:
    # U has shape (n*m, R). Index (i,j).
    # V has shape (m*p, R). Index (j,k).
    # W has shape (p*n, R). Index (k,i).
    # The product is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # This sum must equal 1 if (i,j,k) is a valid term in the multiplication, i.e., always?
    # No, the tensor itself is T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor T has T[i,j,k] = 1 if the pair (A[i,j], B[j,k]) multiplies to contribute to C[i,k].
    # Actually, the tensor T is defined as T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k \text{ where } T_{ijk} = 1 \text{ if } i,k \text{ match the output index}.
    # Wait, the standard definition: T_{ijk} = 1 if i, j, k form a valid triple for the multiplication.
    # In matrix multiplication, every triple (i, j, k) where i is row of A, j is col of A/row of B, k is col of B is a valid term.
    # So T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition satisfies: \sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let's check if the seed construction satisfies this.
    # We computed U_s, V_s, W_s.
    # Let's pick a random (i,j,k) and check the sum.
    
    # We can't check all 3*4*8=96 terms in a tight loop if it's slow, but it's fast enough.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.sum(U_s[idx_u, :] * V_s[idx_v, :] * W_s[idx_w, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U_s, "V": V_s, "W": W_s}
    else:
        return {"U": U, "V": V, "W": W}

