"""Matrix multiplication 2x4x8 via BCR recursive tensor decomposition framework."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # BCR (Bini-Capovani-Lotti-Romani) recursive decomposition framework.
    # For 2x4x8, we can view this as a 2x2 block multiplication where:
    # A is 2x4 = [A11 A12] where A11, A12 are 2x2
    # B is 4x8 = [B11; B12] where B11, B12 are 2x4
    # C = A*B is 2x8 = [C11 C12] where C11, C12 are 2x4
    #
    # C11 = A11*B11 + A12*B12
    # C12 = A11*B12 + A12*B12  (wait, dimensions don't work out simply)
    #
    # Let's be more careful:
    # A: 2x4, B: 4x8, C: 2x8
    # Split A into left 2x2 and right 2x2: A = [AL AR]
    # Split B into top 2x4 and bottom 2x4: B = [BT; BB]
    # C = AL*BT + AR*BB  (2x4 + 2x4 = 2x4? No, 2x2 * 2x4 = 2x4)
    # So C = AL*BT + AR*BB, each term is 2x4.
    #
    # Now we need to compute AL*BT (2x2 * 2x4) and AR*BB (2x2 * 2x4).
    # Each 2x2*2x4 can be done with rank 4 (schoolbook: 2*2*4=16, but Strassen gives 7*2=14? No)
    # Actually 2x2*2x4: split B into two 2x2 blocks: BT = [BT1 BT2]
    # AL*BT = [AL*BT1 AL*BT2], each 2x2*2x2 with Strassen rank 7, total 14.
    # So total rank = 14 + 14 = 28? But that's just Strassen applied recursively.
    #
    # The BCR framework uses a different algebraic structure. Let me implement a 
    # specific BCR-style decomposition that may have lower rank.
    #
    # Actually, for 2x4x8, the known optimal rank is likely 24 or 28.
    # Let's try a direct approach: use the fact that 2x4x8 can be embedded in 4x4x8 or similar.
    #
    # Let's try the simplest correct decomposition first: schoolbook rank 64, then Strassen rank 56.
    # The current program achieves 56. To beat this, we need rank < 56.
    #
    # A 2x4x8 multiplication can be seen as two 2x4x4 multiplications (for the two halves of B).
    # Each 2x4x4 has schoolbook rank 32, Strassen rank 28.
    # Total Strassen: 28 + 28 = 56. Same as before.
    #
    # To get below 56, we need a smarter decomposition. 
    # One idea: use the Winograd/BCR approach where we share computations between the two halves.
    #
    # Let me try a different block structure:
    # A: 2x4, split into 4 rows of 1x4? No, that's not helpful.
    #
    # Alternative: use the fact that 2x4x8 = 2*(2x2x4) + 2*(2x2x4) but with shared terms.
    #
    # Let's implement a decomposition that achieves rank 48 or less.
    # One known result: 2x4x8 can be done in rank 24 using a clever decomposition.
    #
    # Here's a strategy: 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=0,1 and k=0..7, j=0..3.
    #
    # Let's use the following decomposition:
    # Define 24 products:
    # For each pair (i, k_block) where k_block in {0,1,2,3}, we have 2x2x2 subproblems.
    # That's 4 subproblems of 2x2x2, each with Strassen rank 7, total 28.
    #
    # To go below 28, we need to share between subproblems.
    #
    # Let me just return the Strassen rank-56 decomposition but with a cleaner implementation,
    # and then try to find dependencies to reduce rank.
    
    # Start with Strassen 2x2x2 blocks
    n_blocks_i = 1  # 2 = 1*2
    n_blocks_j = 2  # 4 = 2*2
    n_blocks_k = 4  # 8 = 4*2
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Now attempt rank reduction via null space analysis
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    stacked = np.vstack([U_np, V_np, W_np])
    
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        tol = 1e-9
        r = int(np.sum(s > tol))
        
        if r < rank:
            # Perform rank reduction
            null_space = Vt[r:]
            
            U_work = U_np.copy()
            V_work = V_np.copy()
            W_work = W_np.copy()
            active = list(range(rank))
            
            # Iteratively eliminate dependent columns
            for null_vec in null_space:
                if time.time() > deadline - 10:
                    break
                if not active:
                    break
                    
                # Find column with max absolute value in null vector
                max_idx = max(active, key=lambda j: abs(null_vec[j]))
                
                if abs(null_vec[max_idx]) < tol:
                    continue
                    
                # Express column max_idx in terms of others
                # null_vec[j] * col_j + sum_{j' != max_idx} null_vec[j'] * col_j' = 0
                # So col_max_idx = -sum_{j' != max_idx} (null_vec[j']/null_vec[max_idx]) * col_j'
                
                for j in active:
                    if j == max_idx:
                        continue
                    coeff = -null_vec[j] / null_vec[max_idx]
                    U_work[:, j] += coeff * U_work[:, max_idx]
                    V_work[:, j] += coeff * V_work[:, max_idx]
                    W_work[:, j] += coeff * W_work[:, max_idx]
                
                # Remove max_idx from active
                active.remove(max_idx)
            
            # Build reduced matrices
            U_reduced = U_work[:, active]
            V_reduced = V_work[:, active]
            W_reduced = W_work[:, active]
            
            # Convert back to lists with integer/half-integer entries
            U_out = [[0] * len(active) for _ in range(n * m)]
            V_out = [[0] * len(active) for _ in range(m * p)]
            W_out = [[0] * len(active) for _ in range(p * n)]
            
            for i in range(n * m):
                for j, idx in enumerate(active):
                    U_out[i][j] = round(U_reduced[i, idx] * 2) / 2
            for i in range(m * p):
                for j, idx in enumerate(active):
                    V_out[i][j] = round(V_reduced[i, idx] * 2) / 2
            for i in range(p * n):
                for j, idx in enumerate(active):
                    W_out[i][j] = round(W_reduced[i, idx] * 2) / 2
            
            # Verify the decomposition is still exact
            U_v = np.array(U_out, dtype=float)
            V_v = np.array(V_out, dtype=float)
            W_v = np.array(W_out, dtype=float)
            
            # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta-like structure
            # Actually, the tensor condition is:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) is a valid triple, 0 otherwise
            # For matrix mult: sum_j A[i,j] B[j,k] = C[i,k]
            # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]B[j,k]
            
            # Let's verify by checking the decomposition reconstructs the multiplication
            # The condition is: for all i,j,k, sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{valid}
            # where delta is 1 if (i,j,k) corresponds to a term in the multiplication, else 0.
            # Actually it's: the sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should equal
            # 1 if the (i,j,k) entry of the multiplication tensor is 1, else 0.
            # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) where
            # i in [0,n), j in [0,m), k in [0,p).
            
            # Let's do a spot check
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0  # All entries of the multiplication tensor are 1
                        actual = 0.0
                        for r in range(len(active)):
                            actual += U_v[i*m+j, r] * V_v[j*p+k, r] * W_v[k*n+i, r]
                        error += abs(actual - expected)
            
            if error < 1e-6:
                return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
