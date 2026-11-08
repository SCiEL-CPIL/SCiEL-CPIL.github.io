import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 33 solution (current best)
    def get_rank_33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        terms = []
        # Strassen for 2x2
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        # Naive for 2x1x2
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

    # Try to find Rank 30 (3 * 10)
    # Search for rank 10 decomposition of M_{2,3,2}
    # U: 10 x 6, V: 10 x 6
    # Target: M_{2,3,2}
    
    def search_rank_10_m232():
        # Target tensors for M_{2,3,2}
        # A (2x3) flat: a0,a1,a2,a3,a4,a5
        # B (3x2) flat: b0,b1,b2,b3,b4,b5
        # C (2x2) flat: c0,c1,c2,c3 (c0=00, c1=01, c2=10, c3=11)
        
        # Target matrices M_ik (6x6)
        # c0 = a0*b0 + a1*b2 + a2*b4
        # c1 = a0*b1 + a1*b3 + a2*b5
        # c2 = a3*b0 + a4*b2 + a5*b4
        # c3 = a3*b1 + a4*b3 + a5*b5
        
        targets = []
        for ik in range(4):
            M = np.zeros((6, 6))
            if ik == 0:
                M[0,0] = 1; M[1,2] = 1; M[2,4] = 1
            elif ik == 1:
                M[0,1] = 1; M[1,3] = 1; M[2,5] = 1
            elif ik == 2:
                M[3,0] = 1; M[4,2] = 1; M[5,4] = 1
            elif ik == 3:
                M[3,1] = 1; M[4,3] = 1; M[5,5] = 1
            targets.append(M)
            
        # Strategy: Fix V to have sparse structure, then solve for U and W
        # Each term r has U[r] (6-vector) and V[r] (6-vector)
        # M_ik = sum_r W_ik[r] * outer(U[r], V[r])
        
        # Try structured V patterns
        np.random.seed(42)
        
        # Generate candidate V matrices with sparse entries
        for trial in range(20000):
            if time.time() > deadline - 15:
                return None
                
            # V: 10 x 6, entries in {-1, 0, 1} with ~2-3 nonzeros per row
            V = np.zeros((10, 6), dtype=int)
            for r in range(10):
                nnz = np.random.choice([2, 3, 3, 4])
                cols = np.random.choice(6, size=nnz, replace=False)
                for c in cols:
                    V[r, c] = np.random.choice([-1, 1])
            
            # For each term r, M_r = outer(U[r], V[r])
            # We need: for each ik, sum_r W_ik[r] * M_r = target_ik
            # This is: sum_r W_ik[r] * U[r] * V[r]^T = target_ik
            
            # Fix U[r] to be sparse too, or solve for U given V
            # Alternative: fix both U and V to be sparse, solve for W
            
            # Try: U also sparse
            U = np.zeros((10, 6), dtype=int)
            for r in range(10):
                nnz = np.random.choice([2, 3, 3, 4])
                cols = np.random.choice(6, size=nnz, replace=False)
                for c in cols:
                    U[r, c] = np.random.choice([-1, 1])
            
            # Build the linear system for W
            # A_mat[ik, r] = M_r flattened (36 values)
            # Actually: for each ik, we have 36 equations:
            # sum_r W_ik[r] * (U[r] * V[r]^T)[i,j] = target_ik[i,j]
            
            M_r = np.einsum('ri,rj->rij', U.astype(float), V.astype(float))  # 10 x 6 x 6
            M_r_flat = M_r.reshape(10, 36)  # 10 x 36
            
            W_valid = True
            W_vals = []
            for ik in range(4):
                # A_mat: 36 x 10
                A_mat = M_r_flat.T  # 36 x 10
                b_vec = targets[ik].flatten()
                
                # Solve least squares
                W_real, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check residual
                if np.linalg.norm(A_mat @ W_real - b_vec) > 1e-8:
                    W_valid = False
                    break
                    
                # Check if close to half-integers
                W_half = np.round(W_real * 2) / 2
                if not np.allclose(W_real, W_half, atol=1e-8):
                    W_valid = False
                    break
                W_vals.append(W_half)
                
            if W_valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = [w.tolist() for w in W_vals]
                return U_list, V_list, W_list
                
        return None

    sol_10 = search_rank_10_m232()
    
    if sol_10:
        U_10, V_10, W_10 = sol_10
        # Construct M_{2,3,6} from three M_{2,3,2} blocks
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for r in range(10):
                col = block * 10 + r
                # U is same for all blocks (depends on A)
                for i in range(6):
                    U[i][col] = U_10[r][i]
                    
                # V: local index j*2+k -> global j*6 + (block*2+k)
                for j in range(3):
                    for k in range(2):
                        l_idx = j * 2 + k
                        g_idx = j * p + (block * 2 + k)
                        V[g_idx][col] = V_10[r][l_idx]
                        
                # W: local index i*2+k -> global (block*2+k)*n + i
                for i in range(2):
                    for k in range(2):
                        l_idx = i * 2 + k
                        g_idx = (block * 2 + k) * n + i
                        W[g_idx][col] = W_10[l_idx][r]
                        
        # Verify
        if time.time() < deadline - 5:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_arr[:, r], A_flat)
                v_val = np.dot(V_arr[:, r], B_flat)
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += W_arr[k * n + i, r] * u_val * v_val
                        
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}
                
    return get_rank_33()
# EVOLVE-BLOCK-END
