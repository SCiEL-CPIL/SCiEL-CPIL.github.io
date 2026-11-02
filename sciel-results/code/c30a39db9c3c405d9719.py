import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 6, 6, 6
    
    deadline = time.time() + 110.0
    
    # Target tensor for 6x6
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Function to find decomposition for 3x3
    def find_3x3_decomposition(rank):
        if time.time() > deadline - 20: # Leave 20s for 6x6 construction
            return None
            
        n3 = 3
        T3 = np.zeros((n3 * n3, n3 * n3, n3 * n3), dtype=float)
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    u = i * n3 + k
                    v = k * n3 + j
                    w = j * n3 + i
                    T3[u, v, w] = 1.0

        best_res = None
        best_err = float('inf')
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(n3 * n3, rank) * 0.5
            V = np.random.randn(n3 * n3, rank) * 0.5
            W = np.random.randn(n3 * n3, rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_f = x[:n3*n3*rank].reshape(n3*n3, rank)
                V_f = x[n3*n3*rank : n3*n3*rank + n3*n3*rank].reshape(n3*n3, rank)
                W_f = x[n3*n3*rank + n3*n3*rank :].reshape(n3*n3, rank)
                T_recon = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
                err = T_recon - T3
                return np.sum(err * err)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 200})
            except Exception:
                continue
                
            x_opt = res.x
            U_opt = x_opt[:n3*n3*rank].reshape(n3*n3, rank)
            V_opt = x_opt[n3*n3*rank : n3*n3*rank + n3*n3*rank].reshape(n3*n3, rank)
            W_opt = x_opt[n3*n3*rank + n3*n3*rank :].reshape(n3*n3, rank)
            
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            err = np.linalg.norm(T_recon - T3)
            
            if err < 1e-9:
                return {"U": U_snap, "V": V_snap, "W": W_snap}
            
            if err < best_err:
                best_err = err
                best_res = {"U": U_snap, "V": V_snap, "W": W_snap}
                
        return None

    # Try to find 3x3 decomposition with rank 23
    # If found, total rank is 8 * 23 = 184
    # If not, try 22, 21, 20, 19
    # If none, fallback to 189
    
    for r3 in [23, 22, 21, 20, 19]:
        if time.time() > deadline - 30:
            break
        res3 = find_3x3_decomposition(r3)
        if res3:
            # Construct 6x6 from 3x3
            U6 = np.zeros((n * m, 8 * r3), dtype=float)
            V6 = np.zeros((m * p, 8 * r3), dtype=float)
            W6 = np.zeros((p * n, 8 * r3), dtype=float)
            
            r_offset = 0
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # Extract 3x3 blocks
                        # A block (I, K): rows 3I..3I+2, cols 3K..3K+2
                        # U rows correspond to A indices u = i*3 + k
                        # Global u = i_glob * 6 + k_glob
                        # i_glob = 3I + i, k_glob = 3K + k
                        # So u_glob = (3I+i)*6 + (3K+k) = 18I + 6i + 3K + k
                        
                        # V block (K, J): rows 3K..3K+2, cols 3J..3J+2
                        # v = k*3 + j
                        # Global v = (3K+k)*6 + (3J+j) = 18K + 6k + 3J + j
                        
                        # W block (I, J): rows 3I..3I+2, cols 3J..3J+2
                        # w = j*3 + i
                        # Global w = (3J+j)*6 + (3I+i) = 18J + 6j + 3I + i
                        
                        for r in range(r3):
                            u_idx = r_offset + r
                            v_idx = r_offset + r
                            w_idx = r_offset + r
                            
                            # Map U
                            for i in range(3):
                                for k in range(3):
                                    u_local = i * 3 + k
                                    u_glob = (3 * I + i) * 6 + (3 * K + k)
                                    U6[u_glob, u_idx] = res3["U"][u_local, r]
                                    
                            # Map V
                            for k in range(3):
                                for j in range(3):
                                    v_local = k * 3 + j
                                    v_glob = (3 * K + k) * 6 + (3 * J + j)
                                    V6[v_glob, v_idx] = res3["V"][v_local, r]
                                    
                            # Map W
                            for j in range(3):
                                for i in range(3):
                                    w_local = j * 3 + i
                                    w_glob = (3 * J + j) * 6 + (3 * I + i)
                                    W6[w_glob, w_idx] = res3["W"][w_local, r]
                                    
                        r_offset += r3
                        
            # Verify
            T_recon = np.einsum('ir,jr,kr->ijk', U6, V6, W6)
            if np.allclose(T_recon, T, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
            
    # Fallback to 189 (Block Strassen)
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
