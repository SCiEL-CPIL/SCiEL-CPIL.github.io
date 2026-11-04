import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 140  # safety margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        T_computed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_expected = np.ones((n, m, p))
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    def get_strassen_block(i_set, j_set):
        """Get Strassen decomposition for 2x2x2 block with i in i_set, j in j_set."""
        # i_set = {a, b}, j_set = {c, d}
        # Local indices: i0=a, i1=b, j0=c, j1=d
        a, b = i_set
        c, d = j_set
        rank = 7
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Strassen terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U coefficients (for A[i,j])
        U_coeffs = [
            {a: 1, b: 1},  # M1: A00+A11
            {b: 1},         # M2: A10+A11 -> A11 (wait, this is wrong)
        ]
        # Let me use the standard Strassen formulas more carefully.
        # For matrices A (2x2) and B (2x2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U_coeffs = [
            {a*3+c: 1, b*3+d: 1},  # M1
            {b*3+c: 1, b*3+d: 1},  # M2
            {a*3+c: 1},             # M3
            {b*3+d: 1},             # M4
            {a*3+c: 1, a*3+d: 1},  # M5
            {b*3+c: 1, a*3+c: -1}, # M6
            {a*3+d: 1, b*3+d: -1}  # M7
        ]
        V_coeffs = [
            {c*2+0: 1, d*2+1: 1},  # M1
            {c*2+0: 1},             # M2
            {d*2+0: 1, c*2+0: -1}, # M3
            {c*2+1: 1, d*2+1: -1}, # M4
            {d*2+1: 1},             # M5
            {c*2+0: 1, d*2+0: 1},  # M6
            {c*2+1: 1, d*2+1: 1}   # M7
        ]
        W_coeffs = [
            {0*3+a: 1, 1*3+b: 1, 0*3+b: -1, 1*3+a: 1},  # M1: C00+=M1, C11+=M1, C01-=M5... 
            # This is getting complicated. Let me use the standard embedding.
        ]
        
        # Actually, let me just use the same Strassen coefficients as the current program,
        # but with the correct i, j indices.
        # The current program uses i_local = li // 2, j_local = li % 2 for U,
        # which means U[i_local*3 + j_local, col] where i_local, j_local are 0 or 1.
        # For a general placement, we need to map local (0,1) to global (a,b) and (c,d).
        
        # Let me define the mapping:
        # local i=0 -> global i=a, local i=1 -> global i=b
        # local j=0 -> global j=c, local j=1 -> global j=d
        
        # U: local (i, j) -> global (i_map[i], j_map[j])
        i_map = {0: a, 1: b}
        j_map = {0: c, 1: d}
        
        # Standard Strassen U coefficients (local indices)
        U_local = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        for l in range(7):
            for li, coeff in U_local[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_map[i_local]*3 + j_map[j_local], l] += coeff
            for lj, coeff in V_local[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_map[j_local]*2 + k_local, l] += coeff
            for lc, coeff in W_local[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_map[i_local], l] += coeff
        
        return U, V, W
    
    def get_17_rank_block(i_set, j_set):
        """Get rank-17 decomposition for 3x3x2 block with Strassen on i_set, j_set."""
        a, b = i_set
        c, d = j_set
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Strassen part (7 terms)
        U_s, V_s, W_s = get_strassen_block(i_set, j_set)
        U[:, :7] = U_s
        V[:, :7] = V_s
        W[:, :7] = W_s
        
        # Schoolbook part (10 terms)
        col = 7
        i_complement = [i for i in range(3) if i not in i_set]
        j_complement = [j for j in range(3) if j not in j_set]
        
        # i in complement, all j, k in {0,1}
        for i in i_complement:
            for j in range(3):
                for k in range(2):
                    U[i*3+j, col] = 1.0
                    V[j*2+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
        
        # i in i_set, j in complement, k in {0,1}
        for i in i_set:
            for j in j_complement:
                for k in range(2):
                    U[i*3+j, col] = 1.0
                    V[j*2+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    def build_full_from_block(U_block, V_block, W_block, rank_block, i_set, j_set):
        """Assemble full 3x3x8 decomposition from 3x3x2 block."""
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        return U, V, W
    
    def try_optimize_block(rank_block, i_set, j_set, max_time=25):
        """Try to find a rank_block decomposition for 3x3x2 using optimization."""
        if time_left() < max_time + 5:
            return None
        
        n_params = 21 * rank_block
        
        def unpack(params):
            U = params[0:9*rank_block].reshape(3, 3, rank_block)
            V = params[9*rank_block:15*rank_block].reshape(3, 2, rank_block)
            W = params[15*rank_block:21*rank_block].reshape(2, 3, rank_block)
            return U, V, W
        
        def objective(params):
            U, V, W = unpack(params)
            T_computed = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            return np.sum((T_computed - 1.0)**2)
        
        # Initialize with the rank-17 solution
        U17, V17, W17 = get_17_rank_block(i_set, j_set)
        
        params_init = np.zeros(n_params)
        if rank_block <= 17:
            params_init[0:9*rank_block] = U17[:, :rank_block].flatten()
            params_init[9*rank_block:15*rank_block] = V17[:, :rank_block].flatten()
            params_init[15*rank_block:21*rank_block] = W17[:, :rank_block].flatten()
        
        best_params = None
        best_obj = float('inf')
        
        n_restarts = 15
        for restart in range(n_restarts):
            if time_left() < max_time:
                break
            
            if restart == 0:
                x0 = params_init.copy()
            else:
                x0 = params_init.copy()
                x0 += np.random.randn(n_params) * 0.1
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is None or best_obj > 1e-8:
            return None
        
        U, V, W = unpack(best_params)
        T_check = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if np.max(np.abs(T_check - 1.0)) > 1e-8:
            return None
        
        return U.reshape(9, rank_block), V.reshape(6, rank_block), W.reshape(6, rank_block)
    
    # Search over all 9 placements
    best_result = None
    best_rank = 72
    
    placements = []
    for i_set in [(0,1), (0,2), (1,2)]:
        for j_set in [(0,1), (0,2), (1,2)]:
            placements.append((i_set, j_set))
    
    for i_set, j_set in placements:
        if time_left() < 30:
            break
        
        # Try rank 16
        if time_left() > 35:
            result_16 = try_optimize_block(16, i_set, j_set, max_time=20)
            if result_16 is not None:
                U_block, V_block, W_block = result_16
                U, V, W = build_full_from_block(U_block, V_block, W_block, 16, i_set, j_set)
                if validate(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fall back to rank 17
        U_block, V_block, W_block = get_17_rank_block(i_set, j_set)
        U, V, W = build_full_from_block(U_block, V_block, W_block, 17, i_set, j_set)
        if validate(U, V, W):
            if best_result is None:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 68
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
