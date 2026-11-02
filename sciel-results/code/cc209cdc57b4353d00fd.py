import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i

    # First, try to find a rank-9 decomposition for 2x2 x 2x3
    # If found, use: 4 Strassen blocks (28) + rank-9 block (9) + schoolbook (4) = 41
    # If not found, fall back to current 46-rank solution
    
    # Let's try to optimize for a rank-9 decomposition of 2x2 x 2x3
    # The tensor T[i,j,k] should satisfy: T[A_idx(i,j), B_idx(j,k), C_idx(i,k)] = 1
    # and T[other] = 0
    
    # For 2x2 x 2x3, we have 4*2*6 = 48 entries in the tensor
    # But only 2*2*3 = 12 target entries should be 1, rest should be 0
    
    # Let's try to find U, V, W of shape (4,9), (2,9), (6,9) such that
    # einsum('ir,jr,kr->ijk', U, V, W) matches the target tensor
    
    # This is a non-convex optimization problem. Let's try a few random initializations
    
    best_rank = 46
    best_U, best_V, best_W = None, None, None
    
    # Try rank 9 first
    for rank in [9, 10, 11]:
        if time.time() > deadline - 30:
            break
            
        for trial in range(20):
            if time.time() > deadline - 30:
                break
                
            np.random.seed(42 + rank * 100 + trial)
            U_init = np.random.randn(nm, rank)
            V_init = np.random.randn(mp, rank)
            W_init = np.random.randn(pn, rank)
            
            # Create target tensor for 2x2 x 2x3 (first 3 columns)
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(3):  # Only first 3 columns
                        T_target[A_idx(i,j), B_idx(j,k), C_idx(i,k)] = 1.0
            
            def objective(params):
                U_flat = params[:nm*rank].reshape(nm, rank)
                V_flat = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_flat = params[nm*rank+mp*rank:].reshape(pn, rank)
                T = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
                return np.sum((T - T_target)**2)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
            
            if result.fun < 1e-6:
                # Check if this is a valid decomposition
                U_opt = result.x[:nm*rank].reshape(nm, rank)
                V_opt = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_opt = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Verify
                T = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(3):
                            if abs(T[A_idx(i,j), B_idx(j,k), C_idx(i,k)] - 1.0) > 1e-6:
                                valid = False
                if valid:
                    # Check non-target entries
                    target_mask = np.zeros((nm, mp, pn), dtype=bool)
                    for i in range(n):
                        for j in range(m):
                            for k in range(3):
                                target_mask[A_idx(i,j), B_idx(j,k), C_idx(i,k)] = True
                    if np.any(np.abs(T[~target_mask]) > 1e-6):
                        valid = False
                
                if valid:
                    best_rank = rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break
        
        if best_rank <= rank:
            break
    
    # If we found a better decomposition for 3 columns, use it
    if best_rank <= 11:
        # Use: 4 Strassen blocks (k=0-7) + 1 block of 3 cols (rank best_rank) + 1 schoolbook (k=12)
        R_total = 4 * 7 + best_rank + 4
        U = np.zeros((nm, R_total), dtype=float)
        V = np.zeros((mp, R_total), dtype=float)
        W = np.zeros((pn, R_total), dtype=float)
        
        def add_strassen_block(k0, col_offset):
            prods = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}),
                ({(0,0):1}, {(0,1):1,(1,1):-1}),
                ({(1,1):1}, {(1,0):1,(0,0):-1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
            ]
            for t, (a_coeff, b_coeff) in enumerate(prods):
                col = col_offset + t
                for (i, j), c in a_coeff.items():
                    U[A_idx(i, j), col] += c
                for (j, kl), c in b_coeff.items():
                    V[B_idx(j, k0 + kl), col] += c
            c_combos = [
                (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
                (0, k0+1,   [(2,1),(4,1)]),
                (1, k0,     [(1,1),(3,1)]),
                (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
            ]
            for (i, k, terms) in c_combos:
                for (t, coeff) in terms:
                    W[C_idx(i, k), col_offset + t] += coeff
        
        def add_single_block(k, col_offset):
            prods = [
                ((0,0),(0,k)),
                ((0,1),(1,k)),
                ((1,0),(0,k)),
                ((1,1),(1,k)),
            ]
            for t, ((i,j),(jj,kk)) in enumerate(prods):
                col = col_offset + t
                U[A_idx(i, j), col] += 1
                V[B_idx(jj, kk), col] += 1
            W[C_idx(0, k), col_offset+0] += 1
            W[C_idx(0, k), col_offset+1] += 1
            W[C_idx(1, k), col_offset+2] += 1
            W[C_idx(1, k), col_offset+3] += 1
        
        col = 0
        for k0 in range(0, 8, 2):
            add_strassen_block(k0, col)
            col += 7
        
        # Add the optimized block for k=8,9,10
        U[:, col:col+best_rank] = best_U
        V[:, col:col+best_rank] = best_V
        W[:, col:col+best_rank] = best_W
        col += best_rank
        
        add_single_block(12, col)
        col += 4
        assert col == R_total
        
        # Verify
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[A_idx(i,j), B_idx(j,k), C_idx(i,k)] - 1.0) > 1e-9:
                        valid = False
        if valid:
            target_mask = np.zeros((nm, mp, pn), dtype=bool)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target_mask[A_idx(i,j), B_idx(j,k), C_idx(i,k)] = True
            if np.any(np.abs(T[~target_mask]) > 1e-9):
                valid = False
        
        if valid:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to current 46-rank solution
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    def add_strassen_block(k0, col_offset):
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff

    def add_single_block(k, col_offset):
        prods = [
            ((0,0),(0,k)),
            ((0,1),(1,k)),
            ((1,0),(0,k)),
            ((1,1),(1,k)),
        ]
        for t, ((i,j),(jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1

    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"

    T = np.einsum('ir,jr,kr->ijk', U, V, W)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_val = T[A_idx(i,j), B_idx(j,k), C_idx(i,k)]
                if abs(target_val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        target_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_mask[A_idx(i,j), B_idx(j,k), C_idx(i,k)] = True
        non_target = ~target_mask
        if np.any(np.abs(T[non_target]) > 1e-9):
            valid = False

    if not valid:
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
