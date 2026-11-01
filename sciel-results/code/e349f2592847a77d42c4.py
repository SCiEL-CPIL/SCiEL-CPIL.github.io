import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # 1. Fallback: Rank 21 Solution
    # This is 3 independent Strassen decompositions for the 3 pairs of k columns?
    # No, Strassen is for 2x2 * 2x2.
    # Here we have 2x2 * 2x6.
    # We can view the 2x6 matrix as three 2x2 matrices.
    # Let B = [B1 B2 B3] where B1, B2, B3 are 2x2.
    # C = A * B = [A*B1 A*B2 A*B3].
    # Each A*Bk is 2x2.
    # So we can use Strassen for each block.
    # Rank = 3 * 7 = 21.
    
    rank_fb = 21
    strassen = [
        # Strassen 7 terms for 2x2 multiplication
        # U, V, W are 4-vectors (flattened 2x2 matrices)
        # Convention: U[i*2+j], V[i*2+j], W[i*2+j]
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # M1 = (A00+A11)(B00+B11)
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # M2 = (A10+A11)B00
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # M3 = A00(B10-B11)
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # M4 = A11(B00-B01)
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # M5 = (A00+A01)B11
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # M6 = (A10-A00)(B00+B01)
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # M7 = (A01-A11)(B10+B11)
    ]
    
    # Construct U, V, W for Rank 21
    # U: 4 x 21
    # V: 12 x 21
    # W: 12 x 21
    U_fb = np.zeros((4, rank_fb))
    V_fb = np.zeros((12, rank_fb))
    W_fb = np.zeros((12, rank_fb))
    
    for b in range(3): # 3 blocks of 2 columns in B
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            # U part: A is 2x2, index i*2+j. Same for all blocks.
            for idx in range(4):
                U_fb[idx, col] = sc['U'][idx]
            
            # V part: B block b is columns 2*b, 2*b+1 of the 2x6 matrix.
            # B index in V: j'*6 + k.
            # For block b, k is 2*b, 2*b+1.
            # j' is 0, 1.
            # So indices are:
            # j'=0, k=2*b -> 0*6 + 2*b = 2*b
            # j'=0, k=2*b+1 -> 0*6 + 2*b+1 = 2*b+1
            # j'=1, k=2*b -> 1*6 + 2*b = 6+2*b
            # j'=1, k=2*b+1 -> 1*6 + 2*b+1 = 6+2*b+1
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                V_fb[v_indices[idx], col] = sc['V'][idx]
            
            # W part: C is 2x6. Index k*n + i = k*2 + i.
            # For block b, k is 2*b, 2*b+1.
            # i is 0, 1.
            # Indices:
            # k=2*b, i=0 -> 4*b
            # k=2*b, i=1 -> 4*b+1
            # k=2*b+1, i=0 -> 4*b+2
            # k=2*b+1, i=1 -> 4*b+3
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                W_fb[w_indices[idx], col] = sc['W'][idx]

    # 2. Attempt Rank 20
    target_rank = 20
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i*2 + j
                b = j*6 + k
                c = k*2 + i
                T[a, b, c] = 1.0

    def residuals(params):
        # params: flattened U, V, W
        # U: 4 x R, V: 12 x R, W: 12 x R
        R = target_rank
        U_s = params[0:R*4].reshape(R, 4).T # 4 x R
        V_s = params[R*4:R*16].reshape(R, 12).T # 12 x R
        W_s = params[R*16:].reshape(R, 12).T # 12 x R
        
        # S_abc = sum_s U_as V_bs W_cs
        S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
        
        return (S - T).flatten()

    def snap_and_verify(U, V, W):
        # Snap to nearest half-integer
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Verify
        S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
        if np.allclose(S, T, atol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    found = False
    best_solution = None
    
    np.random.seed(42)
    
    # Strategy: 
    # 1. Try truncated rank-21 solutions (drop 1 column, optimize remaining 20)
    # 2. Try random initializations
    
    inits = []
    
    # Truncated Rank-21 inits
    # We have 21 columns. Try dropping each column and optimizing the rest.
    # This is 21 inits.
    for drop_col in range(21):
        mask = [i for i in range(21) if i != drop_col]
        U_init = U_fb[:, mask]
        V_init = V_fb[:, mask]
        W_init = W_fb[:, mask]
        
        # Flatten: U is 4x20, V is 12x20, W is 12x20
        # Reshape to match residuals expectation: params[0:R*4] is U_s.T?
        # My residuals: U_s = params[0:R*4].reshape(R, 4).T -> 4xR
        # So params should be U.T.flatten() + V.T.flatten() + W.T.flatten()
        init_params = np.concatenate([
            U_init.T.flatten(),
            V_init.T.flatten(),
            W_init.T.flatten()
        ])
        inits.append(init_params)
        
        if time.time() > deadline - 30:
            break

    # Random inits
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        U_rand = np.random.randn(20, 4)
        V_rand = np.random.randn(20, 12)
        W_rand = np.random.randn(20, 12)
        init_params = np.concatenate([
            U_rand.T.flatten(),
            V_rand.T.flatten(),
            W_rand.T.flatten()
        ])
        inits.append(init_params)
        
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use 'lm' method for least squares, it's fast for small problems
            # max_nfev limited to keep time under control
            result = least_squares(residuals, init, method='lm', max_nfev=500)
            
            if result.cost < 1e-6: # Small residual
                # Extract solution
                R = target_rank
                U_sol = result.x[0:R*4].reshape(R, 4).T
                V_sol = result.x[R*4:R*16].reshape(R, 12).T
                W_sol = result.x[R*16:].reshape(R, 12).T
                
                valid, U_snap, V_snap, W_snap = snap_and_verify(U_sol, V_sol, W_sol)
                if valid:
                    found = True
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        except Exception:
            continue

    if found:
        U, V, W = best_solution
        # Convert to lists of lists of integers/half-integers
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Return Rank 21 fallback
        U_out = U_fb.tolist()
        V_out = V_fb.tolist()
        W_out = W_fb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
