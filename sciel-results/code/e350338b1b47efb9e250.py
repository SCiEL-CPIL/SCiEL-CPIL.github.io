import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS
    
    # We decompose the problem into 3 blocks of 3x3x2
    # Total rank = 3 * rank_332
    # We aim for rank_332 = 12, so total rank = 36
    
    def find_332_decomposition(target_rank):
        """
        Find a decomposition for 3x3x2 matrix multiplication with given rank.
        Returns U, V, W for 3x3x2.
        U: (9, rank), V: (6, rank), W: (6, rank)
        """
        # Target tensor T for 3x3x2
        T_target = np.zeros((9, 6, 6))
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    a = i * 3 + j
                    b = j * 2 + k
                    c = k * 3 + i
                    T_target[a, b, c] = 1.0
        
        def objective(x):
            U = x[:9*target_rank].reshape(9, target_rank)
            V = x[9*target_rank:9*target_rank + 6*target_rank].reshape(6, target_rank)
            W = x[9*target_rank + 6*target_rank:].reshape(6, target_rank)
            
            T_recon = np.zeros((9, 6, 6))
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
                
            error = T_target - T_recon
            return np.sum(error**2)
        
        def gradient(x):
            U = x[:9*target_rank].reshape(9, target_rank)
            V = x[9*target_rank:9*target_rank + 6*target_rank].reshape(6, target_rank)
            W = x[9*target_rank + 6*target_rank:].reshape(6, target_rank)
            
            T_recon = np.zeros((9, 6, 6))
            for t in range(target_rank):
                T_recon += np.outer(np.outer(U[:,t], V[:,t]), W[:,t])
                
            error = T_target - T_recon
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for t in range(target_rank):
                dU[:,t] = np.einsum('abc,bc->a', error, V[:,t] * W[:,t])
                dV[:,t] = np.einsum('abc,ac->b', error, U[:,t] * W[:,t])
                dW[:,t] = np.einsum('abc,ab->c', error, U[:,t] * V[:,t])
                
            return np.concatenate([-2*dU.flatten(), -2*dV.flatten(), -2*dW.flatten()])

        # Try a few random initializations
        best_sol = None
        best_err = float('inf')
        
        for trial in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(trial * 100 + 42)
            x0 = np.random.randn(9*target_rank + 6*target_rank + 6*target_rank) * 0.5
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'gtol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_sol = res.x
            except Exception:
                continue
        
        if best_sol is not None and best_err < 1e-6:
            U = best_sol[:9*target_rank].reshape(9, target_rank)
            V = best_sol[9*target_rank:9*target_rank + 6*target_rank].reshape(6, target_rank)
            W = best_sol[9*target_rank + 6*target_rank:].reshape(6, target_rank)
            return U, V, W
        return None

    # Try to find rank 12 for 3x3x2
    U_332, V_332, W_332 = find_332_decomposition(12)
    
    if U_332 is not None:
        rank_332 = 12
    else:
        # Fallback to schoolbook for 3x3x2 (rank 18)
        rank_332 = 18
        U_332 = np.zeros((9, 18))
        V_332 = np.zeros((6, 18))
        W_332 = np.zeros((6, 18))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_332[i*3+j, col] = 1.0
                    V_332[j*2+k, col] = 1.0
                    W_332[k*3+i, col] = 1.0
                    col += 1

    # Assemble 3x3x6 decomposition
    # U is same for all blocks: (9, 3*rank_332)
    # V is block diagonal: (18, 3*rank_332) -> 3 blocks of (6, rank_332)
    # W is block diagonal: (18, 3*rank_332) -> 3 blocks of (6, rank_332)
    
    total_rank = 3 * rank_332
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((18, total_rank))
    W_full = np.zeros((18, total_rank))
    
    for block in range(3):
        start_col = block * rank_332
        end_col = start_col + rank_332
        
        # U is identical for all blocks
        U_full[:, start_col:end_col] = U_332
        
        # V: block b corresponds to columns 2*b to 2*b+1 of B?
        # B is 3x6. Indices j*p+k.
        # Block 0: k=0,1 -> j*6+0, j*6+1? No.
        # The V dimension is m*p = 3*6 = 18.
        # The decomposition for 3x3x2 assumes B is 3x2.
        # In the full 3x3x6 problem, B has 18 elements.
        # We split B into three 3x2 matrices: B_0, B_1, B_2.
        # B_0 uses columns 0,1 of B (indices 0,1 in V? No, V index is j*p+k).
        # For 3x3x2, V index is j*2+k.
        # In 3x3x6, V index is j*6+k.
        # Block 0 corresponds to k=0,1? No, the output C is 3x6.
        # C = [C_0, C_1, C_2] where C_i is 3x2.
        # C_i = A B_i.
        # B_i consists of columns 2i and 2i+1 of B.
        # So for block i, the relevant V indices are j*6 + 2i and j*6 + 2i+1.
        
        # V_full row index for block i, local index l (0..5)
        # Local V index l = j*2 + k_local (k_local in 0,1)
        # Global V index = j*6 + (2i + k_local)
        # So for a given local V row l:
        # j = l // 2
        # k_local = l % 2
        # global_v_idx = j*6 + 2*i + k_local
        
        for l in range(6):
            j = l // 2
            k_local = l % 2
            global_v_idx = j * 6 + 2 * block + k_local
            V_full[global_v_idx, start_col:end_col] = V_332[l]
            
        # W: Output C is 3x6.
        # W dimension is p*n = 6*3 = 18.
        # Index c = k*n + i = k*3 + i.
        # Block i corresponds to output columns 2i and 2i+1.
        # So for block i, the relevant W indices are (2i)*3 + r and (2i+1)*3 + r for r=0..2.
        # Local W index l = k_local*3 + r.
        # Global W index = (2i + k_local)*3 + r.
        
        for l in range(6):
            k_local = l // 3
            r = l % 3
            global_w_idx = (2 * block + k_local) * 3 + r
            W_full[global_w_idx, start_col:end_col] = W_332[l]

    # Verify the full decomposition
    T_target_full = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                a = i * 3 + j
                b = j * 6 + k
                c = k * 3 + i
                T_target_full[a, b, c] = 1.0
                
    T_recon_full = np.zeros((9, 18, 18))
    for t in range(total_rank):
        T_recon_full += np.outer(np.outer(U_full[:,t], V_full[:,t]), W_full[:,t])
        
    if not np.allclose(T_recon_full, T_target_full, atol=1e-6):
        # Fallback to schoolbook if verification fails
        total_rank = 54
        U_full = np.zeros((9, 54))
        V_full = np.zeros((18, 54))
        W_full = np.zeros((18, 54))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U_full[i*3+j, col] = 1.0
                    V_full[j*6+k, col] = 1.0
                    W_full[k*3+i, col] = 1.0
                    col += 1

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(1000)
                if abs(frac - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
