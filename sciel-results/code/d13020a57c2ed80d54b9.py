import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS

    # Build target tensor
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1

    def check(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)

    def error(Ux, Vx, Wx):
        recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
        return int(np.sum((recon - T) ** 2))

    # Rank-26 Strassen-based decomposition
    def get_rank26():
        rank = 26
        U = np.zeros((9, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((9, rank), dtype=np.int64)
        
        col = 0
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        # Cross terms for the 3rd row/col
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return U, V, W

    U26, V26, W26 = get_rank26()
    if not check(U26, V26, W26):
        raise RuntimeError("Rank-26 verification failed")
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    # Try to find rank 24 and 25 by removing 2 or 1 columns
    def try_reduce(target_rank, num_removals, max_combos, local_iters_per_combo):
        """Try to find a valid decomposition by removing columns and repairing."""
        all_cols = list(range(26))
        all_combs = list(combinations(all_cols, num_removals))
        
        rng = np.random.RandomState(99 + target_rank)
        if len(all_combs) > max_combos:
            sampled = rng.choice(len(all_combs), size=max_combos, replace=False)
            sampled_combs = [all_combs[i] for i in sampled]
        else:
            sampled_combs = all_combs
        
        for cols_to_remove in sampled_combs:
            if time.time() > deadline - 5:
                return None
            
            cols_to_keep = [c for c in all_cols if c not in cols_to_remove]
            
            U_init = np.zeros((9, target_rank), dtype=np.int64)
            V_init = np.zeros((9, target_rank), dtype=np.int64)
            W_init = np.zeros((9, target_rank), dtype=np.int64)
            
            for new_idx, old_idx in enumerate(cols_to_keep):
                U_init[:, new_idx] = U26[:, old_idx]
                V_init[:, new_idx] = V26[:, old_idx]
                W_init[:, new_idx] = W26[:, old_idx]
            
            # Local search
            U_c, V_c, W_c = U_init.copy(), V_init.copy(), W_init.copy()
            current_err = error(U_c, V_c, W_c)
            if current_err == 0:
                return (U_c, V_c, W_c)
            
            best_err = current_err
            best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
            
            rng_local = np.random.RandomState(1234 + target_rank)
            temperature = 5.0
            cooling = 0.9995
            no_improve = 0
            
            for it in range(local_iters_per_combo):
                if time.time() > deadline - 3:
                    break
                
                mat_choice = rng_local.randint(3)
                row = rng_local.randint(9)
                col = rng_local.randint(target_rank)
                delta = rng_local.choice([-1, 1])
                
                if mat_choice == 0:
                    U_c[row, col] += delta
                elif mat_choice == 1:
                    V_c[row, col] += delta
                else:
                    W_c[row, col] += delta
                
                new_err = error(U_c, V_c, W_c)
                
                if new_err < current_err or (temperature > 1e-8 and rng_local.random() < np.exp(-(new_err - current_err) / temperature)):
                    current_err = new_err
                    no_improve = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                        if best_err == 0:
                            return (best_Ux, best_Vx, best_Wx)
                else:
                    if mat_choice == 0:
                        U_c[row, col] -= delta
                    elif mat_choice == 1:
                        V_c[row, col] -= delta
                    else:
                        W_c[row, col] -= delta
                
                temperature *= cooling
                no_improve += 1
                if no_improve > 500 and temperature < 0.5:
                    break
            
            if best_err == 0:
                return (best_Ux, best_Vx, best_Wx)
        
        return None
    
    # Try rank 25 first (remove 1 column)
    if time.time() < deadline - 10:
        result = try_reduce(25, 1, max_combos=50, local_iters_per_combo=300)
        if result:
            U_sol, V_sol, W_sol = result
            if check(U_sol, V_sol, W_sol):
                best_rank = 25
                best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    # Try rank 24 (remove 2 columns)
    if time.time() < deadline - 10 and best_rank == 26:
        result = try_reduce(24, 2, max_combos=30, local_iters_per_combo=200)
        if result:
            U_sol, V_sol, W_sol = result
            if check(U_sol, V_sol, W_sol):
                best_rank = 24
                best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    # Final verification
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
