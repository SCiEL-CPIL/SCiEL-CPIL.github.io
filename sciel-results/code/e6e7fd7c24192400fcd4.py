import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 140

    # Precompute the list of non-zero tensor entries
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i*m+j, j*p+k, k*n+i))
    
    nz_u = np.array([idx[0] for idx in nz_indices])
    nz_v = np.array([idx[1] for idx in nz_indices])
    nz_w = np.array([idx[2] for idx in nz_indices])

    def verify_fast(U_arr, V_arr, W_arr):
        """Fast exact verification using only non-zero entries."""
        U_int = np.round(U_arr * 2).astype(np.int64)
        V_int = np.round(V_arr * 2).astype(np.int64)
        W_int = np.round(W_arr * 2).astype(np.int64)
        
        # Check non-zero entries: should be 2 (since we use half-integers, 1 = 2/2)
        # Actually, with integer entries, each non-zero entry should be 1
        # But we're scaling by 2, so each entry should be 2
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        
        # Check all nz entries are 2 (since 1*2 = 2 when scaled by 2)
        for idx in range(len(nz_u)):
            if T_actual[nz_u[idx], nz_v[idx], nz_w[idx]] != 2:
                return False
        
        # Check all other entries are 0
        T_check = T_actual.copy()
        for idx in range(len(nz_u)):
            T_check[nz_u[idx], nz_v[idx], nz_w[idx]] = 0
        if not np.all(T_check == 0):
            return False
        return True

    # Build the rank-44 decomposition
    U = [[0] * 44 for _ in range(dim_u)]
    V = [[0] * 44 for _ in range(dim_v)]
    W = [[0] * 44 for _ in range(dim_w)]

    col = 0

    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col

    col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
    col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
    col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
    col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)

    for j in range(4):
        U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
        col += 1
        U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
        col += 1
    for j in range(4):
        U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
        col += 1
        U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
        col += 1

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # First verify the base decomposition is correct
    if not verify_fast(U_np, V_np, W_np):
        # Fallback to schoolbook rank-48
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = 44

    def verify_against_expected(U_arr, V_arr, W_arr):
        """Verify decomposition against expected tensor."""
        return verify_fast(U_arr, V_arr, W_arr)

    # Phase 1: Try removing single columns
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        if current_rank <= 1:
            break

        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            keep_indices = list(range(current_rank))
            keep_indices.pop(r_idx)
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify_against_expected(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                found = True
                break
        
        if not found:
            break

    # Phase 2: Try removing pairs of columns
    if current_rank > 1 and time.time() < deadline - 10:
        found_pair = False
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1+1, current_rank):
                if time.time() > deadline - 10:
                    break
                
                keep_indices = list(range(current_rank))
                keep_indices.remove(r2)
                keep_indices.remove(r1)
                
                U_test = current_U[:, keep_indices]
                V_test = current_V[:, keep_indices]
                W_test = current_W[:, keep_indices]
                
                if verify_against_expected(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 2
                    found_pair = True
                    break
            if found_pair:
                break
        if found_pair:
            # After removing a pair, try single removals again
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                if current_rank <= 1:
                    break
                found = False
                for r_idx in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    keep_indices = list(range(current_rank))
                    keep_indices.pop(r_idx)
                    U_test = current_U[:, keep_indices]
                    V_test = current_V[:, keep_indices]
                    W_test = current_W[:, keep_indices]
                    if verify_against_expected(U_test, V_test, W_test):
                        current_U = U_test
                        current_V = V_test
                        current_W = W_test
                        current_rank -= 1
                        found = True
                        break
                if not found:
                    break

    # Phase 3: Try local perturbation with half-integers
    if current_rank > 1 and time.time() < deadline - 15:
        import random
        random.seed(42)
        
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            r_idx = random.randint(0, current_rank - 1)
            keep_indices = list(range(current_rank))
            keep_indices.pop(r_idx)
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify_against_expected(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
            else:
                # Try perturbing one of the kept columns with half-integers
                pert_col = random.randint(0, len(keep_indices) - 1)
                pert_matrix = random.choice(['U', 'V', 'W'])
                if pert_matrix == 'U':
                    pert_row = random.randint(0, dim_u - 1)
                    U_pert = U_test.copy()
                    U_pert[pert_row, pert_col] += random.choice([-1, 1]) * 0.5
                    if verify_against_expected(U_pert, V_test, W_test):
                        current_U = U_pert
                        current_V = V_test.copy()
                        current_W = W_test.copy()
                        current_rank -= 1
                elif pert_matrix == 'V':
                    pert_row = random.randint(0, dim_v - 1)
                    V_pert = V_test.copy()
                    V_pert[pert_row, pert_col] += random.choice([-1, 1]) * 0.5
                    if verify_against_expected(U_test, V_pert, W_test):
                        current_U = U_test.copy()
                        current_V = V_pert
                        current_W = W_test.copy()
                        current_rank -= 1
                else:
                    pert_row = random.randint(0, dim_w - 1)
                    W_pert = W_test.copy()
                    W_pert[pert_row, pert_col] += random.choice([-1, 1]) * 0.5
                    if verify_against_expected(U_test, V_test, W_pert):
                        current_U = U_test.copy()
                        current_V = V_test.copy()
                        current_W = W_pert
                        current_rank -= 1

    # Final verification
    if current_rank < 44:
        if verify_against_expected(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U, "V": V, "W": W}
