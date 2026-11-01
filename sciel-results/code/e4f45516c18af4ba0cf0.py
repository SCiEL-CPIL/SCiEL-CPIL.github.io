"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    def validate(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_222():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W
    
    # Verify Strassen
    sU, sV, sW = strassen_222()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = np.sum(sU[i*2+j, :] * sV[j*2+k, :] * sW[k*2+i, :])
                assert np.isclose(total, 1.0), f"Strassen failed at ({i},{j},{k}): {total}"
    
    # Search for 2x3x2 decomposition
    def search_232(r_max=12, time_budget=30):
        """Find a low-rank decomposition of the 2x3x2 all-ones tensor."""
        n2, m2, p2 = 2, 3, 2
        best = None
        start = time.time()
        
        for r in range(9, r_max + 1):
            if time.time() - start > time_budget:
                break
            attempts = 200
            for att in range(attempts):
                if time.time() - start > time_budget:
                    break
                
                # Random U, V with entries in {-1, 0, 1}
                U_rand = np.zeros((n2*m2, r), dtype=int)
                V_rand = np.zeros((m2*p2, r), dtype=int)
                
                density = 0.4
                mask_u = np.random.rand(n2*m2, r) < density
                mask_v = np.random.rand(m2*p2, r) < density
                U_rand[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
                V_rand[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
                
                # Build linear system for W
                num_vars = p2 * n2 * r  # W has p2*n2 rows, r cols
                num_eqs = n2 * m2 * p2
                
                A = np.zeros((num_eqs, num_vars))
                b = np.ones(num_eqs)
                
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            eq_idx = i * m2 * p2 + j * p2 + k
                            u_val = U_rand[i*m2+j, :]
                            v_val = V_rand[j*p2+k, :]
                            coeffs = u_val * v_val
                            base = (k * n2 + i) * r
                            A[eq_idx, base:base+r] = coeffs
                
                try:
                    W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                
                W_check = W_vec * 2
                W_rounded = np.round(W_check)
                if np.max(np.abs(W_check - W_rounded)) < 1e-6:
                    W_snap = W_rounded / 2.0
                    W_mat = W_snap.reshape(p2*n2, r)
                    
                    # Validate
                    ok = True
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                total = np.sum(U_rand[i*m2+j, :] * V_rand[j*p2+k, :] * W_mat[k*n2+i, :])
                                if not np.isclose(total, 1.0, atol=1e-9):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        best = (U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r)
                        return best
        
        return best
    
    # Try to find 2x3x2 decomposition
    result_232 = search_232(r_max=12, time_budget=25)
    
    if result_232 is not None:
        U232, V232, W232, r232 = result_232
    else:
        # Fallback: schoolbook for 2x3x2
        r232 = 12
        U232 = [[0]*r232 for _ in range(6)]
        V232 = [[0]*r232 for _ in range(6)]
        W232 = [[0]*r232 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U232[i*3+j][col] = 1
                    V232[j*2+k][col] = 1
                    W232[k*2+i][col] = 1
                    col += 1
    
    # Build full 2x5x6 decomposition
    # Split m=5 into m1=2, m2=3
    # Split p=6 into three groups of 2
    # Part 1: 2x2x6 (three 2x2x2 blocks using Strassen)
    # Part 2: 2x3x6 (three 2x3x2 blocks)
    
    r_strassen = 7
    r_total = 3 * r_strassen + 3 * r232
    
    U_full = [[0] * r_total for _ in range(n * m)]  # 10 x r_total
    V_full = [[0] * r_total for _ in range(m * p)]  # 30 x r_total
    W_full = [[0] * r_total for _ in range(p * n)]  # 12 x r_total
    
    sU, sV, sW = strassen_222()
    
    # Part 1: 2x2x6
    # A rows 0,1; A cols 0,1 (j=0,1); B rows 0,1; B cols split into 3 groups of 2
    for block in range(3):
        # B cols: 2*block to 2*block+1
        offset = block * r_strassen
        for c in range(r_strassen):
            # U: A[i, j] for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U_full[i*m + j][offset + c] = sU[i*2+j, c]
            
            # V: B[j, k] for j in {0,1}, k in {2*block, 2*block+1}
            for j in range(2):
                for k in range(2):
                    V_full[j*p + (2*block + k)][offset + c] = sV[j*2+k, c]
            
            # W: C[i, k] for i in {0,1}, k in {2*block, 2*block+1}
            for i in range(2):
                for k in range(2):
                    W_full[(2*block + k)*n + i][offset + c] = sW[k*2+i, c]
    
    # Part 2: 2x3x6
    # A rows 0,1; A cols 2,3,4 (j=2,3,4); B rows 2,3,4; B cols split into 3 groups of 2
    U232_arr = np.array(U232)
    V232_arr = np.array(V232)
    W232_arr = np.array(W232)
    
    for block in range(3):
        # B cols: 2*block to 2*block+1
        offset = 3 * r_strassen + block * r232
        for c in range(r232):
            # U: A[i, j] for i in {0,1}, j in {2,3,4}
            for i in range(2):
                for j in range(3):
                    U_full[i*m + (2 + j)][offset + c] = U232_arr[i*3+j, c]
            
            # V: B[j, k] for j in {2,3,4}, k in {2*block, 2*block+1}
            for j in range(3):
                for k in range(2):
                    V_full[(2 + j)*p + (2*block + k)][offset + c] = V232_arr[j*2+k, c]
            
            # W: C[i, k] for i in {0,1}, k in {2*block, 2*block+1}
            for i in range(2):
                for k in range(2):
                    W_full[(2*block + k)*n + i][offset + c] = W232_arr[k*2+i, c]
    
    # Validate full decomposition
    if validate(U_full, V_full, W_full, r_total):
        if r_total < best_result["rank"]:
            best_result = {"U": U_full, "V": V_full, "W": W_full, "rank": r_total}
    
    # Try direct search for even lower rank
    if time.time() - start_time < time_limit - 10:
        for r in range(r_total - 1, max(40, r_total - 5), -1):
            if time.time() - start_time > time_limit - 5:
                break
            
            attempts = 100
            for _ in range(attempts):
                if time.time() - start_time > time_limit - 5:
                    break
                
                U_rand = np.zeros((n*m, r), dtype=int)
                V_rand = np.zeros((m*p, r), dtype=int)
                
                mask_u = np.random.rand(n*m, r) < 0.3
                mask_v = np.random.rand(m*p, r) < 0.3
                U_rand[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
                V_rand[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
                
                num_vars = p * n * r
                num_eqs = n * m * p
                A = np.zeros((num_eqs, num_vars))
                b = np.ones(num_eqs)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            coeffs = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                            base = (k * n + i) * r
                            A[eq_idx, base:base+r] = coeffs
                
                try:
                    W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                
                W_check = W_vec * 2
                W_rounded = np.round(W_check)
                if np.max(np.abs(W_check - W_rounded)) < 1e-6:
                    W_snap = W_rounded / 2.0
                    W_mat = W_snap.reshape(p*n, r)
                    
                    if validate(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                        best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_mat.tolist(), "rank": r}
                        break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
