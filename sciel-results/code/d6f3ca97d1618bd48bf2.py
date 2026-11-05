import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 5
    
    def verify_numpy(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        return True
    
    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, rank, max_attempts=200):
        """Attempt to reduce rank by finding linear dependencies among columns."""
        current_rank = rank
        U_cur = [row[:] for row in U]
        V_cur = [row[:] for row in V]
        W_cur = [row[:] for row in W]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Check if any column is a linear combination of others
            # Use numpy for speed
            Ua = np.array(U_cur, dtype=float)
            Va = np.array(V_cur, dtype=float)
            Wa = np.array(W_cur, dtype=float)
            
            # Check if columns are linearly dependent
            # For each column, see if it can be expressed as combination of others
            found_reduction = False
            for r in range(current_rank - 1, -1, -1):
                # Check if column r is in span of other columns
                # Build matrix of all columns except r
                if r == 0:
                    U_other = Ua[:, 1:]
                    V_other = Va[:, 1:]
                    W_other = Wa[:, 1:]
                else:
                    U_other = np.hstack([Ua[:, :r], Ua[:, r+1:]])
                    V_other = np.hstack([Va[:, :r], Va[:, r+1:]])
                    W_other = np.hstack([Wa[:, :r], Wa[:, r+1:]])
                
                U_target = Ua[:, r]
                V_target = Va[:, r]
                W_target = Wa[:, r]
                
                # Try to find coefficients
                try:
                    # Solve least squares
                    coeff_U, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                    coeff_V, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                    coeff_W, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
                    
                    # Check if same coefficients work for all three
                    if np.allclose(coeff_U, coeff_V, atol=1e-10) and np.allclose(coeff_V, coeff_W, atol=1e-10):
                        # Found a dependency! Remove column r
                        coeff = coeff_U
                        
                        # Verify exactly with fractions
                        U_exact = np.array(U_cur, dtype=object)
                        V_exact = np.array(V_cur, dtype=object)
                        W_exact = np.array(W_cur, dtype=object)
                        
                        # Reconstruct column r from others
                        U_recon = np.zeros(n*m, dtype=object)
                        V_recon = np.zeros(m*p, dtype=object)
                        W_recon = np.zeros(p*n, dtype=object)
                        
                        col_idx = 0
                        for r2 in range(current_rank):
                            if r2 != r:
                                for i in range(n*m):
                                    U_recon[i] += Fraction(U_cur[i][r2]) * Fraction(coeff[col_idx])
                                for i in range(m*p):
                                    V_recon[i] += Fraction(V_cur[i][r2]) * Fraction(coeff[col_idx])
                                for i in range(p*n):
                                    W_recon[i] += Fraction(W_cur[i][r2]) * Fraction(coeff[col_idx])
                                col_idx += 1
                        
                        # Check if reconstruction is exact
                        if all(Fraction(U_cur[i][r]) == U_recon[i] for i in range(n*m)) and \
                           all(Fraction(V_cur[i][r]) == V_recon[i] for i in range(m*p)) and \
                           all(Fraction(W_cur[i][r]) == W_recon[i] for i in range(p*n)):
                            # Remove column r
                            U_cur = [row[:r] + row[r+1:] for row in U_cur]
                            V_cur = [row[:r] + row[r+1:] for row in V_cur]
                            W_cur = [row[:r] + row[r+1:] for row in W_cur]
                            current_rank -= 1
                            found_reduction = True
                            break
                except:
                    pass
            
            if not found_reduction:
                break
        
        if current_rank < rank:
            if verify_numpy(U_cur, V_cur, W_cur, current_rank):
                if verify_exact(U_cur, V_cur, W_cur, current_rank):
                    return U_cur, V_cur, W_cur, current_rank
        return None
    
    # Build the rank-36 Strassen-based solution
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W36[gk*n + gi][col] += val_w
                    col += 1
            else:
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W36[gk*n + gi][col] += val_w
                    col += 1
    
    best_result = None
    if col == rank36 and verify_numpy(U36, V36, W36, rank36):
        if verify_exact(U36, V36, W36, rank36):
            best_result = {"U": U36, "V": V36, "W": W36}
            # Try to reduce rank
            reduced = try_reduce_rank(U36, V36, W36, rank36, max_attempts=50)
            if reduced is not None:
                U_red, V_red, W_red, rank_red = reduced
                if rank_red < rank36:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
    
    # Schoolbook fallback
    rank_school = n * m * p
    if best_result is None:
        best_result = {
            "U": [[0]*rank_school for _ in range(n*m)],
            "V": [[0]*rank_school for _ in range(m*p)],
            "W": [[0]*rank_school for _ in range(p*n)]
        }
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_result["U"][i*m+j][c] = 1
                    best_result["V"][j*p+k][c] = 1
                    best_result["W"][k*n+i][c] = 1
                    c += 1
    return best_result
