import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook rank as fallback
    rank_school = n * m * p  # 64
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: For a given rank r, randomly initialize U and V,
    # then solve for W from the linear system.
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is linear in W entries.
    
    # The W variables are W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,r_rank)
    # Total W variables: p*n*r_rank = pn*r_rank
    # Total equations: n*m*p = 64
    # For r_rank >= 64/(pn) = 64/16 = 4, we have more variables than equations.
    
    # But the system must be consistent. Let's try various ranks.
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_school
    
    np.random.seed(12345)
    
    for target_rank in range(36, min(57, rank_school)):  # Try ranks from 36 to 56
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            r = target_rank
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            # Build the linear system for W
            # For each (i,j,k): sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] = 1
            # W variables indexed by (k*n+i, r')
            
            # Number of equations: n*m*p
            # Number of W variables: p*n*r
            num_eq = n * m * p
            num_w_vars = pn * r
            
            # Build A matrix: A[eq_idx, w_var_idx] = U[i*m+j,r'] * V[j*p+k,r']
            # where eq_idx corresponds to (i,j,k) and w_var_idx corresponds to (k*n+i, r')
            
            A = np.zeros((num_eq, num_w_vars))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for rp in range(r):
                            w_var_idx = (k * n + i) * r + rp
                            A[eq_idx, w_var_idx] = U[i * m + j, rp] * V[j * p + k, rp]
                        eq_idx += 1
            
            # Solve the least squares problem
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W = W_flat.reshape(pn, r)
            
            # Verify the solution
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for rp in range(r):
                            val += U[i*m+j, rp] * V[j*p+k, rp] * W[k*n+i, rp]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                found = True
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        if found:
            # Try to go lower
            continue
    
    # If no better solution found, use the Strassen rank-56 solution
    if best_rank >= 56:
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                # Term 1: M1
                U[0*m + j0, col] = 1
                U[1*m + (j0+1), col] = 1
                V[j0*p + k0, col] = 1
                V[(j0+1)*p + (k0+1), col] = 1
                W[k0*n + 0, col] = 1
                W[(k0+1)*n + 1, col] = 1
                col += 1
                
                # Term 2: M2
                U[1*m + j0, col] = 1
                U[1*m + (j0+1), col] = 1
                V[j0*p + k0, col] = 1
                W[k0*n + 1, col] = 1
                W[(k0+1)*n + 1, col] = -1
                col += 1
                
                # Term 3: M3
                U[0*m + j0, col] = 1
                V[j0*p + (k0+1), col] = 1
                V[(j0+1)*p + (k0+1), col] = -1
                W[(k0+1)*n + 0, col] = 1
                W[(k0+1)*n + 1, col] = 1
                col += 1
                
                # Term 4: M4
                U[1*m + (j0+1), col] = 1
                V[(j0+1)*p + k0, col] = 1
                V[j0*p + k0, col] = -1
                W[k0*n + 0, col] = 1
                W[k0*n + 1, col] = 1
                col += 1
                
                # Term 5: M5
                U[0*m + j0, col] = 1
                U[0*m + (j0+1), col] = 1
                V[(j0+1)*p + (k0+1), col] = 1
                W[k0*n + 0, col] = -1
                W[(k0+1)*n + 0, col] = 1
                col += 1
                
                # Term 6: M6
                U[1*m + j0, col] = 1
                U[0*m + j0, col] = -1
                V[j0*p + k0, col] = 1
                V[j0*p + (k0+1), col] = 1
                W[(k0+1)*n + 1, col] = 1
                col += 1
                
                # Term 7: M7
                U[0*m + (j0+1), col] = 1
                U[1*m + (j0+1), col] = -1
                V[(j0+1)*p + k0, col] = 1
                V[(j0+1)*p + (k0+1), col] = 1
                W[k0*n + 0, col] = 1
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Use the found solution
    r = best_rank
    U_out = np.zeros((n * m, r))
    V_out = np.zeros((m * p, r))
    W_out = np.zeros((p * n, r))
    U_out[:, :] = best_U
    V_out[:, :] = best_V
    W_out[:, :] = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
