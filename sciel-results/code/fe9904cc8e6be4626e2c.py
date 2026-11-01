import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def to_list(mat):
        out = []
        for row in mat:
            out_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    out_row.append(int(round(x)))
                elif abs(x*2 - round(x*2)) < 1e-9:
                    out_row.append(int(round(x*2))/2)
                else:
                    out_row.append(float(x))
            out.append(out_row)
        return out
    
    def check_solution(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for j3 in range(m):
                            for k2 in range(p):
                                lhs = 0.0
                                for c in range(r):
                                    lhs += W[k*n+i, c] * U[i2*m+j2, c] * V[j3*p+k2, c]
                                expected = 1.0 if (i2==i and j3==j2 and k2==k) else 0.0
                                if abs(lhs - expected) > tol:
                                    return False
        return True
    
    def solve_for_W(U, V, r):
        """Given U and V of shape (n*m, r) and (m*p, r), solve for W."""
        n, m, p = DIMENSIONS
        # W has shape (p*n, r)
        # Constraints: for each (i,k,i2,j2,j3,k2):
        # sum_c W[k*n+i, c] * U[i2*m+j2, c] * V[j3*p+k2, c] = delta
        # 
        # Let's set up a linear system. Unknowns are W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r).
        # Total unknowns: p*n*r
        
        # Number of constraints: n*p * n*m * m*p = n^2 * m^2 * p^2
        num_constraints = n*n * m*m * p*p
        num_unknowns = p * n * r
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.zeros(num_constraints)
        
        idx = 0
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for j3 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i2==i and j3==j2 and k2==k) else 0.0
                                for c in range(r):
                                    # W[k*n+i, c] is unknown index (k*n+i)*r + c
                                    unknown_idx = (k*n + i) * r + c
                                    A[idx, unknown_idx] = U[i2*m+j2, c] * V[j3*p+k2, c]
                                b[idx] = expected
                                idx += 1
        
        # Solve least squares
        W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape((p*n, r))
        return W
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try ranks from 17 down
    np.random.seed(12345)
    
    for r in range(17, 11, -1):
        if time.time() > deadline:
            break
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Initialize U and V
            if trial == 0:
                # Use schoolbook truncated
                U = U_school[:, :r].copy()
                V = V_school[:, :r].copy()
            else:
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
            
            W = solve_for_W(U, V, r)
            
            if check_solution(U, V, W):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if check_solution(U_r, V_r, W_r):
                    return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                # Return non-rounded if valid
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback
    return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
