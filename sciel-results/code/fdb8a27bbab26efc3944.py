import numpy as np
import time
from scipy.optimize import least_squares
import random

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target rank to try
    # We know 39 works. Let's try 37.
    # If 37 fails, we fall back to 39.
    
    def build_equations(R):
        # Variables: U (4*R), V (22*R), W (22*R)
        # Total vars: 48*R
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        # Equations: For each (i, k, i', j', j'', k'')
        # LHS: sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
        # RHS: 1 if i==i', j'==j'', k==k'' else 0
        # Note: j' and j'' are the same index j in the standard sum.
        # In the bilinear expansion, A_{i', j'} and B_{j'', k''} are independent variables.
        # The term A_{i, j} B_{j, k} appears in C_{i, k}.
        # So we match coefficients of A_{i', j'} B_{j'', k''}.
        # The sum in matrix mult is over j. So only terms where j' == j'' contribute?
        # No, the bilinear map is defined on the full spaces.
        # The identity is:
        # sum_{j} A_{i, j} B_{j, k} = sum_{r} W_{i, k, r} (sum_{i', j'} U_{i', j', r} A_{i', j'}) (sum_{j'', k''} V_{j'', k'', r} B_{j'', k''})
        # Expanding the RHS:
        # sum_{r} W_{i, k, r} sum_{i', j', j'', k''} U_{i', j', r} V_{j'', k'', r} A_{i', j'} B_{j'', k''}
        # We need this to equal sum_{j} A_{i, j} B_{j, k}.
        # So the coefficient of A_{i', j'} B_{j'', k''} in the expansion must be:
        # 1 if i'=i, j'=j'', k''=k.
        # 0 otherwise.
        
        # We generate one equation for each (i, k, i', j', j'', k'')
        # That is 2 * 11 * 4 * 2 * 2 * 11 = 41360 equations.
        # This is too many for least_squares directly if we are not careful.
        # But we can vectorize.
        
        # Let's define the function to compute residuals.
        def residual_func(x):
            # x is flattened [U, V, W]
            U = x[:num_u * R].reshape(num_u, R)
            V = x[num_u * R : num_u * R + num_v * R].reshape(num_v, R)
            W = x[num_u * R + num_v * R :].reshape(num_w, R)
            
            # We need to check:
            # sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r] == delta(i, i') delta(j', j'') delta(k, k'')
            
            # This is a 4-way tensor contraction.
            # Let's iterate over i, k and check the resulting matrix of coefficients.
            # For fixed i, k, we form M_{i', j', j'', k''} = sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
            # This is effectively:
            # M_{i', j'} (j'', k'')? No.
            # Let A_r = U[:, r] (shape 4)
            # Let B_r = V[:, r] (shape 22)
            # Let w_r = W[k*n+i, r]
            # The term is w_r * A_r * B_r^T?
            # A_r has indices (i', j'). B_r has indices (j'', k'').
            # So for fixed i, k, we compute:
            # S = sum_r w_r * outer(A_r, B_r)
            # S has shape (4, 22).
            # Index (i'*m+j', j''*p+k'') of S is sum_r w_r U[i'*m+j', r] V[j''*p+k'', r].
            # We want S[i'*m+j', j''*p+k''] to be 1 if i'=i, j'=j'', k''=k.
            
            residuals = []
            for i in range(n):
                for k in range(p):
                    # Compute S for this (i, k)
                    # S = sum_r W[k*n+i, r] * outer(U[:, r], V[:, r])
                    # U[:, r] is 4x1, V[:, r] is 22x1.
                    # outer is 4x22.
                    # We can do this with einsum or broadcasting.
                    # W_idx = W[k*n+i, :] (shape R)
                    # U (4, R), V (22, R)
                    # S = (U * W_idx[None, :]).T @ V ?
                    # S[i', j', j'', k''] = sum_r U[i', j', r] W[i, k, r] V[j'', k'', r]
                    # This is (U * W[None, :]) @ V.T
                    # U * W[None, :] is 4xR.
                    # V.T is R x 22.
                    # Result is 4x22.
                    w_vec = W[k * n + i, :]
                    S = (U * w_vec[None, :]) @ V.T
                    
                    # Target matrix T_target of shape 4x22
                    # T_target[i'*m+j', j''*p+k''] = 1 if i'=i, j'=j'', k''=k
                    # Note: j' and j'' must be equal.
                    # So for each j in 0..m-1 (0,1):
                    #   row idx = i*m + j
                    #   col idx = j*p + k
                    #   target = 1
                    # All other entries 0.
                    
                    # We extract the residual for this (i, k) block
                    # We want S - T_target to be 0.
                    # Instead of building full T_target, we just subtract the specific entries.
                    # But least_squares needs a flat vector.
                    # We can just append S.ravel() and subtract the targets?
                    # No, we only have non-zero targets at specific spots.
                    # It is more efficient to just check those spots?
                    # But the off-diagonal terms must also be zero.
                    # So we must include all 4*22 = 88 entries for each (i, k).
                    # Total 22 * 88 = 1936 residuals.
                    
                    residuals.append(S.ravel())
            
            return np.concatenate(residuals)
        
        # This function returns residuals of shape (n*p * num_u * num_v, )
        # We need to subtract the target values.
        # Let's modify the function to return S - T.
        
        def residual_func_with_target(x):
            U = x[:num_u * R].reshape(num_u, R)
            V = x[num_u * R : num_u * R + num_v * R].reshape(num_v, R)
            W = x[num_u * R + num_v * R :].reshape(num_w, R)
            
            residuals = np.zeros((n * p, num_u * num_v))
            
            for i in range(n):
                for k in range(p):
                    w_vec = W[k * n + i, :]
                    S = (U * w_vec[None, :]) @ V.T
                    residuals[i * p + k, :] = S.ravel()
            
            # Subtract targets
            # Target for (i, k) is 1 at (i*m+j, j*p+k) for j in 0..m-1
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        row_idx = i * m + j
                        col_idx = j * p + k
                        residuals[i * p + k, row_idx * num_v + col_idx] -= 1.0
            
            return residuals.ravel()

        return residual_func_with_target, num_u * R + num_v * R + num_w * R

    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p # 44
    
    # Try ranks from 37 up to 39
    for R in [37, 38, 39]:
        if time.time() > deadline - 10:
            break
            
        try:
            res_func, num_vars = build_equations(R)
            
            # Initial guess: Random
            # We can use the known 39-rank solution as a start for R=39,
            # but for R=37 we need a random start or a projected start.
            # Let's use random start for 37.
            
            x0 = np.random.randn(num_vars)
            
            # Scale? The solution is integers. Random normal is fine.
            
            # Use least_squares
            # This might be slow. 1936 residuals, ~1800 vars.
            # Jacobian is sparse?
            # We'll use the default 'trf' method.
            
            sol = least_squares(res_func, x0, method='trf', max_nfev=1000)
            
            if sol.cost < 1e-6:
                # Found a solution
                x = sol.x
                num_u = n * m
                num_v = m * p
                num_w = p * n
                U = x[:num_u * R].reshape(num_u, R)
                V = x[num_u * R : num_u * R + num_v * R].reshape(num_v, R)
                W = x[num_u * R + num_v * R :].reshape(num_w, R)
                
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                # Check if the rounded solution satisfies the equations
                # We can just check the residual of the rounded solution
                x_rounded = np.concatenate([U_int.ravel(), V_int.ravel(), W_int.ravel()])
                res_rounded = res_func(x_rounded)
                if np.max(np.abs(res_rounded)) < 1e-9:
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    best_rank = R
                    break # Found minimal rank, stop
                else:
                    # Maybe the solution is not integer.
                    # Keep the real solution? No, task requires integer/half-integer.
                    # If rounding fails, this rank might not have an integer solution.
                    pass
                    
        except Exception as e:
            print(f"Error for R={R}: {e}")
            continue

    # If no rank < 44 found, use the schoolbook 44
    if best_rank == n * m * p:
        # Fallback to 39 construction if 37/38 failed
        # We can hardcode the 39 construction or just return 44.
        # The prompt says "Find a rank-minimal...".
        # If we can't find 37, we should return 39 if we can construct it.
        # Let's construct 39 explicitly.
        
        rank = 39
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # 5 blocks of 2x2x2
        for block in range(5):
            # Strassen for 2x2x2
            # A indices: 0,1,2,3 (i*m+j)
            # B indices: 0..7 (j*p+k) -> for this block, j in 0,1, k in 0,1
            # W indices: 0..3 (k*n+i)
            
            # We map the local 2x2x2 to the global indices.
            # Block b uses B columns 2*b, 2*b+1.
            # A is shared.
            # C rows are 2*b, 2*b+1.
            
            # Strassen terms for A (shared):
            # L1 = a11+a22 -> U[0]+=1, U[3]+=1
            # L2 = a21+a22 -> U[2]+=1, U[3]+=1
            # L3 = a11 -> U[0]+=1
            # L4 = a22 -> U[3]+=1
            # L5 = a11+a12 -> U[0]+=1, U[1]+=1
            # L6 = a21-a11 -> U[2]+=1, U[0]-=1
            # L7 = a12-a22 -> U[1]+=1, U[3]-=1
            
            # B terms for block b:
            # b11 = B[0, 2b] -> V[0*p + 2b]
            # b12 = B[0, 2b+1] -> V[0*p + 2b+1]
            # b21 = B[1, 2b] -> V[1*p + 2b]
            # b22 = B[1, 2b+1] -> V[1*p + 2b+1]
            
            # M1 = L1 * (b11+b22)
            # M2 = L2 * b11
            # M3 = L3 * (b21-b22)
            # M4 = L4 * (b12-b11)
            # M5 = L5 * b22
            # M6 = L6 * (b11+b12)
            # M7 = L7 * (b21+b22)
            
            # W terms for block b:
            # c11 = C[2b, 0] -> W[0*n + 2b]
            # c12 = C[2b, 1] -> W[1*n + 2b]
            # c21 = C[2b+1, 0] -> W[0*n + 2b+1]
            # c22 = C[2b+1, 1] -> W[1*n + 2b+1]
            
            # M1 -> c11 +=, c12 +=, c21 +=, c22 +=
            # M1 = L1 * (b11+b22)
            # c11 = M1 - M4 - M5 + M7
            # c12 = M5 + M6
            # c21 = M2 + M7
            # c22 = M1 - M3 - M6 + M4
            
            # Let's assign columns
            # Term 1: M1
            U[0, col] += 1; U[3, col] += 1
            V[2*b, col] += 1; V[p + 2*b + 1, col] += 1
            W[2*b, col] += 1; W[n + 2*b, col] += 1; W[2*b + 1, col] += 1; W[n + 2*b + 1, col] += 1
            col += 1
            
            # Term 2: M2
            U[2, col] += 1; U[3, col] += 1
            V[2*b, col] += 1
            W[2*b + 1, col] += 1
            col += 1
            
            # Term 3: M3
            U[0, col] += 1
            V[p + 2*b, col] += 1; V[p + 2*b + 1, col] -= 1
            W[2*b + 1, col] -= 1
            col += 1
            
            # Term 4: M4
            U[3, col] += 1
            V[2*b + 1, col] += 1; V[2*b, col] -= 1
            W[2*b, col] -= 1; W[2*b + 1, col] += 1
            col += 1
            
            # Term 5: M5
            U[0, col] += 1; U[1, col] += 1
            V[p + 2*b + 1, col] += 1
            W[2*b, col] += 1; W[2*b + 1, col] += 1
            col += 1
            
            # Term 6: M6
            U[2, col] += 1; U[0, col] -= 1
            V[2*b, col] += 1; V[2*b + 1, col] += 1
            W[2*b, col] += 1; W[2*b + 1, col] -= 1
            col += 1
            
            # Term 7: M7
            U[1, col] += 1; U[3, col] -= 1
            V[p + 2*b, col] += 1; V[p + 2*b + 1, col] += 1
            W[2*b, col] += 1; W[2*b + 1, col] += 1
            col += 1
            
        # Last block: 2x2x1 (vector mult)
        # B column 10.
        # A is shared.
        # C column 10.
        # c1 = a11*x + a12*y
        # c2 = a21*x + a22*y
        # 4 terms: a11*x, a12*y, a21*x, a22*y
        
        # Term 8: a11*x
        U[0, col] += 1
        V[10, col] += 1
        W[10, col] += 1
        col += 1
        
        # Term 9: a12*y
        U[1, col] += 1
        V[p + 10, col] += 1
        W[10, col] += 1
        col += 1
        
        # Term 10: a21*x
        U[2, col] += 1
        V[10, col] += 1
        W[n + 10, col] += 1
        col += 1
        
        # Term 11: a22*y
        U[3, col] += 1
        V[p + 10, col] += 1
        W[n + 10, col] += 1
        col += 1
        
        best_U = U
        best_V = V
        best_W = W
        best_rank = 39

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
