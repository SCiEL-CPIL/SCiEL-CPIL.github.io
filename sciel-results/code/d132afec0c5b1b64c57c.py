"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find lower rank decomposition
    # Known lower bound for 2x4x4 is 25 (based on algebraic complexity literature)
    # We will try ranks from 25 up to 31.
    
    # Variables:
    # U has n*m * R entries
    # V has m*p * R entries
    # W has p*n * R entries
    # Total vars = R * (nm + mp + pn)
    
    # Equations:
    # For each i, k: sum_j (A_ij B_jk) = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a quadratic system.
    # To use ILP, we can fix U and V to simple structures? No, that's restrictive.
    # Alternative: Randomized search for U, V, then solve for W?
    # If we fix U and V, W is determined linearly?
    # W[k,i,r] = (sum_j U[i,j,r] V[j,k,r]) / (some normalization)? No, W is a factor.
    
    # Actually, the equation is:
    # C_ik = sum_r U_ir V_rk W_ri  (using Einstein notation loosely, indices vary)
    # Let's stick to the trace convention: W[k*n+i, r]
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # If we fix U and V, can we solve for W?
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_ij_k? No, it's the multiplication tensor.
    # The multiplication tensor M has M[i, j, k] = 1 if i,j,k are valid indices for the product?
    # No, the tensor represents the map (A, B) -> C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # In tensor notation, M is a 3-tensor where M[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Wait, the standard decomposition is:
    # C[i, k] = sum_r U[i, r] V[r, k] ... no.
    
    # The task defines:
    # U: (n*m, R) -> U[i*m+j, r]
    # V: (m*p, R) -> V[j*p+k, r]
    # W: (p*n, R) -> W[k*n+i, r]
    
    # The condition is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a "valid" triple?
    # No, the matrix multiplication tensor T is defined such that:
    # T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    # Actually, the tensor M of matrix multiplication is such that:
    # M[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # So M[i, j, k] = 1 for all valid i, j, k.
    # Wait, if M[i, j, k] = 1 for all i, j, k, then:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for ALL i, j, k.
    # This implies that the bilinear form defined by the decomposition equals the identity map on the tensor space?
    # No, that's not right.
    
    # Let's re-read the standard definition.
    # Matrix multiplication is a bilinear map: (A, B) -> C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # We want to represent this as sum_r U_r(A) V_r(B) W_r(C)? No.
    # We want C = sum_r U_r(A) * V_r(B) * ...?
    # The rank of a tensor T is the smallest R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # Here T is the multiplication tensor.
    # T(i, j, k) = 1 if the term A[i,j] B[j,k] appears in C[i,k].
    # So T(i, j, k) = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, if T(i, j, k) = 1 for all i, j, k, then the tensor is just the all-ones tensor?
    # No, that would mean C[i,k] = sum_{j,r} A[i,j] B[j,k] ... which is correct.
    # But the indices of the tensor factors must match the structure.
    # U is indexed by (i, j) and r.
    # V is indexed by (j, k) and r.
    # W is indexed by (k, i) and r.
    
    # The condition is:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    
    # This is a system of n*m*p equations.
    # Unknowns: R * (nm + mp + pn).
    # If we fix R, we can try to find integer solutions.
    
    # This is a hard non-linear problem.
    # However, if we fix U and V to be sparse or simple, we can solve for W.
    # But W is also integer/half-integer.
    
    # Alternative approach: Random search for U and V with small integer entries, then check if W exists.
    # For a fixed U, V, the equations are:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    # This is linear in W!
    # Let X[i, j, k, r] = U[i, j, r] V[j, k, r].
    # Then for each (i, k), we have a system over j:
    # sum_j sum_r X[i, j, k, r] W[k, i, r] = 1?
    # No, the sum is over r. The index j is summed in the matrix multiplication, but in the tensor decomposition, the indices are explicit.
    # The tensor equation is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # Since T[i, j, k] = 1 for all i, j, k, we have:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    
    # This is a set of n*m*p equations.
    # Unknowns are W[k, i, r] for all k, i, r.
    # The number of unknowns is p * n * R.
    # The number of equations is n * m * p.
    # If R is small, we might have more equations than unknowns?
    # n=2, m=4, p=4.
    # Equations: 2*4*4 = 32.
    # Unknowns: 4*2*R = 8R.
    # If R=4, unknowns=32. Square system.
    # If R=3, unknowns=24. Underdetermined? No, 32 equations, 24 unknowns. Overdetermined.
    # So R must be at least 4?
    # But the schoolbook rank is 32.
    # Is it possible to have R < 32?
    # Yes, the rank of the 2x4x4 multiplication tensor is known to be 25 (or close to it).
    # Wait, if R=25, unknowns = 8*25 = 200. Equations = 32.
    # So for any R >= 4, the system is underdetermined in terms of W if we fix U and V.
    # But U and V are also unknowns.
    
    # Strategy:
    # 1. Iterate R from 25 to 31.
    # 2. Randomly generate U and V with small integer entries (e.g., -1, 0, 1).
    # 3. Solve the linear system for W.
    # 4. Check if W has integer or half-integer entries.
    # 5. If yes, verify the decomposition.
    
    # The linear system for W:
    # Let A be a matrix of size (n*m*p) x (p*n*R).
    # Rows indexed by (i, j, k).
    # Columns indexed by (k, i, r).
    # A[(i, j, k), (k', i', r')] = U[i, j, r'] V[j, k', r'] if k==k' and i==i' else 0?
    # No, the equation is sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
    # For a fixed (i, j, k), the terms involve W[k, i, r] for r=1..R.
    # So the equation is:
    # sum_{r=1}^R (U[i, j, r] V[j, k, r]) W[k, i, r] = 1.
    
    # This means the equations decouple by (k, i)!
    # For each pair (k, i), we have a system of m equations (one for each j) and R unknowns (W[k, i, 1..R]).
    # Let's check:
    # For fixed k, i, the equations are for j=0..m-1:
    # sum_{r=1}^R U[i, j, r] V[j, k, r] W[k, i, r] = 1.
    # This is a linear system: M_{k,i} W_{k,i} = 1, where M_{k,i} is an m x R matrix.
    # M_{k,i}[j, r] = U[i, j, r] V[j, k, r].
    
    # So we need to find U, V such that for all k, i, the system M_{k,i} W_{k,i} = 1 has a solution W_{k,i} with integer/half-integer entries.
    # And we want to minimize R.
    
    # This is much simpler!
    # We can try to construct U and V such that M_{k,i} is invertible or has a nice structure.
    # If M_{k,i} is square (m=R), we can solve for W.
    # If m=4, R=4, we can try to make M_{k,i} invertible with integer inverse?
    # Or just solve and check if W is rational with small denominator.
    
    # Let's try R=4 first?
    # If R=4, m=4. M_{k,i} is 4x4.
    # We need to find U, V such that for all k, i, M_{k,i} is invertible and W_{k,i} = M_{k,i}^{-1} * 1 has half-integer entries.
    
    # Let's try a structured U, V.
    # U[i, j, r] = delta_{i, r}? (i in 0..1, r in 0..3). So U has non-zero only if r=i.
    # V[j, k, r] = delta_{k, r}? (k in 0..3, r in 0..3).
    # Then M_{k,i}[j, r] = U[i, j, r] V[j, k, r].
    # If r != i, U[i, j, r] = 0. So only r=i contributes.
    # M_{k,i}[j, i] = U[i, j, i] V[j, k, i].
    # We need M_{k,i} to be invertible. But it has only one non-zero column (r=i). So it's not invertible.
    
    # Let's try U[i, j, r] = 1 for all i, j, r.
    # V[j, k, r] = 1 for all j, k, r.
    # Then M_{k,i}[j, r] = 1.
    # M_{k,i} is a 4x4 all-ones matrix. Rank 1. Not invertible.
    
    # We need M_{k,i} to have full rank.
    # Let's try random U, V with entries in {-1, 0, 1}.
    # For each R, generate many random U, V.
    # For each k, i, check if M_{k,i} is invertible (if R>=m) or has a solution (if R>m).
    # If R=m, check if det(M_{k,i}) != 0.
    # Then compute W_{k,i} and check if entries are half-integers.
    
    # Let's implement this search.
    
    def is_half_integer(x, tol=1e-9):
        return np.allclose(x * 2, np.round(x * 2), atol=tol)

    def try_rank(R, max_attempts=1000):
        # U: (n, m, R)
        # V: (m, p, R)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random integer entries
            U = np.random.randint(-1, 2, size=(n, m, R))
            V = np.random.randint(-1, 2, size=(m, p, R))
            
            valid = True
            W = np.zeros((p, n, R))
            
            for k in range(p):
                for i in range(n):
                    # Build M_{k,i} of size (m, R)
                    M = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i, j, r] * V[j, k, r]
                    
                    # Solve M @ W_ki = 1
                    # M is m x R. W_ki is R x 1.
                    try:
                        if R == m:
                            det = np.linalg.det(M)
                            if abs(det) < 1e-9:
                                valid = False
                                break
                            W_ki = np.linalg.solve(M, np.ones(m))
                        else:
                            # Least squares
                            W_ki, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            # Check if solution exists (residuals should be small)
                            if rank < m:
                                valid = False
                                break
                            # Check if M @ W_ki is close to 1
                            if not np.allclose(M @ W_ki, np.ones(m), atol=1e-6):
                                valid = False
                                break
                        
                        if not is_half_integer(W_ki):
                            valid = False
                            break
                            
                        W[k, i, :] = W_ki
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to required format
                U_out = [[0] * R for _ in range(n * m)]
                V_out = [[0] * R for _ in range(m * p)]
                W_out = [[0] * R for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            U_out[i * m + j][r] = U[i, j, r]
                
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            V_out[j * p + k][r] = V[j, k, r]
                
                for k in range(p):
                    for i in range(n):
                        for r in range(R):
                            W_out[k * n + i][r] = W[k, i, r]
                
                return U_out, V_out, W_out
        
        return None

    # Try ranks from 25 down to 4?
    # Actually, let's try small R first to see if it's possible.
    # Literature suggests rank is 25 for 2x4x4? Or is it lower?
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, rank is 14?
    # For 2x4x4, it might be around 20-25.
    # Let's try R from 4 to 31.
    
    for R in range(4, 32):
        if time.time() > deadline - 10:
            break
        result = try_rank(R, max_attempts=500)
        if result:
            U_out, V_out, W_out = result
            # Verify
            # Reconstruct tensor and check
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_out[i * m + j][r] * V_out[j * p + k][r] * W_out[k * n + i][r]
                        T_recon[i, j, k] = val
            
            if np.allclose(T_recon, 1.0):
                best_U, best_V, best_W = U_out, V_out, W_out
                best_rank = R
                break  # Found lowest R, stop

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
