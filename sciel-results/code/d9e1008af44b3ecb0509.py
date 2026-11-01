"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import nnls

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as numpy arrays of shape (4,14), (8,14), (8,14).
        """
        R = 14
        U = np.zeros((4, R), dtype=int)
        V = np.zeros((8, R), dtype=int)
        W = np.zeros((8, R), dtype=int)

        # Left block (cols 0,1 of B): Strassen on A * B_left where B_left is 2x2
        # B_left[0,0]=b00, B_left[0,1]=b01, B_left[1,0]=b10, B_left[1,1]=b11
        # V indices: j*p+k -> j*4+k
        #   b00 -> V[0], b01 -> V[1], b10 -> V[4], b11 -> V[5]
        # W indices: k*n+i -> k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        # L1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # L2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1

        # L3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1

        # L4: a11(b10-b00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1

        # L5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1

        # L6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # L7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1

        # Right block (cols 2,3 of B): Strassen on A * B_right where B_right is 2x2
        # B_right[0,0]=b02, B_right[0,1]=b03, B_right[1,0]=b12, B_right[1,1]=b13
        # V indices:
        #   b02 -> V[2], b03 -> V[3], b12 -> V[6], b13 -> V[7]
        # W indices:
        #   C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7]

        # R1: (a00+a11)(b02+b13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1

        # R2: (a10+a11)b02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1

        # R3: a00(b03-b13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1

        # R4: a11(b12-b02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1

        # R5: (a00+a01)b13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1

        # R6: (a10-a00)(b02+b03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1

        # R7: (a01-a11)(b12+b13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1

        return U, V, W

    def verify_np(U, V, W, atol=1e-9):
        """Fast numpy-based verification using einsum."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        T = np.zeros((n, m, p))
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        if not np.allclose(T, 1.0, atol=atol):
            return False
        return True

    def try_random_rank(R, num_trials=50, seed_base=42):
        """
        Try to find a rank-R decomposition by:
        1. Randomly generating R rank-1 terms (U_r, V_r)
        2. Solving for W_r via linear system
        3. Checking if the decomposition is exact
        Returns (U, V, W) if successful, None otherwise.
        """
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            seed = seed_base + trial * 1000
            rng = np.random.RandomState(seed)
            
            # Generate random U and V matrices with entries in {-1, 0, 1}
            # Use sparse random matrices to encourage structure
            U_rand = np.zeros((nm, R), dtype=int)
            V_rand = np.zeros((mp, R), dtype=int)
            
            for r in range(R):
                # Each column of U: pick 2-4 random positions, set to ±1
                u_nz = rng.choice(nm, size=rng.randint(2, 5), replace=False)
                for idx in u_nz:
                    U_rand[idx, r] = rng.choice([-1, 1])
                
                # Each column of V: pick 2-4 random positions, set to ±1
                v_nz = rng.choice(mp, size=rng.randint(2, 5), replace=False)
                for idx in v_nz:
                    V_rand[idx, r] = rng.choice([-1, 1])
            
            # Build linear system to solve for W
            # For each (i,j,k) in tensor, we need:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            # This is linear in W: for each (i,j,k), sum_r (U[i,j,r]*V[j,k,r]) * W[k*n+i, r] = 1
            # But W[k*n+i, r] appears in multiple equations (for different j)
            
            # Better approach: for each fixed (i,k), the equation for all j is:
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (for each j... no)
            
            # Actually, the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i,j,k] = 1 for all i,j,k
            # For fixed (i,k): sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r] * delta_{j,j}) 
            # Wait, this isn't quite right. Let me re-think.
            
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For each (i,j,k), this must equal 1.
            # The unknowns are W[k*n+i, r] for all k,i,r.
            # For a fixed (i,k), W[k*n+i, r] appears in T[i,j,k] for all j.
            # So for fixed (i,k): T[i,j,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
            # This must equal 1 for all j.
            # So we have m equations for each (i,k) pair, with p*n*R unknowns... no.
            
            # Let me think differently. The system is:
            # For each (i,j,k): sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
            # Number of unknowns: p*n*R = 8*R
            # Number of equations: n*m*p = 16
            
            # For R >= 2, we have more unknowns than equations, so the system is underdetermined.
            # We can try to find a solution.
            
            # Build the linear system: A * W_vec = b
            # where W_vec is a vector of length p*n*R
            # A has shape (n*m*p, p*n*R)
            
            A = np.zeros((n*m*p, pn*R), dtype=float)
            b = np.ones(n*m*p, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            # W[k*n+i, r] is at index (k*n+i)*R + r
                            w_idx = (k*n+i)*R + r
                            A[eq_idx, w_idx] = U_rand[i*m+j, r] * V_rand[j*p+k, r]
                        eq_idx += 1
            
            # Solve the underdetermined system. Use least squares or find a particular solution.
            # Since the system is underdetermined, use np.linalg.lstsq to find minimum-norm solution
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact
            if not np.allclose(A @ W_vec, b, atol=1e-10):
                continue
            
            # Reshape W_vec back to matrix
            W_sol = W_vec.reshape(pn, R)
            
            # Check if W_sol has approximately integer/half-integer entries
            # Round to nearest half-integer and verify
            W_rounded = np.round(W_sol * 2) / 2
            
            # Verify the full decomposition with rounded values
            U_check = U_rand.astype(float)
            V_check = V_rand.astype(float)
            W_check = W_rounded.astype(float)
            
            T = np.zeros((n, m, p))
            for r in range(R):
                u = U_check[:, r].reshape(n, m)
                v = V_check[:, r].reshape(m, p)
                w = W_check[:, r].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            if np.allclose(T, 1.0, atol=1e-9):
                return U_rand.tolist(), V_rand.tolist(), W_rounded.tolist()
            
            # Also try without rounding, in case the solution is already close to integer
            if np.allclose(W_sol, np.round(W_sol), atol=1e-8):
                W_int = np.round(W_sol).astype(int)
                T2 = np.zeros((n, m, p))
                for r in range(R):
                    u = U_rand[:, r].reshape(n, m).astype(float)
                    v = V_rand[:, r].reshape(m, p).astype(float)
                    w = W_int[:, r].reshape(p, n).astype(float)
                    T2 += np.einsum('ij,jk,ki->ijk', u, v, w)
                if np.allclose(T2, 1.0, atol=1e-9):
                    return U_rand.tolist(), V_rand.tolist(), W_int.tolist()
        
        return None

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Try the structured rank-14 decomposition
    U, V, W = get_rank14()
    
    # Verify before returning to ensure correctness
    if verify_np(U.tolist(), V.tolist(), W.tolist()):
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = 14
    else:
        best_result = None
        best_rank = 16
    
    # Try to find a lower-rank decomposition via randomized algebraic elimination
    for R in range(13, 14):  # Try rank 13 (and potentially lower if time permits)
        if time.time() > deadline - 20:
            break
        
        result = try_random_rank(R, num_trials=30, seed_base=R*10000)
        if result is not None:
            U_try, V_try, W_try = result
            if verify_np(U_try, V_try, W_try):
                best_result = {"U": U_try, "V": V_try, "W": W_try}
                best_rank = R
                break  # Found lower rank, stop searching
    
    if best_result is not None:
        return best_result
    
    # Fallback to naive rank 16 if something went wrong (should not happen)
    return naive_decomp()
# EVOLVE-BLOCK-END
