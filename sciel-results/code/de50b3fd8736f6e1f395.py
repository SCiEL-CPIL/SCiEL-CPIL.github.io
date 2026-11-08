"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Formulate as linear algebra over Q (rationals)
    # We want to find rank r such that there exist U (nm x r), V (mp x r), W (pn x r)
    # such that for all i, j, k:
    # sum_{l=0}^{r-1} U[i*m+j, l] * V[j*p+k, l] = W[k*n+i, l]  <-- WAIT, this is wrong.
    # The tensor decomposition definition for matrix multiplication is:
    # T_{i,j,k} = sum_{r=1}^R U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ? No.
    # The standard definition is:
    # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T has indices (i, j, k). T_{i,j,k} = 1 if we are multiplying (i,j) of A by (j,k) of B.
    # A decomposition of rank R is a set of triples (u_r, v_r, w_r) where:
    # u_r is a vector of length nm (indices i,j)
    # v_r is a vector of length mp (indices j,k)
    # w_r is a vector of length pn (indices k,i)
    # Such that T_{i,j,k} = sum_{r=1}^R u_r[i,j] v_r[j,k] w_r[k,i]
    
    # So the condition is:
    # For all i, j, k:
    # sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k}
    # where T_{i,j,k} = 1 for all valid i,j,k (since every term in the product contributes to the sum for that specific (i,k) output, but wait...).
    
    # Let's re-read the tensor definition carefully.
    # Matrix multiplication tensor T_{i,j,k} for (n x m) * (m x p) -> (n x p)
    # The entries of the result C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined on indices (i, j, k).
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in the sum for C_{i,k}.
    # Since every j from 0 to m-1 appears in the sum for every (i,k), T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # So we need:
    # sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # This is a system of equations.
    # Let R be the rank.
    # Unknowns: U (nm*R), V (mp*R), W (pn*R). Total 3*R*(nm+mp+pn)/... no, just 3*R*(nm+mp+pn) variables? No, R columns.
    # Variables are U[:, r], V[:, r], W[:, r] for r=1..R.
    # This is a multilinear system.
    
    # However, we can fix two matrices and solve for the third?
    # If we fix U and V, the equation is:
    # sum_{r=1}^R (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    # This is linear in W!
    # Let A_{(i,j,k), r} = U[i*m+j, r] * V[j*p+k, r].
    # Let b_{i,j,k} = 1.
    # We need to solve A * W_flat = b, where W_flat is the vectorized W.
    # But W has structure: W is a matrix of size (pn x R).
    # The index for W in the sum is (k, i). The variable is W[k*n+i, r].
    # So for each r, W[:, r] is a vector of length pn.
    # The equation couples all r's.
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    
    # This is linear in the entries of W if U and V are fixed.
    # Let's check the dimensions.
    # Number of equations: n*m*p = 48.
    # Number of unknowns in W: p*n*R = 16*R.
    # If 16*R >= 48, i.e., R >= 3, we might have a solution.
    # But U and V are also unknowns.
    
    # Alternative approach:
    # Fix W. Then the equation is bilinear in U and V.
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    
    # Let's try a different perspective.
    # For a fixed r, let u_r, v_r, w_r be the r-th columns.
    # The contribution of rank r is the tensor T^{(r)}_{i,j,k} = u_r[i,j] v_r[j,k] w_r[k,i].
    # We need sum_r T^{(r)} = T.
    
    # Heuristic Search:
    # 1. Assume entries are integers or half-integers.
    # 2. Try small ranks R.
    # 3. For a given R, can we find U, V, W?
    
    # Let's try to exploit the structure.
    # T_{i,j,k} = 1.
    # If we choose W such that w_r[k,i] is non-zero only for specific (k,i), it might simplify.
    
    # Known result: Rank of 2x3x8 matrix multiplication.
    # Lower bound: max(n*m, m*p, p*n) is not the lower bound.
    # Lower bound is max(n, m, p) = 8? No.
    # Strassen showed rank < 7 for 2x2x2.
    # For 2x3x8, what is the rank?
    # Let's try to find a decomposition of rank R.
    
    # Let's use the "fix two, solve one" strategy with random initialization and least squares, checking for exactness.
    # Since we need exact integer/half-integer solutions, we can use rational arithmetic or check residuals.
    
    # Let's try ranks from a lower bound.
    # Lower bound for 2x3x8:
    # Rank >= max(n, m, p) = 8 is weak.
    # Rank >= nm + mp + pn - 2? No.
    # Let's just iterate R from 8 to 47.
    
    # To make it efficient, we can fix one matrix (say W) randomly with small integers,
    # and then the system for U and V is bilinear.
    # Actually, if we fix W, the equation is:
    # sum_r w_r[k,i] u_r[i,j] v_r[j,k] = 1
    # This is still hard.
    
    # Better: Fix U and V. Solve for W linearly.
    # Then check if the resulting W has integer/half-integer entries.
    # If not, perturb U or V?
    
    # Let's try a specific structured ansatz.
    # Maybe U and V can be sparse?
    
    # Let's try R=8, 9, ... up to some limit.
    # For each R:
    # 1. Randomly initialize U and V with entries from {-1, 0, 1}.
    # 2. Construct the linear system for W.
    #    A_{(i,j,k), (k',i',r)} = U[i*m+j, r] * V[j*p+k, r] if (k',i') == (k,i) else 0?
    #    Wait, the index of W is (k,i). The equation is for specific (i,j,k).
    #    The term involving W is W[k*n+i, r].
    #    So the unknowns are W[k*n+i, r] for all k,i,r.
    #    The coefficient for W[k*n+i, r] in the equation for (i,j,k) is:
    #    U[i*m+j, r] * V[j*p+k, r]
    #    So A is a matrix of size (n*m*p) x (p*n*R).
    #    A[(i*m*p + j*p + k), (k*n + i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
    #    And b[(i*m*p + j*p + k)] = 1.
    # 3. Solve A W_flat = b.
    # 4. If a solution exists, check if entries are close to integers/half-integers.
    # 5. If yes, snap to nearest half-integer and verify exactly.
    
    # Let's implement this.
    
    def check_exact(U, V, W):
        # Verify if U, V, W exactly decompose the tensor
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should be 1 for all i,j,k
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # We will use numpy for the linear solve, but convert to Fraction for exact checking if needed.
    # Or just use high precision floats and snap.
    
    for R in range(8, 30):  # Try ranks 8 to 29
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations for U and V
        # Sparse random initialization might be better
        max_attempts = 50
        if R > 15:
            max_attempts = 10
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V
            # Use {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Construct A matrix for W
            # Size: (n*m*p) x (p*n*R)
            # Rows indexed by (i,j,k)
            # Cols indexed by (k,i,r)
            A = np.zeros((n*m*p, p*n*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i*m*p + j*p + k
                        for r in range(R):
                            col_idx = (k*n + i)*R + r
                            A[row_idx, col_idx] = U[i*m+j, r] * V[j*p+k, r]
            
            b = np.ones(n*m*p)
            
            # Solve A W_flat = b
            try:
                # Use least squares
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if residuals are small
                if len(residuals) == 0 or residuals[0] < 1e-10:
                    W = W_flat.reshape(p*n, R)
                    
                    # Check if W entries are close to integers or half-integers
                    # Snap to nearest 0.5
                    W_snapped = np.round(W * 2) / 2
                    
                    # Verify exactness with snapped W
                    if check_exact(U, V, W_snapped):
                        # Found a valid decomposition
                        # Convert to list of lists of floats (or ints if integer)
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W_snapped.tolist()
                        
                        # Ensure types are correct (floats are fine if they are .0 or .5)
                        # The problem allows integer/half-integer.
                        
                        if R < best_rank:
                            best_rank = R
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            # We can break early if we find a very low rank, but let's try to find even lower?
                            # The loop goes up, so we start from low. If we find R=8, that's great.
                            # But we are iterating R from 8 upwards.
                            # So if we find a solution for R, it's the current best.
                            # We can continue to see if we find a lower one? No, R is increasing.
                            # So the first R we find is the minimal rank found so far.
                            # We can break the R loop if we are confident? No, maybe R=8 fails but R=9 works?
                            # Wait, if R=8 works, it's better than R=9.
                            # So if we find a solution for R, we should return it immediately?
                            # No, because we are iterating R from 8 upwards.
                            # If we find a solution for R=8, it's the best possible (assuming lower bound is 8).
                            # But what if R=8 doesn't work, but R=9 does?
                            # Then we continue.
                            # So we just update best_result and continue searching for lower R?
                            # No, R is increasing. So the first solution found is the lowest R.
                            # So we can return immediately?
                            # Yes, if we iterate R from lower_bound upwards, the first valid solution is the minimal rank.
                            return best_result
            except np.linalg.LinAlgError:
                pass

    # If no better rank found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
