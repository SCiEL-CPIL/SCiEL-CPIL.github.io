"""Matrix multiplication 3,4,7: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, n * m * p), dtype=object)
    V_sb = np.zeros((m * p, n * m * p), dtype=object)
    W_sb = np.zeros((p * n, n * m * p), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = Fraction(1)
                V_sb[j * p + k, c] = Fraction(1)
                W_sb[k * n + i, c] = Fraction(1)
                c += 1
    
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using the linear system approach
    # For a given rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (i, j, k):
    #   sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Fix U and V, then solve for W. The equations are:
    # For each (i, k), and each j:
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is linear in W. Let's write it as:
    #   For each (i, k): A_{(i,k), (k*n+i, r)} * W[k*n+i, r] ... 
    # Actually, the equations couple different W entries.
    
    # Let me re-examine. The condition is:
    # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # 
    # For fixed (i, k), we have m equations (one for each j) but they all involve
    # the same W row W[k*n+i, :]. Let's denote w_{ik} = W[k*n+i, :] (a vector of length R).
    # Then the condition for fixed (i, k) and varying j is:
    #   sum_{r} (U[i*m+j, r] * V[j*p+k, r]) * w_{ik, r} = 1
    # This is m equations in R unknowns (the components of w_{ik}).
    # If R >= m, these can potentially be satisfied.
    #
    # But wait, the w_{ik} is shared across all j for fixed (i,k). So we need:
    #   For all j: sum_{r} (U[i*m+j, r] * V[j*p+k, r]) * w_{ik, r} = 1
    #
    # This means the vector v_{ik} = [sum_{r} U[i*m+1,r]*V[1*p+k,r]*w_{ik,r}, ..., sum_{r} U[i*m+m,r]*V[m*p+k,r]*w_{ik,r}]
    # must be all-ones.
    #
    # Equivalently: let M_{ik} be an m x R matrix where M_{ik}[j, r] = U[i*m+j, r] * V[j*p+k, r].
    # Then we need M_{ik} * w_{ik} = 1_m (vector of ones of length m).
    #
    # So for each (i, k), we need to find w_{ik} such that M_{ik} * w_{ik} = 1_m.
    # This is possible if 1_m is in the column space of M_{ik}.
    #
    # If we choose U and V randomly, M_{ik} will be a random m x R matrix (with entries being products of U and V entries).
    # If R >= m, M_{ik} will typically have full row rank m, so a solution w_{ik} exists.
    #
    # So the algorithm is:
    # 1. For R from some minimum up to schoolbook rank:
    #    a. Generate random U, V with integer entries (e.g., -1, 0, 1)
    #    b. For each (i, k), solve M_{ik} * w_{ik} = 1_m
    #    c. If all solutions exist and are "nice" (rational with small denominators), we have a valid decomposition
    #    d. Check if the resulting W has nice entries
    #
    # The minimum possible rank: 
    # Lower bound: the rank of the tensor M_{n,m,p} is at least max(n*m, m*p, p*n)? No.
    # Actually, a simple lower bound is n*m*p / max(n,m,p) but that's not tight.
    # For 2x2x2, rank is 7, while n*m*p/max = 8/2 = 4.
    # 
    # Let's try R from 20 up to 84 (schoolbook). We want the minimum R.
    
    np.random.seed(42)
    
    for R in range(17, best_rank + 1):  # Start from 17 (above the trivial lower bound of 16)
        if time.time() > deadline - 5:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Generate random U, V with entries in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # For each (i, k), solve M_{ik} * w_{ik} = 1_m
            W_rand = np.zeros((p * n, R), dtype=object)
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build M_{ik}: m x R
                    M_ik = np.zeros((m, R), dtype=object)
                    for j in range(m):
                        for r in range(R):
                            M_ik[j, r] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                    
                    # Solve M_ik * w = 1_m
                    # Use numpy's lstsq for a least-squares solution
                    M_float = M_ik.astype(float)
                    ones = np.ones(m)
                    try:
                        w_sol, residuals, rank_mat, sv = np.linalg.lstsq(M_float, ones, rcond=None)
                    except Exception:
                        valid = False
                        break
                    
                    # Check if the solution is exact (residual ~ 0)
                    if residuals and residuals[0] > 1e-10:
                        valid = False
                        break
                    # Check rank
                    if rank_mat < m:
                        valid = False
                        break
                    
                    # Convert to Fraction for exact arithmetic
                    w_frac = [Fraction(w_sol[r]).limit_denominator(1000000) for r in range(R)]
                    # Verify with exact arithmetic
                    for j in range(m):
                        s = Fraction(0)
                        for r in range(R):
                            s += M_ik[j, r] * w_frac[r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                    
                    W_rand[k * n + i, :] = w_frac
            
            if valid:
                # We have a valid decomposition!
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    # Convert to lists of lists
    return {
        "U": best_U if isinstance(best_U, list) else best_U.tolist(),
        "V": best_V if isinstance(best_V, list) else best_V.tolist(),
        "W": best_W if isinstance(best_W, list) else best_W.tolist()
    }
