"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook rank
    rank_sb = n * m * p  # 105
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i' else 0
    dim1 = n * m  # 15
    dim2 = m * p  # 35
    dim3 = p * n  # 21
    
    # Non-zero entries: for each (i, j, k) with i in [n], j in [m], k in [p]
    # a = i*m + j, b = j*p + k, c = k*n + i
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    # Schoolbook decomposition
    def schoolbook():
        rank = rank_sb
        U = np.zeros((dim1, rank), dtype=float)
        V = np.zeros((dim2, rank), dtype=float)
        W = np.zeros((dim3, rank), dtype=float)
        for col, (a, b, c) in enumerate(nz_indices):
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[c, col] = 1.0
        return U, V, W
    
    # Try to find a lower rank decomposition using randomized search
    # The tensor equation: for each (a,b,c), T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is a multilinear system. We can fix U and V and solve for W, or use iterative methods.
    
    # For a given rank r, we need to find U (dim1 x r), V (dim2 x r), W (dim3 x r)
    # such that for all (a,b,c) in support: U[a,:].V[b,:].W[c,:] (element-wise sum) = 1
    # and for all other (a,b,c): the sum = 0
    
    # This is a hard NP-hard problem in general. Let's try a simple heuristic.
    
    # Lower bound: max(dim1, dim2, dim3) = max(15, 35, 21) = 35
    # But actual rank is likely higher.
    
    # Try a few ranks with randomized initialization and check feasibility
    # Use the fact that we can set up a linear system for W given U, V
    
    best_U, best_V, best_W = schoolbook()
    best_rank = rank_sb
    
    # Try ranks from a reasonable range
    # Use time budget to try a few
    np.random.seed(42)
    
    for r in range(50, min(rank_sb, 80)):  # Try ranks 50-79
        if time.time() > deadline - 10:
            break
        
        # Randomized attempt: initialize U, V randomly, solve for W
        # For each c, W[c,:] should satisfy: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b)
        # This is a linear system in W[c,:] for each c
        
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randn(dim1, r)
            V_rand = np.random.randn(dim2, r)
            
            # For each c in range(dim3), solve for W[c,:]
            # The equation: for all (a,b) such that (a,b,c) is in support, 
            # sum_r U[a,r]*V[b,r]*W[c,r] = 1
            # For all (a,b) such that (a,b,c) is NOT in support, sum = 0
            
            # Build the matrix: for fixed c, we need W[c,:] such that
            # M_c @ W[c,:] = t_c
            # where M_c has rows indexed by (a,b) and M_c[(a,b), r] = U[a,r]*V[b,r]
            # t_c[(a,b)] = 1 if (a,b,c) in support else 0
            
            # This is overdetermined (dim1*dim2 = 525 rows, r columns)
            # Use least squares and check if residual is small
            
            success = True
            W_sol = np.zeros((dim3, r))
            
            for c in range(dim3):
                # Build M_c: (dim1*dim2) x r
                # M_c[a*dim2+b, r] = U_rand[a,r] * V_rand[b,r]
                M_c = np.zeros((dim1 * dim2, r))
                t_c = np.zeros(dim1 * dim2)
                
                for a in range(dim1):
                    for b in range(dim2):
                        M_c[a * dim2 + b, :] = U_rand[a, :] * V_rand[b, :]
                        # Check if (a, b, c) is in support
                        # a = i*m+j, b = j'*p+k, c = k'*n+i'
                        # Support: j==j' and k==k' and i==i'
                        i = a // m
                        j = a % m
                        jp = b // p
                        k = b % p
                        kp = c // n
                        ip = c % n
                        if j == jp and k == kp and i == ip:
                            t_c[a * dim2 + b] = 1.0
                
                # Solve M_c @ w = t_c in least squares sense
                # Check if there's an exact solution
                sol, residuals, rank_m, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                W_sol[c, :] = sol
                
                # Check residual
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    success = False
                    break
            
            if success:
                # Verify the solution
                max_err = 0
                for (a, b, c) in nz_indices:
                    val = np.dot(U_rand[a, :], V_rand[b, :] * W_sol[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                # Check a few non-support entries
                for _ in range(100):
                    a = np.random.randint(dim1)
                    b = np.random.randint(dim2)
                    c = np.random.randint(dim3)
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    if not (j == jp and k == kp and i == ip):
                        val = np.dot(U_rand[a, :], V_rand[b, :] * W_sol[c, :])
                        max_err = max(max_err, abs(val))
                
                if max_err < 1e-6:
                    best_U, best_V, best_W = U_rand, V_rand, W_sol
                    best_rank = r
                    break
        
        if best_rank < r:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest half-integer
    def to_half_int(arr):
        return [[round(x * 2) / 2 for x in row] for row in arr.tolist()]
    
    U_list = to_half_int(best_U)
    V_list = to_half_int(best_V)
    W_list = to_half_int(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
