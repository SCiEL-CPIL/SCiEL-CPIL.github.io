"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def is_half_integer(x):
        f = Fraction(x)
        return (f * 2).denominator == 1

    def verify(U, V, W):
        # Check that U @ V^T == W @ I (tensor equality)
        # Actually, the decomposition requires:
        # T = sum_r U_r \otimes V_r \otimes W_r
        # In matrix form, for all i,j,k:
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T has T[i,j,k] = delta_{i,k} (in the sense of multiplication tensor)
        # Actually, the multiplication tensor for (n x m) * (m x p) -> (n x p) is:
        # T_{i,k,j} = 1 if we're multiplying A[i,j] * B[j,k]
        # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = T[i,k,j]
        # So we need: for all i,k,j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] == (1 if the product term exists else 0)
        # Actually the target tensor for matrix multiplication M_{n,m,p}:
        # T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
        # Wait, no. The matrix multiplication tensor is defined as:
        # (C = A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor T has entries T_{i,j,k} = 1 for all i,j,k (it's the identity in the sense that
        # each product A_{i,j} * B_{j,k} contributes to C_{i,k})
        # 
        # The standard form: we want U, V, W such that:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] = 1 for all i,j,k
        # 
        # But wait, the W convention is W[k*n+i, r] per the task description.
        # Let me re-read: "W[k*n+i,r]"
        # So W is indexed as W[k*n + i, r] where k is the output column and i is the output row?
        # That means W has shape (p*n, rank) and W[k*n+i, r] corresponds to output position (i,k).
        #
        # So the decomposition is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        # Compute the tensor product sum
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank, timeout_frac=0.0):
        """Try to find a decomposition of given rank using structured search."""
        # For small ranks, use the known structure
        # The schoolbook has rank n*m*p = 63
        # Try to find better using the fact that we can combine terms
        
        # A simple approach: start with schoolbook and try to merge columns
        # This is a greedy approach - try to find linear dependencies
        
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        
        # Initialize with identity-like structure for the first min(rank, n*m*p) terms
        num_terms = min(rank, n*m*p)
        for r in range(num_terms):
            i = r // (m*p)
            rem = r % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j][r] = 1
            V[j*p+k][r] = 1
            W[k*n+i][r] = 1
        
        if rank >= n*m*p:
            return U, V, W
        
        # For rank < n*m*p, we need to find a non-trivial decomposition
        # This is hard. Let's try a simple approach: assume the structure has
        # some columns that are sums of schoolbook columns
        
        # Actually, for 3x3x7, the known lower bound is around 34-40
        # and upper bound is 63. Without known identities, we can't easily do better.
        # Let's just return the schoolbook truncated (which won't verify) 
        # or return None to indicate failure.
        return None, None, None

    # First, always have the schoolbook as fallback
    rank_school = n * m * p
    U = [[0]*rank_school for _ in range(nm)]
    V = [[0]*rank_school for _ in range(mp)]
    W = [[0]*rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try to improve using a simple column merging strategy
    # Check if any two columns can be combined (i.e., their tensor product difference
    # can be expressed as another valid column)
    
    # A simpler approach: try rank from lower bound up, using random initialization
    # and checking. But this is expensive.
    
    # For this specific problem (3,3,7), let's try a few known structured approaches.
    # One approach: exploit the symmetry in the 3x3 part.
    
    # Try rank 54 (save 9) by merging columns where i is fixed
    if time.time() < deadline:
        # Try a structured reduction: for each fixed i, we have m*p = 21 terms
        # Can we reduce the 3x7 multiplication to fewer than 21 terms?
        # 3x7 multiplication has known lower bounds around 15-17
        # Total would be around 3 * 17 = 51
        
        # Let's try to find a 3x7 matrix multiplication with rank < 21
        # using random search over small entries
        
        def find_3x7_decomposition(target_rank, max_iters=1000):
            """Find U,V,W for 3x7 matrix multiplication with given rank."""
            nn, mm, pp = 3, 3, 7
            for _ in range(max_iters):
                if time.time() > deadline:
                    return None, None, None
                
                U_rand = np.zeros((nn*mm, target_rank))
                V_rand = np.zeros((mm*pp, target_rank))
                W_rand = np.zeros((pp*nn, target_rank))
                
                # Initialize with random values in {-1, 0, 1}
                for r in range(target_rank):
                    # Random pattern
                    mask_u = np.random.choice([0, 1], size=nn*mm, p=[0.5, 0.5])
                    mask_v = np.random.choice([0, 1], size=mm*pp, p=[0.5, 0.5])
                    mask_w = np.random.choice([0, 1], size=pp*nn, p=[0.5, 0.5])
                    U_rand[:, r] = mask_u * np.random.choice([-1, 1], size=nn*mm)
                    V_rand[:, r] = mask_v * np.random.choice([-1, 1], size=mm*pp)
                    W_rand[:, r] = mask_w * np.random.choice([-1, 1], size=pp*nn)
                
                # Verify
                valid = True
                for i in range(nn):
                    for k in range(pp):
                        for j in range(mm):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_rand[i*mm+j, r] * V_rand[j*pp+k, r] * W_rand[k*nn+i, r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    # Convert to lists
                    return (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
            return None, None, None
        
        # Try various ranks for the 3x7 subproblem
        found = False
        for sub_rank in range(15, 21):
            if time.time() > deadline:
                break
            U_sub, V_sub, W_sub = find_3x7_decomposition(sub_rank, max_iters=200)
            if U_sub is not None:
                # Build full decomposition by tiling over i
                total_rank = 3 * sub_rank
                U_full = [[0]*total_rank for _ in range(nm)]
                V_full = [[0]*total_rank for _ in range(mp)]
                W_full = [[0]*total_rank for _ in range(pn)]
                
                for i in range(3):
                    for r in range(sub_rank):
                        full_r = i * sub_rank + r
                        for j in range(3):
                            U_full[i*3+j][full_r] = U_sub[i*3+j][r]
                        for k in range(7):
                            V_full[j*7+k][full_r] = V_sub[j*7+k][r]
                            W_full[k*3+i][full_r] = W_sub[k*3+i][r]
                
                if verify(U_full, V_full, W_full):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = total_rank
                    found = True
                    break
        
        if found:
            pass  # Already updated best
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        U = [[0]*rank_school for _ in range(nm)]
        V = [[0]*rank_school for _ in range(mp)]
        W = [[0]*rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank_school
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
