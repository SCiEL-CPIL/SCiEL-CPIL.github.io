"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140  # 140s deadline with 10s margin
    
    # Start with schoolbook rank 72
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank using randomized search
    # The tensor T has entries T[i,j,k] = 1 if the (i,j) element of A times (j,k) of B contributes to (i,k) of C
    # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # and 0 otherwise (but for matrix mult, it's always 1 when indices align)
        
        # Actually for matrix mult, the tensor is:
        # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
        # T[a, b, c] = 1 if a, b, c correspond to valid (i,j,k) with matching j
        # More precisely: a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a,b,c] = 1 if j==j' and k==k' and i==i', else 0
        
        # The decomposition requires: for all a,b,c:
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # Efficient check using matrix operations
        # For each r, the contribution is U[:,r] * V[:,r] * W[:,r] (outer products)
        # We need sum_r to equal T
        
        # Build T as a 3D array
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Compute sum_r U[:,r] * V[:,r] * W[:,r]
        # This is a tensor contraction
        result = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            result += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        # Check if result matches T
        if not np.allclose(result, T, atol=1e-10):
            return False
        
        # Also check that entries are integer or half-integer
        for arr in [U_arr, V_arr, W_arr]:
            for val in arr:
                if val != 0 and val % 0.5 != 0:
                    return False
        
        return True
    
    # Randomized search for lower rank
    import random
    random.seed(42)
    
    for target_rank in range(60, best_rank):
        if time.time() > deadline:
            break
        
        max_iter = 200
        found = False
        
        for iter_num in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U_rand = [[0] * target_rank for _ in range(n * m)]
            V_rand = [[0] * target_rank for _ in range(m * p)]
            W_rand = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with random ±1 values for a subset
            for r in range(target_rank):
                # Choose random non-zero pattern
                nz_U = random.sample(range(n*m), min(random.randint(3,8), n*m))
                nz_V = random.sample(range(m*p), min(random.randint(3,8), m*p))
                nz_W = random.sample(range(p*n), min(random.randint(3,8), p*n))
                
                for idx in nz_U:
                    U_rand[idx][r] = random.choice([-1, 1])
                for idx in nz_V:
                    V_rand[idx][r] = random.choice([-1, 1])
                for idx in nz_W:
                    W_rand[idx][r] = random.choice([-1, 1])
            
            if verify_decomposition(U_rand, V_rand, W_rand, target_rank):
                found = True
                best_rank = target_rank
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break
        
        if found:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
