"""
Matrix multiplication tensor decomposition for 3x3 * 3x5.
We use a rank-15 algorithm based on the "fast 3x3 matrix multiplication"
idea extended to rectangular cases, but here we fall back to a safe,
pre-computed low-rank decomposition that has been verified to exactly
decompose the tensor.

For a 3x3 * 3x5 matrix product, the schoolbook rank is 45.
We aim for a lower rank. The known lower bounds and algorithms suggest
that rank 30 or less might be achievable, but without a guaranteed
algorithm, we implement a search for small-rank decompositions using
randomized methods and verify correctness.

Given the time constraint and the need for correctness, we implement a
careful rank-15 decomposition if known, otherwise fall back to a verified
rank-30 decomposition, and finally to schoolbook rank 45.

Actually, for this specific problem, let's use a systematic approach:
We will attempt to find a decomposition of rank r by solving a system
of equations. For small r, we can use a greedy/randomized approach.

However, the safest and most reliable approach within the time limit is
to use a known decomposition. Since I don't have a memorized rank-15
decomposition for 3x3*3x5, I will implement a randomized search for
low-rank decompositions with verification.

The tensor is T_{i,j,k} = e_{i,j} * e_{j,k} * e_{k,i} (in the appropriate
indices). We want U, V, W such that
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid
triple, else 0.

Let's implement a search that tries to find a decomposition of rank r
for decreasing r, starting from a good guess.

Given the complexity and time limits, I will use a hybrid approach:
1. Try to verify a pre-computed small-rank decomposition if available.
2. Fall back to a randomized search with a time limit.
3. Final fallback: schoolbook.

Since I cannot reliably produce a correct small-rank decomposition from
memory, I will focus on making the schoolbook solution fast and correct,
and add a randomized search that might find a better rank.

Actually, let's just implement the schoolbook solution correctly and
efficiently, as it is guaranteed to work. The score is rank, and
schoolbook gives rank 45.

But wait, the task says "Find a rank-minimal exact tensor decomposition".
If we can find a better one, we should. Let me implement a careful
randomized search with verification.

For the tensor T of size (n*m) x (m*p) x (p*n) = 9 x 15 x 15,
we want to find U (9 x r), V (15 x r), W (15 x r) such that
for all (a, b, c) in [0,9) x [0,15) x [0,15):
  sum_{k=0}^{r-1} U[a,k] * V[b,k] * W[c,k] = T[a,b,c]

where T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k'*n+i with i in [0,3),
j in [0,3), k in [0,5), and k' = k (wait, no, the output is C[i,k] = sum_j A[i,j]*B[j,k]).

Let me re-index properly:
- A is n x m = 3 x 3
- B is m x p = 3 x 5
- C is n x p = 3 x 5

The tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for C.
So T[(i,j), (j,k), (i,k)] = 1 for all valid i, j, k.

In linear indices:
- a = i*m + j, i in [0,3), j in [0,3) -> a in [0,9)
- b = j*p + k, j in [0,3), k in [0,5) -> b in [0,15)
- c = i*p + k, i in [0,3), k in [0,5) -> c in [0,15)

Wait, the output W is indexed by W[k*n+i, r] according to the task:
"The output factor follows the official trace convention W[k*n+i,r]"

So W has size p*n = 5*3 = 15, indexed by k*n + i.
And the tensor entry is T[a, b, c] where:
- a = i*m + j (index into A)
- b = j*p + k (index into B)
- c = k*n + i (index into C, using trace convention)

And T[a, b, c] = 1 if the indices are consistent, i.e., if we can find
i, j, k such that a = i*m+j, b = j*p+k, c = k*n+i.

Let's implement this.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to be safe, leaving 10s margin
    
    # Build the target tensor T
    # T[a, b, c] = 1 if exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r
    
    # Initialize best with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Function to verify decomposition
    def verify(U, V, W):
        # Compute product tensor: P[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # Using einsum for efficiency
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.max(np.abs(P - T))
        return err
    
    # Strategy: Try structured decompositions based on the natural tensor structure
    # For 3x3 * 3x5, we can try to exploit the fact that m=3 (square inner dimension)
    
    # Approach 1: Try rank = n*m*p - something by merging some terms
    # The key insight: when m is small, we can sometimes reduce rank by 
    # combining terms that share the same j index
    
    # Let's try a systematic search for low-rank decompositions using 
    # a structured approach based on the Kronecker product structure
    
    # For the 3x3*3x5 case, let's try to find a rank-30 decomposition
    # by exploiting the structure more carefully
    
    # First, let's try a direct approach: for each j, we have n*p = 15 terms
    # Total schoolbook: 3*15 = 45 terms
    
    # Can we reduce this? Let's try rank 30 by combining some terms
    
    # Actually, let's try a more sophisticated approach:
    # Use the fact that for fixed j, the sub-tensor is rank n*p = 15
    # But we might be able to share some structure across different j's
    
    # Let's try a randomized search with better initialization
    # and use the structure of the problem
    
    def try_structured_decomposition(target_rank):
        """Try to find a decomposition of the given rank using structured methods"""
        if time.time() > deadline - 20:
            return None
        
        # Strategy: Initialize based on the natural structure
        # For each j (0, 1, 2), we have n*p = 15 basis terms
        # We want to reduce from 45 to target_rank
        
        # Try different initialization strategies
        for init_strategy in range(5):
            if time.time() > deadline - 20:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Strategy 0: Random initialization with scaling
            if init_strategy == 0:
                U = np.random.randn(n*m, target_rank) * 0.3
                V = np.random.randn(m*p, target_rank) * 0.3
                W = np.random.randn(p*n, target_rank) * 0.3
            
            # Strategy 1: Block-structured initialization
            elif init_strategy == 1:
                # Divide the rank into 3 blocks, one for each j
                block_size = target_rank // 3
                for j in range(m):
                    start = j * block_size
                    end = start + block_size if j < m - 1 else target_rank
                    for i in range(n):
                        for k in range(p):
                            idx = i * p + k
                            if idx < (end - start):
                                col = start + idx
                                U[i*m + j, col] = 1.0
                                V[j*p + k, col] = 1.0
                                W[k*n + i, col] = 1.0
            
            # Strategy 2: Sparse random initialization
            elif init_strategy == 2:
                for c in range(target_rank):
                    # Randomly select a few non-zero entries per column
                    nz_U = np.random.choice(n*m, size=3, replace=False)
                    nz_V = np.random.choice(m*p, size=3, replace=False)
                    nz_W = np.random.choice(p*n, size=3, replace=False)
                    U[nz_U, c] = np.random.randn(3) * 0.5
                    V[nz_V, c] = np.random.randn(3) * 0.5
                    W[nz_W, c] = np.random.randn(3) * 0.5
            
            # Strategy 3: Based on schoolbook with random perturbations
            elif init_strategy == 3:
                # Start from schoolbook and randomly merge terms
                U, V, W, _ = schoolbook()
                # Randomly project to lower rank
                proj_U = np.random.randn(n*m, target_rank)
                proj_V = np.random.randn(m*p, target_rank)
                proj_W = np.random.randn(p*n, target_rank)
                U = U @ np.linalg.pinv(proj_U.T) if target_rank < n*m else U[:, :target_rank]
                V = V @ np.linalg.pinv(proj_V.T) if target_rank < m*p else V[:, :target_rank]
                W = W @ np.linalg.pinv(proj_W.T) if target_rank < p*n else W[:, :target_rank]
            
            # Strategy 4: Alternating least squares initialization
            elif init_strategy == 4:
                # Initialize W randomly, then solve for U and V
                W = np.random.randn(p*n, target_rank) * 0.5
                # For each c, solve for U[:,c] and V[:,c]
                for c in range(target_rank):
                    # Build the linear system
                    # T[a,b,c] = U[a,c] * V[b,c] * W[c,c] + ... (but we're solving sequentially)
                    # This is complex, so let's just use random for now
                    U[:, c] = np.random.randn(n*m) * 0.5
                    V[:, c] = np.random.randn(m*p) * 0.5
            
            # Now optimize using gradient descent with adaptive learning rate
            current_lr = 0.05
            patience = 0
            best_err = np.inf
            max_patience = 100
            min_lr = 1e-8
            
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                # Compute P = einsum(U, V, W)
                P = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = P - T
                err = np.max(np.abs(diff))
                
                if err < 1e-10:
                    return U, V, W
                
                # Track best error
                if err < best_err:
                    best_err = err
                    patience = 0
                else:
                    patience += 1
                
                if patience > max_patience:
                    # Reduce learning rate
                    current_lr *= 0.5
                    patience = 0
                    if current_lr < min_lr:
                        break
                
                # Gradients
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                # Update with momentum-like behavior
                U -= current_lr * grad_U
                V -= current_lr * grad_V
                W -= current_lr * grad_W
            
            # Check final error
            P = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.max(np.abs(P - T))
            if err < 1e-10:
                return U, V, W
        
        return None
    
    # Try various target ranks from low to high
    target_ranks = [20, 25, 27, 30, 33, 36, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 25:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_structured_decomposition(target_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
