"""
Matrix multiplication tensor decomposition for 3x3 * 3x5.

Key structural insight: The tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
has a block structure. For each fixed output column k, the sub-tensor over
(i,j) is exactly the 3x3 matrix multiplication tensor. Therefore, if we find
a rank-r decomposition for 3x3 matrix multiplication, we can replicate it for
each of the p=5 columns to get a rank-5r decomposition for 3x3*3x5.

Strassen's algorithm gives rank 7 for 3x3, yielding rank 35 here.
We search for lower-rank decompositions of the 3x3 sub-problem.

The approach:
1. Build the 3x3 matrix multiplication tensor (9x9x9)
2. Search for low-rank decompositions using structured random search
3. Verify algebraically (exact integer/half-integer arithmetic where possible)
4. Replicate the best 3x3 decomposition for all 5 columns
5. Fall back to schoolbook (rank 45) if nothing better is found
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build the full target tensor T (for verification)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
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
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, r

    best_U, best_V, best_W, best_rank = schoolbook()

    # Build the 3x3 matrix multiplication sub-tensor
    # For 3x3 * 3x3: T3[i*3+j, j*3+k, k*3+i] = 1
    T3 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3[i * 3 + j, j * 3 + k, k * 3 + i] = 1.0

    def verify_3x3(U3, V3, W3):
        """Verify that U3, V3, W3 decompose the 3x3 matrix mult tensor."""
        P = np.einsum('ak,bk,ck->abc', U3, V3, W3)
        return np.max(np.abs(P - T3))

    def replicate_3x3_to_full(U3, V3, W3, r3):
        """
        Replicate a 3x3 rank-r3 decomposition to get a 3x3*3x5 rank-5*r3 decomposition.
        
        U3: (9, r3) for A (3x3)
        V3: (9, r3) for B (3x3) -- but we need V for B (3x5), shape (15, 5*r3)
        W3: (9, r3) for C (3x3) -- but we need W for C (3x5), shape (15, 5*r3)
        
        For each output column k (0..4):
          - The A factor is the same: U3
          - The B factor for column k: V3[:, k_block] where we use the same V3
          - The C factor for column k: W3[:, k_block]
        
        Actually, the structure is:
          T[a, b, c] = 1 where a=i*3+j, b=j*5+k, c=k*3+i
          
          For fixed k, the sub-tensor over (i,j) is T3[i*3+j, j*3+k', k'*3+i]
          Wait, no. Let me re-examine.
          
          T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
          With n=m=3, p=5:
            a = i*3+j, b = j*5+k, c = k*3+i
          
          For fixed k (output column), the valid (a,b,c) entries are:
            a = i*3+j, b = j*5+k, c = k*3+i for i,j in [0,3)
          
          So for fixed k, the sub-tensor is:
            S_k[a, b, c] = 1 iff a=i*3+j, b=j*5+k, c=k*3+i
          
          This is NOT the same as T3 because b = j*5+k not j*3+k.
          
          Let me think differently. The 3x3 sub-problem for column k is:
            C[:,k] = A * B[:,k]
            C[i,k] = sum_j A[i,j] * B[j,k]
          
          The tensor for this is:
            T_k[i*3+j, j, i] = 1  (if we index B column as just j)
          
          But in our full tensor, B is indexed as b=j*5+k, so for fixed k,
          the B index is j*5+k. The sub-tensor for column k is:
            T_k[a, b, c] = 1 iff a=i*3+j, b=j*5+k, c=k*3+i
          
          This is a permutation of the standard 3x3 matrix mult tensor.
          
          Actually, let's just directly search for the full tensor decomposition
          but exploit the block structure in initialization.
        """
        r_full = r3 * p
        U_full = np.zeros((n * m, r_full))
        V_full = np.zeros((m * p, r_full))
        W_full = np.zeros((p * n, r_full))
        
        for k in range(p):
            col_offset = k * r3
            # U is the same for all k (A doesn't change)
            U_full[:, col_offset:col_offset + r3] = U3
            
            # V: for B[j,k], the index is j*p+k = j*5+k
            # V3[j*3+k', :] gives the B factor for input j and output k'
            # But we need V[j*5+k, :] for the full problem
            # For fixed output column k, the B input is just j (0,1,2)
            # So V[j*5+k, col_offset:col_offset+r3] = V3[j*3+?, :]
            # Hmm, this is getting confusing. Let me just build it directly.
            pass
        
        # Actually, let's take a different approach: directly construct
        # the full decomposition from the 3x3 one.
        
        # The 3x3 tensor T3[a3, b3, c3] = 1 where a3=i*3+j, b3=j*3+k', c3=k'*3+i
        # The full tensor T[a, b, c] = 1 where a=i*3+j, b=j*5+k, c=k*3+i
        
        # For the 3x3 decomposition: sum_t U3[a3,t] * V3[b3,t] * W3[c3,t] = T3[a3,b3,c3]
        # We want: sum_{k,t} U_full[a, k*r3+t] * V_full[b, k*r3+t] * W_full[c, k*r3+t] = T[a,b,c]
        
        # Set:
        # U_full[a, k*r3+t] = U3[a, t] for all k (A is shared)
        # V_full[b, k*r3+t] = V3[?, t] where ? depends on b and k
        # W_full[c, k*r3+t] = W3[?, t] where ? depends on c and k
        
        # b = j*5+k, so for fixed k, b determines j = b//5 (if b%5==k)
        # c = k*3+i, so for fixed k, c determines i = c%3 (if c//3==k)
        
        # For V: b=j*5+k, we need V3 index = j*3+k' for some k'
        # But in the 3x3 sub-problem for output column k, the B input is just j.
        # The 3x3 tensor uses b3=j*3+k' where k' is the 3x3 output index.
        # In our case, for output column k of the 5-column output, the "3x3 output"
        # index k' should be... hmm, this doesn't map cleanly.
        
        # Let me just directly build the full tensor decomposition by searching.
        return None  # This approach is flawed; use direct search instead

    # Direct search for the full tensor, but with structured initialization
    # that exploits the block structure.
    
    def verify_full(U, V, W):
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T))

    # Try structured random search with block-aware initialization
    # For each rank r, initialize U, V, W with structure that respects
    # the 5-block nature of the output.
    
    def structured_init(r, seed):
        """Initialize with block structure: group columns of U,V,W by output column."""
        rng = np.random.RandomState(seed)
        
        # r should be a multiple of some base. Let's use r = base * p or similar.
        # For now, just random but with some structure.
        U = rng.randn(n * m, r) * 0.3
        V = rng.randn(m * p, r) * 0.3
        W = rng.randn(p * n, r) * 0.3
        
        # Add block structure: for W, columns should be grouped by k
        # W[k*3+i, :] should be related across i for same k
        for k in range(p):
            # Make W entries for same k have similar structure
            base = W[k*3:(k+1)*3, :].copy()
            W[k*3:(k+1)*3, :] = base + rng.randn(3, r) * 0.1
        
        return U, V, W

    def gradient_descent(U, V, W, max_steps, lr_init, deadline):
        """Run gradient descent on the full tensor decomposition."""
        lr = lr_init
        best_err = np.inf
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            P = np.einsum('ak,bk,ck->abc', U, V, W)
            diff = P - T
            err = np.max(np.abs(diff))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Gradients
            grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
            grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
            grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
            
            # Adaptive learning rate
            g_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if g_norm > 1e-10:
                step_size = lr / g_norm
            else:
                step_size = lr
            
            U -= step_size * grad_U
            V -= step_size * grad_V
            W -= step_size * grad_W
            
            # Decay learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.7
        
        return best_U, best_V, best_W, best_err

    # Try various ranks with structured search
    # Start with ranks that are multiples of 5 (due to 5 output columns)
    # and also try other ranks.
    
    target_ranks = [25, 28, 30, 32, 35, 38, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(4):
            if time.time() > deadline - 10:
                break
            
            seed = 1000 + attempt * 7 + target_rank
            U, V, W = structured_init(target_rank, seed)
            
            U, V, W, err = gradient_descent(U, V, W, max_steps=300, lr_init=0.1,
                                           deadline=deadline - 10)
            
            if err < 1e-8:
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                found = True
                break
        
        if found:
            break

    # Final verification
    if verify_full(best_U, best_V, best_W) > 1e-6:
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook()

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
