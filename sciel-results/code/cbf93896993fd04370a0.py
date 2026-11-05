import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # 1. Construct the target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    # List of non-zero indices (i, j, k) in the flattened tensor space
    # Actually, we just need to fill T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # 2. Helper: Verify if U, V, W decompose T
    def verify(U, V, W):
        # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
        # Compute S = sum_r U[:,r] V[:,r] W[:,r]
        # Efficient check: Flatten T to a matrix and check if it's in the span?
        # No, we must check the 3-way reconstruction.
        # S = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This is expensive for large r. 
        # Alternative: Check if the set of non-zero entries of T is covered and zeros are zero.
        # Since T is sparse, we can check specific entries.
        # But we need to ensure ALL entries match.
        # For r up to ~96, dim_u=12, dim_v=32, dim_w=24.
        # S size is 12*32*24 = 9216.
        # einsum is fast enough.
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(S, T, atol=1e-9)

    # 3. Helper: Build schoolbook decomposition (Rank = n*m*p)
    def build_schoolbook():
        r = n * m * p
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    # 4. Randomized Rank Minimization Algorithm
    # Strategy:
    # Start with r = n*m*p (Schoolbook rank).
    # We want to find a rank-r decomposition.
    # We can start with a random rank-R decomposition that exactly reconstructs T?
    # No, random rank-R doesn't exactly reconstruct T.
    
    # Alternative Strategy: "Greedy Peeling" / "Sparse Recovery"
    # We know the support of T is S = {(a,b,c) | T[a,b,c]!=0}.
    # We want to find min r such that T = sum_{l=1}^r u_l v_l w_l.
    
    # Let's try to find a low-rank decomposition by solving for U, V, W given a random initialization and refining?
    # No, the problem is non-convex.
    
    # Better Strategy: "Structured Random Search"
    # We know that the rank of M_{3,4,8} is likely close to the schoolbook rank or slightly lower.
    # Known bounds: 
    # M_{2,2,2} rank 7.
    # M_{3,3,3} rank 23 (Winograd) or 21?
    # M_{4,4,4} rank 21 (Laderman) or 22?
    
    # Let's try a specific structural decomposition first, then random search if time permits.
    # But the directive says "DIFFERENT strategy". Randomized numerical search is different from block tiling.
    
    # Algorithm:
    # 1. Start with r = n*m*p.
    # 2. Generate random U0, V0, W0 of rank r.
    # 3. Project to the affine space of decompositions of T?
    #    The set of decompositions of T is not an affine space.
    
    # Let's use a simpler heuristic:
    # 1. Try r = n*m*p - 1, n*m*p - 2, ...
    # 2. For each r, try to find U, V, W.
    #    How?
    #    Fix U and V randomly. Solve for W?
    #    T = sum U[:,l] V[:,l] W[:,l]^T (matrix outer products flattened)
    #    This is a linear system for W if U, V are fixed?
    #    Let M_{a,b} = T[a,b,:] (vector of length dim_w).
    #    M_{a,b} = sum_l U[a,l] V[b,l] W[:,l]
    #    This is a tensor equation.
    #    If we fix U and V, we have a linear system for W.
    #    The number of equations is dim_u * dim_v = 12 * 32 = 384.
    #    The number of unknowns is r * dim_w = r * 24.
    #    If r=96, unknowns = 2304. Equations = 384. Underdetermined.
    #    We can find a least-squares solution.
    #    Then check if the solution exactly reconstructs T.
    #    If not, try different random U, V.
    
    # This is a valid randomized approach.
    
    best_rank = n * m * p
    best_U, best_V, best_W = build_schoolbook()
    
    # Try ranks from best_rank down to some lower bound
    # Lower bound: max(dim_u, dim_v, dim_w) = 32? No, rank can be higher.
    # Actually, rank >= max(dim_u, dim_v, dim_w) is not true for MM tensors.
    # Rank of M_{2,2,2} is 7, max dim is 4.
    # Let's try down to 60.
    
    # To speed up, we can use the fact that we just need *a* solution.
    
    for r in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations for this rank
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random U, V
            # Scale them to be reasonable
            U_rand = np.random.randn(dim_u, r)
            V_rand = np.random.randn(dim_v, r)
            
            # Construct the linear system for W
            # T[a,b,c] = sum_l U_rand[a,l] V_rand[b,l] W[c,l]
            # Let A_{(a,b), (l,c)} = U_rand[a,l] V_rand[b,l]
            # Then T_flat = A W_flat
            # T_flat has length dim_u * dim_v
            # W_flat has length r * dim_w
            
            # We only need to satisfy the equations for non-zero T entries?
            # No, we must satisfy ALL entries (zeros must be zero).
            # So we use all dim_u * dim_v equations.
            
            T_flat = T.flatten()
            
            # Build A
            # A has shape (dim_u * dim_v, r * dim_w)
            # A[(a*dim_v + b), (l*dim_w + c)] = U_rand[a,l] * V_rand[b,l]
            # This is a huge matrix: 384 x (r*24).
            # For r=90, 384 x 2160.
            # Building this explicitly is slow.
            
            # Alternative: Solve iteratively or use structure.
            # Actually, we can just check if the least squares solution works.
            
            # Let's construct A efficiently.
            # A is a Khatri-Rao product of U_rand and V_rand, reshaped.
            # K = np.kron(U_rand.T, V_rand.T) ? No.
            # Let's just build it if r is small enough.
            
            if r * dim_w > 5000:
                # Too big for simple dense LSQ in Python loop
                continue
                
            A = np.zeros((dim_u * dim_v, r * dim_w))
            for a in range(dim_u):
                for b in range(dim_v):
                    idx = a * dim_v + b
                    for l in range(r):
                        A[idx, l*dim_w:(l+1)*dim_w] = U_rand[a,l] * V_rand[b,l]
            
            # Solve A W_flat = T_flat
            # Use lstsq
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, T_flat, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape(r, dim_w).T # W is (dim_w, r)
            
            # Verify
            if verify(U_rand, V_rand, W_sol):
                found = True
                best_rank = r
                best_U = U_rand
                best_V = V_rand
                best_W = W_sol
                break
                
        if found:
            # We found a solution for r. Try smaller r?
            # The loop continues to r-1.
            pass

    # If we found a better rank, return it.
    # Otherwise, return schoolbook.
    
    # One more check: verify the best solution
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook just in case
        best_U, best_V, best_W = build_schoolbook()
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
