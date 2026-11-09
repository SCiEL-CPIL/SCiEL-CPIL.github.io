"""Matrix multiplication 2x3x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start = time.time()
DEADLINE = 115.0  # safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS
    # We need to find a rank-minimal decomposition for M_{2,3,6}
    # The schoolbook rank is n*m*p = 36
    # We want to find if we can do better
    
    # The tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # A rank-r decomposition gives U (nm x r), V (mp x r), W (pn x r) such that
    # T = sum_{l=1}^{r} u_l ⊗ v_l ⊗ w_l
    # i.e., for each (a,b,c) with a=i*m+j, b=j*p+k, c=k*n+i:
    #   T[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
    
    # For n=2, m=3, p=6:
    # U is 6 x r, V is 18 x r, W is 12 x r
    # The constraint is: for each i in {0,1}, j in {0,1,2}, k in {0,...,5}:
    #   sum_l U[i*3+j, l] * V[j*6+k, l] * W[k*2+i, l] = 1
    # And for all other (a,b,c) triples, the sum must be 0.
    
    # Let's try to find a rank < 36 decomposition.
    # 
    # Key insight: Since n=2, we can think of this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For a fixed i, this is a linear combination of the rows of B weighted by A[i,:].
    # The "trick" for low-rank decompositions often involves sharing computations.
    
    # Let's try rank 24 first (a natural candidate: 2*3*6 = 36, maybe we can save 1/3)
    # Actually, let me think about this differently.
    
    # The bilinear complexity of 2x3x6 multiplication:
    # Known results: M_{2,n,p} for various n,p.
    # For M_{2,3,6}, I believe the rank is 24 based on known constructions.
    
    # Construction for rank 24:
    # Group the 6 columns of B into 3 pairs: (B[:,0],B[:,1]), (B[:,2],B[:,3]), (B[:,4],B[:,5])
    # For each pair, we compute 2x2 matrix products with shared structure.
    # Actually, let me try a different approach.
    
    # Alternative: Use the fact that we can write the multiplication as:
    # For each of the 3 "column groups" of size 2, we need to compute:
    #   C[i, 2g] = sum_j A[i,j] * B[j, 2g]
    #   C[i, 2g+1] = sum_j A[i,j] * B[j, 2g+1]
    # 
    # This is like 3 separate 2x3x2 multiplications.
    # The rank of M_{2,3,2} is... let's compute. Schoolbook would be 2*3*2=12.
    # Can we do better? For 2x3x2, the rank is actually 6 (I think).
    # If so, total rank = 3*6 = 18? That seems too low.
    
    # Let me verify: M_{2,3,2} rank.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] for i=0,1; k=0,1
    # That's 4 output entries, each a sum of 3 products = 12 products total.
    # Can we share? 
    # Precompute: P0 = A[0,0]*B[0,0], P1 = A[0,0]*B[0,1], etc.
    # No obvious sharing for 2x3x2.
    # Actually, I think rank(M_{2,3,2}) = 6 is not right. Let me think again.
    
    # Actually, for M_{2,2,2}, the rank is 7 (Strassen).
    # For M_{2,3,2}, I'm not sure of the exact rank.
    
    # Let me just try to construct a valid decomposition with rank < 36.
    # I'll use a systematic approach: try to find U, V, W with integer entries.
    
    # Let me try rank 24 with a structured approach:
    # For each of the 3 column-pairs (g=0,1,2), use 8 terms.
    # Total: 3*8 = 24.
    
    # For column pair g (columns 2g and 2g+1):
    # We need to compute:
    #   C[0,2g] = A[0,0]*B[0,2g] + A[0,1]*B[1,2g] + A[0,2]*B[2,2g]
    #   C[0,2g+1] = A[0,0]*B[0,2g+1] + A[0,1]*B[1,2g+1] + A[0,2]*B[2,2g+1]
    #   C[1,2g] = A[1,0]*B[0,2g] + A[1,1]*B[1,2g] + A[1,2]*B[2,2g]
    #   C[1,2g+1] = A[1,0]*B[0,2g+1] + A[1,1]*B[1,2g+1] + A[1,2]*B[2,2g+1]
    
    # That's 4 outputs, each needing 3 multiplications = 12 multiplications per pair.
    # With 3 pairs: 36 total (schoolbook).
    
    # To reduce to 8 per pair, we need to share 4 multiplications per pair.
    # This is like doing a 2x3x2 multiplication in rank 8 instead of 12.
    
    # Hmm, that's a 33% reduction. Let me see if that's possible.
    
    # Actually, let me try a completely different approach. Let me use random search
    # to find a valid decomposition with rank < 36.
    
    # First, let's return schoolbook as a fallback.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower-rank decomposition using iterative refinement
    # We'll try ranks from 18 up to 35 and see if we can find a valid one.
    
    # The constraints are:
    # For each (i,j,k) with i in range(n), j in range(m), k in range(p):
    #   sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    # For all other (a,b,c) in [0,nm)x[0,mp)x[0,pn):
    #   sum_l U[a, l] * V[b, l] * W[c, l] = 0
    
    # This is a system of polynomial equations in the entries of U, V, W.
    # Let's try a simple approach: fix U and W, solve for V, or use gradient descent.
    
    # Actually, let me try a more structured approach.
    # 
    # Known: The rank of M_{2,3,6} is at most 24 by the following construction:
    # 
    # Write the multiplication as:
    # C = A * B where A is 2x3, B is 3x6
    # 
    # Partition B into B1, B2, B3 each of size 3x2.
    # C = [A*B1, A*B2, A*B3]
    # 
    # For each 2x3x2 multiplication, use rank 8 (if possible).
    # Total rank: 3*8 = 24.
    
    # Let me try to construct a rank-8 decomposition for 2x3x2.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] for i=0,1; k=0,1
    # 
    # 8 terms to compute 4 outputs.
    # 
    # Let me try:
    # t0 = A[0,0]*B[0,0]
    # t1 = A[0,1]*B[1,0]
    # t2 = A[0,2]*B[2,0]
    # t3 = A[1,0]*B[0,1]
    # t4 = A[1,1]*B[1,1]
    # t5 = A[1,2]*B[2,1]
    # t6 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])  -- this doesn't help directly
    # 
    # This is getting complicated. Let me just use numerical optimization.
    
    # Use a simple approach: random initialization + gradient descent on the loss
    # loss = sum of (constraint_violations)^2
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try various ranks
    for target_rank in [24, 21, 18, 16, 15, 14, 12]:
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(20):
            if time.time() > DEADLINE - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > DEADLINE - 3:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Constraints: T[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
                # For (a,b,c) = (i*m+j, j*p+k, k*n+i): should be 1
                # For all other: should be 0
                
                # Efficient computation using tensor operations
                # T = sum_l U[:,l] ⊗ V[:,l] ⊗ W[:,l]
                # This is expensive to compute fully. Let's compute only the needed entries.
                
                # For the 1-constraints:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = np.sum(U[a,:] * V[b,:] * W[c,:])
                            loss += (val - 1)**2
                            # d/dU[a,l] = 2*(val-1)*V[b,l]*W[c,l]
                            dU[a,:] += 2*(val-1)*V[b,:]*W[c,:]
                            dV[b,:] += 2*(val-1)*U[a,:]*W[c,:]
                            dW[c,:] += 2*(val-1)*U[a,:]*V[b,:]
                
                # For the 0-constraints (sample a subset for speed):
                num_zero_constraints = 50
                for _ in range(num_zero_constraints):
                    a = random.randint(0, n*m-1)
                    b = random.randint(0, m*p-1)
                    c = random.randint(0, p*n-1)
                    # Check if this is a "1" constraint
                    i = a // m
                    j = a % m
                    if i >= n or j >= m:
                        continue
                    j2 = b // p
                    k = b % p
                    if j2 != j or j2 >= m or k >= p:
                        continue
                    i2 = c % n
                    k2 = c // n
                    if i2 != i or k2 != k or k2 >= p:
                        continue
                    # This is a 1-constraint, skip
                    continue
                    
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    loss += val**2
                    dU[a,:] += 2*val*V[b,:]*W[c,:]
                    dV[b,:] += 2*val*U[a,:]*W[c,:]
                    dW[c,:] += 2*val*U[a,:]*V[b,:]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-10:
                    found = True
                    break
            
            if found:
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            val = np.sum(U[a,:] * V[b,:] * W[c,:])
                            max_err = max(max_err, abs(val-1))
                
                if max_err < 1e-8:
                    # Round to nearest integers or half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded version
                    max_err_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m + j
                                b = j*p + k
                                c = k*n + i
                                val = np.sum(U_rounded[a,:] * V_rounded[b,:] * W_rounded[c,:])
                                max_err_r = max(max_err_r, abs(val-1))
                    
                    if max_err_r < 1e-10:
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        best_rank = target_rank
                        return best_result
        
        if time.time() > DEADLINE - 5:
            break
    
    return best_result
# EVOLVE-BLOCK-END
