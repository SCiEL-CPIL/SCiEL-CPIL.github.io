"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook baseline: rank = n*m*p = 60
    # Known best: decompose 2x5x6 as 2x1x6 (rank 12) + 2x4x6 (rank 48) = 60
    # Better: 2x4x6 can use Strassen-like: 2x2x6 blocks.
    # 2x2x6 has rank 18 (known optimal for 2x2x6 is 18? Actually 2x2x2 is 7, 2x2x3 is 11, 2x2x4 is 16, 2x2x5 is 20, 2x2x6 is 24? Let me think.)
    # Actually the known result from the prompt: "2x2x6 uses 3 Strassen instances (rank 21)"
    # Wait, re-reading: "Decomposes 2x5x6 into 2x1x6 (rank 12) and 2x4x6 (rank 42 via 2x2x6 blocks). 2x2x6 uses 3 Strassen instances (rank 21)."
    # So 2x4x6 = 2 blocks of 2x2x6, each rank 21, total 42. Plus 2x1x6 rank 12. Total 54.
    # But wait, 2x5x6: split the 5 dimension into 1+4. 
    # 2x1x6: rank = 2*1*6 = 12 (schoolbook for this sub-block)
    # 2x4x6: split into two 2x2x6 blocks. Each 2x2x6 has rank 21 per the prompt.
    # So total rank = 12 + 21 + 21 = 54.
    # Score would be 60/54 = 1.111...
    #
    # Let me verify: the prompt says "Measured: 1 -> 1.111" which corresponds to rank 54.
    #
    # Now, can we do better? Let me think about known optimal ranks.
    # For 2x2x6: the rank is known to be 18? Or 21?
    # Actually, let me reconsider. The prompt says 2x2x6 uses 3 Strassen instances (rank 21).
    # Strassen for 2x2x2 has rank 7. 3 instances would be 21. But that seems like a specific construction.
    #
    # Let me just implement the known rank-54 construction and see if I can improve.
    # Actually, let me first implement the rank-54 solution properly.
    
    # Construction for 2x5x6 with rank 54:
    # Split m=5 into m1=1, m2=4
    # Block 1: 2x1x6, schoolbook rank 12
    # Block 2: 2x4x6, split into two 2x2x6 blocks, each rank 21
    
    # For 2x2x6 with rank 21: this is a known result. Let me construct it.
    # The 2x2x6 tensor T[i,j,k] = delta_{i,j,k} for i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5}
    # We need rank-21 decomposition.
    #
    # Actually, let me use a different approach. Let me try to find a rank-54 decomposition
    # using known constructions, and then try to improve via optimization.
    
    # Let me implement the rank-54 construction:
    # U has shape (n*m, rank) = (10, 54)
    # V has shape (m*p, rank) = (30, 54)  
    # W has shape (p*n, rank) = (12, 54)
    
    # For the 2x1x6 block (i in {0,1}, j=0, k in {0..5}):
    # Schoolbook: 12 terms, each with U[i*5+0, c]=1, V[0*6+k, c]=1, W[k*2+i, c]=1
    
    # For each 2x2x6 block (i in {0,1}, j in {j0, j0+1}, k in {0..5}):
    # Need rank-21 decomposition of the 2x2x6 multiplication tensor.
    
    # Known rank-21 for 2x2x6: I'll use a construction based on Strassen-like ideas.
    # Actually, let me look up: the rank of 2x2x6 is known to be 18.
    # Wait, I should verify. Let me just try rank 18 first for 2x2x6.
    
    # For now, let me implement a known valid decomposition. The prompt confirms rank 54 works.
    # Let me construct it explicitly.
    
    # I'll use a systematic approach: build the full tensor and find a decomposition.
    # The multiplication tensor T[i,j,k] for (n,m,p)=(2,5,6) has T[i,j,k]=1 iff the corresponding
    # entry in the product is (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the appropriate sense).
    # Actually no: T is defined such that (A*B)[i,k] = sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where the original tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, that's not right either. The matrix multiplication tensor has:
    # T[i,j,k] = 1 if we're computing entry (i,k) using row i of A, column j of A/B, row j of B, column k of B.
    # Actually, the standard definition: the multiplication tensor mu_{n,m,p} has
    # mu_{n,m,p}[i,j,k] = 1 for all i in [n], j in [m], k in [p], and it's a 3-tensor.
    # The decomposition satisfies: mu[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (with the trace convention W[k*n+i, r] as specified).
    
    # So the target tensor is all-ones of shape (n, m, p) = (2, 5, 6).
    
    # Let me build a rank-54 decomposition using the block structure.
    
    def make_2x2x6_rank21():
        """Return (U_block, V_block, W_block) for 2x2x6 with rank 21.
        U_block: (2*2, 21) = (4, 21)
        V_block: (2*6, 21) = (12, 21)
        W_block: (6*2, 21) = (12, 21)
        Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2,m=2,p=6
        """
        # I'll use a known construction. For 2x2x6, one approach:
        # Use the fact that 2x2x6 can be decomposed using rank-21.
        # Let me use a linear algebra approach: find 21 vectors that span the tensor.
        
        # Actually, let me use a simpler approach: I'll construct it from known results.
        # The 2x2x6 tensor can be seen as three 2x2x2 tensors (for k=0,1 and k=2,3 and k=4,5)?
        # No, that doesn't work directly.
        
        # Let me use a different strategy: I'll use random initialization + optimization
        # to find a rank-r decomposition, starting from r=18 and going up.
        
        # For now, let me just use schoolbook for 2x2x6 (rank 24) and see if total is 12+24+24=60.
        # That's the same as the baseline. I need to do better.
        
        # Let me try a specific known rank-21 construction for 2x2x6.
        # One known construction uses the identity:
        # (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)/2 + ... no, that's not right.
        
        # Actually, for 2x2x2, Strassen gives rank 7. For 2x2x6, we can view it as
        # computing 12 products. A rank-21 decomposition would save 3 ranks per 2x2x6 block.
        
        # Let me use a practical approach: I'll set up the problem as finding a low-rank
        # decomposition and use optimization.
        
        # For now, return schoolbook (rank 24) as a fallback.
        n_b, m_b, p_b = 2, 2, 6
        rank_b = n_b * m_b * p_b  # 24
        U_b = np.zeros((n_b * m_b, rank_b))
        V_b = np.zeros((m_b * p_b, rank_b))
        W_b = np.zeros((p_b * n_b, rank_b))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, c] = 1
                    V_b[j * p_b + k, c] = 1
                    W_b[k * n_b + i, c] = 1
                    c += 1
        return U_b, V_b, W_b
    
    # Build the full rank-60 (schoolbook) or better decomposition
    # Let me try to optimize for a lower rank.
    
    # Strategy: Use optimization to find a rank-r decomposition of the 2x5x6 tensor.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let me try rank values from 50 upward and use a simple optimization.
    
    def try_decomposition(rank, max_iter=5000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent."""
        # Initialize with schoolbook-like structure
        np.random.seed(42)
        
        # U: (n*m, rank) = (10, rank)
        # V: (m*p, rank) = (30, rank)
        # W: (p*n, rank) = (12, rank)
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Also try a structured initialization based on the block decomposition
        # Initialize U, V, W with the schoolbook pattern for the first min(rank, 60) columns
        init_rank = min(rank, n * m * p)
        for c in range(init_rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        # Remaining columns: random
        if rank > init_rank:
            U[:, init_rank:] = np.random.randn(n * m, rank - init_rank) * 0.5
            V[:, init_rank:] = np.random.randn(m * p, rank - init_rank) * 0.5
            W[:, init_rank:] = np.random.randn(p * n, rank - init_rank) * 0.5
        
        # Gradient descent
        # Loss: sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        # where T[i,j,k] = 1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            # Compute current tensor
            # T_comp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This is expensive for large rank. Let me use a more efficient computation.
            
            # For each c: outer product of U[:,c], V[:,c], W[:,c] in the appropriate indices
            # Actually, let me compute the loss and gradients more carefully.
            
            # T_comp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let me compute this using einsum-like operations
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # T_comp[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            # This is a contraction over c.
            
            # Compute using loops over c for efficiency with small rank
            T_comp = np.zeros((n, m, p))
            for c in range(rank):
                # U3[:, :, c] has shape (n, m)
                # V3[:, :, c] has shape (m, p)
                # W3[:, :, c] has shape (p, n)
                # Contribution: for each i,j,k: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                # = np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
                T_comp += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
            
            error = T_comp - 1.0  # target is all ones
            loss = np.sum(error ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dL/dU[i*m+j, c] = -2 * sum_{k} error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
            # dL/dV[j*p+k, c] = -2 * sum_{i} error[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
            # dL/dW[k*n+i, c] = -2 * sum_{j} error[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                U3c = U3[:, :, c]  # (n, m)
                V3c = V3[:, :, c]  # (m, p)
                W3c = W3[:, :, c]  # (p, n)
                
                # dU[i*m+j, c] = -2 * sum_k error[i,j,k] * V3[j,k,c] * W3[k,i,c]
                # = -2 * sum_k error[i,j,k] * V3c[j,k] * W3c[k,i]
                # For fixed i,j: sum_k error[i,j,k] * V3c[j,k] * W3c[k,i]
                # = (error[i,j,:] * V3c[j,:]) @ W3c[:,i]  ... not quite
                # Let me compute: for each i,j: sum_k error[i,j,k] * V3c[j,k] * W3c[k,i]
                # = sum_k (error[i,j,:] * V3c[j,:]) [k] * W3c[k,i]
                # = np.einsum('k,ki->', error[i,j,:] * V3c[j,:], W3c)
                
                # More efficiently:
                # temp[i,j] = sum_k error[i,j,k] * V3c[j,k] * W3c[k,i]
                # = np.einsum('ijk,jk,ki->ij', error, V3c, W3c)
                dU3 = np.einsum('ijk,jk,ki->ij', error, V3c, W3c)
                dU[i * m:(i+1) * m, c] = -2 * dU3.ravel()  # Wait, this isn't right
                # Actually dU[i*m+j, c] for all i,j:
                dU[:, c] = -2 * dU3.ravel()
                
                # dV[j*p+k, c] = -2 * sum_i error[i,j,k] * U3[i,j,c] * W3[k,i,c]
                # = -2 * sum_i error[i,j,k] * U3c[i,j] * W3c[k,i]
                # temp[j,k] = sum_i error[i,j,k] * U3c[i,j] * W3c[k,i]
                # = np.einsum('ijk,ij,ki->jk', error, U3c, W3c)
                dV3 = np.einsum('ijk,ij,ki->jk', error, U3c, W3c)
                dV[:, c] = -2 * dV3.ravel()
                
                # dW[k*n+i, c] = -2 * sum_j error[i,j,k] * U3[i,j,c] * V3[j,k,c]
                # = -2 * sum_j error[i,j,k] * U3c[i,j] * V3c[j,k]
                # temp[k,i] = sum_j error[i,j,k] * U3c[i,j] * V3c[j,k]
                # = np.einsum('ijk,ij,jk->ki', error, U3c, V3c)
                dW3 = np.einsum('ijk,ij,jk->ki', error, U3c, V3c)
                dW[:, c] = -2 * dW3.ravel()
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_U, best_V, best_W, best_loss
    
    # Try different ranks
    best_result = None
    best_rank = n * m * p  # schoolbook rank 60
    
    # First, let me just return the schoolbook decomposition as a safe fallback
    # and try to improve if time permits.
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
                c += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank_found = rank_school
    
    # Try to find better decompositions
    for rank in range(50, min(rank_school, 60) + 1):
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple initializations
        for seed in [42, 123, 456]:
            if time.time() > DEADLINE - 10:
                break
            
            np.random.seed(seed)
            U, V, W, loss = try_decomposition(rank, max_iter=2000, lr=0.005)
            
            if loss < 1e-8:
                # Found a valid decomposition!
                if rank < best_rank_found:
                    best_rank_found = rank
                    best_U, best_V, best_W = U, V, W
                break  # No need to try more seeds for this rank
    
    # Convert to lists of lists for the return format
    def to_list(arr):
        return [[arr[i, j] for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    # Round to reduce floating point noise
    best_U = np.round(best_U, 10)
    best_V = np.round(best_V, 10)
    best_W = np.round(best_W, 10)
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
