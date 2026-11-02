import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    nU = n * m
    nV = m * p
    nW = p * n
    num_targets = n * m * p
    
    # Precompute target indices
    a_idx = []
    b_idx = []
    c_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx.append(i * m + j)
                b_idx.append(j * p + k)
                c_idx.append(k * n + i)
                
    a_idx = np.array(a_idx)
    b_idx = np.array(b_idx)
    c_idx = np.array(c_idx)
    
    # Try to find a low-rank decomposition using SDP relaxation
    # The problem is:
    # Minimize Tr(G)
    # Subject to:
    #   G >= 0 (PSD)
    #   G_ii <= 1 (or some bound, actually we minimize trace so it will be tight)
    #   sum_{r} U[i,r]^2 * V[j,r]^2 * W[k,r]^2 ... wait, the constraint is on the tensor T.
    # The tensor T is rank 1 if T = u v w^T.
    # For general rank R, T = sum_{r=1}^R u_r v_r w_r^T.
    # The SDP relaxation for tensor rank (specifically, the "flatness" or "trace" bound) 
    # often involves looking for a PSD matrix G such that its diagonal elements correspond to the 
    # squared norms of the vectors in a decomposition, but the off-diagonals capture correlations.
    
    # A simpler SDP relaxation for the rank of a tensor T is:
    # Minimize Tr(G)
    # Subject to:
    #   G is PSD
    #   G_{(i,j,k), (i',j',k')} = <T_{(i,j,k)}, T_{(i',j',k')}> ... no, that's fixed.
    
    # Actually, the standard SDP for tensor rank is:
    # Minimize Tr(G)
    # Subject to:
    #   G >= 0
    #   G_{(i,j,k), (i,j,k)} = 1 for all (i,j,k) in support? No.
    
    # Let's use the "Gram Matrix" formulation.
    # If T = sum_{r=1}^R u_r v_r w_r^T, we can define vectors x_{i,j} = u_i v_j, y_{j,k} = v_j w_k, z_{k,i} = w_k u_i.
    # This is getting complicated.
    
    # Let's stick to the direct SDP relaxation for the existence of a decomposition of rank R.
    # We want to know if there exist U, V, W of rank R such that T = sum u v w^T.
    # This is equivalent to checking if the tensor T lies in the convex hull of rank-1 tensors? No, sum of rank 1.
    
    # A known SDP bound is:
    # Minimize Tr(G)
    # Subject to:
    #   G is PSD
    #   G_{(i,j,k), (i',j',k')} = T_{(i,j,k)} T_{(i',j',k')} ... no.
    
    # Let's try a different SDP:
    # Minimize Tr(G)
    # Subject to:
    #   G is PSD
    #   G_{(i,j), (i',j')} = U_{i,r} U_{i',r} ... no.
    
    # Given the complexity and the 150s limit, solving a large SDP for each rank candidate
    # using standard libraries (like CVXPY if available, but only NumPy/SciPy are guaranteed)
    # is risky. SciPy has `scipy.optimize.minimize` but not a general SDP solver.
    # However, we can use the fact that for small dimensions, we might be able to use
    # a custom interior-point method or rely on the fact that the rank is likely not much
    # lower than 96 for 3x4x8.
    
    # Actually, for 3x4x8, the schoolbook rank is 96.
    # Known lower bounds for 3x4x8:
    # The rank of 2x2x2 is 7.
    # The rank of 3x3x3 is 23 (Strassen).
    # For 3x4x8, it's less standard.
    
    # Since we don't have a guaranteed SDP solver in the standard library, 
    # and the directive asks for a "different strategy", let's implement a 
    # "Structured Search" that exploits the block structure of the matrix multiplication.
    
    # Matrix multiplication (3x4) * (4x8) can be viewed as blocks.
    # A = [A1 A2 A3] where A1, A2, A3 are 3x? No, A is 3x4.
    # Let's partition the 4x4 middle dimension.
    # A (3x4) * B (4x8) = C (3x8).
    # Split B into B1 (4x4) and B2 (4x4).
    # C = A * [B1 B2] = [A*B1 A*B2].
    # This doesn't reduce rank directly unless we find a shared decomposition.
    
    # Alternative Strategy: Use the "Strassen-like" block decomposition.
    # If we can decompose the 4x4 multiplication part with lower rank, we can scale it.
    # Rank of 4x4x4 is 22 (known).
    # If we treat the 3x4x8 as three 3x4x4 multiplications?
    # No, the output is 3x8.
    
    # Let's try a heuristic search for a rank < 96 using a smarter initialization 
    # and a more robust optimization method than L-BFGS-B on the full tensor.
    
    # We will use a "Block Coordinate Descent" approach.
    # Fix U, V, optimize W. Then fix U, W, optimize V. Then fix V, W, optimize U.
    # This is equivalent to minimizing || T - sum u v w^T ||^2.
    # This is a least squares problem at each step, which is much faster and more stable
    # than L-BFGS-B on the full nonlinear problem.
    
    def compute_residual(U, V, W):
        # U: (nU, R), V: (nV, R), W: (nW, R)
        # T_approx = sum_{r} u_r v_r w_r^T
        # Residual for index (i,j,k) is 1 - U[i,r] V[j,r] W[k,r]
        # We want sum (1 - U[i,r] V[j,r] W[k,r])^2 to be 0.
        # This is not a linear least squares because of the product of three variables.
        # However, if we fix two, it is linear in the third.
        
        # Let's fix U and V. We want to find W such that:
        # sum_r U[i,r] V[j,r] W[k,r] = 1 for all (i,j,k).
        # Let M_{i,j,k} = sum_r U[i,r] V[j,r].
        # Then we need sum_r M_{i,j,k} W[k,r] = 1.
        # This is a linear system for W?
        # W has shape (nW, R).
        # The equation is: for each (i,j,k), sum_{r=1}^R (U[i,r] V[j,r]) W[k,r] = 1.
        # Note that the coefficient of W[k,r] depends on i, j.
        # This is a bilinear system.
        
        pass

    # Since we can't easily solve the bilinear system, let's stick to the 
    # "Different Strategy" of using a known algebraic decomposition if possible, 
    # or a very aggressive random search with early termination.
    
    # Given the constraints, the most likely "improvement" is finding a rank 
    # slightly below 96. 
    # Let's try rank 90, 85, 80.
    
    # We will use a "Gradient Descent with Restart" approach, but with 
    # a much better initialization: 
    # Initialize U, V, W with random values, but scaled appropriately.
    
    # Actually, let's try to use the "Strassen" algorithm for 2x2x2 blocks 
    # and see if we can compose it.
    # 3x4x8 can be seen as 3 blocks of 1x4x8? No.
    
    # Let's implement a simple, robust search for rank R.
    
    def try_rank_sdp_like(R, time_budget):
        # We will use a "Alternating Least Squares" style approach.
        # Although the objective is non-convex, ALS can be effective.
        
        # Initialize
        U = np.random.randn(nU, R) * 0.1
        V = np.random.randn(nV, R) * 0.1
        W = np.random.randn(nW, R) * 0.1
        
        # To make it "SDP-like", we could project onto the PSD cone? 
        # No, U, V, W are not PSD matrices.
        
        # Let's just do a fast gradient descent with momentum.
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_err = float('inf')
        
        for step in range(1000):
            if time.time() > start_time + time_budget:
                break
                
            # Compute gradients
            # Error E = sum_{i,j,k} (1 - sum_r U[i,r]V[j,r]W[k,r])^2
            # dE/dW[k,r] = -2 * (1 - val_{i,j,k}) * U[i,r] * V[j,r]
            # This requires iterating over all i,j,k.
            
            # Efficient computation:
            # Let P[i,j,r] = U[i,r] * V[j,r]
            # Let Q[i,j,k] = sum_r P[i,j,r] * W[k,r]
            # Residual R[i,j,k] = 1 - Q[i,j,k]
            
            P = U[a_idx] * V[b_idx] # (num_targets, R)
            Q = (P * W[c_idx]).sum(axis=1) # (num_targets,)
            Res = 1.0 - Q
            
            # Gradient for W:
            # dE/dW[k,r] = sum_{i,j} -2 * Res[i,j,k] * P[i,j,r]
            # We can compute this by grouping by k.
            # But k is part of the index.
            
            # Let's compute grad_W
            # grad_W has shape (nW, R)
            # For each target (i,j,k), the contribution to grad_W[k,r] is -2 * Res * P[:,r]
            
            # This is a scatter operation.
            # grad_W = np.zeros((nW, R))
            # np.add.at(grad_W, (c_idx, slice(None)), -2 * Res[:, np.newaxis] * P)
            
            # Similarly for U and V.
            
            # This might be slow in Python loops if not vectorized.
            # Let's use np.add.at
            
            grad_W = np.zeros((nW, R))
            np.add.at(grad_W, (c_idx, np.arange(R)[None, :]), -2 * Res[:, None] * P)
            
            grad_U = np.zeros((nU, R))
            # dE/dU[i,r] = sum_{j,k} -2 * Res * V[j,r] * W[k,r]
            # Let S[j,k,r] = V[j,r] * W[k,r]
            # Then grad_U[i,r] = sum_{j,k} -2 * Res[i,j,k] * S[j,k,r]
            # This is also a scatter.
            
            # To save time, let's just update W first, then U, then V.
            
            # Update W
            vW = momentum * vW - lr * grad_W
            W += vW
            
            # Recompute P, Q, Res
            P = U[a_idx] * V[b_idx]
            Q = (P * W[c_idx]).sum(axis=1)
            Res = 1.0 - Q
            err = np.sum(Res**2)
            
            if err < best_err:
                best_err = err
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
                
            if err < 1e-6:
                break
            
            # Update U
            # grad_U[i,r] = sum_{j,k} -2 * Res[i,j,k] * V[j,r] * W[k,r]
            # We need V[b_idx, :] * W[c_idx, :]
            # This is (num_targets, R)
            S = V[b_idx] * W[c_idx]
            grad_U = np.zeros((nU, R))
            np.add.at(grad_U, (a_idx, np.arange(R)[None, :]), -2 * Res[:, None] * S)
            
            vU = momentum * vU - lr * grad_U
            U += vU
            
            # Update V
            # grad_V[j,r] = sum_{i,k} -2 * Res[i,j,k] * U[i,r] * W[k,r]
            S = U[a_idx] * W[c_idx]
            grad_V = np.zeros((nV, R))
            np.add.at(grad_V, (b_idx, np.arange(R)[None, :]), -2 * Res[:, None] * S)
            
            vV = momentum * vV - lr * grad_V
            V += vV
            
        # Snap to half-integers
        U_int = np.round(U_best * 2) / 2
        V_int = np.round(V_best * 2) / 2
        W_int = np.round(W_best * 2) / 2
        
        P = U_int[a_idx] * V_int[b_idx]
        Q = (P * W_int[c_idx]).sum(axis=1)
        Res = 1.0 - Q
        err_int = np.sum(Res**2)
        
        if err_int < 1e-8:
            return (U_int, V_int, W_int), err_int
        else:
            return None, best_err

    # Try a few ranks
    candidates = [90, 85, 80]
    
    for rank in candidates:
        if time.time() > deadline - 10:
            break
            
        result, err = try_rank_sdp_like(rank, 30)
        
        if result is not None:
            U, V, W = result
            U_list = [[float(val) for val in row] for row in U]
            V_list = [[float(val) for val in row] for row in V]
            W_list = [[float(val) for val in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
