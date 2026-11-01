import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The multiplication tensor T has shape (n, m, p)
    # T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the standard layout for the evaluator:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The tensor being decomposed is T such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k in product)
    # Actually, the tensor T_{i,j,k} for matrix multiplication C=AB is 1 if we consider the operation.
    # Let's define the tensor M of size (n*m) x (m*p) x (p*n) ? No.
    
    # Standard Tensor Rank definition for Matrix Mult:
    # We want to represent the bilinear map.
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)}? 
    # Let's look at the indices:
    # Input A: (n, m) -> flattened to n*m
    # Input B: (m, p) -> flattened to m*p
    # Output C: (n, p) -> flattened to n*p. 
    # The task specifies W shape (p*n, rank) and index W[k*n+i, r]. This is (p, n) flattened? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # C is n x p. C[i, k]. 
    # If W index is k*n + i, that corresponds to row k, col i in a p x n matrix?
    # Or is it C^T? 
    # Let's check the schoolbook in the prompt:
    # W[k * n + i][column] = 1
    # Loop: for i in n, for j in m, for k in p.
    # U[i*m+j], V[j*p+k], W[k*n+i].
    # This implies the third mode of the tensor is indexed by (k, i) with index k*n+i.
    # So the tensor T has dimensions (n*m, m*p, p*n).
    # T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's construct this target tensor T_target
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T_target = np.zeros((dim_u, dim_v, dim_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0

    # We search for the minimal rank r.
    # Lower bound for rank of 2x2 by 2x11 mult:
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x2x2x...x2, rank grows.
    # Here p=11.
    # A trivial bound is n*m*p = 44.
    # Strassen on 2x2 blocks: 
    # We can view 2x2 by 2x11 as 2x2 by (2x2) x (2x2) x ... 
    # Actually, let's just start from a reasonable lower bound.
    # Rank of 2x2x2 is 7.
    # Rank of 2x2x4 is at least ? 
    # Let's start r from 30 up to 44.
    
    best_result = None
    
    # We will use ALS to approximate T_target with rank r.
    # If the error drops below a tiny threshold (e.g. 1e-10), we assume it's exact (or close enough for integer/half-integer rounding).
    
    for r in range(30, 45):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize random tensors U, V, W
        # Use small random values to avoid convergence issues initially
        U = np.random.randn(dim_u, r) * 0.1
        V = np.random.randn(dim_v, r) * 0.1
        W = np.random.randn(dim_w, r) * 0.1
        
        converged = False
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
                
            # Update U: U[:, c] = T_target[:, :, c] dot (V[:, c] * W[:, c]) / ||V[:, c] * W[:, c]||^2
            # This is a bit slow if done naively. 
            # T_target is sparse? No, it's 0s and 1s.
            # We can compute the contraction.
            
            # For each column c:
            # U[:, c] = matvec(T_target, V[:, c] * W[:, c])
            # This is equivalent to:
            # U = T_target @ (V * W)  -- where * is element-wise on columns?
            # Let's do it efficiently.
            # T_target shape (I, J, K). V shape (J, r). W shape (K, r).
            # We want U (I, r).
            # U[i, c] = sum_j,k T[i,j,k] V[j,c] W[k,c]
            # This is a tensor contraction.
            
            # Reshape T to (I, J*K)
            T_flat = T_target.reshape(dim_u, dim_v * dim_w)
            
            # Construct the combined vector for V and W for all c at once?
            # V[:, c] * W[:, c] is not a single vector in R^{J*K} unless we outer product them.
            # Actually, for fixed c, the vector is V_c \otimes W_c.
            # So we need T_flat @ (V_c \otimes W_c).
            
            # This can be computed as:
            # U_c = (T_target * (V_c[None, :, None] * W_c[None, None, :])).sum(axis=(1,2))
            
            # Let's update all columns.
            # This might be memory heavy if r is large, but r<=44, dims are small (4, 22, 22).
            # 4*22*22 = 1936. Very small.
            
            # Update U
            for c in range(r):
                # Contract T with V[:,c] and W[:,c]
                # T[i, j, k] * V[j, c] * W[k, c]
                # Sum over j, k
                # Use np.einsum or reshape
                # T_target (I, J, K)
                # V_c (J)
                # W_c (K)
                # Result (I)
                U[:, c] = np.einsum('ijk,jc,kc->ic', T_target, V, W)[:, c]
                
            # Update V
            # V[j, c] = sum_i,k T[i, j, k] U[i, c] W[k, c]
            for c in range(r):
                V[:, c] = np.einsum('ijk,ic,kc->jc', T_target, U, W)[:, c]
                
            # Update W
            # W[k, c] = sum_i,j T[i, j, k] U[i, c] V[j, c]
            for c in range(r):
                W[:, c] = np.einsum('ijk,ic,jc->kc', T_target, U, V)[:, c]
                
            # Check convergence / error
            # Error = || T_target - sum_c U_c \otimes V_c \otimes W_c ||_F
            # We can compute the reconstructed tensor
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_target - T_recon)
            
            if err < 1e-9:
                converged = True
                break
                
        if converged:
            # We found a rank r decomposition.
            # Now we need to convert to integer/half-integer entries.
            # The problem asks for integer/half-integer entries.
            # ALS finds real values. 
            # If the decomposition is exact for the integer tensor, the real values might be integers or half-integers if we scale appropriately?
            # Actually, tensor rank decompositions over Q (rationals) exist if they exist over R for integer tensors?
            # Not necessarily, but for small ranks, often yes.
            # However, the evaluator checks "integer/half-integer".
            # We can try to round U, V, W to nearest 0.5 and check if the decomposition still holds.
            
            # Scale factors? 
            # Usually, if an exact rational decomposition exists, ALS might converge to it up to scaling.
            # Let's try rounding to nearest 0.5.
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_recon_rounded = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
            err_rounded = np.linalg.norm(T_target - T_recon_rounded)
            
            if err_rounded < 1e-9:
                # Success!
                # Convert to lists
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break # Found minimal r in the loop order (increasing r), so this is minimal.
        
    if best_result is None:
        # Fallback to schoolbook if numerical method fails or times out
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
        best_result = {"U": U, "V": V, "W": W}
        
    return best_result
