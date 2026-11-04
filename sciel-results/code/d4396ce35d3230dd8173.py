"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # The target tensor T is defined by T[i, j, k] = 1 if the index corresponds to a valid 
    # multiplication term, but actually for matrix multiplication:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # In tensor form, the multiplication tensor M has entries M[i, j, k] = 1 
    # for all i in [0, n), j in [0, m), k in [0, p).
    # Wait, the standard tensor for matrix multiplication (n, m, p) has size n*m*p.
    # The decomposition is:
    # C = A * B
    # C[i, k] = sum_r U[i, m+j_r?] ... 
    # Let's stick to the standard definition provided in the prompt:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The constraint is: for all i, j, k:
    #   delta_{i,j,k} (which is 1 for all i,j,k in the support? No.)
    #   
    # The matrix multiplication tensor $M_{n,m,p}$ is a tensor in $R^{n \times m} \otimes R^{m \times p} \otimes R^{p \times n}$?
    # No, the standard formulation is:
    # We want to compute C = A B.
    # A is n x m, B is m x p, C is n x p.
    # The tensor of multiplication is defined such that:
    # C_{i,k} = \sum_{j} A_{i,j} B_{j,k}
    # This can be written as a contraction of a rank-R decomposition:
    # C_{i,k} = \sum_{r=1}^R U_{i, m+j?, r} ... 
    #
    # Let's look at the factor shapes:
    # U: (n*m, R). Index u = i*m + j.
    # V: (m*p, R). Index v = j*p + k.
    # W: (p*n, R). Index w = k*n + i.
    #
    # The decomposition condition is:
    # For all i, j, k:
    #   The coefficient of A_{i,j} B_{j,k} in C_{i,k} must be 1.
    #   The coefficient of A_{i,j} B_{j',k} for j != j' must be 0? No.
    #   The coefficient of A_{i',j} B_{j,k} for i != i' must be 0?
    #
    # Actually, the tensor $T$ has entries $T_{(i,j), (j,k), (k,i)}$?
    # The prompt says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "W[k*n+i,r]".
    #
    # The exact decomposition condition is:
    # \sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}?
    # No.
    #
    # The matrix multiplication tensor $M$ has components $M_{i,j,k} = 1$ for all $i,j,k$?
    # No, that would be the all-ones tensor.
    #
    # Let's derive it carefully.
    # We want $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # Let the decomposition be:
    # $C_{i,k} = \sum_r ( \sum_{j'} U[i*m+j', r] A[i, j'] ) ( \sum_{k'} V[j'*p+k', r] B[j', k'] ) W[k*n+i, r]$?
    # This doesn't look right because the indices in U and V must match the summation index j.
    #
    # Standard Tensor Decomposition for MatMul:
    # The tensor is $T \in \mathbb{R}^{(n \times m)} \otimes \mathbb{R}^{(m \times p)} \otimes \mathbb{R}^{(p \times n)}$?
    # Usually it is $T \in \mathbb{R}^n \otimes \mathbb{R}^m \otimes \mathbb{R}^p \otimes \mathbb{R}^n$?
    #
    # Let's look at the shapes again:
    # U: (n*m, R). Row index is $i \cdot m + j$.
    # V: (m*p, R). Row index is $j \cdot p + k$.
    # W: (p*n, R). Row index is $k \cdot n + i$.
    #
    # The condition is that the "reconstructed" matrix multiplication tensor matches the standard one.
    # The standard tensor $S$ has $S_{(i,j), (j,k), (k,i)} = 1$ if the indices are consistent?
    #
    # Actually, the condition is:
    # For all $i, j, k$:
    #   $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$
    # And for all "inconsistent" indices (where the j in U doesn't match j in V, or k in V doesn't match k in W, or i in W doesn't match i in U), the sum must be 0.
    #
    # Specifically:
    # The term $U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]$ contributes to the "slot" defined by the outer indices.
    # The tensor is defined on the space of triplets of indices $( (i,j), (j',k), (k',i') )$.
    # The target tensor $T$ has $T_{(i,j), (j',k), (k',i')} = 1$ if $i=i', j=j', k=k'$, and 0 otherwise.
    #
    # So we need to find U, V, W such that:
    # $\sum_r U_{u, r} V_{v, r} W_{w, r} = T_{u, v, w}$
    # where $u = i*m+j$, $v = j'*p+k$, $w = k'*n+i'$.
    # $T_{u,v,w} = 1$ iff $j = j'$ and $k = k'$ and $i = i'$.
    #
    # This is a rank-R decomposition of a specific 3-way tensor of size $(n*m) \times (m*p) \times (p*n)$.
    # For n=4, m=4, p=8:
    # Size: 16 x 32 x 32.
    #
    # We will use ALS to minimize the Frobenius error:
    # $E = \sum_{u,v,w} ( \sum_r U_{u,r} V_{v,r} W_{w,r} - T_{u,v,w} )^2$
    
    # Construct the target tensor T
    # T is sparse. Only 16 * 8 * 4? No.
    # u ranges 0..15, v ranges 0..31, w ranges 0..31.
    # T[u,v,w] = 1 if:
    #   i = u // m, j = u % m
    #   j2 = v // p, k = v % p
    #   k2 = w // n, i2 = w % n
    #   i == i2 and j == j2 and k == k2
    #
    # Number of non-zeros:
    # i in 0..3, j in 0..3, k in 0..7.
    # For each (i,j,k), there is exactly one u, v, w.
    # So there are 4*4*8 = 128 non-zeros.
    
    # We will store the non-zero indices for efficient ALS updates.
    # nz_u, nz_v, nz_w are lists of indices where T=1.
    nz_u = []
    nz_v = []
    nz_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nz_u.append(u)
                nz_v.append(v)
                nz_w.append(w)
                
    nz_u = np.array(nz_u)
    nz_v = np.array(nz_v)
    nz_w = np.array(nz_w)
    
    # Try to find a decomposition with rank R < 128.
    # Start with R=98 (Strassen bound for 4x4x4 is 49, but this is rectangular).
    # Actually, let's try a range of ranks.
    # The theoretical minimum is not known, but schoolbook is 128.
    # Strassen for 4x4x4 is 49.
    # For 4x4x8, we can view it as two 4x4x4 multiplications? No, the middle dim is 4.
    # It's 4x4 * 4x8.
    # We can split B into two 4x4 matrices B1, B2.
    # C = A*B1 + A*B2.
    # If we use Strassen for 4x4x4, rank is 49.
    # So 49 + 49 = 98.
    # Can we do better?
    # Let's try ranks from 98 down to 80.
    
    best_rank = 128
    best_factors = None
    
    # Pre-compute some matrices for speed?
    # ALS update for U:
    # U[:, r] = (T[:, v, w] * V[v, r] * W[w, r]) / (V[v, r]^2 * W[w, r]^2)
    # This is tricky because the target T is sparse.
    
    # Let's use a simpler approach: Random initialization and Gradient Descent?
    # Or just try to solve the linear system for each factor?
    
    # Actually, for a fixed V and W, U is determined by:
    # U[u, r] = \sum_{v,w} T[u,v,w] V[v,r] W[w,r] / \sum_{v,w} (V[v,r] W[w,r])^2 ?
    # No, that's for least squares on the full tensor.
    
    # Let's implement a basic ALS loop.
    # To speed up, we only iterate over the non-zero entries of T.
    # But the "zero" entries also constrain the solution.
    # If we only fit the non-zeros, we might get a solution that is 1 on non-zeros but garbage on zeros.
    # We need to enforce the zeros too.
    
    # Alternative: Use the fact that the tensor is a "block" tensor.
    # But let's stick to the numerical search as requested by the strategy.
    
    # We will use a "masked" least squares approach.
    # We define a mask M where M[u,v,w] = 1 if T[u,v,w] is non-zero, else 0.
    # We want to minimize || U V W - T ||_F^2.
    # This includes the zeros.
    
    # Since the tensor is large (16x32x32 = 16384 elements), we can actually store the full tensor.
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    T[nz_u, nz_v, nz_w] = 1.0
    
    # ALS for rank R
    def run_als(R, max_iters=50, tol=1e-9):
        # Initialize randomly
        # Use small random values to avoid blow-up
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > _DEADLINE:
                return None, np.inf
                
            # Update U
            # For each r, U[:, r] minimizes || T - \sum_r' U[:, r'] V[r'] W[r'] ||^2
            # This is a linear least squares problem for each column of U?
            # No, U, V, W are coupled.
            # Standard ALS:
            # Fix V, W.
            # The term involving U[:, r] is U[:, r] * (V[r] W[r])^T (in tensor form).
            # Let Z_r = V[:, r] * W[:, r] (outer product? No, element-wise product of the two vectors flattened?)
            # The tensor term is \sum_r U[:, r] \otimes V[:, r] \otimes W[:, r].
            # For fixed V, W, the residual for U is:
            # R_U = T - \sum_r' U[:, r'] \otimes V[:, r'] \otimes W[:, r']
            # We want to minimize || R_U ||^2 with respect to U[:, r].
            # This is equivalent to solving:
            # \sum_{u,v,w} ( \sum_r' U[u, r'] V[v, r'] W[w, r'] - T[u, v, w] )^2
            # Derivative w.r.t U[u, r]:
            # 2 \sum_{v,w} ( \sum_r' U[u, r'] V[v, r'] W[w, r'] - T[u, v, w] ) V[v, r] W[w, r] = 0
            # \sum_{v,w} U[u, r] V[v, r] W[w, r] V[v, r'] W[w, r'] ...
            
            # Let's use a matrix formulation.
            # Reshape T to matrix T_mat of size (n*m) x ((m*p)*(p*n))
            # T_mat[u, v*p*n + w] = T[u, v, w]
            # The model is U * (V \otimes W)^T?
            # (V \otimes W)[r, v*p*n + w] = V[v, r] * W[w, r]
            # So T_mat \approx U * Z^T, where Z is (R x (m*p*p*n))
            
            T_mat = T.reshape(n*m, -1)
            
            for r in range(R):
                # Construct Z for this r? No, Z depends on all V, W.
                # Let's construct Z once per iteration.
                pass
            
            # Construct Z: (R, (m*p)*(p*n))
            # Z[r, v*(p*n) + w] = V[v, r] * W[w, r]
            # This is expensive to construct every time.
            # V is (m*p, R), W is (p*n, R).
            # We can compute Z using broadcasting.
            # Z = (V.T * W.T).reshape(R, -1)?
            # V.T is (R, m*p). W.T is (R, p*n).
            # We need element-wise product for each r, then flatten.
            
            # Let's optimize:
            # V_flat = V.T # (R, m*p)
            # W_flat = W.T # (R, p*n)
            # Z = np.einsum('ir,is->irs', V_flat, W_flat).reshape(R, -1)
            
            Z = np.einsum('ir,is->irs', V.T, W.T).reshape(R, -1)
            
            # Solve U = T_mat * Z^T * (Z * Z^T)^-1 ?
            # Minimize || T_mat - U Z^T ||_F^2
            # U = T_mat Z^T (Z Z^T)^{-1}
            
            try:
                # Regularize to avoid singularity
                reg = 1e-6
                ZtZ = Z @ Z.T + reg * np.eye(R)
                U_new = (T_mat @ Z.T) @ np.linalg.inv(ZtZ)
                U = U_new
            except np.linalg.LinAlgError:
                pass
                
            # Update V
            # Reshape T to (m*p, (n*m)*(p*n))
            # T_mat_v[v, u*(p*n) + w] = T[u, v, w]
            T_mat_v = T.transpose(1, 0, 2).reshape(m*p, -1)
            
            # Z_v[r, u*(p*n) + w] = U[u, r] * W[w, r]
            Z_v = np.einsum('ir,is->irs', U.T, W.T).reshape(R, -1)
            
            try:
                Zv_tZv = Z_v @ Z_v.T + reg * np.eye(R)
                V_new = (T_mat_v @ Z_v.T) @ np.linalg.inv(Zv_tZv)
                V = V_new
            except np.linalg.LinAlgError:
                pass
                
            # Update W
            # Reshape T to (p*n, (n*m)*(m*p))
            T_mat_w = T.transpose(2, 0, 1).reshape(p*n, -1)
            
            Z_w = np.einsum('ir,is->irs', U.T, V.T).reshape(R, -1)
            
            try:
                Zw_tZw = Z_w @ Z_w.T + reg * np.eye(R)
                W_new = (T_mat_w @ Z_w.T) @ np.linalg.inv(Zw_tZw)
                W = W_new
            except np.linalg.LinAlgError:
                pass
                
            # Calculate error
            # Reconstruct T_rec
            # T_rec[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
            # Use einsum for speed
            T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_rec - T)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Check if error is small enough
        if err < 1e-6:
            # Try to round to integers/half-integers
            # Scale factors?
            # The prompt allows integer/half-integer entries.
            # Let's try rounding to nearest 0.5
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            T_rec_int = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
            err_int = np.linalg.norm(T_rec_int - T)
            
            if err_int < 1e-9:
                return (U_int, V_int, W_int), err_int
            else:
                # Maybe scaling is needed?
                # Or maybe just return the float solution if it's "exact" enough?
                # The prompt says "integer/half-integer entries".
                # If we can't round, this rank might not be achievable with these constraints.
                # But let's return the float one for now and check validity in the wrapper?
                # No, the evaluator checks validity.
                # Let's try a few random restarts or different scalings.
                pass
                
        return None, err
        
    # Search for minimal rank
    # We know 128 is possible.
    # Try 98, 90, 80...
    
    for R in [98, 90, 85, 80, 75, 70, 64]:
        if time.time() > _DEADLINE:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > _DEADLINE:
                break
                
            factors, err = run_als(R, max_iters=100, tol=1e-10)
            if factors is not None:
                U, V, W = factors
                # Convert to lists for the return format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify one last time
                U_np = np.array(U_list)
                V_np = np.array(V_list)
                W_np = np.array(W_list)
                T_rec = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
                if np.allclose(T_rec, T, atol=1e-9):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
