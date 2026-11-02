import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145

    n, m, p = DIMENSIONS  # 2, 2, 13
    N, M, P = n * m, m * p, p * n  # 4, 26, 26

    # Field-theoretic decomposition over F_13
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # 
    # delta_{j,j'} = sum_{s=0}^{1} f_j(s) * f_{j'}(s)  [2 terms]
    #   f_0 = [1, 0], f_1 = [0, 1] (identity)
    # delta_{i,i'} = sum_{r=0}^{1} g_i(r) * g_{i'}(r)  [2 terms, but we can use 1 with sign]
    #   Actually for n=2: delta_{i,i'} = (1 + (-1)^{i+i'})/2... no
    #   Use: g_0 = [1], g_1 = [-1] gives delta via (1 + g_i*g_i')/2... not rank-1
    #   Better: g_i(r) for r in {0,1}: g_0=[1,0], g_1=[0,1] -> 2 terms
    #   OR: use a single term with g_0=1, g_1=-1: then g_i*g_i' = 1 if i=i', -1 if i!=i'
    #        delta_{i,i'} = (1 + g_i*g_i')/2, which needs 2 terms (constant + product)
    #   So minimum 2 terms for i-delta when n=2.
    # delta_{k,k'} = sum_{d=0}^{12} omega^{d*k} * omega^{-d*k'}  [13 terms, Vandermonde]
    #
    # Total rank: 2 * 2 * 13 = 52? No, that's schoolbook.
    # 
    # Wait, the decomposition is a TENSOR product of the three deltas:
    # T = T_j ⊗ T_i ⊗ T_k
    # Rank of tensor product = product of ranks? No!
    # 
    # Actually, T[i*m+j, j'*p+k, k'*n+i'] factors as:
    #   U_part[i*m+j, t_j] for j-delta
    #   V_part[j'*p+k, t_k] for k-delta (k is in V's index)
    #   W_part[k'*n+i', t_i] for i-delta (i is in W's index)
    #
    # The tensor is a Kronecker product of three smaller tensors.
    # T = T_j ⊗ T_k ⊗ T_i (in appropriate order)
    # 
    # For a Kronecker product of tensors, the rank is at most the product of ranks.
    # But we can do better if we share terms.
    #
    # Actually, let me reconsider. The tensor is:
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # This is exactly the Kronecker product:
    # T = D_m ⊗ D_p ⊗ D_n  (with appropriate index mappings)
    # where D_m is the m×m identity tensor, etc.
    #
    # The rank of D_m (as a 3-way tensor with indices (j, j', ?)) is... 
    # Actually D_m here is just the identity matrix viewed as a tensor.
    # 
    # Let me think of it differently. Define:
    # T_j[j, j'] = delta_{j,j'} (an m×m matrix)
    # T_i[i, i'] = delta_{i,i'} (an n×n matrix)  
    # T_k[k, k'] = delta_{k,k'} (a p×p matrix)
    #
    # The full tensor T[a,b,c] = T_j[j(a), j(b)] * T_i[i(c), i(a)] * T_k[k(b), k(c)]
    #
    # This is a "structured" tensor. Its rank can be computed.
    #
    # Key insight: T_j[j,j'] * T_k[k,k'] can be combined into V's contribution:
    # V[j'*p+k, t] needs to encode both j' and k information.
    #
    # Let's use: for the (j',k) part of V, and the (j) part of U, and (i,i',k') split between W and others.
    #
    # Approach: 
    # - Use p=13 terms for the k-delta via Vandermonde over F_13
    # - Use m=2 terms for the j-delta
    # - Use n=2 terms for the i-delta
    # - Total: 13 * 2 * 2 = 52 (schoolbook again)
    #
    # But wait! The j-delta and k-delta both involve V's index (j',k). 
    # We can combine them: V[j'*p+k, t] where t encodes (j', k) via a combined rank.
    # 
    # The tensor T_j ⊗ T_k has rank at most m*p = 26 (schoolbook for that sub-problem).
    # But T_j ⊗ T_k is actually the identity tensor of size m×p, which has rank m*p.
    #
    # Hmm, let me try a different decomposition.
    #
    # Alternative: Note that T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j=j' AND i=i' AND k=k'
    # 
    # This means: for fixed (i,j), the only nonzero entries are at (j'*p+k, k'*n+i') 
    # where j'=j, k=k, i'=i. So T[i*m+j, j*p+k, k*n+i] = 1 and all other entries in 
    # row i*m+j are 0.
    #
    # This is a permutation-like tensor. Each row of U has exactly one nonzero "pattern".
    #
    # Let me try: rank = n * m * p / something? 
    #
    # Actually, for the specific case n=2, m=2, p=13:
    # Known result: the rank of the matrix multiplication tensor M_{n,m,p} 
    # is at most n*m*p, and there are better bounds.
    #
    # For M_{2,2,13}: 
    # - Schoolbook: 52
    # - Can we do better?
    #
    # Let me try the field approach more carefully.
    # 
    # Work over F_13. Let omega = 2 (a primitive root mod 13).
    # omega^1=2, omega^2=4, omega^3=8, omega^4=16=3, omega^5=6, omega^6=12,
    # omega^7=24=11, omega^8=22=9, omega^9=18=5, omega^10=10, omega^11=20=7, omega^12=14=1
    #
    # delta_{k,k'} = (1/13) * sum_{d=0}^{12} omega^{d(k-k')}
    #
    # For the j-delta (m=2): delta_{j,j'} = sum_{s=0}^{1} f_j(s) f_{j'}(s)
    #   f_0 = [1, 0], f_1 = [0, 1]
    #
    # For the i-delta (n=2): delta_{i,i'} = sum_{r=0}^{1} g_i(r) g_{i'}(r)
    #   g_0 = [1, 0], g_1 = [0, 1]
    #
    # Now, the full decomposition:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # = [sum_s f_j(s) f_{j'}(s)] [sum_r g_i(r) g_{i'}(r)] [sum_d (1/13) omega^{d(k-k')}]
    #
    # = sum_{s,r,d} f_j(s) f_{j'}(s) g_i(r) g_{i'}(r) (1/13) omega^{dk} omega^{-dk'}
    #
    # Each (s,r,d) triple gives a rank-1 term:
    # U[i*m+j, (s,r,d)] = f_j(s) * g_i(r)
    # V[j'*p+k, (s,r,d)] = f_{j'}(s) * omega^{dk}
    # W[k'*n+i', (s,r,d)] = g_{i'}(r) * (1/13) * omega^{-dk'}
    #
    # Rank = m * n * p = 2 * 2 * 13 = 52. Still schoolbook!
    #
    # The issue is that the tensor is a full Kronecker product, so its rank is the product.
    #
    # BUT WAIT. Let me re-examine. The entries must be integers or half-integers.
    # 1/13 is not an integer or half-integer!
    #
    # So the field approach with 1/13 doesn't work directly. We need integer/half-integer entries.
    #
    # Alternative: use the fact that over the integers, delta_{k,k'} can be represented
    # differently. But for p=13, the minimal rank for the identity tensor of size 13x13
    # (as a 2-way tensor) is 13, but as part of a 3-way tensor...
    #
    # Let me reconsider the problem. Maybe the rank IS 52 and that's optimal for this case.
    # But the task says "rank-minimal" and the current score is 1.0 (seed), implying there
    # might be room for improvement.
    #
    # Actually, re-reading: "score is the rank" and "seed/value" where seed is the initial rank.
    # If seed is 52, then score 1.0 means rank 52. To improve, we need rank < 52.
    #
    # Let me think about whether rank < 52 is possible.
    #
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    #
    # This is equivalent to: for each (i,j), the row i*m+j of the "unrolled" tensor has
    # a 1 at position (j*p+k, k*n+i) for each k, and 0 elsewhere.
    #
    # So row i*m+j has p nonzeros (one for each k).
    #
    # The tensor can be seen as a 3-way array. Its rank is the minimum number of rank-1
    # tensors that sum to it.
    #
    # Lower bound: The tensor has "support" of size n*m*p = 52 (the number of nonzero entries).
    # Each rank-1 tensor can have at most... well, it's a product of three vectors, so its
    # support can be large.
    #
    # Actually, a better lower bound: consider the "slice" where we fix j=j', i=i'.
    # Then we get a p×p identity tensor (in the k,k' indices). The rank of the identity
    # matrix of size p is p. But this is a 2-slice, not a direct lower bound on the 3-way rank.
    #
    # Another approach: the tensor T is the "multiplication table" tensor. For M_{n,m,p},
    # the rank is known to be at most n*m*p, and for small cases, exact values are known.
    #
    # For M_{2,2,13}: I believe the rank is exactly 52, because the tensor is a 
    # "combinatorial" tensor that doesn't admit Strassen-like shortcuts when p is prime
    # and larger than 4.
    #
    # However, let me try a different approach: can we use the fact that n=2 and m=2
    # to get some savings?
    #
    # Consider: T[i*m+j, j'*p+k, k'*n+i'] 
    # If we fix i and i', the sub-tensor over (j, j', k, k') is:
    #   delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # which is either 0 (if i≠i') or the identity tensor of size 2×13.
    #
    # The rank of the identity tensor of size m×p (as a 2-way tensor embedded in 3-way)
    # is m*p.
    #
    # I think for this specific case, rank 52 is indeed optimal. Let me just make sure
    # the schoolbook decomposition is correct and return it.
    #
    # Actually, wait. Let me reconsider the problem statement. It says "exactly decompose
    # the matrix multiplication tensor" with "integer/half-integer entries". 
    #
    # The schoolbook decomposition uses all 1s, which are integers. So it's valid.
    #
    # Let me just verify the schoolbook is correct and return it.

    def build_schoolbook(rank):
        """Build schoolbook decomposition with given rank (must be >= n*m*p)."""
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                actual = float(np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :]))
                                err = abs(actual - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol

    def tensor_loss(params, rank):
        """Loss function for tensor decomposition optimization."""
        U = params[:N * rank].reshape(N, rank)
        V = params[N * rank:N * rank + M * rank].reshape(M, rank)
        W = params[N * rank + M * rank:].reshape(P, rank)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                actual = float(np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :]))
                                loss += (actual - expected) ** 2
        return loss

    def try_optimize(rank, max_iter=500, tol=1e-8, seed=42):
        """Try to find a decomposition of given rank via optimization."""
        np.random.seed(seed)
        
        # Initialize with scaled random + schoolbook-like structure
        U0 = np.random.randn(N, rank) * 0.1
        V0 = np.random.randn(M, rank) * 0.1
        W0 = np.random.randn(P, rank) * 0.1
        
        # Add schoolbook structure if rank >= n*m*p
        if rank >= n * m * p:
            U0, V0, W0 = build_schoolbook(rank)
            # Perturb slightly to escape local minima
            U0 += np.random.randn(N, rank) * 0.01
            V0 += np.random.randn(M, rank) * 0.01
            W0 += np.random.randn(P, rank) * 0.01
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Use L-BFGS-B for optimization
        result = minimize(
            tensor_loss, params0, args=(rank,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < 1e-6:
            U_opt = result.x[:N * rank].reshape(N, rank)
            V_opt = result.x[N * rank:N * rank + M * rank].reshape(M, rank)
            W_opt = result.x[N * rank + M * rank:].reshape(P, rank)
            
            if verify_decomposition(U_opt, V_opt, W_opt, tol=tol):
                # Round to half-integers if possible
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify_decomposition(U_round, V_round, W_round, tol=1e-6):
                    return U_round, V_round, W_round
                
                return U_opt, V_opt, W_opt
        
        return None

    # Start with schoolbook (rank 52) as guaranteed fallback
    U_fb, V_fb, W_fb = build_schoolbook(n * m * p)
    
    # Try to find lower rank decompositions
    # Based on known bounds, try ranks below 52
    for rank in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
        
        result = try_optimize(rank, max_iter=300, seed=rank)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if verify_decomposition(U_opt, V_opt, W_opt, tol=1e-8):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # If optimization didn't find better, return schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
