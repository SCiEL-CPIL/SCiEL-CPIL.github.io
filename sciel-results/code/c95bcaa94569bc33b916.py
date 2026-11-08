import numpy as np
import scipy.optimize as opt
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 44
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randint(-2, 3, size=(n * m, target_rank))
    V = np.random.randint(-2, 3, size=(m * p, target_rank))
    W = np.random.randint(-2, 3, size=(p * n, target_rank))
    
    # Flatten for optimization
    def flatten(x):
        return np.concatenate([x['U'].flatten(), x['V'].flatten(), x['W'].flatten()])
    
    def unflatten(x):
        u_size = n * m * target_rank
        v_size = m * p * target_rank
        w_size = p * n * target_rank
        U_flat = x[:u_size]
        V_flat = x[u_size:u_size + v_size]
        W_flat = x[u_size + v_size:]
        return {
            'U': U_flat.reshape(n * m, target_rank),
            'V': V_flat.reshape(m * p, target_rank),
            'W': W_flat.reshape(p * n, target_rank)
        }
    
    # Residual function
    def residual(x):
        params = unflatten(x)
        U = params['U']
        V = params['V']
        W = params['W']
        
        # Compute tensor T = sum_t U_t outer V_t outer W_t
        # T[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We want T to match the multiplication tensor.
        # Multiplication tensor M[i, j, k] is 1 if j is the column index of A and row index of B?
        # No, the tensor is defined by C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor T_{i,j,k} should be 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # In the decomposition C = sum_t U_t V_t W_t, the term for (i,j,k) is U[i,j,t] V[j,k,t] W[k,i,t].
        # So we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 if (i,j,k) is a valid multiplication term?
        # Actually, the multiplication tensor has 1 at (i, j, k) for all i, j, k?
        # No. The tensor is $T_{i, j, k} = \delta_{j, j'} \delta_{k, k'}$?
        # The standard matrix multiplication tensor $M_{n,m,p}$ has entries $T_{i, j, k} = 1$ for all $i,j,k$?
        # No. The tensor is defined such that $\langle T, A \otimes B \otimes C \rangle = \text{tr}(C A B)$.
        # Or $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # This corresponds to $T_{i, j, k} = 1$ for all $i, j, k$?
        # Let's check: $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}$.
        # If $T_{i,j,k} = 1$, then $\sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = \text{tr}(C A B)$.
        # Yes, the tensor is all ones?
        # No, the decomposition is $T = \sum_t U_t \otimes V_t \otimes W_t$.
        # So $T_{i,j,k} = \sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}$.
        # We want this to equal the multiplication tensor.
        # The multiplication tensor for $C = AB$ is $T_{i,j,k} = 1$ for all $i,j,k$?
        # Let's check schoolbook:
        # $U_{i,j,t} = 1$ if $t$ corresponds to $(i,j,k)$.
        # $V_{j,k,t} = 1$ if $t$ corresponds to $(i,j,k)$.
        # $W_{k,i,t} = 1$ if $t$ corresponds to $(i,j,k)$.
        # Then $T_{i',j',k'} = \sum_t U_{i',j',t} V_{j',k',t} W_{k',i',t}$.
        # If $t$ is index for $(i,j,k)$, then $U_{i',j',t} = \delta_{i',i} \delta_{j',j}$.
        # $V_{j',k',t} = \delta_{j',j} \delta_{k',k}$.
        # $W_{k',i',t} = \delta_{k',k} \delta_{i',i}$.
        # Product is $\delta_{i',i} \delta_{j',j} \delta_{j',j} \delta_{k',k} \delta_{k',k} \delta_{i',i} = \delta_{i',i} \delta_{j',j} \delta_{k',k}$.
        # Sum over $t$ (which is $(i,j,k)$) gives 1 if $i'=i, j'=j, k'=k$.
        # So $T_{i,j,k} = 1$ for all $i,j,k$.
        
        # So the target tensor is all ones.
        
        # Compute T from U, V, W
        # T[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Residual is T - 1
        # We can flatten T and subtract 1
        
        # Efficient computation:
        # T = einsum('it,jt,kt->ijk', U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        # But indices are permuted.
        # U index: i*m+j
        # V index: j*p+k
        # W index: k*n+i
        
        # Let's just loop for simplicity in the residual, as n,m,p are small.
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(target_rank):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    T[i,j,k] = val
        return (T - 1).flatten()
    
    x0 = flatten({'U': U, 'V': V, 'W': W})
    
    try:
        result = opt.least_squares(residual, x0, method='lm', max_nfev=10000)
        # Check residual
        if result.cost < 1e-6:
            params = unflatten(result.x)
            # Round to half-integers
            U_int = np.round(params['U'] * 2) / 2
            V_int = np.round(params['V'] * 2) / 2
            W_int = np.round(params['W'] * 2) / 2
            
            # Verify exactness
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for t in range(target_rank):
                            val += U_int[i*m+j, t] * V_int[j*p+k, t] * W_int[k*n+i, t]
                        T[i,j,k] = val
            if np.allclose(T, 1):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
    except Exception:
        pass
    
    # Fallback to schoolbook
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
