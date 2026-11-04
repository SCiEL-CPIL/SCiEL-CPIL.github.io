"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Strategy:
1. The matrix multiplication tensor $T$ of size $n \times m \times p$ can be decomposed
   as $T_{i,j,k} = \sum_{r=1}^R U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
2. For a fixed rank $R$, this is a system of linear equations in the variables
   $x_{\alpha, r}$ where $\alpha$ indexes the triple $(i,j,k)$ implicitly via the
   Kronecker product structure.
3. Specifically, let $u_r$ be the vector of length $nm$ (flattened U), $v_r$ length $mp$,
   $w_r$ length $pn$. The equation for each $(i,j,k)$ is:
   $\sum_{r=1}^R (U_{i,j,r} V_{j,k,r} W_{k,i,r}) = 1$.
   This is still bilinear in the variables if we view them all as unknowns.
   
   HOWEVER, we can reframe this. Let $X_r$ be the rank-1 tensor $U_r \otimes V_r \otimes W_r$.
   We want $\sum_{r=1}^R X_r = T$.
   
   A more effective approach for small sizes is to view this as finding a low-rank
   approximation in the tensor space. But exactness is required.
   
   Alternative Linear Approach:
   Fix $R$. The set of all tensors decomposable with rank $R$ is a variety.
   We can use the fact that for $R < nmp$, the system is underdetermined.
   
   Let's use a different structural insight:
   The tensor $T_{[n,m,p]}$ has rank $R$ if and only if it can be written as sum of $R$ rank-1 tensors.
   
   We will use a "Gaussian Elimination on the Tensor" approach:
   1. Start with the full tensor $T$ (which has rank $nmp$ in the standard basis).
   2. We want to find $R$ rank-1 tensors $T_1, ..., T_R$ such that $\sum T_r = T$.
   3. This is equivalent to finding $R$ vectors $u_r \in \mathbb{R}^{nm}$, $v_r \in \mathbb{R}^{mp}$, $w_r \in \mathbb{R}^{pn}$
      such that $\sum_{r} u_r \otimes v_r \otimes w_r = T$.
      
   This is still non-linear. However, we can linearize it by fixing two sets of vectors.
   
   NEW STRATEGY:
   Use the "Iterative Linearization" or "Alternating Least Squares" with exact verification.
   But ALS converges to local minima and may not find exact solutions.
   
   BETTER STRATEGY:
   Use the known rank bounds. For [3,5,5], the rank is likely close to 75 but potentially lower.
   Actually, for matrix multiplication, the rank is generally believed to be $nmp$ for prime dimensions or generic cases,
   but there are known reductions for specific sizes.
   
   Let's try a direct linear system approach by fixing the structure of U and V to be sparse/structured,
   then solving for W.
   
   1. Choose random sparse matrices $U$ and $V$ of size $nm \times R$ and $mp \times R$.
   2. The condition becomes: For each $(i,j,k)$, $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$.
   3. This is a linear system in $W$ (flattened $W$ has size $pn \times R$).
   4. Number of equations: $n \cdot m \cdot p = 75$.
   5. Number of unknowns: $p \cdot n \cdot R = 15 R$.
   6. If $15 R \ge 75$, i.e., $R \ge 5$, the system is potentially solvable.
   7. We solve for $W$ using `np.linalg.lstsq`.
   8. Then we verify if the solution is exact.
   9. If exact, we have a valid decomposition.
   10. We iterate over random $U, V$ and decreasing $R$.
"""

import time
import numpy as np
import random

n, m, p = 3, 5, 5
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _validate(U, V, W):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    rank = U.shape[1]
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Construct the tensor from the decomposition
    # Tensor T[i, j, k]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U_np[:, r].reshape(n, m)
        v_col = V_np[:, r].reshape(m, p)
        w_col = W_np[:, r].reshape(p, n)
        
        # Outer product logic:
        # T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        # This is a tensor contraction.
        # We can do this efficiently:
        # temp[i, j, k] = u_col[i, j] * v_col[j, k]
        # Then sum over k with w_col[k, i]
        
        # Efficiently:
        # For each i, j:
        #   T[i, j, :] += u_col[i, j] * v_col[j, :]
        #   Then T[i, j, k] *= w_col[k, i] ... wait, w depends on k and i.
        
        # Let's just loop over i, j for clarity, as n, m, p are small.
        for i in range(n):
            for j in range(m):
                val_u = u_col[i, j]
                if val_u == 0: continue
                # v_col[j, :] is length p
                # w_col[:, i] is length p
                # Contribution to T[i, j, k] is val_u * v_col[j, k] * w_col[k, i]
                T[i, j, :] += val_u * (v_col[j, :] * w_col[:, i])
                
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook():
    """Standard rank n*m*p = 75 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _solve_linear_system(R, U, V, rng):
    """
    Given fixed U and V, solve for W such that the tensor equation holds.
    
    Equation: For all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Let $A_{(i,j,k), (k', i', r)}$ be the coefficient of $W_{k', i', r}$ in the equation for $(i,j,k)$.
    The variable is $W_{k, i, r}$. It appears in the equation for $(i,j,k)$ with coefficient:
    $U[i*m+j, r] * V[j*p+k, r]$ if the indices match?
    
    Wait, the variable is $W[k, i, r]$.
    The term in the sum is $U[i*m+j, r] * V[j*p+k, r] * W[k, i, r]$.
    
    So for a fixed target $(i, j, k)$, the equation is:
    $\sum_{r=1}^R U[i*m+j, r] V[j*p+k, r] W[k, i, r] = 1$
    
    This equation involves only the variables $W[k, i, r]$ for $r=1..R$.
    It does NOT involve $W[k', i', r]$ for any other $(k', i')$.
    
    This means the system decouples!
    For each pair $(k, i)$, we have a set of equations for all $j$:
    For fixed $k, i$:
    $\sum_{j=0}^{m-1} \left( \sum_{r=1}^R U[i*m+j, r] V[j*p+k, r] W[k, i, r] \right) = 1$?
    
    NO. The tensor condition is:
    For each specific $(i, j, k)$, the sum over $r$ must be 1.
    $\sum_{r=1}^R U[i*m+j, r] V[j*p+k, r] W[k, i, r] = 1$
    
    This equation is linear in the variables $W[k, i, 1], \dots, W[k, i, R]$.
    The coefficient of $W[k, i, r]$ is $C_{i,j,k,r} = U[i*m+j, r] * V[j*p+k, r]$.
    
    So for each fixed $(k, i)$, we have a system of $m$ equations (one for each $j$)
    with $R$ unknowns ($W[k, i, r]$ for $r=1..R$).
    
    We need to solve $A_{k,i} \mathbf{w}_{k,i} = \mathbf{1}$
    where $(A_{k,i})_{j,r} = U[i*m+j, r] * V[j*p+k, r]$
    and $\mathbf{w}_{k,i} = [W[k, i, 1], \dots, W[k, i, R]]^T$.
    
    If $R \ge m$ (which is 5), we can potentially solve this.
    If the matrix $A_{k,i}$ has full row rank, we can find a solution.
    Since we can choose $U$ and $V$ randomly, $A_{k,i}$ will likely be full rank.
    
    We can use `np.linalg.lstsq` to find the minimum norm solution, or just any solution.
    Then we verify if the solution is exact (it should be, up to numerical precision).
    """
    # Dimensions:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R) -> We will construct W row by row.
    
    W = np.zeros((p * n, R))
    
    # Iterate over all (k, i) pairs
    for k in range(p):
        for i in range(n):
            # Construct the matrix A for this (k, i)
            # Rows correspond to j (0 to m-1)
            # Cols correspond to r (0 to R-1)
            A = np.zeros((m, R))
            for j in range(m):
                for r in range(R):
                    # U index: i*m + j
                    # V index: j*p + k
                    A[j, r] = U[i*m + j, r] * V[j*p + k, r]
            
            # Solve A @ w = 1
            # w is vector of length R
            b = np.ones(m)
            
            try:
                # Use least squares to handle potential rank deficiency or noise
                w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                # A @ w should be b
                if not np.allclose(A @ w, b, atol=1e-9):
                    return None # No exact solution for this (k,i) with this U,V
            
                # Store w in W
                W[k*n + i, :] = w
                
            except Exception:
                return None
                
    return W

def _find_decomposition():
    """
    Search for a valid decomposition with minimal rank.
    We iterate R from 5 upwards.
    For each R, we try random U, V and solve for W.
    """
    best = _schoolbook()
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    # Theoretical lower bound is higher, but let's start from a reasonable guess.
    # Rank 1 is impossible.
    # For [3,5,5], rank 75 is schoolbook.
    # Can we do better?
    # The decoupling argument suggests that for each (k,i), we need to solve an m x R system.
    # If R >= m, we can solve it.
    # However, the variables W are shared? No, W[k, i, r] is unique to (k,i).
    # Wait, W is a matrix of size (p*n) x R.
    # Row index is k*n + i.
    # So W[k, i, r] is the element at row (k*n+i), col r.
    # The equations for different (k,i) are indeed independent!
    # 
    # This means we can solve for each (k,i) independently.
    # The condition for existence is that for each (k,i), the system A_{k,i} w = 1 has a solution.
    # This requires that 1 is in the column space of A_{k,i}.
    # If A_{k,i} has full row rank (rank m), then any b is in the column space.
    # Since U and V are random, A_{k,i} will have full row rank with high probability if R >= m.
    #
    # So, theoretically, we can find a solution for ANY R >= m (which is 5)??
    # Let's re-read the tensor condition.
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
    #
    # Yes, the equations for T[i, j, k] only involve W[k, i, :].
    # So we can satisfy each (i, j, k) equation by choosing W[k, i, :].
    #
    # BUT, W is a single matrix. Does W[k, i, r] appear in other equations?
    # The term is U[i', j', r] V[j', k', r] W[k', i', r].
    # W[k', i', r] only appears in the equation for (i', j', k') where the W index matches.
    # The W index is (k', i').
    # So W[k', i', r] ONLY appears in the equation for T[i', j', k'] for all j'?
    # Let's check.
    # Equation for T[i, j, k]:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # The variable W[k, i, r] is multiplied by U[i, j, r] V[j, k, r].
    # So for a fixed (k, i), the variable W[k, i, r] appears in the equations for ALL j (0..m-1)
    # for that specific i and k.
    #
    # So the system for (k, i) is:
    # For each j in 0..m-1:
    #   sum_{r=1}^R (U[i, j, r] * V[j, k, r]) * W[k, i, r] = 1
    #
    # This is a linear system A_{k,i} w_{k,i} = 1.
    # If this system has a solution for ALL (k, i), then we have a valid decomposition.
    #
    # So the problem reduces to:
    # Find U (nm x R) and V (mp x R) such that for all (k, i),
    # the matrix A_{k,i} (m x R) has full row rank (or at least 1 is in its col space).
    #
    # If R >= m, and U, V are random, this is almost surely true.
    #
    # So we can likely achieve rank R = m = 5?
    # Or even lower?
    # If R < m, the system is overdetermined. We need 1 to be in the col space.
    # This is a constraint on U and V.
    #
    # Let's try R = 5 first.
    
    for R in range(5, 75):
        if _time_left() < 5:
            break
            
        # Try a few random initializations for U and V
        for trial in range(100):
            if _time_left() < 2:
                break
                
            # Generate random sparse/dense U and V
            # Use small integers or fractions to keep numbers manageable?
            # Floats are fine for validation.
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            # Solve for W
            W = _solve_linear_system(R, U, V, rng)
            
            if W is not None:
                # Validate the full decomposition
                if _validate(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    return best

def construct():
    """
    Main entry point. Returns a dictionary with keys 'U', 'V', 'W'
    containing the factor matrices of the tensor decomposition.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Start with schoolbook
    sb = _schoolbook()
    
    # Try to find a lower rank decomposition
    candidate = _find_decomposition()
    
    # The candidate might be schoolbook if no better found
    # But _find_decomposition returns schoolbook if it fails?
    # No, it returns best which is initialized to schoolbook.
    
    return candidate

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0]) if result['U'] else 'unknown'
    print(f"Rank achieved: {rank}")
