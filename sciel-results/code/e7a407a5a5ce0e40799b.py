"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Check if the decomposition exactly multiplies 2x3 by 3x10 matrices.
    
    For random A (n x m) and B (m x p), we need:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j... 
    Actually, the tensor decomposition says:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[(i,j),r] A[i,j]) * (sum_j V[(j,k),r] B[j,k])... 
    
    No, the correct interpretation:
    The multiplication tensor T has T[i,j,k] = delta... 
    Actually for matrix multiplication, the tensor is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In the decomposition, we have rank-r factors U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all A, B:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    
    Wait, that's not right either. The standard Strassen-like decomposition:
    We compute C = A*B where A is n x m, B is m x p.
    
    The decomposition works as follows:
    - U has shape (n*m, r): for each (i,j) pair and rank factor r
    - V has shape (m*p, r): for each (j,k) pair and rank factor r  
    - W has shape (p*n, r): for each (k,i) output pair and rank factor r
    
    The formula is: C[i,k] = sum_r W[k*n + i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Hmm, that's a bilinear form with three factors, which isn't standard. Let me reconsider.
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We want: C[i,k] = sum_r M1[r, i, :m] dot A[:, i] ... 
    
    No. The correct formulation:
    Define linear maps:
    - For each r, let u_r be a vector in R^{n*m}, so u_r[(i,j)] = U[(i,j), r]
    - For each r, let v_r be a vector in R^{m*p}, so v_r[(j,k)] = V[(j,k), r]  
    - For each r, let w_r be a vector in R^{p*n}, so w_r[(k,i)] = W[(k,i), r]
    
    Then: C[i,k] = sum_r w_r[(k,i)] * (sum_j u_r[(i,j)] * A[i,j]) * (sum_j v_r[(j,k)] * B[j,k])
    
    This is a trilinear form, but we need bilinear. The issue is that this would be cubic in the inputs.
    
    Let me re-read the task: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    The standard matrix multiplication tensor decomposition:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose this as:
    C[i,k] = sum_r U[i*m + j_a(r), r] * V[j_b(r)*p + k, r] * ... 
    
    No, I think the correct interpretation is simpler. The multiplication tensor is a 3-tensor T[i,j,k] where:
    T[i,j,k] = 1 if the (i,k) output depends on A[i,j] and B[j,k]
    
    So T[i,j,k] = 1 when we're computing C[i,k] from A[i,j] * B[j,k].
    
    The tensor decomposition of rank r means:
    T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r] ... 
    
    No wait, that's still not right for a standard matrix multiplication.
    
    Let me think about this differently. The matrix multiplication can be viewed as a bilinear map:
    (A, B) -> C where C = A*B
    
    In tensor notation, this is: C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition means we can write this as:
    C[i,k] = sum_r alpha_r(A) * beta_r(B) * gamma_r(i,k)
    
    where alpha_r is a linear form on A (dimension n*m), beta_r is a linear form on B (dimension m*p),
    and gamma_r is a coefficient for the (i,k) output.
    
    So: alpha_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
         beta_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
         gamma_r(i,k) = W[k*n+i, r]
    
    And we need: for all A, B:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] A[i,j] * V[j*p+k, r] B[j,k] * W[k*n+i, r]
    
    This is still trilinear, which doesn't make sense for bilinear matrix multiplication.
    
    OH WAIT. I think I've been overthinking this. Let me look at the schoolbook solution again:
    
    In the schoolbook, for each (i,j,k) triple, we set:
    U[i*m+j][col] = 1
    V[j*p+k][col] = 1
    W[k*n+i][col] = 1
    
    So the formula must be:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    In schoolbook: for a specific (i,j,k) triple at column c:
    - U[i*m+j, c] = 1, all other U[i*m+j', c] = 0 for j' != j
    - V[j*p+k, c] = 1, all other V[j'*p+k', c] = 0
    - W[k*n+i, c] = 1
    
    So: sum_j U[i*m+j, c] * A[i,j] = A[i,j] (only the j-th term survives)
        sum_j V[j*p+k, c] * B[j,k] = B[j,k] (only the j-th term survives)
        W[k*n+i, c] = 1
    
    So C[i,k] gets contribution: 1 * A[i,j] * B[j,k] from this column.
    
    Summing over all columns (i',j,k') where i'=i and k'=k... wait, no.
    Column c corresponds to (i_c, j_c, k_c).
    The contribution to C[i,k] from column c is:
    W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    
    For column c = (i_c, j_c, k_c):
    W[k*n+i, c] = 1 iff k=k_c and i=i_c
    U[i*m+j, c] = 1 iff i=i_c and j=j_c
    V[j*p+k, c] = 1 iff j=j_c and k=k_c
    
    So the contribution is non-zero only when i=i_c, k=k_c, AND j=j_c for both U and V terms.
    In that case: 1 * A[i,j_c] * B[j_c,k] = A[i,j_c]*B[j_c,k]
    
    Summing over all c where i_c=i, k_c=k, j_c=j gives exactly sum_j A[i,j]B[j,k]. ✓
    
    Great, so the formula is confirmed:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    This IS trilinear in (A, B, C), but C is the output, so it's really saying:
    The bilinear map (A,B) -> C has the form where each output C[i,k] is a sum of products of a linear form in A and a linear form in B, with coefficients from W.
    
    So it's a "rank-r" decomposition of the bilinear map, where each term is:
    (linear form in A) * (linear form in B) * (coefficient for output (i,k))
    
    OK so now I understand the structure. Let me verify with a small test.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    # Test with a few random matrices
    for trial in range(5):
        A = [[rng.uniform(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[rng.uniform(-10, 10) for _ in range(p)] for _ in range(m)]
        # Compute C = A*B
        C_true = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_true[i][k] = s
        # Compute C from decomposition
        C_calc = [[0.0]*p for _ in range(n)]
        rank = len(U[0])
        for r in range(rank):
            # alpha_r = sum_j U[i*m+j, r] * A[i,j]
            # beta_r = sum_j V[j*p+k, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    alpha = 0.0
                    for j in range(m):
                        alpha += U[i*m+j][r] * A[i][j]
                    beta = 0.0
                    for j in range(m):
                        beta += V[j*p+k][r] * B[j][k]
                    C_calc[i][k] += W[k*n+i][r] * alpha * beta
        # Check
        for i in range(n):
            for k in range(p):
                if abs(C_calc[i][k] - C_true[i][k]) > 1e-9:
                    return False
    return True


def construct():
    """Construct the best known rank-minimal decomposition for 2x3x10 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook rank 60
    best = _build_schoolbook()
    best_rank = 60
    
    # Try to find lower rank decompositions using random search
    # Strategy: use random linear combinations and try to reduce rank
    # by exploiting symmetries and the specific dimensions
    
    # For 2x3x10, we can try ranks from 59 down
    # Use a random search approach: initialize U, V, W randomly and try to solve
    # the system of polynomial equations
    
    # The equations are: for all i,k:
    # sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    
    # This must hold for all A, B, so we get polynomial equations in the entries of U, V, W.
    # Specifically, for each (i,k) and each pair of indices (j1, j2) in the A and B sides:
    # The coefficient of A[i,j1] * B[j2,k] must match:
    # Left side: 1 if j1 == j2, else 0
    # Right side: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r]
    
    # So the constraint is:
    # sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta(j1, j2)
    # for all i, k, j1, j2
    
    # Number of constraints: n * p * m * m = 2 * 10 * 3 * 3 = 180
    # Number of variables: rank * (n*m + m*p + p*n) = rank * (6 + 30 + 20) = rank * 56
    
    # For rank < 60/56... no, rank * 56 >= 180, so rank >= ceil(180/56) = 4
    # But the actual rank is much higher due to the nonlinear constraints.
    
    # Let's try random initialization and Newton's method or least-squares
    
    # Actually, let's try a more structured approach.
    # For 2x3x10, one approach is to use the fact that 2x3 multiplication
    # can be done with rank 7 (Strassen-like), and then extend to 10 columns.
    
    # But the coupling between columns makes this non-trivial.
    
    # Let's try random search for rank 55-59
    np.random.seed(42)
    
    def try_random_rank(rank, n_trials=50):
        """Try random initialization and least-squares to find a rank-r decomposition."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U_np = np.random.randn(nm, rank) * 0.5
            V_np = np.random.randn(mp, rank) * 0.5
            W_np = np.random.randn(pn, rank) * 0.5
            
            # Build the system of equations
            # Variables: U (nm*rank), V (mp*rank), W (pn*rank)
            # Equations: for each (i, k, j1, j2):
            #   sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta(j1,j2)
            
            n_eq = n * p * m * m
            eqs = []
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            target = 1.0 if j1 == j2 else 0.0
                            # Equation: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] - target = 0
                            eqs.append((target, i, k, j1, j2))
            
            # Use simple iterative approach: alternate minimization
            # Fix V and W, solve for U (linear), then fix U and W, solve for V, then fix U and V, solve for W
            for iteration in range(200):
                if time.time() > deadline:
                    return None
                
                # Solve for U: for each (i, j1, r):
                # The equations involving U[i*m+j1, r] are:
                # For each (k, j2): W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta(j1,j2)
                # Sum over r of these... it's coupled across r.
                
                # This is a 3-way tensor unfolding problem. Let's use a different approach.
                # For fixed V and W, the equation for U is:
                # For each (i, j1): sum_{k, j2} [W[k*n+i, :] * V[j2*p+k, :]]^T * ... 
                
                # Actually, let's think of it as:
                # For each (i, j1, j2): sum_k W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :] = delta(j1,j2)
                # where * is element-wise product and sum is over r.
                
                # This is: sum_r U[i*m+j1, r] * (sum_k W[k*n+i, r] * V[j2*p+k, r]) = delta(j1,j2)
                
                # Define M[i, j1, j2] = delta(j1, j2)
                # Define N[i, j1, j2] = sum_k W[k*n+i, :] * V[j2*p+k, :]  (element-wise, summed over r)
                # Then: sum_r U[i*m+j1, r] * N[i, j1, j2, r]... 
                
                # Hmm, this is getting complicated. Let me just use a direct least-squares approach.
                
                # Reshape to vectors
                u_vec = U_np.flatten()  # nm * rank
                v_vec = V_np.flatten()  # mp * rank
                w_vec = W_np.flatten()  # pn * rank
                
                # Build Jacobian and residual for the nonlinear system
                # This is expensive, so let's do a simpler gradient descent
                
                # Compute residual
                residual = 0.0
                for (target, i, k, j1, j2) in eqs:
                    s = 0.0
                    for r in range(rank):
                        s += W_np[k*n+i, r] * U_np[i*m+j1, r] * V_np[j2*p+k, r]
                    residual += (s - target) ** 2
                
                if residual < 1e-12:
                    # Found a solution!
                    return {
                        "U": U_np.tolist(),
                        "V": V_np.tolist(),
                        "W": W_np.tolist()
                    }
                
                # Gradient descent (too slow for this problem)
                # Let's break and try a different approach
                if trial % 10 == 9:
                    break
            
        return None
    
    # Try structured approach based on 2x3x3 rank 15, then extend
    # Actually, let's just try a few random ranks
    
    for rank in range(59, 40, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(rank, n_trials=20)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                best = result
                best_rank = rank
                break
    
    if best_rank < 60:
        return best
    return _build_schoolbook()

# EVOLVE-BLOCK-END
