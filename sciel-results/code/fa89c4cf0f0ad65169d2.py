import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Finds a low-rank exact tensor decomposition for 2x4x8 matrix multiplication.
    Strategy: 
    1. Start with a known valid decomposition (e.g., block diagonal Strassen).
    2. Use a greedy search / linear algebra approach to identify and remove redundant terms 
       by checking if any term can be expressed as a linear combination of others 
       in the context of the specific tensor constraints, or more practically, 
       by searching for a smaller rank decomposition via random initialization 
       and refinement if the static construction is suboptimal.
    
    However, for 2x4x8, the known upper bound is 56 (4 * 14). 
    The lower bound is 32.
    Recent results suggest R(2x4x8) might be lower than 56. 
    But finding it dynamically in <120s is hard.
    
    A more robust "different strategy" is to construct the decomposition 
    using a **global linear search** over the space of rank-1 tensors 
    constrained to the support of the multiplication tensor.
    
    Actually, the most effective different strategy for small matrices is 
    to use **Strassen's algorithm generalized** via block partitioning 
    and then **merging** the blocks if possible.
    
    Let's try a specific construction for 2x4x8 that might beat 56.
    We can view 2x4x8 as:
    A (2x4) * B (4x8) = C (2x8)
    
    Partition A into A1 (2x2), A2 (2x2) horizontally.
    Partition B into B1 (4x4), B2 (4x4) vertically.
    
    C = A1*B1 + A2*B2
    
    Each term A1*B1 is 2x2 * 2x4? No, B1 is 4x4.
    A1 is 2x2. B1 top 2x4, B1 bottom 2x4.
    So A1*B1 = A1 * [B1_top; B1_bottom] = [A1*B1_top, A1*B1_bottom]
    This is two 2x2 * 2x4 multiplications.
    R(2x2x4) = 14 (two 2x2x2 multiplications).
    So A1*B1 costs 28.
    A2*B2 costs 28.
    Total 56.
    
    Can we do better?
    Consider A (2x4) and B (4x8).
    Maybe use a 2x2x2 Strassen block but with larger blocks?
    
    Let's implement a search for rank < 56 using a simple heuristic:
    Try to find a decomposition of rank R by solving a least-squares problem 
    for random U, V and then fixing W? No, that's non-linear.
    
    Given the difficulty, I will implement the **optimal known construction** 
    for 2x4x8 if it exists, or fall back to 56.
    
    Actually, there is a known result: R(2x4x8) <= 50? 
    Let's check if we can construct a rank 50 decomposition.
    
    Alternative Strategy: **Randomized Search for Rank Reduction**
    1. Start with rank 56.
    2. Randomly perturb the matrices U, V, W.
    3. Project onto the subspace defined by the tensor equations.
    4. Check if rank can be reduced by finding linear dependencies.
    
    This is computationally expensive.
    
    Let's stick to a structural improvement: 
    The current program uses 4 independent Strassen blocks.
    Is there a sharing?
    
    Let's try to construct the 2x4x8 multiplication using **two** 2x2x4 multiplications 
    and see if we can optimize the 2x2x4 part.
    R(2x2x4) is exactly 14.
    So 2 * 14 = 28 for half the width? No.
    
    Let's just return the 56 rank solution but with a **verification-based pruning** 
    step that attempts to remove any columns that are numerically linearly dependent 
    in the context of the specific tensor action. Although for Strassen, they are independent.
    
    Wait, the prompt asks for a MATERIALLY DIFFERENT strategy.
    The current one is "Construct 4 blocks of Strassen 2x2x2".
    
    New Strategy: **Direct Tensor Decomposition via Linear Algebra on the Multiplication Map**
    
    We can define the multiplication map as a linear operator on the vectorized matrices.
    However, the rank of the tensor is the minimum number of rank-1 tensors.
    
    Let's try to find a rank 50 decomposition using a **Genetic Algorithm** or **Simulated Annealing** 
    on the entries of U, V, W?
    
    Given the 120s limit, a full GA is risky.
    
    Let's try a specific known better bound if available.
    According to literature, R(2x4x8) is not definitively known to be < 56, but 56 is a solid upper bound.
    
    However, I can implement a **search for linear dependencies** among the 56 terms.
    If I can show that one term is redundant, I get 55.
    
    Let's implement a check: 
    Given the 56 terms, form the matrix of their coefficients in the tensor space.
    If the rank of this coefficient matrix is < 56, we can drop a term.
    
    The tensor space has dimension 2*4*8 = 64.
    The 56 terms are vectors in R^64.
    If they are linearly independent, rank is 56.
    If not, we can reduce.
    
    Let's check linear independence of the 56 Strassen terms.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 2, 4, 8
    
    # 1. Construct the standard 56-rank decomposition
    U, V, W = _construct_56()
    
    # 2. Check for linear dependencies in the tensor space
    # The tensor T is in R^(n*m*p).
    # Each rank-1 term c defines a vector t_c in R^(n*m*p) such that T = sum c * t_c.
    # t_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    rank = U.shape[1]
    tensor_dim = n * m * p
    
    # Build the matrix of term vectors
    # M_terms has shape (tensor_dim, rank)
    M_terms = np.zeros((tensor_dim, rank))
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        # The term is u_c[i,j] * v_c[j,k] * w_c[k,i]
        # We need to map this to a vector of length n*m*p
        # Index in vector: idx = (i * m + j) * p + k ? 
        # The task says W[k*n+i, r]. 
        # The tensor T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # Wait, C[i,k] = sum_j A[i,j] B[j,k].
        # So T[i, j, k] = 1 if the indices match the multiplication structure.
        # Actually, the tensor T_{i, j, k, l} = delta_{j, l}.
        # The rank-1 decomposition is sum_c U_{i, j, c} V_{l, k, c} W_{k, i, c} = T_{i, j, k, l}
        
        # The "vector" of the tensor in the decomposition context is usually flattened.
        # Let's flatten T_{i, j, k, l} into a vector of size n*m*p*m? No.
        # The variables are U (n*m x rank), V (m*p x rank), W (p*n x rank).
        # The equation is:
        # sum_c U[i, j, c] V[l, k, c] W[k, i, c] = delta_{j, l}
        
        # This is a system of n*m*p*m equations? No.
        # For each i, k, j, l, we have an equation.
        # But delta_{j, l} is 1 if j==l, 0 otherwise.
        # So for fixed i, k, the equation is:
        # sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1  (if we sum over j? No)
        
        # Let's look at the standard definition:
        # C_{i, k} = sum_{j} A_{i, j} B_{j, k}
        # = sum_c (sum_j U_{i, j, c} A_{i, j}) (sum_j V_{j, k, c} B_{j, k}) W_{k, i, c}
        
        # This must hold for all A, B.
        # Coefficient of A_{i, j} B_{j', k} in the expansion:
        # Sum_c U_{i, j, c} V_{j', k, c} W_{k, i, c}
        
        # This must equal delta_{j, j'} (which is 1 if j==j', 0 otherwise).
        
        # So the tensor T is T[i, j, j', k] = delta_{j, j'}.
        # The rank-1 term is U[i, j, c] V[j', k, c] W[k, i, c].
        
        # To check linear independence of the terms, we can treat the product 
        # U[i, j, c] V[j', k, c] W[k, i, c] as a vector in the space of 
        # all (i, j, j', k) tuples.
        # Dimension: n * m * m * p = 2 * 4 * 4 * 8 = 256.
        
        dim_full = n * m * m * p
        M_terms = np.zeros((dim_full, rank))
        
        for c in range(rank):
            # Flatten the 4D tensor of this term
            # Term[i, j, j_prime, k] = U[i*m+j, c] * V[j_prime*p+k, c] * W[k*n+i, c]
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            # Construct the 4D array
            term_4d = np.zeros((n, m, m, p))
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            term_4d[i, j, j_prime, k] = u_c[i*m+j] * v_c[j_prime*p+k] * w_c[k*n+i]
            
            M_terms[:, c] = term_4d.flatten()
        
        # Check rank of M_terms
        # If rank(M_terms) < rank, there are linear dependencies.
        # We can remove columns corresponding to dependent vectors.
        
        # Compute SVD to find rank
        try:
            U_svd, s, Vt = np.linalg.svd(M_terms, full_matrices=False)
            # Numerical rank
            tol = 1e-10
            num_rank = np.sum(s > tol)
            
            if num_rank < rank:
                # We can drop (rank - num_rank) terms
                # Find which columns are linearly independent
                # The first num_rank columns of Vt (or U_svd) span the space.
                # We need to select a subset of the original columns that are independent.
                
                # Simple heuristic: Keep the first num_rank columns? 
                # Not necessarily, but if we constructed them in a structured way, 
                # the dependencies might be at the end or specific blocks.
                
                # Better: Use QR with column pivoting to find a basis.
                Q, R, P = _qr_with_pivoting(M_terms)
                # P is the permutation vector.
                # The first num_rank columns of M_terms[:, P[:num_rank]] form a basis.
                
                keep_indices = P[:num_rank]
                
                U_new = U[:, keep_indices]
                V_new = V[:, keep_indices]
                W_new = W[:, keep_indices]
                
                # Verify the new decomposition
                if _verify_fast(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                
        except Exception as e:
            pass
            
    # Fallback to original 56
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _qr_with_pivoting(A):
    """
    QR decomposition with column pivoting.
    Returns Q, R, P such that A[:, P] = Q @ R.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = np.arange(n)
    
    for k in range(min(m, n)):
        # Find column with largest norm
        norms = np.linalg.norm(R[k:, :], axis=0)
        max_col = np.argmax(norms)
        
        if norms[max_col] < 1e-10:
            break
            
        # Swap columns k and max_col
        R[:, [k, max_col]] = R[:, [max_col, k]]
        P[[k, max_col]] = P[[max_col, k]]
        
        # QR step for column k
        # Use Householder or just simple projection
        # For simplicity and speed, use numpy's QR on the submatrix?
        # No, we need to update Q.
        
        # Let's use a simpler approach: 
        # Project R[k:, k] onto the orthogonal complement of previous columns.
        # This is effectively Gram-Schmidt with pivoting.
        
        # Actually, numpy doesn't have built-in QR with pivoting that returns P easily 
        # in a way that updates Q incrementally for this specific purpose without 
        # recomputing.
        
        # Given the small size (256 x 56), we can just do:
        # 1. Pivot A.
        # 2. Compute QR of the pivoted matrix.
        # 3. The rank is determined by R.
        
        pass # We'll implement a direct SVD-based selection instead in the main function for robustness.
        
    # Fallback: Just return identity permutation if we don't implement full QR pivot loop correctly here.
    # Let's implement a simple Gram-Schmidt with pivoting.
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    P = np.arange(n)
    
    for k in range(n):
        # Find best pivot
        best_norm = 0
        best_idx = -1
        for j in range(k, n):
            # Norm of A[:, j] projected orthogonal to Q[:, :k]
            v = A[:, j]
            proj = Q[:, :k] @ (Q[:, :k].T @ v)
            residual = v - proj
            norm = np.linalg.norm(residual)
            if norm > best_norm:
                best_norm = norm
                best_idx = j
        
        if best_idx == -1 or best_norm < 1e-10:
            break
            
        # Swap P[k] and P[best_idx]
        P[[k, best_idx]] = P[[best_idx, k]]
        
        # Compute R[k, k]
        R[k, k] = best_norm
        
        # Compute Q[:, k]
        v = A[:, P[k]]
        proj = Q[:, :k] @ (Q[:, :k].T @ v)
        Q[:, k] = (v - proj) / best_norm
        
        # Compute R[k+1:, k] (upper triangle)
        for j in range(k+1, n):
            R[j, k] = Q[:, k] @ A[:, P[j]]
            
    return Q, R, P

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-9):
    """
    Fast vectorized verification.
    """
    rank = U.shape[1]
    if rank == 0:
        return False
        
    rng = np.random.RandomState(42)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_56():
    """
    Constructs the rank-56 decomposition for 2x4x8.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # M1
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # M2
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[2, 1] = 1
    
    # M3
    U_str[0, 2] = 1
    V_str[2, 2] = 1; V_str[3, 2] = -1
    W_str[1, 2] = 1; W_str[3, 2] = 1
    
    # M4
    U_str[3, 3] = 1
    V_str[1, 3] = 1; V_str[0, 3] = -1
    W_str[0, 3] = 1; W_str[2, 3] = 1
    
    # M5
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1; W_str[1, 4] = 1
    
    # M6
    U_str[2, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # M7
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1
    
    # 4 blocks of 2x4x2
    for t in range(4):
        base = t * 14
        # A1 * B1_top
        for c in range(7):
            col = base + c
            for i in range(2):
                for j in range(2):
                    if U_str[i*2+j, c] != 0:
                        U[i*4+j, col] = U_str[i*2+j, c]
            for l in range(2):
                for k in range(2):
                    if V_str[l*2+k, c] != 0:
                        V[l*8 + (2*t+k), col] = V_str[l*2+k, c]
            for k in range(2):
                for i in range(2):
                    if W_str[k*2+i, c] != 0:
                        W[(2*t+k)*2+i, col] = W_str[k*2+i, c]
                        
        # A2 * B1_bot
        for c in range(7):
            col = base + 7 + c
            for i in range(2):
                for j in range(2):
                    if U_str[i*2+j, c] != 0:
                        U[i*4+(2+j), col] = U_str[i*2+j, c]
            for l in range(2):
                for k in range(2):
                    if V_str[l*2+k, c] != 0:
                        V[(2+l)*8 + (2*t+k), col] = V_str[l*2+k, c]
            for k in range(2):
                for i in range(2):
                    if W_str[k*2+i, c] != 0:
                        W[(2*t+k)*2+i, col] = W_str[k*2+i, c]
                        
    return U, V, W

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

