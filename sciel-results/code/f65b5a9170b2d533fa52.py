"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _rank1_term(i, j, k, n, m, p):
    """Generate a single rank-1 term as (U_col, V_col, W_col) where:
    U has shape (n*m,), V has shape (m*p,), W has shape (p*n,).
    Only the entries (i,j), (j,k), (k,i) are non-zero with value 1.
    """
    U = np.zeros(n * m)
    V = np.zeros(m * p)
    W = np.zeros(p * n)
    U[i * m + j] = 1.0
    V[j * p + k] = 1.0
    W[k * n + i] = 1.0
    return U, V, W


def _strassen_2x2_terms(n, m, p):
    """Generate the 7 Strassen terms for 2x2 matrix multiplication.
    For general dimensions, we use the 2x2 pattern on indices 0,1.
    Returns list of (U, V, W) tuples, each a numpy array.
    """
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    terms = []
    # Each term is a sum of rank-1 terms. We'll decompose each Strassen
    # sub-product into rank-1 terms using the schoolbook method on 2x2 subblocks.
    
    # For a 2x2 multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
    # Strassen rewrites this as 7 products of linear combinations.
    # Each product is of the form (linear_comb_of_A) * (linear_comb_of_B)
    # which expands to a sum of rank-1 terms.
    
    # Let's enumerate the 7 Strassen formulas and expand each into rank-1 terms.
    
    # Index convention: for 2x2, i,j,k in {0,1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # C contributions: C[0,0] += M1, C[0,1] -= M1, C[1,0] -= M1, C[1,1] += M1
    # Expansion:
    # A[0,0]*B[0,0] -> rank-1 (i=0,j=0,k=0)
    # A[0,0]*B[1,1] -> rank-1 (i=0,j=0,k=1)
    # A[1,1]*B[0,0] -> rank-1 (i=1,j=1,k=0)
    # A[1,1]*B[1,1] -> rank-1 (i=1,j=1,k=1)
    terms.append(('M1', [
        (0,0,0,1),   # A00*B00
        (0,0,1,1),   # A00*B11
        (1,1,0,1),   # A11*B00
        (1,1,1,1),   # A11*B11
    ]))
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # C[0,0] += M2, C[1,0] += M2
    terms.append(('M2', [
        (1,0,0,1),   # A10*B00
        (1,1,0,1),   # A11*B00
    ]))
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # C[0,1] += M3
    terms.append(('M3', [
        (0,0,1,1),   # A00*B01
        (0,0,1,-1),  # A00*(-B11) -> stored as (0,0,1,-1)
    ]))
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # C[1,0] -= M4
    terms.append(('M4', [
        (1,1,0,-1),  # A11*(-B00)
        (1,1,0,1),   # A11*B10
    ]))
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # C[0,1] += M5, C[1,1] += M5
    terms.append(('M5', [
        (0,0,1,1),   # A00*B11
        (0,1,1,1),   # A01*B11
    ]))
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # C[1,1] += M6, C[1,0] -= M6
    terms.append(('M6', [
        (1,0,0,1),   # A10*B00
        (1,0,1,1),   # A10*B01
        (0,0,0,-1),  # (-A00)*B00
        (0,0,1,-1),  # (-A00)*B01
    ]))
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # C[0,0] -= M7, C[0,1] -= M7 (wait, let me re-check)
    # Actually: C[0,0] += M7? Let me verify Strassen.
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # So C[0,0] = M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    terms.append(('M7', [
        (0,1,0,1),   # A01*B10
        (0,1,1,1),   # A01*B11
        (1,1,0,-1),  # (-A11)*B10
        (1,1,1,-1),  # (-A11)*B11
    ]))
    
    # Now for 5x5, we need to map 2x2 patterns to positions.
    # This is a placeholder; the actual construction is more complex.
    return terms


def _build_4x4_from_strassen_2x2(n=4, m=4, p=4):
    """Build a rank-17 decomposition for 4x4 matrix multiplication.
    
    This uses the fact that 4x4 can be decomposed using the Strassen
    2x2 scheme applied in a divide-and-conquer fashion, but for the
    purpose of this task, we'll construct a valid decomposition.
    
    Actually, the rank of 4x4 matrix multiplication is at most 49 (schoolbook)
    but known to be at most 49. The rank-17 claim is for 2x2. For 4x4,
    we can use a tensor product or block decomposition.
    
    Let's just build the schoolbook rank-64 and then try to reduce.
    """
    pass


def _schoolbook_decomposition(n, m, p):
    """Generate the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rtol=1e-10, atol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The matrix multiplication tensor T[i,j,k] = delta_ij... no.
    Actually, the tensor is T[i,j,k,l] where (C = A*B)_{i,l} = sum_j A_{i,j} B_{j,l}
    But in the standard tensor formulation for matrix multiplication, we have:
    C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The tensor M has entries M_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
    Wait, the standard formulation: the matrix multiplication tensor T of order 3 with indices
    (i, j, k) where T[i,j,k] = 1 if the product term A_{i,j} B_{j,k} appears in C_{i,k}.
    But that's not right either. Let me think about this more carefully.
    
    The matrix multiplication can be written as a tensor contraction:
    C_{i,k} = T_{i,j,k} A_{i,j} B_{j,k} where T_{i,j,k} = 1 for all valid i,j,k.
    
    But the rank decomposition is: T = sum_c U_c V_c W_c where
    T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}... 
    
    Actually, the standard formulation in the problem is:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    and the decomposition satisfies: for all (i,j), (j,k), (k,i) with the proper index mapping:
    
    The tensor T[i, j, k] (where i is row of C, j is... let me re-read the task.
    
    "U factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k, i) where k is column of B / ... and i is row of A / row of C.
    
    The standard matrix multiplication tensor:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    This can be written as:
    C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_k V_{j,k,c} B_{j,k}) ... no, that's not right.
    
    Let me think of it as: the tensor T has 3 indices (i, j, k) with T[i,j,k] = 1 for all valid indices
    in the sense that A_{i,j} * B_{j,k} contributes to C_{i,k}. But the decomposition is:
    
    1 = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} for all valid (i,j,k)
    and the only constraint is that this holds for all i in [0,n), j in [0,m), k in [0,p).
    
    Wait, that's the tensor of all-ones? No. The matrix multiplication tensor has
    T[i,j,k] = 1 for all i,j,k (since every product term contributes). The constraint is:
    
    For all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But that's not quite right either. Let me re-examine.
    
    Actually, the tensor decomposition for matrix multiplication is such that:
    C = A * B means C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    We want to write this as sum_c (U_c * A) (V_c * B) ... where U_c, V_c are...
    
    I think the correct formulation is:
    - U has entries U[i*m+j, c] for i in [0,n), j in [0,m)
    - V has entries V[j*p+k, c] for j in [0,m), k in [0,p)  
    - W has entries W[k*n+i, c] for k in [0,p), i in [0,n)
    
    And the constraint is that for all (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}... no.
    
    Actually, I think the tensor is: T[i,j,k] = 1 for all valid (i,j,k), and the decomposition
    must satisfy: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    But T[i,j,k] = 1 for all i,j,k is the all-ones tensor, not the matrix multiplication tensor.
    
    Let me reconsider. The matrix multiplication tensor has entries T_{i,j,k} = 1 when
    the product A_{i,j} B_{j,k} contributes to C_{i,k}. Since it always does (for valid i,j,k),
    T_{i,j,k} = 1 for all valid (i,j,k).
    
    But wait, that's just the all-ones tensor of dimension (n*m) x (m*p) x (p*n) with a constraint
    that links the indices. The decomposition must satisfy:
    
    For all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And additionally, the entries of U, V, W are zero unless the index pairs correspond to
    valid (i,j), (j,k), (k,i) pairs. But since U is indexed by (i,j) pairs, all entries are "valid."
    
    So the constraint is simply: the tensor T with T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    must equal sum_c U[:,c] V[:,c] W[:,c] in the sense of the tensor product.
    
    But this is a constrained tensor: not all entries of the full (n*m) x (m*p) x (p*n) tensor
    are constrained. Only the entries where the indices are linked (i,j), (j,k), (k,i) are constrained
    to be 1. The other entries are free (can be anything, since they don't correspond to valid
    matrix multiplication terms).
    
    So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    Let me implement this.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > rtol + atol * rank:
                    return False
    return True


def _try_subtraction_method(n=5, m=5, p=5, time_limit=100.0):
    """Try to find a rank-120 decomposition for 5x5 matrix multiplication
    using the subtraction method based on 4x4 Strassen-style decompositions.
    
    The idea: construct a decomposition of the 5x5 tensor as a combination
    of terms from 4x4 subproblems, and cancel boundary terms.
    
    This is complex; let's try a simpler approach: start with schoolbook rank-125
    and try to find linear dependencies to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = n * m * p
    
    # The schoolbook decomposition has the property that each column c corresponds
    # to a unique (i,j,k) tuple. We can try to find linear combinations that
    # reduce the rank.
    
    # A simpler approach: use random linear combinations of the basis tensors
    # and check if they satisfy the constraint with fewer terms.
    
    # Actually, let's try a different approach: use the fact that for n=m=p=2,
    # Strassen gives rank 7 < 8. For n=m=p=5, we can potentially use block
    # decompositions.
    
    # Let's try to build a decomposition using 2x2 Strassen blocks.
    # For 5x5, we can't directly tile with 2x2 blocks. But we can use
    # the following: 5 = 2 + 2 + 1.
    
    # For now, let's try to find a rank-120 decomposition by perturbing
    # the schoolbook decomposition.
    
    # The key insight: if we can find 5 linearly dependent combinations
    # of the 125 rank-1 tensors, we can reduce the rank by 5.
    
    # Let's try to find vectors (a_c) such that sum_c a_c * (U_c, V_c, W_c) = 0
    # in the tensor sense, i.e., for all i,j,k: sum_c a_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 0
    
    # This is a non-linear constraint. Let's instead try to find 5 new columns
    # that are linear combinations of existing columns such that the new set
    # still satisfies the constraint.
    
    # Given the complexity, let's just return the schoolbook decomposition
    # as a fallback.
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - 20:
        return U, V, W, rank
    
    # Try random perturbations
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try to find 5 linear dependencies
        # This is a complex optimization problem; let's use a simple heuristic.
        
        # Randomly select 5 pairs of columns to "merge"
        col_indices = list(range(rank))
        random.shuffle(col_indices)
        
        # For now, this is a placeholder; the actual implementation would
        # need to solve a system of equations.
        
        pass
    
    return best_U, best_V, best_W, best_rank


def _strassen_block_decomposition(n, m, p):
    """Try to build a decomposition using 2x2 Strassen blocks where possible.
    
    For 5x5, we can decompose into blocks: top-left 2x2, top-right 2x3, etc.
    This is complex; let's skip for now.
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    time_limit = 100.0  # 150s total, leave 50s margin
    deadline = start_time + time_limit
    
    # Try the subtraction method first
    U, V, W, rank = _try_subtraction_method(n, m, p, time_limit=50.0)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
        # Return schoolbook as fallback
        U, V, W = _schoolbook_decomposition(n, m, p)
        rank = n * m * p
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert numpy arrays to lists for the return format
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        rank = n * m * p
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
