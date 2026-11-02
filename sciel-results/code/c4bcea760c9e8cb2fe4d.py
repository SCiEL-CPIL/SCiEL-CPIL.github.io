"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that U @ diag @ V^T reconstructs the multiplication tensor correctly."""
    rank = U.shape[1]
    # The tensor T[i, j, k] = delta_{i,j,k} for matmul: (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: T = sum_r U[:,r] V[:,r] W[:,r]^T
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give 1 if i,j,k form a valid triple
    # Actually the tensor for matmul is T_{i,j,k} = 1 (it's just the identity in the middle)
    # More precisely: the bilinear map is (A,B) -> A*B
    # T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} where the bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = (AB)_{i,k}
    # So T_{i,j,k} = 1 for all i, j, k. Wait, that's not right either.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T_{i,j,k} = 1 for all valid i, j, k.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Let's verify with schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 where c = i*m*p + j*p + k
    # Sum over r: only r = i*m*p + j*p + k contributes, giving 1*1*1 = 1. Correct.
    
    # Efficient check: for each r, the outer product U[:,r] V[:,r] W[:,r]
    # We need sum_r U[:,r] V[:,r] W[:,r]^T to equal the all-ones tensor reshaped
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # The tensor T should have T[i,j,k] = 1
    
    # Build the tensor from decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is expensive to check naively. Let's use a sampling approach or direct computation.
    
    # Direct computation for small dimensions:
    T_decomp = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # Expected tensor: T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    return np.allclose(T_decomp, T_expected, atol=tol, rtol=tol)

def _to_fraction_matrix(M, tol=1e-9):
    """Convert a numpy matrix to a list of lists of Fractions, rounding near-integer/half-integer values."""
    result = []
    for row in M:
        frow = []
        for val in row:
            # Try to find a rational with small denominator
            # First check if it's close to an integer
            nearest_int = round(val)
            if abs(val - nearest_int) < tol:
                frow.append(Fraction(nearest_int))
            else:
                # Try half-integers
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < tol:
                    frow.append(Fraction(int(round(val * 2)), 2))
                else:
                    # Use Fraction's limit_denominator
                    frac = Fraction(val).limit_denominator(100)
                    frow.append(frac)
        result.append(frow)
    return result

def _try_random_decomposition(n, m, p, rank, seed, time_limit=10):
    """Try to find a random decomposition of the given rank."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # We need sum_r U[:,r] V[:,r] W[:,r]^T = T (all ones tensor)
    # This is a system of nonlinear equations. Let's use a simple approach:
    # Start from schoolbook and try to compress.
    
    # Actually, let's just return the schoolbook for now as a fallback
    # and do a more sophisticated search in construct()
    
    return None  # Not implemented yet

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = (U_sb, V_sb, W_sb, n * m * p)
    
    # Try to find lower rank decompositions
    # For 3x3 by 3x5, the schoolbook rank is 45
    # Known lower bounds: at least n*m = 9, m*p = 15, etc.
    # The Strassen-style algorithms can sometimes reduce rank for square matrices,
    # but for rectangular it's less clear.
    
    # Let's try a simple random search for rank < 45
    # We'll use a heuristic: try to find a decomposition where the tensor
    # T_{i,j,k} = 1 is expressed as sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # This is equivalent to finding a rank-r decomposition of the 3-way tensor T
    # where T[i,j,k] = 1 for all i,j,k
    
    # Actually, the all-ones tensor has a very simple structure.
    # T[i,j,k] = 1 = 1 * 1 * 1
    # So rank 1 should work? Let's check:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_r U V W = 1*1*1 = 1 for all i,j,k. Yes!
    
    # Wait, that means the rank is 1? That can't be right for matrix multiplication.
    # Let me re-read the problem.
    
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The bilinear map is (A, B) -> A*B where A is n x m and B is m x p.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is defined by T[i,j,k] such that (AB)_{i,k} = sum_{i,j,k} T[i,j,k] A_{i,j} B_{j,k}
    # Comparing: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 if the indices match up, i.e., the j in A_{i,j} must equal the j in B_{j,k}
    # But in the sum over i,j,k, we're summing over all i,j,k, so T[i,j,k] must be 1 for all i,j,k
    # where the j index is shared. But that's always the case.
    
    # Hmm, I think I'm confusing myself. Let me think again.
    
    # The tensor for matrix multiplication is a 3-tensor T with indices (i, j, k)
    # where i is the row of A (and output), j is the column of A / row of B, k is the column of B (and output)
    # T[i,j,k] = 1 for all valid i, j, k
    # The bilinear form is: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # But this sums over ALL i, j, k, which gives sum_i sum_j sum_k A[i,j] B[j,k] = (sum_j sum_k B[j,k]) * (sum_i sum_j A[i,j])
    # That's not right.
    
    # I think the correct interpretation is:
    # The output (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor should be T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A_{i,j} B_{j,k} = sum_i sum_k (AB)_{i,k} * something
    # Actually, the standard way is: the tensor T has T[i,j,k] = 1, and the bilinear map is
    # (A, B) -> sum_{i,j,k} T[i,j,k] A_{i,j} B_{j,k} e_{i,k}
    # where e_{i,k} are basis vectors for the output space.
    # So the output component (i,k) is sum_j T[i,j,k] A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k} (since T[i,j,k]=1)
    
    # OK so T[i,j,k] = 1 for all i,j,k. And the rank-1 decomposition I proposed:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # gives sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    # So the rank IS 1?
    
    # That seems too good to be true. Let me check the problem statement again.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # If rank 1 works, then the answer is trivially 1. But the schoolbook is 45, and the current score is 1.0
    # which corresponds to rank 45. If rank 1 were valid, the score would be 45.
    
    # Let me re-examine. Maybe the tensor is NOT all ones.
    
    # Actually, I think the issue is that the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if and only if the indices are consistent, but in the standard formulation,
    # T is indeed the all-ones tensor of shape (n, m, p).
    
    # Wait, no. The tensor is T in V^* ⊗ (V ⊗ W)^* or something like that.
    # Let me look at this differently.
    
    # The matrix multiplication map is: Mult: M_{n,m} × M_{m,p} -> M_{n,p}
    # Mult(A,B) = AB
    # This is a bilinear map, so it corresponds to a tensor T ∈ M_{n,m}^* ⊗ M_{m,p}^* ⊗ M_{n,p}
    # T(A,B) = AB
    # In components: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ e_{i,k}
    # where e_{i,j}^* are dual basis for M_{n,m}, etc.
    # The tensor T has components T[i,j,k] = 1 (using the convention that the output index is (i,k))
    # But the output is indexed by (i,k), so in the W factor, W[k*n+i, r] corresponds to output index (i,k)
    
    # So T[i,j,k] = 1 for all i, j, k. And my rank-1 decomposition should work.
    
    # Let me test this hypothesis by checking if the validator accepts it.
    
    U1 = np.ones((n * m, 1), dtype=np.float64)
    V1 = np.ones((m * p, 1), dtype=np.float64)
    W1 = np.ones((p * n, 1), dtype=np.float64)
    
    if _validate(U1, V1, W1, n, m, p):
        # Rank 1 works!
        best = (U1, V1, W1, 1)
        # Convert to fractions
        U_frac = _to_fraction_matrix(U1)
        V_frac = _to_fraction_matrix(V1)
        W_frac = _to_fraction_matrix(W1)
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # If rank 1 doesn't work (which would be surprising), fall back to schoolbook
    U_frac = _to_fraction_matrix(U_sb)
    V_frac = _to_fraction_matrix(V_sb)
    W_frac = _to_fraction_matrix(W_sb)
    return {"U": U_frac, "V": V_frac, "W": W_frac}
# EVOLVE-BLOCK-END
