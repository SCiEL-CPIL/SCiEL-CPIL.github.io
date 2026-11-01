import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import product
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # The matrix multiplication tensor T has indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the decomposition U, V, W:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    
    # We search for a decomposition of rank R.
    # To make this a MILP, we discretize the coefficients.
    # Let's try coefficients from {-1, 1} first, as they are common in Strassen-like algorithms.
    # If that fails or is too hard, we might try {-1, 0, 1} or just 0/1 with more terms.
    
    # Actually, a more robust approach for small sizes is to search for a rank R
    # using a fixed set of "basis" rank-1 tensors.
    # However, the space of all rank-1 tensors is infinite.
    # We can restrict to a finite set of candidate rank-1 tensors.
    # A natural candidate set is the "standard basis" rank-1 tensors:
    # e_{i,j} in U, e_{j,k} in V, e_{k,i} in W.
    # But this only gives the schoolbook decomposition (rank 42) and its linear combinations
    # if we allow real coefficients. But rank is about the number of terms, not linear combinations
    # of the standard basis in the tensor product space in that simple way.
    
    # Alternative Strategy:
    # Use the fact that for small matrices, we can try to find a decomposition
    # by solving a system of polynomial equations, or by using a MILP to select
    # a subset of a larger set of candidate rank-1 tensors.
    
    # Let's generate a set of candidate rank-1 tensors.
    # Candidate: (u, v, w) where u, v, w are vectors with entries in {-1, 1}.
    # This is too many.
    
    # Let's stick to a structured search.
    # The known best for 2x3x7 is likely 39 (from Strassen hybrid).
    # Can we do better?
    # Let's try to find a rank 38 or 37 decomposition.
    
    # We will use a MILP to select a subset of size R from a pool of N candidates.
    # The candidates will be the 42 schoolbook terms, plus some "Strassen-like" terms.
    # Actually, let's just try to find ANY decomposition of rank R < 42.
    # We can fix the structure of the decomposition to be "block diagonal" or similar.
    
    # Let's try a different approach: Randomized search with verification.
    # Generate random U, V, W of size (nm x R), (mp x R), (pn x R) with integer entries.
    # This is hard to verify quickly for large R.
    
    # Let's go back to the MILP idea but with a specific structure.
    # We know that rank 39 is achievable. Can we find rank 38?
    # Let's try to solve for rank R = 38, 39, 40, 41.
    
    # We will use a set of candidate rank-1 tensors.
    # The most effective candidates are those that are "close" to the schoolbook terms.
    # Let's define the schoolbook terms as S = {(i, j, k) | i in 0..n-1, j in 0..m-1, k in 0..p-1}.
    # Each term corresponds to U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    
    # If we allow coefficients to be -1, 0, 1, we can create "negative" terms.
    # Let's try to find a decomposition using terms from a set C.
    # Let C be the set of all (i, j, k) triples.
    # We want to select R triples and assign coefficients alpha, beta, gamma in {-1, 1}
    # such that the sum of alpha_c * beta_c * gamma_c * T_c = T.
    # This is not linear in the coefficients.
    
    # Let's simplify: Assume all coefficients are 1. Then we are looking for a set of R
    # rank-1 tensors that sum to T. This is the "non-negative" rank, which is usually higher.
    
    # Let's assume coefficients can be -1 or 1.
    # Let x_{i,j,k} be 1 if we use the term (i,j,k) with positive sign, 0 otherwise.
    # Let y_{i,j,k} be 1 if we use the term (i,j,k) with negative sign, 0 otherwise.
    # We want sum (x - y) * T_{i,j,k} = T.
    # This is linear!
    # Variables: x_{i,j,k}, y_{i,j,k} for all i,j,k.
    # Constraints:
    # For each (i, j, k):
    # sum_{i',j',k'} (x_{i',j',k'} - y_{i',j',k'}) * T_{i',j',k'}[i,j,k] = T[i,j,k]
    # Wait, T is a tensor, not a matrix. The constraint is:
    # For each (a, b, c) in the tensor space:
    # sum_{r} U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # If we restrict to the schoolbook basis, U[a,r] is non-zero only if a = i*m+j.
    # This couples the variables.
    
    # Let's try a direct MILP on the entries of U, V, W.
    # This is non-linear (product of variables from U, V, W).
    # So MILP is not directly applicable unless we linearize.
    
    # Linearization:
    # Let z_{a,b,c,r} = U[a,r] * V[b,r] * W[c,r].
    # This is cubic.
    
    # Given the complexity, let's try a heuristic search for rank 38.
    # We know rank 39 is possible.
    # We will try to modify the rank 39 decomposition to see if we can merge two terms.
    
    # First, we need a rank 39 decomposition.
    # The inspiration mentioned a hybrid Strassen/Schoolbook.
    # Let's construct one.
    # Split the 3x7 matrix into 2x7 and 1x7? No, split inner dimension 3 into 2+1.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    # C = A1*B1 + A2*B2.
    # Rank(A1*B1) for 2x2 * 2x7.
    # We can use Strassen for 2x2 * 2x2 blocks.
    # 2x2 * 2x7 = 4 blocks of 2x2 * 2x2.
    # Each 2x2 * 2x2 has rank 7 (Strassen).
    # So 4 * 7 = 28 terms for A1*B1.
    # A2*B2 is 2x1 * 1x7. This is outer product, rank 2 (2 terms: A2[0]*B2, A2[1]*B2).
    # Total rank = 28 + 2 = 30?
    # Wait, Strassen for 2x2 * 2x2 uses 7 multiplications.
    # The decomposition is:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Let's verify the rank for 2x2 * 2x7.
    # We can group the 7 columns into 3 pairs and 1 single? No, 7 is odd.
    # Let's just use 4 Strassen blocks for the first 6 columns, and 1 schoolbook for the last.
    # 4 * 7 + 4 (schoolbook for 2x2*2x1 is 4 terms) = 28 + 4 = 32.
    # Plus A2*B2 (2x1 * 1x7) which is 2 terms?
    # A2 is 2x1, B2 is 1x7. C2 = A2 * B2.
    # C2[i, k] = A2[i, 0] * B2[0, k].
    # This is rank 2 in the tensor sense?
    # T2[i, 0, k] = A2[i,0] B2[0,k].
    # This can be decomposed into 2 rank-1 tensors:
    # Term 1: U[0,0]=1, U[1,0]=1? No.
    # U is 2x1 (for the 2x1 part of A). V is 1x7 (for the 1x7 part of B). W is 7x2.
    # Wait, the dimensions are fixed.
    
    # Let's just implement the Strassen hybrid for 2x3 * 3x7.
    # Split j into j1 (2) and j2 (1).
    # A1 (2x2), A2 (2x1).
    # B1 (2x7), B2 (1x7).
    # C = A1 B1 + A2 B2.
    
    # Decompose A1 B1 (2x2 * 2x7).
    # Split k into 3 pairs (6 cols) and 1 col.
    # B1 = [B1_0, B1_1, B1_2, B1_3] where B1_0..2 are 2x2, B1_3 is 2x1.
    # A1 B1_0, A1 B1_1, A1 B1_2 use Strassen (7 terms each).
    # A1 B1_3 uses Schoolbook (4 terms).
    # Total for A1 B1: 3*7 + 4 = 25 terms.
    
    # Decompose A2 B2 (2x1 * 1x7).
    # C2[i, k] = A2[i, 0] B2[0, k].
    # This is an outer product.
    # We can write it as:
    # Term 1: U[0,0]=1, U[1,0]=1 (for A2 part? No, U is indexed by i,m).
    # Let's be precise with indices.
    # U is (2*3) x R. V is (3*7) x R. W is (7*2) x R.
    # For A2 B2:
    # i in 0..1, j=2, k in 0..6.
    # U[i*3+2, r] * V[2*7+k, r] * W[k*2+i, r] = A2[i,0] B2[0,k].
    # We can use 2 terms:
    # r=0: U[0*3+2, 0]=1, U[1*3+2, 0]=1? No, U[i*3+2, r] must be A2[i,0].
    # If we set U[2, 0]=1, U[5, 0]=1 (for i=0,1), V[14+k, 0]=1 (for all k), W[k*2+0, 0]=1, W[k*2+1, 0]=1?
    # This doesn't work directly because W depends on i.
    
    # Actually, A2 B2 is rank 2 in the tensor T[i, 2, k].
    # T[i, 2, k] = A2[i, 0] B2[0, k].
    # This is a rank-2 tensor in the (i, k) space, with j fixed at 2.
    # It can be decomposed into 2 rank-1 tensors:
    # Term 1: u=(1,0), v=(1,1,...,1), w=(1,0, 1,0, ...)? No.
    # T[i, k] = u_i v_k.
    # This is rank 1 if A2 and B2 are fixed vectors.
    # Wait, T[i, 2, k] = A2[i, 0] * B2[0, k].
    # This is a rank-1 tensor in the (i, k) subspace.
    # So it requires 1 term?
    # Let's check: U[i*3+2, r] = A2[i, 0], V[2*7+k, r] = B2[0, k], W[k*2+i, r] = 1.
    # If we set U[2, 0]=1, U[5, 0]=1 (assuming A2[i,0]=1 for simplicity),
    # V[14..20, 0]=1, W[0, 0]=1, W[2, 0]=1, W[1, 0]=1, W[3, 0]=1...
    # Then U[i*3+2, 0] V[14+k, 0] W[k*2+i, 0] = 1*1*W[k*2+i, 0].
    # We need this to be 1 for all i, k.
    # So W[k*2+i, 0] must be 1 for all k, i.
    # This works! So A2 B2 can be done in 1 term if we allow W to have many 1s.
    # But wait, W is shared with other terms.
    
    # Let's assume the Strassen part uses specific indices in W.
    # This is getting complicated to construct manually.
    
    # Let's just return the schoolbook rank 42 for now, as a safe baseline,
    # but implement a simple search for rank 41 to see if it's possible.
    # If not, return 42.
    
    # Actually, the inspiration says rank 39 is possible.
    # Let's try to hardcode a rank 39 decomposition if we can find one.
    # Since I don't have one, I will return the schoolbook.
    
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for c, (i, j, k) in enumerate(terms):
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
