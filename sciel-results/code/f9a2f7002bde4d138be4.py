import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # The tensor T is defined by T[i, j, k] = 1 if i == k else 0.
    # We are looking for U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Strategy:
    # The tensor is a sum of n*p "identity" slices in the (i,k) plane, each of size m.
    # Actually, T[i, j, k] = delta_{ik} * 1.
    # This can be viewed as: For each (i,k) pair, we need to represent the vector e_j (all 1s in j dim? No.)
    # Let's look at the structure again.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # T_{ijk} = 1 if i==k, 0 otherwise.
    
    # We can decompose this into independent contributions for each (i,k) pair.
    # However, the rank is global. We can use the same rank index r for multiple (i,k) pairs if the
    # U, V, W factors allow it.
    
    # A better perspective:
    # T = sum_{i=0}^{n-1} E_{i,i} \otimes J_m
    # where E_{i,i} is the matrix unit in the (n x p) space (since i is row of A, k is col of B, and i==k implies i==k in C).
    # Wait, C is n x p. A is n x m. B is m x p.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T_{ijk} maps A_{ij}, B_{jk} to C_{ik}.
    # T_{ijk} = 1 if i==k, else 0.
    
    # This tensor is a sum of n terms? No, i ranges 0..n-1, k ranges 0..p-1.
    # The non-zero entries are only when i==k. Since n=3, p=5, the indices i,k go up to 2 and 4.
    # So non-zero only for (i,k) in {(0,0), (1,1), (2,2)}.
    # For each such (i,k), the slice T_{i,:,k} is a vector of length m=5, all entries 1.
    # So T = sum_{i=0}^{2} (E_{ii}^{(n,p)} \otimes \mathbf{1}_m)
    # where E_{ii} is the matrix with 1 at (i,i) and 0 elsewhere, and \mathbf{1}_m is the all-ones vector.
    
    # We want to decompose T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r is in C^{n*m}, v_r in C^{m*p}, w_r in C^{p*n}.
    
    # Let's try to construct a low-rank decomposition.
    # We know the trivial rank is n*m*p = 75.
    # We can do better by sharing factors.
    
    # Consider the vector \mathbf{1}_m. It has rank 1 in the "slice" sense?
    # The tensor slice for a fixed (i,k) is constant in j.
    # This suggests that V_{j, r} might be constant in j for a given r, or U_{i,j, r} might be constant in j.
    
    # Let's try a decomposition based on the fact that the "j" dimension is fully symmetric for the non-zero slices.
    # If we set V_{j, r} = 1 for all j, then the contribution to T_{ijk} from term r is U_{i,j,r} * 1 * W_{k,i,r}.
    # Summing over j: sum_j U_{i,j,r} W_{k,i,r} ... wait, the sum is over r, not j.
    # T_{ijk} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    
    # If we choose V_{j*p+k, r} = 1 for all j, then:
    # T_{ijk} = sum_r U_{i*m+j, r} W_{k*n+i, r}.
    # This must equal 1 if i==k, 0 otherwise.
    # This looks like a matrix multiplication where the "j" index is summed? No, j is fixed in T_{ijk}.
    # The equation must hold for ALL j.
    # So, for a fixed i, k, the vector (over j) of LHS must be the constant vector (1,1,1,1,1) if i==k, else 0.
    # Let A_r = vector of U_{:, r} for fixed i (length m).
    # Let B_r = scalar W_{k*n+i, r}.
    # Then LHS_j = sum_r A_{r,j} B_r.
    # We need sum_r A_{r,j} B_r = C_{ik} for all j, where C_{ik} is 1 if i==k else 0.
    # This implies that the vector sum_r A_r B_r is a constant vector.
    
    # This structure suggests we can decompose the "identity" part in the (i,k) space using rank-R matrices,
    # and handle the j-dimension by ensuring the U and V factors produce a constant sum.
    
    # Let's try to find the minimal rank for the (i,k) part first.
    # The (i,k) tensor is I_3 (3x3 identity) embedded in 3x5.
    # Rank of identity matrix I_n is n.
    # So we might need at least 3 "independent" directions in the (i,k) space.
    
    # If we use R=3, can we do it?
    # Let r=0,1,2.
    # We need W_{k*n+i, r} to form the identity.
    # Let W_{k*n+i, r} = delta_{ik} delta_{ir}? No, W depends on k, i, r.
    # Let's set W_{k*n+i, r} = 1 if i==k==r, else 0.
    # Then for a fixed i, k:
    # If i!=k, W is 0 for all r, so sum is 0. Correct.
    # If i==k, W_{k*n+i, r} = 1 if r==i, else 0.
    # So only term r=i contributes.
    # T_{ijk} = U_{i*m+j, i} * V_{j*p+i, i} * 1.
    # We need this to be 1 for all j.
    # So U_{i*m+j, i} * V_{j*p+i, i} = 1 for all j.
    # We can set U_{i*m+j, i} = 1 and V_{j*p+i, i} = 1.
    # What about other entries of U and V?
    # U_{i'*m+j', r} for r != i' should be 0 to avoid interference?
    # If r != i', W_{k*n+i', r} is 0 unless i'=k=r.
    # If we set U_{idx, r} = 0 unless the i-index of idx matches r, and V_{idx, r} = 0 unless the k-index of idx matches r?
    # Let's check.
    # U_{i*m+j, r} is non-zero only if i=r.
    # V_{j*p+k, r} is non-zero only if k=r.
    # W_{k*n+i, r} is non-zero only if i=k=r.
    
    # If i=k=r:
    # Term r contributes: U_{r*m+j, r} * V_{j*p+r, r} * W_{r*n+r, r}.
    # We set these to 1.
    # Terms r' != r:
    # If r' != r, then for term r' to contribute to T_{r,j,r}, we need W_{r*n+r, r'} != 0.
    # But W_{k*n+i, r'} is non-zero only if i=k=r'.
    # Here i=r, k=r. So i=k=r != r'. Thus W_{r*n+r, r'} = 0.
    # So no interference.
    
    # This gives a valid decomposition of rank R=3?
    # Let's verify the shapes.
    # U is (n*m, R) = (15, 3).
    # V is (m*p, R) = (25, 3).
    # W is (p*n, R) = (15, 3).
    
    # Construction:
    # For r in 0..2:
    #   U[i*m+j, r] = 1 if i==r else 0.
    #   V[j*p+k, r] = 1 if k==r else 0.
    #   W[k*n+i, r] = 1 if i==k==r else 0.
    
    # Let's check T_{ijk}:
    # sum_{r=0}^2 U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Term r is non-zero only if:
    # 1. i==r (from U)
    # 2. k==r (from V)
    # 3. i==k==r (from W)
    # So term r is non-zero only if i=k=r.
    # If i=k=r, value is 1*1*1 = 1.
    # If i!=k, no term is non-zero (since i==r and k==r implies i==k).
    # If i=k but i!=r, then for term i, U and V are 1, but W_{i*n+i, i}=1?
    # Wait. If i=k=0, we look at term r=0.
    # U_{0*m+j, 0} = 1 (since i=0=r).
    # V_{j*p+0, 0} = 1 (since k=0=r).
    # W_{0*n+0, 0} = 1 (since i=0,k=0,r=0).
    # Sum = 1.
    # What about term r=1?
    # U_{0*m+j, 1} = 0 (since i=0 != 1).
    # So sum is 1. Correct.
    
    # If i=0, k=1.
    # Term r=0: U=1, V=0 (k=1!=0), W=0 (i!=k). Product 0.
    # Term r=1: U=0 (i=0!=1), V=1, W=0. Product 0.
    # Term r=2: U=0, V=0, W=0. Product 0.
    # Sum 0. Correct.
    
    # So Rank 3 is a valid exact decomposition!
    # The score is 1/3 = 0.333...? No, score is seed/value. Seed is 75.
    # Value is 3. Score = 75/3 = 25.
    # Wait, the prompt says "score is the rank". "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If raw value is rank, then lower rank is better.
    # Seed program returns rank 75. Score 1.0.
    # If I return rank 3, score should be 75/3 = 25.0.
    
    # Let's double check if there are any constraints I missed.
    # "integer/half-integer entries". 1 is integer.
    # "exact tensor decomposition". Verified.
    # "output factor follows official trace convention W[k*n+i,r]".
    # My W index: W[k*n+i, r]. Yes.
    
    # Is it possible to go lower than 3?
    # The tensor has non-zero slices for (0,0), (1,1), (2,2).
    # These are linearly independent in the (i,k) space?
    # The rank of the tensor is at least the rank of any of its "slices" or contractions.
    # Consider the contraction over j.
    # S_{ik} = sum_j T_{ijk} = sum_j delta_{ik} = 5 * delta_{ik}.
    # This is a 3x5 matrix with 5 on the diagonal (first 3 entries) and 0 elsewhere.
    # The rank of this matrix is 3.
    # The tensor rank is at least the matrix rank of any contraction?
    # Yes, the tensor rank is >= the rank of the matrix obtained by contracting any index pair?
    # Actually, tensor rank >= spectral norm? No.
    # Tensor rank >= rank of any matricization?
    # The matricization of T as a (n*m) x (m*p) matrix? No.
    # The matricization as (n*p) x (m) ?
    # Let M_{(ik), j} = T_{ijk}.
    # M is 15 x 5.
    # Rows corresponding to (0,0), (1,1), (2,2) are all-ones vectors.
    # Other rows are zero.
    # The rank of this matrix is 1 (since all non-zero rows are identical).
    # This doesn't help.
    
    # Let's look at the matricization as (n) x (m*p).
    # M_{i, (jk)} = T_{ijk}.
    # Row i is non-zero only at positions (j, i) for all j.
    # Row 0: 1 at (0,0), (1,0), (2,0), (3,0), (4,0). Zeros elsewhere.
    # Row 1: 1 at (0,1), (1,1), (2,1), (3,1), (4,1). Zeros elsewhere.
    # Row 2: 1 at (0,2), (1,2), (2,2), (3,2), (4,2). Zeros elsewhere.
    # These 3 rows are linearly independent.
    # So the rank of this matricization is 3.
    # Since the tensor rank is at least the rank of any matricization, Rank(T) >= 3.
    # We found a decomposition of rank 3.
    # So the minimal rank is exactly 3.
    
    # Let's construct the matrices.
    
    R = 3
    U = np.zeros((n * m, R), dtype=object)
    V = np.zeros((m * p, R), dtype=object)
    W = np.zeros((p * n, R), dtype=object)
    
    for r in range(R):
        # U[i*m + j, r] = 1 if i == r else 0
        for i in range(n):
            if i == r:
                for j in range(m):
                    U[i * m + j, r] = 1
        
        # V[j*p + k, r] = 1 if k == r else 0
        for k in range(p):
            if k == r:
                for j in range(m):
                    V[j * p + k, r] = 1
        
        # W[k*n + i, r] = 1 if i == k == r else 0
        for k in range(p):
            for i in range(n):
                if i == r and k == r:
                    W[k * n + i, r] = 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
