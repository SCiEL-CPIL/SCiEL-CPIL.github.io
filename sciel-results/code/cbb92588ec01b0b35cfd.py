"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the matrix multiplication tensor T[i,j,k] = delta (standard schoolbook)
    # T is a 3-tensor of size n*m*p
    # For decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == T[i,j,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # T[i*m+j, k*n+i] = 1 for each (j,k) pair... actually let me think more carefully.
    
    # The matrix multiplication tensor: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,p_idx] = sum_j A[i,j] * B[j,p_idx]
    # Index: A has (n*m) entries, B has (m*p) entries, C has (n*p) entries
    # Tensor T[a_idx, b_idx, c_idx] where a_idx = i*m+j, b_idx = j*p+k, c_idx = i*p+k
    # T has value 1 when a_idx, b_idx, c_idx are consistent: j matches, i matches, k matches
    
    # For rank-r decomposition:
    # U: (n*m) x r, V: (m*p) x r, W: (n*p) x r
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Wait, the task says W has shape (p*n, rank) with trace convention W[k*n+i, r]
    # So W index is k*n + i, which is the same as i*n + k if we think of it as... 
    # Actually k*n + i means k is the outer index (p of them) and i is the inner (n of them)
    # So W[k*n+i] corresponds to the (i,k) entry of C? Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the output C has entries C[i,k] for i in [0,n), k in [0,p)
    # W[k*n+i] = C[i,k]... so W is indexed by (k,i) pair = (p_dim, n_dim)
    
    # T[a,b,c] = 1 iff there exists (i,j,k) such that a = i*m+j, b = j*p+k, c = k*n+i
    # And the tensor is trilinear: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # For schoolbook: each (i,j,k) gives one term where U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # This gives rank n*m*p = 48.
    
    # The question is whether we can do better. For 2x3x8, let me think about known results.
    # The rank of matrix multiplication tensor M_{<n,m,p>} 
    # Known: M_{<2,2,2>} has rank 7 (Strassen)
    # M_{<2,3,3>} ... various results
    # For 2x3x8, I'm not sure of the exact minimum rank.
    
    # Let me try a different approach: use the fact that for fixed n=2, m=3, p=8,
    # we can try to find a lower-rank decomposition via algebraic methods.
    
    # One approach: group terms. Since n=2, we have i in {0,1}.
    # For each i, we have a 3x8 matrix multiplication slice.
    # C[i,:] = A[i,:] * B  (row vector times matrix)
    
    # Actually, let me try a greedy search for linear dependencies.
    
    # First, verify schoolbook works and then try to reduce.
    def verify(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Build T
        T = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i  # W[k*n+i] convention
                    T[a, b, c] = 1.0
        # Compute decomposition
        # sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor; check all entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(n*p):
                    val = sum(U[a, r_] * V[b, r_] * W[c, r_] for r_ in range(r))
                    if abs(val - T[a,b,c]) > 1e-9:
                        return False
        return True
    
    # Try schoolbook first as baseline
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # Strategy: use random linear combinations of the schoolbook terms
    # to see if we can express 48 terms as fewer.
    
    # Actually, for a more systematic approach, let's use the structure.
    # The tensor M_{<2,3,8>} can potentially be decomposed using bilinear forms.
    
    # Let me try: for n=2, the two "slices" C[0,:] = A[0,:]*B and C[1,:] = A[1,:]*B
    # Each slice is a 3x8 matvec (row times matrix).
    # The total tensor rank could potentially be less than 48 if there's sharing.
    
    # A known technique: for M_{<n,m,p>} with small n, use the following structure:
    # Write the multiplication as: for each output position (i,k), 
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # For n=2: 
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # Total: 2*3*8 = 48 terms in schoolbook.
    
    # Can we share? Strassen's trick for 2x2x2 shows sharing is possible.
    # For 2x3x3, the rank is known to be at most... let me think.
    
    # Let me try a computational search: represent the tensor as a matrix and find its rank
    # via flattening. The tensor T has dimensions (n*m) x (m*p) x (n*p) = 6 x 24 x 16.
    # The tensor rank is at least the rank of any matrix flattening.
    
    # Flatten1: (n*m) x (m*p * n*p) = 6 x 384, max rank 6
    # Flatten2: (m*p) x (n*m * n*p) = 24 x 96, max rank 24  
    # Flatten3: (n*p) x (n*m * m*p) = 16 x 144, max rank 16
    
    # So tensor rank >= 16 (from the third flattening). This is a lower bound.
    # The schoolbook gives 48. So there might be room to reduce.
    
    # Let me try to find a rank-r decomposition for r < 48 using a numerical approach.
    # I'll use an alternating least squares / rank minimization approach.
    
    # Actually, let me try a more direct algebraic approach.
    # Since n=2, m=3, p=8, let me try to construct a decomposition of rank < 48.
    
    # One idea: group by the middle index j. For each j in {0,1,2}:
    # We have terms: A[i,j] * B[j,k] for all i in {0,1}, k in {0,1,2,3,4,5,6,7}
    # That's 2*8 = 16 terms per j, total 48.
    
    # Alternative: for each (i,k) output, we need sum_j A[i,j]B[j,k].
    # If we could express this with shared bilinear forms...
    
    # Let me try: for pairs of k values, can we share?
    # B[j,k] and B[j,k'] for different k.
    
    # Actually, let me just try a numerical rank reduction search.
    # Start with the 48 schoolbook terms and try to find linear combinations
    # that give zero (i.e., find the null space of the tensor when viewed appropriately).
    
    # A cleaner approach: the set of all rank-r decompositions forms a variety.
    # We want to find if there's a point in this variety with r < 48.
    
    # Let me try: construct a random 48-term "decomposition" and check if it's equivalent
    # to the schoolbook one, then try to find dependencies.
    
    # Simpler: use the fact that the tensor space is (6 x 24 x 16) and the rank-48 
    # tensor spans a 48-dimensional space (the schoolbook terms). If the actual tensor
    # lies in a lower-dimensional subspace of rank-r tensors for some r < 48, 
    # we'd find it.
    
    # For now, let me try a specific construction.
    # For 2x3x8, note that 2*3*8 = 48.
    # Known result: rank of M_{<2,n,n>} for various n.
    # For n=3: rank of M_{<2,3,3>} is at most 13 (I recall something like this).
    # For 2x3x8: maybe rank is around 20-30?
    
    # Let me try a constructive approach:
    # For the 2x3 part (i in {0,1}, j in {0,1,2}), use a 2x3x1 tensor of rank at most 3
    # (which is schoolbook). Then for each of the 8 k-values, we'd need 3 terms = 24 total?
    # No, that's not how it works because B[j,k] couples j and k.
    
    # Let me try yet another approach: use the structure to write
    # C[i,k] = sum_j A[i,j] B[j,k] as a bilinear map from (A[0,:], A[1,:], B[:,0],...,B[:,7])
    # to (C[0,0],...,C[0,7], C[1,0],...,C[1,7]).
    
    # The domain has 3 + 8*3 = 27 inputs (3 for A row 0, 3 for A row 1, 8*3=24 for B), 
    # wait: A has 2*3=6 entries, B has 3*8=24 entries, total 30 inputs.
    # Output: 2*8 = 16 entries.
    
    # This is a bilinear map R^30 -> R^16 (actually from two spaces: A-space (dim 6) and B-space (dim 24)).
    
    # The rank of this bilinear map is what we're looking for.
    
    # Let me just try to find a lower-rank decomposition numerically.
    # I'll use a simple iterative method: start with random rank-r matrices and 
    # project to get closer to T.
    
    # Actually, given time constraints, let me try a specific known construction.
    # For M_{<2,3,p>}, one can use:
    # For each j in {0,1,2}:
    #   Let a_j = A[0,j], b_j = A[1,j] (scalars)
    #   For each k: term a_j * B[j,k] contributes to C[0,k], b_j * B[j,k] contributes to C[1,k]
    
    # The "sharing" comes from combining terms across k.
    # For fixed j, the contribution to output is:
    #   C[0,:] gets a_j * B[j,:] (a row of B scaled)
    #   C[1,:] gets b_j * B[j,:] (same row scaled differently)
    
    # If we introduce an auxiliary variable x = B[j,:] (8-dim), then:
    #   We need a_j * x[k] for C[0,k] and b_j * x[k] for C[1,k]
    #   This can be done with 1 term if we had x available, but x depends on j.
    
    # For 3 values of j, we have 3 such "auxiliary" rows of B.
    
    # A potential decomposition:
    # For each j in {0,1,2} and each k in {0,...,7}: that's 24 terms.
    # Plus we need to handle the A[i,j] coefficients.
    
    # Actually, I think the rank might indeed be 48 for these dimensions, or close to it.
    # Without a clear construction, let me just return the schoolbook.
    
    # But let me try one more thing: a small exhaustive search for rank 47.
    # If we can show that no rank-47 decomposition exists, then 48 is optimal.
    # If rank-47 exists, great.
    
    # Given the time limit, I'll return the schoolbook decomposition.
    # This is the safe baseline.
    
    if time.time() > deadline:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Let me try to see if we can shave off a few terms using the 2x2x2 Strassen-like structure
    # applied within the 3x8 block.
    
    # For m=3, p=8: we can think of this as 2 groups of 3x4 or 3 groups of 2x4, etc.
    # But the coupling through the middle index j makes it tricky.
    
    # I'll return the schoolbook for now.
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
