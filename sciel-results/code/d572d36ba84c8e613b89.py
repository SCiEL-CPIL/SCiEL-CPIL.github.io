"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if the product C[i,k] depends on A[i,j]*B[j,k]
    # For standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (each term contributes)
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 iff 
    # the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # For standard multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1.
    
    # The tensor is of shape (n, m, p) with T[i,j,k] = 1 for all i,j,k.
    # This is the all-ones tensor of shape (4, 4, 7).
    
    # Wait, that's not right. The matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if and only if the product involves A[i,j] * B[j,k] -> C[i,k]
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i,k), we sum over j. The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # This means T is the all-ones tensor of shape (n, m, p) = (4, 4, 7).
    
    # The rank of the all-ones tensor of shape (n, m, p) is n*m*p if we use schoolbook,
    # but there might be lower rank decompositions.
    
    # Actually, let me reconsider. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the index structure is (i, j, k)
    # with i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all i,j,k.
    
    # This is just the all-ones tensor. Its tensor rank is min(n,m,p) * max(n,m,p)... 
    # No, the rank of the all-ones tensor of shape (a,b,c) is max(a,b,c) if we can use 
    # rank-1 tensors that are all ones on their supports.
    
    # Actually, the all-ones tensor of shape (a,b,c) has rank min(a,b,c) if we can decompose it.
    # Let me think more carefully.
    
    # The all-ones tensor T of shape (a,b,c) can be written as:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # If we set U[i,r] = 1 for r <= min(a,b,c), V[j,r] = 1 for r <= min(a,b,c), 
    # W[k,r] = 1 for r <= min(a,b,c), and 0 otherwise, then:
    # sum_r U[i,r] V[j,r] W[k,r] = min(a,b,c)
    # So T[i,j,k] = min(a,b,c) * (all ones) / min(a,b,c) = 1.
    # So we need rank = min(a,b,c) with U,V,W having entries 1/min(a,b,c).
    
    # But wait, we need integer/half-integer entries. 1/4 is not half-integer.
    # 1/7 is not half-integer either.
    
    # Hmm, let me reconsider. The problem says "integer/half-integer entries".
    # Half-integers are numbers of the form k/2 for integer k.
    # So 1/4 is not allowed, but 1/2 is.
    
    # For shape (4, 4, 7), min(4,4,7) = 4.
    # If we use rank 4 with U,V,W entries of 1/4, that's not allowed.
    # If we use rank 8 with entries of 1/2, that might work.
    
    # Let me think about this differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # We want T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    # One approach: use rank = n*m = 16 or m*p = 28 or n*p = 28.
    # The schoolbook gives rank = n*m*p = 112.
    
    # But we can do better. Let's think about the structure.
    
    # Actually, I realize I may be overthinking this. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix 
    # by a 4 by 7 matrix."
    
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] where i indexes rows of A (and C), j indexes columns of A (and rows of B),
    # k indexes columns of B (and C).
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k].
    
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is indeed the all-ones tensor of shape (n, m, p) = (4, 4, 7).
    
    # The tensor rank of the all-ones tensor of shape (a, b, c) is:
    # It's known that the rank is min(a,b,c) if we allow arbitrary real entries.
    # But with the constraint of integer/half-integer entries, it might be different.
    
    # Let me try a different approach. Instead of trying to find the minimal rank theoretically,
    # let me use the peeling algorithm or a search to find a good decomposition.
    
    # For now, let me start with the schoolbook decomposition and try to improve it.
    
    # Schoolbook: rank = n * m * p = 4 * 4 * 7 = 112
    # U has shape (n*m, rank) = (16, 112)
    # V has shape (m*p, rank) = (28, 112)
    # W has shape (p*n, rank) = (28, 112)
    
    # Wait, the interface says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # And W follows the trace convention W[k*n+i, r].
    
    # Let me reconsider the tensor structure. The multiplication is:
    # C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has dimensions (n, m, p) with T[i,j,k] = 1.
    
    # In the decomposition:
    # U has shape (n*m, rank), so U[i*m+j, r] corresponds to A[i,j]
    # V has shape (m*p, rank), so V[j*p+k, r] corresponds to B[j,k]
    # W has shape (p*n, rank), so W[k*n+i, r] corresponds to C[i,k]
    
    # The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the all-ones tensor, T[i,j,k] = 1 for all i,j,k.
    
    # Let me try to find a low-rank decomposition.
    
    # Approach: Use the fact that T[i,j,k] = 1 can be written as:
    # T[i,j,k] = (1/n) * sum_{i'} delta_{i,i'} * (1/m) * sum_{j'} delta_{j,j'} * (1/p) * sum_{k'} delta_{k,k'}
    # This doesn't help directly.
    
    # Alternative: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let me try rank = 4 (min dimension):
    # U[i*m+j, r] = 1/4 for all i,j,r
    # V[j*p+k, r] = 1 for all j,k,r
    # W[k*n+i, r] = 1 for all k,i,r
    # Then sum_r U*V*W = sum_r (1/4)*1*1 = 4*(1/4) = 1. 
    # But 1/4 is not a half-integer.
    
    # Let me try rank = 8:
    # We need entries that are multiples of 1/2.
    # If we use U[i*m+j, r] = 1/2 for r < 4, 0 otherwise
    # V[j*p+k, r] = 1 for all
    # W[k*n+i, r] = 1 for all
    # Then sum_r = 4 * (1/2) * 1 * 1 = 2. Not 1.
    
    # If we use U[i*m+j, r] = 1/2 for r < 2, 0 otherwise
    # Then sum_r = 2 * (1/2) = 1. 
    # So rank = 2 might work?
    
    # Wait, let me check:
    # U[i*m+j, r] = 1/2 for r in {0, 1}, 0 for r >= 2
    # V[j*p+k, r] = 1 for all j,k,r
    # W[k*n+i, r] = 1 for all k,i,r
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = sum_{r=0}^{1} (1/2) * 1 * 1 = 1. 
    # So rank = 2 works!
    
    # But wait, is this correct? Let me verify with the actual matrix multiplication.
    
    # If U[i*m+j, r] = 1/2 for r in {0,1}, then for each (i,j), U has two non-zero entries.
    # V[j*p+k, r] = 1 for all (j,k) and r in {0,1}.
    # W[k*n+i, r] = 1 for all (k,i) and r in {0,1}.
    
    # The product is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_j sum_{r=0}^{1} (1/2) * 1 * 1
    # = sum_j 1
    # = m = 4
    
    # That's not right. The issue is that the decomposition should give:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In the tensor decomposition, the formula is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And T[i,j,k] = 1 for the standard multiplication tensor.
    
    # So the decomposition is correct if T[i,j,k] = 1 for all i,j,k.
    
    # But when we actually compute C = A * B using this decomposition, we get:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    
    # Wait, I think I'm confusing the tensor decomposition with the actual computation.
    
    # Let me re-read the task more carefully.
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer 
    # entries that exactly decompose the matrix multiplication tensor"
    
    # So the tensor T of shape (n*m, m*p, p*n) or (n, m, p) is what we're decomposing.
    
    # The tensor for matrix multiplication (n x m) * (m x p) is:
    # T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    
    # For standard multiplication, T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    # With U[i*m+j, r] = 1/2 for r in {0,1}, V[j*p+k, r] = 1, W[k*n+i, r] = 1:
    # sum_r = 2 * (1/2) * 1 * 1 = 1. 
    # So this works!
    
    # But wait, 1/2 is a half-integer, so it's allowed.
    
    # So the rank is 2? That seems too good to be true.
    
    # Let me double-check. The tensor T has shape (4, 4, 7) with all entries 1.
    # Can we really decompose it with rank 2?
    
    # T[i,j,k] = sum_{r=0}^{1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # If U[i*m+j, r] = 1/2 for all i,j and r in {0,1},
    # V[j*p+k, r] = 1 for all j,k and r in {0,1},
    # W[k*n+i, r] = 1 for all k,i and r in {0,1},
    
    # Then T[i,j,k] = (1/2)*1*1 + (1/2)*1*1 = 1. 
    # Yes, this works!
    
    # So the rank is 2. But is this minimal? Can we do rank 1?
    
    # For rank 1:
    # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # We need this to be 1 for all i,j,k.
    # So U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    
    # Let U[i*m+j, 0] = a_{i,j}, V[j*p+k, 0] = b_{j,k}, W[k*n+i, 0] = c_{k,i}.
    # Then a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    
    # This is possible if a_{i,j} = 1, b_{j,k} = 1, c_{k,i} = 1.
    # So rank 1 works!
    
    # Wait, that can't be right. Let me check again.
    
    # If U[i*m+j, 0] = 1 for all i,j,
    # V[j*p+k, 0] = 1 for all j,k,
    # W[k*n+i, 0] = 1 for all k,i,
    
    # Then T[i,j,k] = 1 * 1 * 1 = 1. 
    # So rank 1 works!
    
    # But that would mean the tensor rank of the all-ones tensor of shape (4,4,7) is 1,
    # which is not true in general. The issue is that the tensor here is not a standard 
    # 3D tensor; it has a specific structure related to matrix multiplication.
    
    # Actually, I think I'm making an error. Let me reconsider.
    
    # The matrix multiplication tensor is not the all-ones tensor in the usual sense.
    # The tensor T for multiplying (n x m) by (m x p) is defined such that:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    # The tensor T has components T[i, j, k] where:
    # - i indexes the row of A and C
    # - j indexes the column of A and row of B
    # - k indexes the column of B and C
    
    # And T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] appears in C[i,k].
    
    # For standard matrix multiplication, every term A[i,j] * B[j,k] appears in C[i,k] 
    # for the appropriate i,k. Specifically, A[i,j] * B[j,k] contributes to C[i,k].
    
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # But the key point is that the indices are linked: the j in A[i,j] must match 
    # the j in B[j,k]. This is already captured by the tensor structure.
    
    # So T is indeed the all-ones tensor of shape (n, m, p) = (4, 4, 7).
    
    # And I've shown that rank 1 works: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    
    # But this can't be right because the tensor rank of the all-ones tensor of shape 
    # (a, b, c) is not 1 in general.
    
    # Let me reconsider. The issue might be in how the tensor is indexed.
    
    # The tensor T has shape (n*m, m*p, p*n) when flattened, or (n, m, p) when 
    # considering the three modes.
    
    # In the decomposition:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # If U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, then:
    # T[i, j, k] = 1 * 1 * 1 = 1.
    
    # So the rank is 1. But this seems too simple.
    
    # I think the issue is that I'm not accounting for the fact that the same 
    # U[i*m+j, r] value is used for different (i,j) pairs, and similarly for V and W.
    
    # Let me try a specific example. Suppose n=2, m=2, p=2.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # C[0,0] = a11*b11 + a12*b21
    # C[0,1] = a11*b12 + a12*b22
    # C[1,0] = a21*b11 + a22*b21
    # C[1,1] = a21*b12 + a22*b22
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # If we use rank 1 with U[i*2+j, 0] = 1, V[j*2+k, 0] = 1, W[k*2+i, 0] = 1,
    # then T[i,j,k] = 1 for all i,j,k.
    
    # But when we compute C using this decomposition:
    # C[i,k] = sum_j sum_r U[i*2+j, r] * A[i,j] * V[j*2+k, r] * B[j,k] * W[k*2+i, r]
    # = sum_j A[i,j] * B[j,k] * (1 * 1 * 1)
    # = sum_j A[i,j] * B[j,k]
    
    # So it works! The rank is 1.
    
    # But this is clearly wrong because the tensor rank of matrix multiplication 
    # is known to be much higher than 1.
    
    # I think the issue is in the interpretation of the tensor decomposition.
    # Let me re-read the task.
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer 
    # entries that exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor is a 3-tensor of shape (n*m, m*p, p*n) or 
    # equivalently (n, m, p) with the understanding that the indices are linked.
    
    # The decomposition is:
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that's not right either. The tensor T is of shape (n, m, p), not 
    # (n*m, m*p, p*n).
    
    # Let me look at the schoolbook decomposition in the current program:
    # U[i * m + j][column] = 1
    # V[j * p + k][column] = 1
    # W[k * n + i][column] = 1
    
    # So for each (i,j,k), we have a rank-1 component where:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # And the tensor component T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the schoolbook, T[i,j,k] = 1 for each (i,j,k) because there's exactly one 
    # column c where all three are 1.
    
    # So the tensor T has T[i,j,k] = 1 for all i,j,k.
    
    # And I've shown that rank 1 works: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    
    # But this gives T[i,j,k] = 1 for all i,j,k, which is correct.
    
    # So why is the schoolbook rank n*m*p = 112?
    
    # I think the issue is that the tensor is not the all-ones tensor. Let me reconsider.
    
    # Actually, I think I've been making a fundamental error. The matrix multiplication 
    # tensor is not T[i,j,k] = 1 for all i,j,k. Instead, the tensor encodes the 
    # bilinear map from (A, B) to C.
    
    # The correct tensor is:
    # T[i, j, k] = 1 if and only if the (i,j) entry of A and the (j,k) entry of B 
    # contribute to the (i,k) entry of C.
    
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k].
    # So A[i,j] and B[j,k] contribute to C[i,k] for the same i,k.
    
    # The tensor T has T[i, j, k] = 1 for all i,j,k because every combination 
    # (i,j,k) corresponds to a term in the sum.
    
    # But the key is that the tensor is of shape (n, m, p), and the decomposition 
    # uses U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    
    # The relationship is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And T[i,j,k] = 1 for all i,j,k.
    
    # So my earlier analysis was correct, and rank 1 should work.
    
    # But this contradicts the known result that the tensor rank of matrix multiplication 
    # is much higher than 1.
    
    # I think the resolution is that the tensor rank of the all-ones tensor of shape 
    # (n, m, p) is indeed 1, but the matrix multiplication tensor is not the all-ones 
    # tensor in the usual tensor rank sense.
    
    # Actually, I think I've been confusing two different things. The matrix multiplication 
    # tensor is a 3-tensor of shape (n*p, n*m, m*p) or something similar, not (n, m, p).
    
    # Let me look at this more carefully.
    
    # The bilinear map is: (A, B) -> C where A is n x m, B is m x p, C is n x p.
    # The tensor for this bilinear map has shape (n*p, n*m, m*p) where:
    # - The first mode indexes the entries of C (n*p entries)
    # - The second mode indexes the entries of A (n*m entries)
    # - The third mode indexes the entries of B (m*p entries)
    
    # And the tensor component T[c, a, b] = 1 if and only if A[a] and B[b] contribute 
    # to C[c].
    
    # For standard matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So C[i,k] depends on A[i,j] and B[j,k] for all j.
    
    # In terms of the flattened indices:
    # c = i*p + k (or k*n + i, depending on the convention)
    # a = i*m + j
    # b = j*p + k
    
    # The tensor T[c, a, b] = 1 if and only if there exist i,j,k such that:
    # c = i*p + k, a = i*m + j, b = j*p + k.
    
    # This is a specific structure, not the all-ones tensor.
    
    # So the tensor is not the all-ones tensor of shape (n*p, n*m, m*p). It's a 
    # sparse tensor with a specific pattern.
    
    # Now, the decomposition in the task uses:
    # U of shape (n*m, rank)
    # V of shape (m*p, rank)
    # W of shape (p*n, rank)
    
    # And the tensor component is:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Where a indexes A entries, b indexes B entries, c indexes C entries.
    
    # And T[a, b, c] = 1 if and only if A[a] and B[b] contribute to C[c].
    
    # For the schoolbook decomposition:
    # For each (i,j,k), we have a = i*m+j, b = j*p+k, c = k*n+i (or c = i*p+k).
    # U[a, c_idx] = 1, V[b, c_idx] = 1, W[c, c_idx] = 1.
    
    # So T[a, b, c] = 1 if and only if there's a column where U[a, col] = V[b, col] = W[c, col] = 1.
    
    # This matches the structure of the matrix multiplication tensor.
    
    # Now, can we find a lower rank decomposition?
    
    # The tensor T has T[a, b, c] = 1 if and only if:
    # a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k.
    
    # This is a very specific pattern. Let me count the number of non-zero entries.
    # For each (i,j,k), we have one non-zero entry. So there are n*m*p = 112 non-zero entries.
    
    # The rank of this tensor is at least the maximum of:
    # - The number of non-zero entries in any slice
    # - The tensor rank of the underlying structure
    
    # For now, let me just implement the schoolbook decomposition and see if I can 
    # find a better one through search.
    
    # Given the time constraint, let me start with the schoolbook and try to improve it.
    
    # Schoolbook: rank = n * m * p = 112
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Now, let me try to find a lower rank decomposition.
    # The peeling algorithm or a search might help.
    
    # For now, let me return the schoolbook decomposition as a baseline.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
