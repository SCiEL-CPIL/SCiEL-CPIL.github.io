import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    
    # For 2x3 * 3x8 matrix multiplication:
    # A is 2x3, B is 3x8, C = A*B is 2x8
    # 
    # The tensor has dimensions (2*3) x (3*8) x (8*2) = 6 x 24 x 16
    #
    # We can decompose this by splitting the 8 columns of B into 4 groups of 2.
    # For each group of 2 columns, we have a 2x3 * 3x2 sub-problem.
    #
    # For 2x3 * 3x2: A is 2x3, B is 3x2, C is 2x2
    # The schoolbook rank is 3*2 = 6 (one per (j,k) pair where j in {0,1,2}, k in {0,1})
    # 
    # Can we do better than 6 for 2x3 * 3x2?
    # The tensor is (2*3) x (3*2) x (2*2) = 6 x 6 x 4
    # 
    # Actually, let's think about this differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For the 2x3 * 3x2 case:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # This is a 2x2 matrix product with "thick" operands.
    # We can view this as: for each row i of A and column k of C:
    # C[i,k] = dot(A[i,:], B[:,k])
    #
    # The tensor is sum over j of A[i,j] * B[j,k] * C[i,k]
    # = sum_j (A[i,j]) (B[j,k]) (C[i,k])
    #
    # This is exactly the standard matrix multiplication tensor for 2x3 * 3x2.
    # The rank of this tensor... let's think.
    #
    # Actually, we can use the fact that this is equivalent to computing 2 independent
    # dot products of length 3 (one for each column of C, for each row of A).
    # But they share the A factors.
    #
    # Let me try a direct construction. The schoolbook uses 6 terms:
    # for j in 0,1,2 and k in 0,1:
    #   term: A[i,j] * B[j,k] -> C[i,k]
    #
    # That's 3*2 = 6 terms per (i,k) output... no wait.
    # Each term is (i,j) in A, (j,k) in B, (i,k) in C.
    # So the terms are indexed by (j, i, k) where j in {0,1,2}, i in {0,1}, k in {0,1}
    # Total: 3*2*2 = 12 terms? No...
    #
    # Wait, the tensor is T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # For 2x3*3x2: i in {0,1}, j in {0,1,2}, k in {0,1}
    # So there are 2*3*2 = 12 non-zero entries, but the rank can be less.
    #
    # Actually, the rank of the matrix multiplication tensor [n,m,p] is the minimum
    # number of rank-1 tensors needed. For [2,3,2], the schoolbook gives 3*2=6... 
    # no, schoolbook gives n*m*p = 2*3*2 = 12? No.
    #
    # Schoolbook: for each (i,j,k) triple, we have one term. That's 2*3*2 = 12 terms.
    # But each term is U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1.
    # So schoolbook rank = n*m*p = 12 for [2,3,2].
    #
    # Hmm, but that's not right either. Let me reconsider.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 where:
    # - i indexes rows of A (0..n-1)
    # - j indexes columns of A / rows of B (0..m-1)  
    # - k indexes columns of B (0..p-1)
    # And the output is C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # In the factorization, we write:
    # C[i,k] = sum_t U[i*m+j... wait, no.
    #
    # The factorization is:
    # C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
    #
    # No, that's not right either. Let me be more careful.
    #
    # The standard formulation: we want to compute C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition writes:
    # C[i,k] = sum_{t=1}^{r} W[i,k,t] * (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k])
    #
    # Wait, that's not standard either. Let me look at this more carefully.
    #
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)} where:
    # T[(i,j), (j',k), (k',i')] = 1 if i=i' and j=j' and k=k', else 0.
    #
    # So T is a 3-way tensor with indices (i*m+j, j*p+k, k*n+i).
    # T is non-zero only when the three indices are consistent (same i, j, k).
    #
    # A rank-r decomposition is: T = sum_t U_t ⊗ V_t ⊗ W_t
    # where U_t is in R^{n*m}, V_t in R^{m*p}, W_t in R^{p*n}.
    #
    # The condition is: for all (i,j,k):
    # 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # And for all inconsistent triples (i,j,k) where the indices don't match:
    # 0 = sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t]
    #
    # Actually, the tensor is defined as:
    # T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 when a, b, c come from the same (i,j,k), and 0 otherwise.
    #
    # So the condition for the decomposition is:
    # For all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}:
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # And for all other combinations of (a, b, c) that don't correspond to a valid (i,j,k):
    #   sum_t U[a, t] * V[b, t] * W[c, t] = 0
    #
    # This is a strong condition. The "other combinations" include cases where
    # a = i*m+j, b = j'*p+k, c = k'*n+i' with j != j' or k != k' etc.
    #
    # For the schoolbook, each term t corresponds to a specific (i,j,k):
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # All other entries are 0.
    #
    # This gives rank = n*m*p.
    #
    # For [2,3,8]: schoolbook rank = 2*3*8 = 48.
    #
    # Can we do better? Yes, by sharing terms.
    #
    # For [2,3,2]: schoolbook rank = 12. Can we do better?
    # 
    # Let me think about [1,1,1]: rank 1.
    # [1,1,p]: C[0,k] = A[0,0]*B[0,k], so C[0,k] = A[0,0]*B[0,k].
    # This is rank p: for each k, one term with U[0, t_k]=1, V[k, t_k]=1, W[k, t_k]=1.
    # Actually, can we do better? C[0,k] = A[0,0] * B[0,k].
    # The tensor is: T[0, k, k] = 1 for each k.
    # This is a set of p independent rank-1 tensors, so rank = p.
    #
    # [2,1,1]: C[i,0] = A[i,0]*B[0,0].
    # T[i*1+0, 0*1+0, 0*2+i] = T[i, 0, i] = 1 for i in {0,1}.
    # Rank = 2.
    #
    # [2,2,2]: This is the 2x2 matrix multiplication tensor.
    # Known rank: 7 (Strassen).
    #
    # [2,3,2]: Let me think. We can view this as two 2x1 multiplications... no.
    # Actually, [2,3,2] can be seen as: for each of the 2 columns of B (k=0,1),
    # we compute C[:,k] = A * B[:,k], which is a 2x3 times 3x1 multiplication.
    # The 2x3 * 3x1 tensor has rank 3 (schoolbook: sum_j A[i,j]*B[j,0]).
    # But the two columns share the A factors.
    #
    # For 2x3 * 3x1: C[i,0] = sum_j A[i,j] B[j,0]
    # T[i*3+j, j*1+0, 0*2+i] = T[i*3+j, j, i] = 1
    # Rank = 3 (one for each j).
    #
    # For 2x3 * 3x2: we have two such "columns" (k=0 and k=1).
    # If they were independent, rank would be 3+3=6.
    # But they share the A factors, so maybe we can do better?
    # Actually, no - the B factors are different for each k, so the terms are distinct.
    # The rank should be 6 for [2,3,2].
    #
    # Wait, let me verify. The tensor for [2,3,2] has non-zero entries at:
    # (i*3+j, j*2+k, k*2+i) for i in {0,1}, j in {0,1,2}, k in {0,1}
    # That's 2*3*2 = 12 entries.
    # 
    # Each rank-1 term can cover at most one such entry (since the indices are all different
    # for different (i,j,k) triples... let me check).
    #
    # For (i,j,k) = (0,0,0): indices are (0, 0, 0)
    # For (i,j,k) = (0,0,1): indices are (0, 1, 2)
    # For (i,j,k) = (0,1,0): indices are (1, 2, 0)
    # etc.
    #
    # A rank-1 term U ⊗ V ⊗ W has U[a]*V[b]*W[c] non-zero for many (a,b,c).
    # The condition is that for valid (i,j,k), the sum equals 1, and for invalid, equals 0.
    #
    # This is complex. Let me just try to construct a low-rank decomposition computationally.
    #
    # Actually, let me try a different approach. Let me use the known result that
    # the rank of [n,m,p] matrix multiplication is at most n*m*p, and often much less.
    #
    # For our problem [2,3,8], let me try to find a good decomposition by optimization.
    
    # Let me try a direct optimization approach.
    # We want to find U (6 x r), V (24 x r), W (16 x r) such that:
    # For all i in {0,1}, j in {0,1,2}, k in {0,...,7}:
    #   sum_t U[i*3+j, t] * V[j*8+k, t] * W[k*2+i, t] = 1
    # And for all other (a,b,c) combinations:
    #   sum_t U[a, t] * V[b, t] * W[c, t] = 0
    
    # This is a non-convex problem. Let me try a small rank and see if I can solve it.
    
    # Actually, let me think about this more carefully using block structure.
    #
    # C = A * B where A is 2x3, B is 3x8.
    # Write A = [A_left (2x2), A_right (2x1)]
    # Write B = [B_top (2x8); B_bottom (1x8)]
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left is 2x2, B_top is 2x8.
    # A_right is 2x1, B_bottom is 1x8.
    #
    # For A_right * B_bottom: this is a 2x1 * 1x8 multiplication.
    # C_part[i,k] = A_right[i,0] * B_bottom[0,k]
    # This is rank 8 (one term per k, with U[i*3+2, t_k] = 1, V[2*8+k, t_k] = 1, W[k*2+i, t_k] = 1).
    # Actually, we can share: for each k, we need U[i*3+2, t] and V[2*8+k, t] and W[k*2+i, t].
    # For a fixed k, the terms for i=0 and i=1 are:
    #   (U[2, t], V[2*8+k, t], W[k*2, t]) for i=0
    #   (U[5, t], V[2*8+k, t], W[k*2+1, t]) for i=1
    # We can use 2 terms per k (one for each i), giving 16 terms. 
    # Or we can use 8 terms if we can combine... but U[2,t] and U[5,t] are different rows,
    # so we need separate terms. So 16 terms for A_right * B_bottom.
    #
    # Wait, no. For a fixed k, the contribution to C[0,k] is A[0,2]*B[2,k] and to C[1,k] is A[1,2]*B[2,k].
    # These are two separate products. We need:
    #   term for C[0,k]: U[0*3+2, t] = 1, V[2*8+k, t] = 1, W[k*2+0, t] = 1
    #   term for C[1,k]: U[1*3+2, t] = 1, V[2*8+k, t] = 1, W[k*2+1, t] = 1
    # That's 2 terms per k, so 16 terms total for the A_right * B_bottom part.
    #
    # For A_left * B_top: A_left is 2x2, B_top is 2x8.
    # C_part[i,k] = A_left[i,0]*B_top[0,k] + A_left[i,1]*B_top[1,k]
    # This is a 2x2 * 2x8 multiplication.
    # We can split B_top into 4 blocks of 2x2: B_top = [B0, B1, B2, B3] where each Bj is 2x2.
    # For each 2x2 * 2x2 block, we can use Strassen (rank 7).
    # So 4 * 7 = 28 terms for A_left * B_top.
    #
    # Total: 28 + 16 = 44 terms.
    #
    # So rank 44 is achievable! Let me construct it.
    
    rank = 44
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Part 1: A_left * B_top (2x2 * 2x8)
    # A_left[i,j] = A[i,j] for i in {0,1}, j in {0,1}
    # B_top[j,k] = B[j,k] for j in {0,1}, k in {0,...,7}
    #
    # Split B_top into 4 blocks of 2x2: block b has columns 2b, 2b+1
    # For block b:
    #   B_top[0, 2b] = b00, B_top[0, 2b+1] = b01
    #   B_top[1, 2b] = b10, B_top[1, 2b+1] = b11
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    for b in range(4):
        # M1 = (a00+a11)*(b00+b11)
        # a00 = A[0,0], a11 = A[1,1]
        # b00 = B[0,2b], b11 = B[1,2b+1]
        U[0*3+0, col] = 1
        U[1*3+1, col] = 1
        V[0*8+2*b, col] = 1
        V[1*8+2*b+1, col] = 1
        W[(2*b)*2+0, col] = 1      # C[0, 2b]
        W[(2*b+1)*2+1, col] = 1    # C[1, 2b+1]
        col += 1
        
        # M2 = (a10+a11)*b00
        # a10 = A[1,0], a11 = A[1,1]
        # b00 = B[0,2b]
        U[1*3+0, col] = 1
        U[1*3+1, col] = 1
        V[0*8+2*b, col] = 1
        W[(2*b)*2+1, col] = 1  # C[1, 2b]
        col += 1
        
        # M3 = a00*(b10-b11)
        # a00 = A[0,0]
        # b10 = B[1,2b], b11 = B[1,2b+1]
        U[0*3+0, col] = 1
        V[1*8+2*b, col] = 1
        V[1*8+2*b+1, col] = -1
        W[(2*b+1)*2+0, col] = 1  # C[0, 2b+1]
        col += 1
        
        # M4 = a11*(b01-b10)
        # a11 = A[1,1]
        # b01 = B[0,2b+1], b10 = B[1,2b]
        U[1*3+1, col] = 1
        V[0*8+2*b+1, col] = 1
        V[1*8+2*b, col] = -1
        W[(2*b)*2+0, col] = 1      # C[0, 2b]
        W[(2*b)*2+1, col] = 1      # C[1, 2b]
        col += 1
        
        # M5 = (a00+a01)*b11
        # a00 = A[0,0], a01 = A[0,1]
        # b11 = B[1,2b+1]
        U[0*3+0, col] = 1
        U[0*3+1, col] = 1
        V[1*8+2*b+1, col] = 1
        W[(2*b)*2+0, col] = -1     # C[0, 2b]
        W[(2*b+1)*2+0, col] = 1    # C[0, 2b+1]
        col += 1
        
        # M6 = (a10-a00)*(b00+b10)
        # a10 = A[1,0], a00 = A[0,0]
        # b00 = B[0,2b], b10 = B[1,2b]
        # M6 is not used in the standard Strassen output formulas I listed.
        # Let me recheck. The standard Strassen formulas are:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not in any of these! That's strange.
        
        # Actually, I think I have the wrong formulas. Let me rederive.
        # Strassen's algorithm:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Let me verify: C00 = A00*B00 + A01*B10
        # M1 - M5 + M7 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B01+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
        # = A00*B00 + A11*B00 - A11*B01 + A01*B01
        # That's not A00*B00 + A01*B10. So my formulas are wrong.
        
        # Let me look up the correct Strassen formulas.
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B10)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        #
        # C00 = M1 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify C00:
        # M1 - M5 + M7 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B01+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + A11B00 - A11B01 + A01B01
        # But C00 = A00B00 + A01B10. These don't match.
        
        # I think the issue is that I'm mixing up the block layout. Let me be very careful.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]]
        #
        # So C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # Let me re-derive Strassen correctly.
        # M1 = (A00+A11)(B00+B11) = A00B00+A00B11+A11B00+A11B11
        # M2 = (A10+A11)B00 = A10B00+A11B00
        # M3 = A00(B10-B11) = A00B10-A00B11
        # M4 = A11(B01-B10) = A11B01-A11B10
        # M5 = (A00+A01)B11 = A00B11+A01B11
        # M6 = (A10-A00)(B00+B10) = A10B00+A10B10-A00B00-A00B10
        # M7 = (A01-A11)(B01+B11) = A01B01+A01B11-A11B01-A11B11
        
        # C00 = A00B00+A01B10
        # Try: M1 - M5 + M7 = A00B00+A00B11+A11B00+A11B11 - A00B11-A01B11 + A01B01+A01B11-A11B01-A11B11
        # = A00B00 + A11B00 + A01B01 - A11B01
        # Not matching.
        
        # Try: M1 + M4 - M5 - M7
        # = A00B00+A00B11+A11B00+A11B11 + A11B01-A11B10 - A00B11-A01B11 - A01B01-A01B11+A11B01+A11B11
        # = A00B00 + A11B00 + A11B01 - A11B10 - A01B01 - A01B11 + A11B01 + A11B11 + A11B11 - A00B11 + A00B11
        # This is getting messy. Let me just use a known correct set of formulas.
        
        # From Wikipedia:
        # C00 = M1 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify C11:
        # M1 - M2 + M3 + M6 = A00B00+A00B11+A11B00+A11B11 - A10B00-A11B00 + A00B10-A00B11 + A10B00+A10B10-A00B00-A00B10
        # = A11B11 + A10B10
        # But C11 = A10B01+A11B11. Not matching.
        
        # I think the Wikipedia formulas might use a different block layout. Let me just use
        # a verified implementation.
        
        # Actually, let me just use the schoolbook for the 2x2*2x2 blocks to be safe,
        # and accept rank 48 for now. I can optimize later.
        
        # Let me restart with a clean approach: use schoolbook for everything.
        pass
    
    # Reset and use schoolbook
    U = np.zeros((nm, 48))
    V = np.zeros((mp, 48))
    W = np.zeros((pn, 48))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
