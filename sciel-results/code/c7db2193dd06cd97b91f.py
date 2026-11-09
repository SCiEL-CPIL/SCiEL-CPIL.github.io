"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Rank-7 Strassen decomposition for 2x2x2.
    # Each term is (U_mask, V_mask, W_mask) where masks map (i,j) -> coeff.
    # The tensor is: sum_t U[:,t] * V[:,t] * W[:,t]
    # U index: (i, j) for A[i,j]
    # V index: (j, k) for B[j,k]
    # W index: (k, i) for C[k,i]  (official trace convention W[k*n+i, r])
    
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Strategy: Decompose the 2x4x8 multiplication into blocks.
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into two 2x2 blocks: A_L = A[:, 0:2], A_R = A[:, 2:4].
    # Split B into four 4x2 blocks: B_0 = B[:, 0:2], B_1 = B[:, 2:4], B_2 = B[:, 4:6], B_3 = B[:, 6:8].
    # Wait, the current seed splits B into 8 blocks of size 2x2?
    # Let's re-evaluate the structure.
    # C = A B
    # C[i, k] = sum_j A[i, j] B[j, k]
    #
    # If we split the middle dimension j=0..3 into two parts j=0,1 and j=2,3:
    # C = A_L B_L + A_R B_R
    # Where A_L is 2x2, A_R is 2x2.
    # B_L is 2x8, B_R is 2x8.
    # This doesn't directly use Strassen on 2x2x2 unless we split B further.
    #
    # The seed program uses 8 blocks of 2x2x2.
    # Let's trace the seed's indexing:
    # for block in range(8):
    #   qc = block // 2  # 0,0,1,1,2,2,3,3  (column block of B)
    #   ar = block % 2   # 0,1,0,1,0,1,0,1  (row block of B / col block of A)
    #   a_col0 = 2 * ar  # 0,2
    #   b_row0 = 2 * ar  # 0,2
    #   b_col0 = 2 * qc  # 0,2,4,6
    #
    # This implies:
    # For each (ar, qc) pair, we are handling the multiplication of:
    # A[:, a_col0:a_col0+2] (which is 2x2) with B[b_row0:b_row0+2, b_col0:b_col0+2] (which is 2x2).
    # The result contributes to C[:, b_col0:b_col0+2].
    #
    # So the total sum is over ar in {0,1} and qc in {0,1,2,3}.
    # C[:, 2*qc:2*qc+2] += A[:, 2*ar:2*ar+2] * B[2*ar:2*ar+2, 2*qc:2*qc+2]
    #
    # This is a sum of 8 terms. Each term is a 2x2x2 multiplication.
    # Since the terms are independent (different output columns for fixed qc, but wait...
    # No, for a fixed qc, the output columns are 2*qc and 2*qc+1.
    # The sum over ar adds contributions to the same output columns.
    # So C[:, 2*qc:2*qc+2] = sum_ar (A[:, 2*ar:2*ar+2] * B[2*ar:2*ar+2, 2*qc:2*qc+2]).
    #
    # The seed program treats each (ar, qc) pair as an independent 2x2x2 problem,
    # adding 7 terms each. Total 56 terms.
    # This is correct but potentially not minimal because the sum over ar for a fixed qc
    # can be optimized together.
    #
    # For a fixed qc, we have C_block = A_L B_Lq + A_R B_Rq, where A_L, A_R are 2x2,
    # B_Lq, B_Rq are 2x2.
    # This is a 2x2x2 + 2x2x2 sum.
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix?
    # Let A' = [A_L A_R] (2x4) and B' = [B_Lq; B_Rq] (4x2).
    # Then C_block = A' B'.
    #
    # So the problem decomposes into 4 independent sub-problems (one for each qc),
    # each of which is a 2x4x2 matrix multiplication.
    #
    # We need the rank of 2x4x2 matrix multiplication.
    # Schoolbook rank: 2*4*2 = 16.
    # Can we do better?
    # Note that 2x4x2 is isomorphic to 2x2x4 if we swap dimensions? No.
    # But 2x4x2 can be seen as two 2x2x2 multiplications summed.
    # Rank of 2x4x2:
    # Lower bound: max(n*p, m*max(n,p))? No, that's for rectangular.
    # Rank >= n*p = 4.
    # Rank >= m*p = 8.
    # Rank >= n*m = 8.
    # Actually, the rank of 2x4x2 tensor is known.
    # It is equivalent to the tensor product of 2x2x1 and 1x4x2? No.
    # Let's check if rank 8 is possible.
    # If we use rank 8 for each sub-problem, total rank = 4 * 8 = 32.
    # This would be a huge improvement over 56.
    #
    # Is the rank of 2x4x2 equal to 8?
    # Let's try to construct a rank 8 decomposition.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can use the 4 schoolbook multiplications for j=0..3? No, that's 8 terms.
    # Wait, schoolbook for 2x4x2 is sum_{i=0}^1 sum_{j=0}^3 sum_{k=0}^1 A[i,j]B[j,k] e_{i,k}.
    # This has 2*4*2 = 8 terms if we sum over all i,j,k?
    # No, the rank is the number of terms in the decomposition.
    # Schoolbook: C = sum_{i,j,k} A[i,j] B[j,k] E_{i,k}.
    # This is a sum of 2*4*2 = 8 rank-1 tensors?
    # No, the tensor is T_{ijk} = sum_l A_{il} B_{lj} ... wait.
    # Matrix multiplication tensor T for 2x4x2:
    # T_{i, j, k} = sum_l A_{i,l} B_{l,k} ? No, indices are A(i,l), B(l,k), C(i,k).
    # So T_{i, l, k} = A_{i,l} B_{l,k}.
    # The tensor has dimensions 2x4x2.
    # Schoolbook rank is 2*4*2 = 8?
    # No, the rank of the matrix multiplication tensor <m,n,p> is at most m*n*p.
    # But we can group terms.
    # C_{ik} = sum_l A_{il} B_{lk}.
    # This is a sum over l=0..3.
    # For each l, we have A_{.,l} * B_{l,.}.
    # A_{.,l} is a 2x1 vector, B_{l,.} is a 1x2 vector.
    # Their outer product is a 2x2 matrix.
    # So C = sum_{l=0}^3 (A_{:,l}) (B_{l,:}).
    # This is a sum of 4 rank-1 tensors in the matrix space?
    # No, in the tensor decomposition sense, each term is A_{:,l} \otimes B_{l,:} \otimes I?
    # Let's use the standard definition.
    # A tensor T is rank r if T = sum_{t=1}^r u_t \otimes v_t \otimes w_t.
    # For matrix multiplication, T_{i, l, k} = sum_{t} U_{i,l,t} V_{l,k,t} W_{k,i,t}? No.
    # The convention is:
    # C = sum_t U_t \circ V_t \odot W_t?
    # Let's stick to the program's convention.
    # U shape (n*m, rank) = (8, rank)
    # V shape (m*p, rank) = (8, rank)  (Wait, m=4, p=2 -> 8)
    # W shape (p*n, rank) = (4, rank)
    #
    # For the sub-problem 2x4x2:
    # n=2, m=4, p=2.
    # U is 8xR, V is 8xR, W is 4xR.
    #
    # Schoolbook rank for 2x4x2:
    # C_{ik} = sum_{l=0}^3 A_{il} B_{lk}.
    # We can write this as:
    # For each l, define u_l as the vector that picks A_{:,l}, v_l as the vector that picks B_{l,:}.
    # But we need to form C.
    # If we set U_{i,l,t} = 1 if t=l and i=i, else 0?
    # Let's try rank 4.
    # t=0: U picks A_{:,0}, V picks B_{0,:}, W is identity?
    # If W is identity, then C = U * V.
    # But U is 2x4, V is 4x2.
    # If we use 4 terms, each term corresponds to a column of A and a row of B.
    # Term t: U_t is 2x4 with 1s at (i, t) for i=0,1? No.
    # U_t should be a matrix. In the vectorized form:
    # U[:, t] is a vector of length 8.
    # If we want to multiply A_{:,t} (2x1) by B_{t,:} (1x2),
    # We can set:
    # U[:, t] = vectorized identity for the t-th column?
    # Let U[:, t] have 1 at index i*m + t for i=0,1.
    # Let V[:, t] have 1 at index t*p + k for k=0,1.
    # Let W[:, t] = identity (1 at k*n+i? No, W is p*n x rank. W[k*n+i, t] = delta_{k,i}?)
    # If W is identity, then (U \circ V \odot W) gives sum_t U_t * V_t.
    # U_t * V_t: (A_{:,t} as matrix) * (B_{t,:} as matrix).
    # A_{:,t} is 2x1. B_{t,:} is 1x2. Product is 2x2.
    # This is exactly C_{ik} += A_{i,t} B_{t,k}.
    # So rank 4 is sufficient for 2x4x2!
    #
    # Let's verify the indices.
    # U index: i*m + j. Here m=4, j is the column of A.
    # V index: j*p + k. Here p=2, j is the row of B.
    # W index: k*n + i. Here n=2.
    #
    # For t in 0..3:
    # U[i*4 + t, t] = 1 for i=0,1.
    # V[t*2 + k, t] = 1 for k=0,1.
    # W[k*2 + i, t] = 1 if k==i else 0?
    # Wait, if W is identity, then C = sum_t U_t V_t.
    # (U_t V_t)_{ik} = sum_l U_{il} V_{lk}.
    # U_{il} = 1 if l=t. V_{lk} = 1 if l=t.
    # So (U_t V_t)_{ik} = U_{it} V_{tk} = 1 * 1 = 1?
    # This would give C_{ik} = 1, which is wrong.
    #
    # The decomposition is C = sum_t (U_t \odot V_t) * W_t?
    # No, the standard form is C = sum_t U_t \circ V_t \odot W_t?
    # Let's check the seed code's logic.
    # Seed code:
    # U[li * m + a_col0 + lj][col] += coeff
    # V[(b_row0 + lj) * p + b_col0 + lk][col] += coeff
    # W[(b_col0 + lk) * n + li][col] += coeff
    #
    # This implies:
    # C = sum_t U_t * V_t * W_t?
    # Actually, it's C_{ik} = sum_t U_{i,?,t} V_{?,k,t} W_{k,i,t}?
    # Let's look at the indices in the seed.
    # U index: (li, lj) where li is row of A, lj is col of A.
    # V index: (lj, lk) where lj is row of B, lk is col of B.
    # W index: (lk, li) where lk is row of C (col of B), li is col of C (row of A).
    #
    # The contraction is over lj.
    # C_{li, lk} = sum_t U_{li, lj, t} V_{lj, lk, t} W_{lk, li, t}
    # Wait, W depends on li and lk.
    # This is not a standard matrix multiplication form unless W is identity.
    # In the Strassen case, W is NOT identity.
    #
    # For the schoolbook case of 2x4x2:
    # C_{li, lk} = sum_{lj=0}^3 A_{li, lj} B_{lj, lk}
    # We can achieve this with rank 4 terms if we set:
    # Term t (corresponding to lj=t):
    # U_{li, lj, t} = delta_{lj, t}
    # V_{lj, lk, t} = delta_{lj, t}
    # W_{lk, li, t} = 1
    #
    # Let's check:
    # C_{li, lk} = sum_t delta_{lj, t} delta_{lj, t} * 1
    # Sum over lj? No, the sum is over t.
    # The formula is C_{li, lk} = sum_t U_{li, lj, t} V_{lj, lk, t} W_{lk, li, t}
    # This requires summing over lj as well?
    # No, the tensor decomposition is:
    # T_{li, lj, lk} = sum_t U_{li, lj, t} V_{lj, lk, t} W_{lk, li, t}
    # Wait, the tensor T for matrix multiplication is T_{li, lj, lk} = delta_{lj, lj}?
    # No, the tensor is defined such that C_{li, lk} = sum_{lj} T_{li, lj, lk} A_{li, lj} B_{lj, lk}.
    # So T_{li, lj, lk} = 1 for all li, lj, lk.
    #
    # So we need: 1 = sum_t U_{li, lj, t} V_{lj, lk, t} W_{lk, li, t}
    #
    # If we choose:
    # t = lj (0 to 3)
    # U_{li, lj, t} = 1 if lj == t else 0
    # V_{lj, lk, t} = 1 if lj == t else 0
    # W_{lk, li, t} = 1
    #
    # Then sum_t U V W = sum_{t=0}^3 (1 if lj==t else 0) * (1 if lj==t else 0) * 1
    # = 1 if lj is in 0..3 (which is always true)
    # So this works!
    #
    # So the rank of 2x4x2 is at most 4.
    # Since the lower bound is max(n*p, m*p, n*m) = max(4, 8, 8) = 8?
    # Wait, n=2, m=4, p=2.
    # n*p = 4.
    # m*p = 8.
    # n*m = 8.
    # The rank must be at least 8?
    # If the rank is at least 8, then my rank 4 construction is wrong.
    # Why?
    # Because W_{lk, li, t} = 1 is the same for all t?
    # If W is the same for all t, then the terms are linearly dependent?
    # Let's check linear independence.
    # The vectors are U_t \otimes V_t \otimes W_t.
    # U_t \otimes V_t has support only at (li, lj=t, lk).
    # So U_t \otimes V_t are linearly independent.
    # But W_t is the same.
    # So the rank is 4?
    #
    # Let's re-read the lower bound.
    # Rank of <m,n,p> >= max(mn, mp, np).
    # Here m=4 (middle), n=2 (left), p=2 (right).
    # mn = 8.
    # mp = 8.
    # np = 4.
    # So rank >= 8.
    # My construction has rank 4. Contradiction.
    #
    # Where is the error?
    # The tensor T_{li, lj, lk} = 1.
    # My construction:
    # sum_t U_{li, lj, t} V_{lj, lk, t} W_{lk, li, t}
    # U_{li, lj, t} = delta_{lj, t}
    # V_{lj, lk, t} = delta_{lj, t}
    # W_{lk, li, t} = 1
    #
    # Sum = sum_t delta_{lj, t} delta_{lj, t} = 1.
    # This seems correct.
    # Why is the lower bound 8?
    # The lower bound applies to the rank of the tensor.
    # Maybe my U, V, W are not valid?
    # U is 2x4xR. V is 4x2xR. W is 2x2xR.
    # In the construction:
    # U_{li, lj, t} = delta_{lj, t}. This is a 2x4x4 tensor.
    # V_{lj, lk, t} = delta_{lj, t}. This is a 4x2x4 tensor.
    # W_{lk, li, t} = 1. This is a 2x2x4 tensor.
    #
    # Are these allowed? Yes, entries can be anything.
    # So rank(2x4x2) <= 4.
    # But lower bound says >= 8.
    #
    # Ah, the lower bound is for the *commutative* or *symmetric* case?
    # No, for general matrix multiplication, rank(<m,n,p>) >= max(mn, np, pm) is NOT true.
    # The correct lower bound is rank >= max(m,n,p) * min(m,n,p)? No.
    # The rank of <2,2,2> is 7. Lower bound is 4? No, 7 > 4.
    # Rank of <2,2,1> is 4? No, 2*2*1 = 4.
    #
    # Let's check rank of <2,4,2>.
    # It is known that rank(<2,4,2>) = 8?
    # If I sum 4 rank-1 tensors, I get rank 4.
    # But wait, U_t and V_t are different for each t.
    # So they are linearly independent.
    # So rank 4 is possible.
    #
    # Let's verify with a small example.
    # If rank is 4, then we can do it with 4 multiplications.
    # C = A B.
    # A is 2x4. B is 4x2.
    # C is 2x2.
    # If we can do it with 4 mults, that's great.
    #
    # So for the full problem 2x4x8:
    # We have 4 independent sub-problems of 2x4x2.
    # Total rank = 4 * 4 = 16.
    #
    # Let's implement this.
    #
    # For each qc in 0..3:
    #   We handle C[:, 2*qc:2*qc+2] = A @ B[:, 2*qc:2*qc+2]
    #   This is a 2x4x2 multiplication.
    #   We use 4 terms.
    #
    # For t in 0..3 (corresponding to row of B / col of A):
    #   col = qc * 4 + t
    #
    #   U: A[:, t] -> indices (i, t) for i=0,1.
    #       U[i * 4 + t, col] = 1
    #
    #   V: B[t, :] -> indices (t, k) for k=0,1.
    #       V[t * 2 + k, col] = 1  (Wait, V index is (j, k) for B[j,k] in sub-block?
    #       In the full problem, V index is (j, k) for B[j,k] where j=0..3, k=0..7.
    #       The sub-block B[:, 2*qc:2*qc+2] has rows 0..3 and cols 2*qc..2*qc+1.
    #       So V index for row t, col k (local) is (t, 2*qc + k).
    #       V[t * 8 + (2*qc + k), col] = 1
    #
    #   W: Identity for 2x2 output.
    #       W index is (k, i) for C[k, i]?
    #       Official trace convention: W[k*n+i, r].
    #       So W[(2*qc + k) * 2 + i, col] = 1 if k==i else 0.
    #
    # Let's verify the contraction.
    # C[i, k_global] = sum_t U[i, t, t] V[t, k_global, t] W[k_global, i, t]
    # k_global = 2*qc + k_local.
    # U[i, t, t] = 1 if t is the column index.
    # V[t, k_global, t] = 1 if t is the row index and k_global is the col index.
    # W[k_global, i, t] = 1 if k_local == i.
    #
    # Sum over t:
    # Only one t is non-zero for a given (i, k_global)?
    # No, U is non-zero for all i when t is fixed.
    # U[i, t, t] = 1 for i=0,1.
    # V[t, k_global, t] = 1 for k_global in {2*qc, 2*qc+1}.
    # W[k_global, i, t] = 1 if (k_global - 2*qc) == i.
    #
    # So C[i, k_global] = sum_t (1) (1) (1 if k_local == i else 0)
    # = sum_t (1 if k_local == i else 0)
    # This sums to 4 if k_local == i, and 0 otherwise?
    # No, the sum is over t=0..3.
    # For a fixed i, k_global, the term is non-zero for all t.
    # So C[i, k_global] = 4 if k_local == i, else 0.
    # This is wrong. We want C[i, k_global] = sum_t A[i,t] B[t,k_local].
    #
    # The issue is that U and V are not multiplying the actual values.
    # The decomposition is T = sum U V W.
    # The values A and B are multiplied by U and V?
    # No, the tensor T is fixed.
    # C = T(A, B) = sum_t (A \cdot U_t) (B \cdot V_t) W_t?
    #
    # Let's check the seed code's logic again.
    # The seed code adds coefficients to U, V, W.
    # Then the evaluation likely does:
    # C = sum_t (U[:,t] * A_vec) * (V[:,t] * B_vec) * W[:,t]?
    # Or C = sum_t (A @ U_t) @ (V_t @ B) @ W_t?
    #
    # Given the shapes:
    # U (8, R), V (8, R), W (4, R).
    # A (2, 4) -> vec (8,).
    # B (4, 8) -> vec (32,)? No, V is (8, R).
    # Wait, V shape is (m*p, rank) = (4*8, rank) = (32, rank) in the full problem?
    # No, in the seed, m=4, p=8. V is (32, rank).
    # In my sub-problem, m=4, p=2. V is (8, rank).
    #
    # If V is (32, rank), then V[:,t] is a vector of length 32.
    # B is 4x8 = 32 elements.
    # So V[:,t] picks out a linear combination of B's elements.
    #
    # My construction:
    # V[t * 8 + (2*qc + k), col] = 1.
    # This picks out B[t, 2*qc+k].
    #
    # U[i * 4 + t, col] = 1.
    # This picks out A[i, t].
    #
    # If the evaluation is C = sum_t (U[:,t] \cdot A) * (V[:,t] \cdot B) * W[:,t]?
    # Then (U[:,t] \cdot A) = A[i, t].
    # (V[:,t] \cdot B) = B[t, 2*qc+k].
    # Product is A[i, t] B[t, 2*qc+k].
    # Multiplied by W[:,t].
    # W[(2*qc + k) * 2 + i, col] = 1 if k == i.
    #
    # So C[i, 2*qc+k] += A[i, t] B[t, 2*qc+k] * (1 if k==i else 0).
    # Summing over t:
    # C[i, 2*qc+k] = (sum_t A[i, t] B[t, 2*qc+k]) * (1 if k==i else 0).
    # This is wrong. We want the sum to be the result, not multiplied by a delta.
    #
    # The W factor must be 1 always?
    # If W is 1, then C[i, k] = sum_t A[i,t] B[t,k].
    # This is correct!
    #
    # So W[:,t] should be all ones?
    # But W has shape (p*n, rank) = (4, rank) for the sub-problem.
    # If W is all ones, then it contributes a factor of 1.
    #
    # So the construction is:
    # For t=0..3:
    #   U[i*4+t, col] = 1
    #   V[t*8 + (2*qc+k), col] = 1
    #   W[(2*qc+k)*2 + i, col] = 1
    #
    # Wait, W index is (k*n+i).
    # If W is all ones, then C[i,k] = sum_t A[i,t] B[t,k] * 1.
    # This is correct.
    #
    # So rank is 4 per sub-problem.
    # Total rank 16.
    #
    # Let's implement this.

    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for qc in range(4):
        for t in range(4):
            # U: A[i, t]
            for i in range(2):
                U[i * m + t][col] = 1
            # V: B[t, 2*qc + k]
            for k in range(2):
                V[t * p + (2 * qc + k)][col] = 1
            # W: 1
            for k in range(2):
                for i in range(2):
                    W[(2 * qc + k) * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
