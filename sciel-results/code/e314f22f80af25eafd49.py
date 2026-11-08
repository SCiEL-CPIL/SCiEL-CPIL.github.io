"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    n, m, p = 2, 4, 5

    # The multiplication tensor T for (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = 1 if the (i,k) entry of A*B involves A[i,j] * B[j,k]
    # In the standard decomposition: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor entries: T_{i,j,k} where the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # Actually for matrix multiplication, the tensor is:
    # T[i, j, k] = delta_{i,j} * delta_{j,k} ... no.
    # Let me think more carefully.
    # (A B)[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor T has T[i, j, k] = 1 for all valid i, j, k where the product contributes.
    # Actually the standard matrix multiplication tensor M_{n,p,m} (or similar) is:
    # T[i, j, k] = 1 if the (i,k) output depends on A[i,j] and B[j,k]
    # For (n x m) * (m x p): i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to some output)
    # Wait, that's the full rank n*m*p decomposition.
    # The tensor T in U⊗V⊗W decomposition:
    # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # where T[i,j,k] = 1 for all i,j,k in the 2x4x5 grid
    # No wait - for matrix multiplication, T[i,j,k] is not all 1s.
    # Let me re-derive: We want (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is: sum_{i,k} (AB)[i,k] * C[i,k] where C is arbitrary
    # = sum_{i,k} sum_j A[i,j] B[j,k] C[i,k]
    # = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # So the tensor T[i,j,k] = C[i,k]... no, T is fixed.
    # The multiplication tensor is T[i,j,k] = 1 for all i, j, k (with i in [0,n), j in [0,m), k in [0,p))
    # Because the coefficient of A[i,j] * B[j,k] in the bilinear expression is 1 (it contributes to output (i,k)).
    # Actually yes: for matrix multiplication A (n×m) times B (m×p), the tensor is
    # T[i,j,k] = 1 for all i∈{0,...,n-1}, j∈{0,...,m-1}, k∈{0,...,p-1}
    # This is the "multiplication tensor" and it has n*m*p non-zero entries, all equal to 1.
    #
    # Now, a rank-r decomposition: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # T[i, j, k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For the schoolbook case R = n*m*p = 40, each term is a single (i,j,k) with U,V,W = 1.
    #
    # Strategy: Use Strassen for a 2x2x2 block (rank 7) and find a low-rank decomposition
    # for the remaining 2x2x3 block, then combine.
    #
    # But actually, let's think about the structure. We have n=2, m=4, p=5.
    # We can partition the 4 columns of A and 4 rows of B into groups.
    #
    # Alternative: Just use a known low-rank algorithm. The best known for 2x4x5...
    # Actually, let's try a different approach: use the fact that matrix multiplication
    # 2x4x5 can be decomposed using block structure.
    #
    # Let me try: partition A as [A1 | A2] where A1 is 2x2, A2 is 2x2
    # Partition B as [B1; B2] where B1 is 2x5, B2 is 2x5
    # Then AB = A1*B1 + A2*B2
    # Each term is 2x2 times 2x5.
    # A 2x2*2x5 multiplication can be done with rank 7 (Strassen for 2x2) + 3*1 = 10?
    # Actually 2x2*2x5: think of B as 5 columns of 2x1. 
    # (A*B)[i,k] = sum_j A[i,j] B[j,k] for j=0,1.
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The tensor is 2x2x5 with all entries 1. Rank? 
    # This is the tensor T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}.
    # We can write this as: for each k, we have a 2x2 matrix of 1s.
    # The 2x2x1 tensor (all ones) has rank 1 (it's u⊗v⊗w with u=v=w=[1,1]).
    # Wait no: T[i,j,0]=1 for i,j in {0,1}. This is [1,1]⊗[1,1]⊗[1]. Rank 1.
    # So 2x2x5 tensor = sum_{k=0}^{4} [1,1]⊗[1,1]⊗[e_k] where e_k is unit vector in 5D.
    # That gives rank 5. But we can do better by combining.
    # Actually T[i,j,k] = 1 for all i,j,k. This is [1,1]⊗[1,1]⊗[1,1,1,1,1]. Rank 1!
    # Wait, that's the all-ones tensor. But for matrix multiplication 2x2*2x5,
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,4}. Yes! All ones.
    # So the 2x2*2x5 multiplication tensor is just the all-ones 2x2x5 tensor, which has rank 1.
    # That can't be right... Let me recheck.
    #
    # (A*B)[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The bilinear form: sum_{i,k} (AB)[i,k] C[i,k] = sum_{i,k}(A[i,0]B[0,k] + A[i,1]B[1,k])C[i,k]
    # = sum_{i,k} A[i,0] B[0,k] C[i,k] + sum_{i,k} A[i,1] B[1,k] C[i,k]
    # The tensor T[i,j,k] (where i is row of A, j is col of A / row of B, k is col of B):
    # T[i,j,k] = 1 for all i,j,k. Yes, because every product A[i,j]*B[j,k] appears with coefficient 1.
    # So T is the all-ones 2x2x5 tensor = 1_2 ⊗ 1_2 ⊗ 1_5, which has rank 1.
    #
    # Hmm, but that means 2x2*2x5 multiplication can be done with rank 1?? That's clearly wrong.
    # Let me reconsider. The issue is that the "tensor" for matrix multiplication is NOT all ones.
    #
    # I'm confusing myself. Let me be very explicit.
    # For A (2x2) and B (2x5), (AB) is 2x5.
    # (AB)[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # (AB)[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # etc.
    #
    # The multiplication tensor M is a 3-tensor with indices (i, j, k) where:
    # i indexes rows of A (0..1), j indexes cols of A = rows of B (0..1), k indexes cols of B (0..4)
    # M[i,j,k] = 1 for all i,j,k.
    #
    # A rank-1 term u⊗v⊗w gives contribution u[i]*v[j]*w[k] to position (i,j,k).
    # For M[i,j,k]=1 for all i,j,k, we need sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k.
    # One solution: r=1, u=[1,1], v=[1,1], w=[1,1,1,1,1]. Then u[i]*v[j]*w[k]=1 for all i,j,k. ✓
    #
    # So the 2x2*2x5 tensor really does have rank 1?! But that would mean we can compute
    # all 10 products with just 1 scalar multiplication, which is impossible.
    #
    # OH WAIT. I think I'm confusing the tensor representation. Let me look at this differently.
    #
    # The standard matrix multiplication tensor M_{n,p,m} (using the convention from the problem):
    # For multiplying (n×m) matrix A by (m×p) matrix B to get (n×p) matrix C:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The tensor T has dimensions (n*m) × (m*p) × (n*p) in the U⊗V⊗W representation.
    # T[(i*m+j), (j*p+k), (k*n+i)] ... no, that's the schoolbook.
    #
    # Actually, in the U⊗V⊗W decomposition:
    # C[i,k] = sum_r U[i*m + j... 
    # 
    # Let me re-read the problem. The interface is:
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (20, rank)  
    # W: shape (p*n, rank) = (10, rank)
    # W[k*n+i, r] is the convention.
    #
    # The decomposition must satisfy: for all i in [0,n), k in [0,p):
    # (AB)[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r]... 
    #
    # No wait. The standard way: The bilinear map is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * D[i,k] for arbitrary D
    # where T[i,j,k] = 1 for all valid i,j,k.
    #
    # In the U,V,W representation:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # = sum_r (sum_i U[i*m+?, r] A[i,?]) ... 
    #
    # I think the correct formulation is:
    # Define the tensor T with T[i,j,k] = 1 for all i∈[0,n), j∈[0,m), k∈[0,p).
    # Then T = sum_r u_r ⊗ v_r ⊗ w_r where:
    # u_r ∈ R^{n*m} indexed by (i,j) -> u_r[i*m+j]
    # v_r ∈ R^{m*p} indexed by (j,k) -> v_r[j*p+k]  
    # w_r ∈ R^{n*p} indexed by (i,k) -> w_r[k*n+i] (using the W convention from the problem)
    #
    # And we need: T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k.
    #
    # So for 2x2x5: we need u_r[i*2+j] * v_r[j*5+k] * w_r[k*2+i] summing to 1 for all i∈{0,1}, j∈{0,1}, k∈{0,1,2,3,4}.
    #
    # With r=1: u[0]=u[1]=u[2]=u[3]=1, v[0..4]=v[5..9]=... wait v has size m*p=2*5=10.
    # v[j*5+k] for j∈{0,1}, k∈{0,...,4}. Set all v=1. w[k*2+i] for k∈{0,...,4}, i∈{0,1}. w has size p*n=10.
    # Set all w=1. Then sum_r = 1*1*1 = 1. ✓
    #
    # So the 2x2*2x5 tensor has rank 1 in this representation?? That seems too good.
    # 
    # Let me verify with a concrete example. Suppose A = [[a00, a01], [a10, a11]], B = [[b00,b01,b02,b03,b04],[b10,b11,b12,b13,b14]].
    # (AB)[0,0] = a00*b00 + a01*b10
    # (AB)[1,0] = a10*b00 + a11*b10
    #
    # Using rank-1: C[i,k] = sum_r U[i*m+j, r]... 
    # 
    # I think I'm still confused about how U,V,W combine to give C. Let me re-derive.
    #
    # The claim is: if T = sum_r u_r ⊗ v_r ⊗ w_r, then the bilinear computation is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_r (sum_{i,j} u_r[i*m+j] A[i,j]) (sum_{j,k} v_r[j*p+k] B[j,k]) (sum_{i,k} w_r[k*n+i] ...)
    # 
    # No, that's not right either. The standard interpretation:
    # A rank-1 decomposition term u ⊗ v ⊗ w means we compute:
    # s = (sum_{i,j} u[i*m+j] * A[i,j]) * (sum_{j,k} v[j*p+k] * B[j,k])
    # and then add s to the appropriate output C[i,k] with coefficient w[k*n+i].
    # 
    # So: C[i,k] = sum_r w_r[k*n+i] * (sum_{i',j} u_r[i'*m+j] * A[i',j]) * (sum_{j',k'} v_r[j'*p+k'] * B[j',k'])
    #
    # And we need this to equal sum_j A[i,j]*B[j,k] for all i,k.
    #
    # For the all-ones tensor T[i,j,k]=1:
    # We need: sum_r w_r[k*n+i] * (sum_{i',j} u_r[i'*m+j] A[i',j]) * (sum_{j',k'} v_r[j'*p+k'] B[j',k']) = sum_j A[i,j] B[j,k]
    #
    # For n=m=p=2 (2x2x2), this is the Strassen problem. The all-ones 2x2x2 tensor does NOT have rank 1
    # in this sense, because a rank-1 term gives:
    # w[i] * (sum u[i',j] A[i',j]) * (sum v[j',k'] B[j',k'])
    # which is a rank-1 bilinear form, but sum_j A[i,j]B[j,k] is not rank-1 as a bilinear form in (A,B).
    #
    # So my earlier analysis was wrong because I was treating T[i,j,k] as a trilinear form in three separate vectors,
    # but the actual constraint is that the bilinear map (A,B) -> C must match.
    #
    # The correct tensor equation is:
    # For the multiplication tensor M_{n,m,p} (A is n×m, B is m×p, C is n×p):
    # M[(i,j), (j',k'), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} 
    # NO. Let me look at this more carefully.
    #
    # The matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector for A[i,j], e_{j,k} for B[j,k], e_{i,k} for C[i,k].
    #
    # So M has a 1 at position ((i,j), (j,k), (i,k)) and 0 elsewhere.
    # This is NOT the all-ones tensor! It's a very sparse tensor.
    #
    # For n=2, m=4, p=5:
    # M[(i*4+j), (j*5+k), (k*2+i)] = 1, and 0 elsewhere.
    # Total non-zero entries: n*m*p = 40.
    #
    # The schoolbook decomposition uses rank 40: for each (i,j,k), one term with
    # U[i*4+j, r] = 1, V[j*5+k, r] = 1, W[k*2+i, r] = 1.
    #
    # Now, to reduce rank, we need to find u_r, v_r, w_r such that:
    # sum_r u_r[i*4+j] * v_r[j*5+k] * w_r[k*2+i] = delta_{(i,j),(i',j'),(k',k)} ... 
    #
    # More precisely: for each triple (i,j,k) with i∈[0,2), j∈[0,4), k∈[0,5):
    # sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
    # And for all other combinations of indices (not matching the pattern):
    # sum_r U[a, r] * V[b, r] * W[c, r] = 0 where (a,b,c) doesn't correspond to a valid (i,j,k).
    #
    # Actually, the constraint is:
    # For all a ∈ [0,8), b ∈ [0,20), c ∈ [0,10):
    # sum_r U[a,r] * V[b,r] * W[c,r] = M[a,b,c]
    # where M[a,b,c] = 1 if a = i*4+j, b = j*5+k, c = k*2+i for some i,j,k, else 0.
    #
    # This is a system of 8*20*10 = 1600 equations (but only 40 are non-zero) in R*8 + R*20 + R*10 variables.
    #
    # This is a hard algebraic problem. Let me try a constructive approach.
    #
    # Strategy: Use the block structure. Write A as [A1 | A2] (2x2 | 2x2) and B as [B1; B2] (2x5; 2x5).
    # Then AB = A1*B1 + A2*B2.
    #
    # A1*B1: 2x2 * 2x5. The tensor for this is:
    # M1[(i*2+j), (j*5+k), (k*2+i)] = 1 for i∈{0,1}, j∈{0,1}, k∈{0,...,4}.
    # This is a sub-tensor of the full problem.
    #
    # Similarly A2*B2: 2x2 * 2x5, same structure.
    #
    # So the full tensor = M1 + M2 where M1 and M2 are 2x2x5 multiplication tensors.
    #
    # Now, what's the rank of a 2x2x5 multiplication tensor?
    # M1[(i*2+j), (j*5+k), (k*2+i)] = 1 for i,j∈{0,1}, k∈{0,...,4}.
    #
    # Let me try to find a low-rank decomposition of this 2x2x5 tensor.
    # Variables: U has size 4 (i*2+j for i,j∈{0,1}), V has size 10 (j*5+k), W has size 10 (k*2+i).
    #
    # Try rank 5: one term per k value.
    # For fixed k, we need: sum_r U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1 for i,j∈{0,1}.
    # And for k' ≠ k: sum_r U[i*2+j, r] * V[j*5+k', r] * W[k*2+i, r] = 0.
    #
    # With 5 terms (one per k): let term r=k.
    # U[i*2+j, k] * V[j*5+k, k] * W[k*2+i, k] = 1 for i,j∈{0,1}.
    # And U[i*2+j, k'] * V[j*5+k, k'] * W[k*2+i, k'] = 0 for k'≠k.
    #
    # Simple solution: U[i*2+j, k] = 1 for all i,j; V[j*5+k, k] = 1 for all j; W[k*2+i, k] = 1 for all i.
    # For k'≠k: V[j*5+k, k'] = 0.
    # Then: for term r=k: 1*1*1 = 1 ✓
    # For term r=k'≠k: 1*0*1 = 0 ✓
    # So rank 5 works for 2x2x5!
    #
    # Total rank for 2x4x5 = 5 + 5 = 10.
    #
    # But can we do better? The schoolbook is 40. Strassen-based might give something like 7+7=14 for 2x2x2 blocks,
    # but our 2x2x5 blocks have rank 5 each, total 10.
    #
    # Can we get below 10? Let me try to find a rank-4 or rank-5 decomposition of the 2x2x5 tensor.
    #
    # Actually, let me just construct the rank-10 solution and verify it.
    # If I can find something better, great, but 10 is already a 4x improvement over 40.

    # Construct rank-10 decomposition:
    # Split j (column of A / row of B) into j=0,1 (for A1*B1) and j=2,3 (for A2*B2).
    # For A1*B1: i∈{0,1}, j∈{0,1}, k∈{0,...,4}
    # For A2*B2: i∈{0,1}, j∈{2,3}, k∈{0,...,4}
    
    rank = 10
    U = [[0]*rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0]*rank for _ in range(20)]  # m*p = 4*5 = 20
    W = [[0]*rank for _ in range(10)]  # p*n = 5*2 = 10

    # For the first block (j=0,1): terms 0-4
    for k in range(5):
        r = k
        for i in range(2):
            for j in range(2):  # j=0,1
                U[i*4 + j][r] = 1
        for j in range(2):  # j=0,1
            for kk in range(5):
                if kk == k:
                    V[j*5 + kk][r] = 1
        for i in range(2):
            W[k*2 + i][r] = 1

    # For the second block (j=2,3): terms 5-9
    for k in range(5):
        r = 5 + k
        for i in range(2):
            for j in range(2, 4):  # j=2,3
                U[i*4 + j][r] = 1
        for j in range(2, 4):  # j=2,3
            for kk in range(5):
                if kk == k:
                    V[j*5 + kk][r] = 1
        for i in range(2):
            W[k*2 + i][r] = 1

    # Verify correctness
    def verify(U, V, W, n, m, p, rank):
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for valid (i,j,k)
        # and 0 for invalid combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False, f"Valid (i,j,k)=({i},{j},{k}) gives {s}, expected 1"
        
        # Check invalid combinations
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Decode
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    # Valid if j_a == j_b and i_a == i_c
                    if j_a == j_b and i_a == i_c:
                        continue  # valid, already checked
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False, f"Invalid (a,b,c)=({a},{b},{c}) gives {s}, expected 0"
        
        return True, "OK"

    valid, msg = verify(U, V, W, n, m, p, rank)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
