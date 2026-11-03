"""Rewritten program for 3x4x5 matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# Known optimal rank for 2x2 matrix multiplication is 7 (Strassen)
# We can try to exploit structure: 3x4x5 = (1+2)x(2+2)x(2+3) or similar partitions
# But a more direct approach: use the fact that rank(3x4x5) <= rank(3x3x3)*... 
# Actually, let's think about known bounds.
# The schoolbook rank is 3*4*5 = 60.
# We know that rank of matrix multiplication tensor [m,n,p] satisfies:
#   rank([m,n,p]) <= rank([m1,n1,p1]) + rank([m2,n2,p2]) if we can partition.
# More usefully, we can embed smaller known decompositions.

# Known results:
# rank([2,2,2]) = 7 (Strassen)
# rank([3,3,3]) is between 19 and 23. Best known upper bound is 23 (Landsberg), lower bound 19.
# Actually, the best known upper bound for [3,3,3] is 23.
# 
# For [3,4,5]:
# We can write 3 = 1+2, 4 = 2+2, 5 = 2+3
# But it's cleaner to use: rank([m,n,p]) <= rank([m,n,p']) + rank([m,0,p-p'])... no.
# 
# Better: use the block decomposition. If we partition the matrices:
# A is 3x4, B is 4x5, C = AB is 3x5.
# We can split A into A1 (3x2) and A2 (3x2), B into B1 (2x5) and B2 (2x5).
# Then C = A1*B1 + A2*B2.
# This gives rank([3,2,5]) + rank([3,2,5]).
# 
# Alternatively, we can use the identity:
# rank([m,n,p]) <= rank([m,n,p1]) + rank([m,n-p1,p2]) ... this isn't standard.
# 
# The standard subadditivity is:
# rank([m1+m2, n, p]) <= rank([m1, n, p]) + rank([m2, n, p])
# rank([m, n1+n2, p]) <= rank([m, n1, p]) + rank([m, n2, p])
# rank([m, n, p1+p2]) <= rank([m, n, p1]) + rank([m, n, p2])
# 
# So: rank([3,4,5]) <= rank([3,2,5]) + rank([3,2,5]) = 2*rank([3,2,5])
# And rank([3,2,5]) <= rank([3,2,2]) + rank([3,2,3])
# rank([3,2,2]) <= rank([1,2,2]) + rank([2,2,2]) = 4 + 7 = 11? 
# Wait, rank([1,n,p]) = n*p (since it's just a linear map from n*p to 1... no, it's a bilinear form).
# Actually [1,n,p] corresponds to multiplying a 1xn matrix by an nxp matrix. The rank is n*p because each output entry is independent.
# So rank([1,2,2]) = 4.
# rank([2,2,2]) = 7.
# So rank([3,2,2]) <= 4 + 7 = 11.
# 
# rank([3,2,3]) <= rank([1,2,3]) + rank([2,2,3]) = 6 + rank([2,2,3])
# rank([2,2,3]) <= rank([2,2,1]) + rank([2,2,2]) = 4 + 7 = 11? 
# rank([2,2,1]) = 4 (multiply 2x2 by 2x1, output is 2x1, each of 2 entries is sum of 2 products... wait)
# [m,n,p]: A is m x n, B is n x p, C is m x p.
# rank([2,2,1]): A is 2x2, B is 2x1, C is 2x1. C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0].
# This is a bilinear map from (2,2) x (2,1) to (2,1). The tensor has rank 4? Let's see:
# T = e0⊗f0⊗g0 + e0⊗f1⊗g0 + e1⊗f0⊗g1 + e1⊗f1⊗g1. Yes, rank 4.
# So rank([2,2,3]) <= rank([2,2,1]) + rank([2,2,2]) = 4 + 7 = 11.
# Then rank([3,2,3]) <= 6 + 11 = 17.
# Then rank([3,2,5]) <= 11 + 17 = 28.
# Then rank([3,4,5]) <= 28 + 28 = 56.
# 
# Can we do better? Let's try a different partition.
# rank([3,4,5]) <= rank([3,4,2]) + rank([3,4,3])
# rank([3,4,2]) <= rank([1,4,2]) + rank([2,4,2]) = 8 + rank([2,4,2])
# rank([2,4,2]) <= rank([2,2,2]) + rank([2,2,2]) = 7 + 7 = 14
# So rank([3,4,2]) <= 8 + 14 = 22
# rank([3,4,3]) <= rank([1,4,3]) + rank([2,4,3]) = 12 + rank([2,4,3])
# rank([2,4,3]) <= rank([2,2,3]) + rank([2,2,3]) = 11 + 11 = 22
# So rank([3,4,3]) <= 12 + 22 = 34
# Total: 22 + 34 = 56. Same.
# 
# Let's try: rank([3,4,5]) <= rank([3,1,5]) + rank([3,3,5])
# rank([3,1,5]) = 15 (each of 3*5=15 outputs is a single product)
# rank([3,3,5]) <= rank([3,3,2]) + rank([3,3,3])
# rank([3,3,2]) <= rank([1,3,2]) + rank([2,3,2]) = 6 + rank([2,3,2])
# rank([2,3,2]) <= rank([2,1,2]) + rank([2,2,2]) = 4 + 7 = 11
# So rank([3,3,2]) <= 6 + 11 = 17
# rank([3,3,3]) <= 23 (best known upper bound)
# So rank([3,3,5]) <= 17 + 23 = 40
# Total: 15 + 40 = 55.
# 
# Can we improve rank([3,3,3])? The best known upper bound is 23. Let's assume we can achieve 23.
# Actually, let's verify: is 23 achievable? Yes, Landsberg showed rank([3,3,3]) <= 23.
# 
# So we can get rank 55. Let's try to construct this.
# 
# But wait, can we do even better? 
# rank([3,4,5]) <= rank([2,4,5]) + rank([1,4,5]) = rank([2,4,5]) + 20
# rank([2,4,5]) <= rank([2,4,2]) + rank([2,4,3]) = 14 + 22 = 36
# Total: 36 + 20 = 56. Worse.
# 
# Another: rank([3,4,5]) <= rank([3,3,5]) + rank([3,1,5]) = 40 + 15 = 55. Same as before.
# 
# Let's try to optimize rank([3,3,5]) further.
# rank([3,3,5]) <= rank([3,3,1]) + rank([3,3,4])
# rank([3,3,1]) = 9
# rank([3,3,4]) <= rank([3,3,2]) + rank([3,3,2]) = 17 + 17 = 34
# Total: 9 + 34 = 43. Worse than 40.
# 
# rank([3,3,5]) <= rank([3,1,5]) + rank([3,2,5]) = 15 + 28 = 43. Worse.
# 
# So 40 seems best for [3,3,5] with this approach.
# 
# Let's try a completely different approach: use the known rank-7 decomposition for [2,2,2] and combine.
# 
# Actually, let me just implement the rank-55 construction and see if it works.
# Then I can try to improve further.

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) where W[k*2+i, c] follows the convention.
    """
    # Strassen's algorithm for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: A00, A01, A10, A11 (index: i*2+j)
    # V rows: B00, B01, B10, B11 (index: j*2+k)
    # W rows: C00, C01, C10, C11 (index: k*2+i)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The multiplication tensor T[i,j,k] = delta[i,k]... no.
    # T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
    # So T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], else 0.
    # Actually, the tensor is T = sum_{i,j,k} e_i^U ⊗ e_j^V ⊗ e_k^W where the index is (i,j,k) -> C[i,k] += A[i,j]*B[j,k].
    # In the decomposition, we have T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c].
    # So for each (i,j,k), T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c].
    # And T[i,j,k] should be 1 if this corresponds to a valid multiplication term, i.e., always 1 for the standard multiplication.
    # Wait, the multiplication tensor is: for each i in [n], j in [m], k in [p], we have a term A[i,j]*B[j,k] contributing to C[i,k].
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid (i,j,k).
    # Actually no, the tensor is defined on the space of (A, B, C) such that C = A*B.
    # The tensor is T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the coefficient is 1.
    # In the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # U[i*m+j, c] corresponds to A[i,j].
    # V[j*p+k, c] corresponds to B[j,k].
    # W[k*n+i, c] corresponds to C[i,k].
    # So the tensor entry for (i, j, k) is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This should equal 1 for all i, j, k.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank-55 construction:
    # rank([3,4,5]) <= rank([3,1,5]) + rank([3,3,5])
    # rank([3,3,5]) <= rank([3,3,2]) + rank([3,3,3])
    # rank([3,3,2]) <= rank([1,3,2]) + rank([2,3,2])
    # rank([2,3,2]) <= rank([2,1,2]) + rank([2,2,2])
    
    # Let's build this step by step.
    # First, get the rank-7 decomposition for [2,2,2].
    U222, V222, W222 = _strassen_2x2()
    
    # rank([2,1,2]) = 4: schoolbook
    U212 = np.eye(4)  # A is 2x1, so 2*1=2 entries... wait, U has shape (n*m, rank) = (2*1, 4) = (2,4)
    # A[i,j] where i in [2], j in [1]. So 2 entries.
    # B[j,k] where j in [1], k in [2]. So 2 entries.
    # C[i,k] where i in [2], k in [2]. So 4 entries.
    # U: (2, 4), V: (2, 4), W: (4, 4)
    U212 = np.zeros((2, 4))
    V212 = np.zeros((2, 4))
    W212 = np.zeros((4, 4))
    for c in range(4):
        i = c // 2
        k = c % 2
        U212[i, c] = 1
        V212[0, c] = 1
        W212[k*2+i, c] = 1
    
    # Combine [2,1,2] and [2,2,2] to get [2,3,2]:
    # [2,3,2] = [2,1,2] ⊕ [2,2,2]
    U232 = np.hstack([U212, U222])  # shape (2, 4+7) = (2, 11)
    V232 = np.vstack([V212, V222])  # shape (1*2 + 2*2, 11) = (6, 11)... wait
    # V for [2,1,2] has shape (m*p, rank) = (1*2, 4) = (2, 4)
    # V for [2,2,2] has shape (m*p, rank) = (2*2, 7) = (4, 7)
    # For [2,3,2]: m=3, p=2, so V has shape (3*2, rank) = (6, rank)
    # The first 2 rows of V correspond to j=0 (from [2,1,2]), the next 4 rows correspond to j=1,2 (from [2,2,2])
    V232 = np.zeros((6, 11))
    V232[0:2, 0:4] = V212
    V232[2:6, 4:11] = V222
    W232 = np.hstack([W212, W222])  # shape (4, 11)
    
    # Verify [2,3,2]
    if not _verify(U232, V232, W232, 2, 3, 2):
        pass  # Should be valid
    
    # rank([1,3,2]) = 6: schoolbook
    U132 = np.zeros((1*3, 6))  # (3, 6)
    V132 = np.zeros((3*2, 6))  # (6, 6)
    W132 = np.zeros((2*1, 6))  # (2, 6)
    for c in range(6):
        j = c // 2
        k = c % 2
        U132[j, c] = 1
        V132[j*2+k, c] = 1
        W132[k, c] = 1
    
    # Combine [1,3,2] and [2,3,2] to get [3,3,2]:
    U332 = np.vstack([U132, U232])  # (3+2, 17) = (5, 17)... wait
    # U for [1,3,2] has shape (1*3, 6) = (3, 6)
    # U for [2,3,2] has shape (2*3, 11) = (6, 11)
    # For [3,3,2]: n=3, m=3, so U has shape (3*3, rank) = (9, rank)
    # The first 3 rows correspond to i=0 (from [1,3,2]), the next 6 rows correspond to i=1,2 (from [2,3,2])
    U332 = np.zeros((9, 17))
    U332[0:3, 0:6] = U132
    U332[3:9, 6:17] = U232
    V332 = np.hstack([V132, V232])  # (6, 17)
    W332 = np.hstack([W132, W232])  # (2, 17)
    
    # Verify [3,3,2]
    if not _verify(U332, V332, W332, 3, 3, 2):
        pass
    
    # Now we need [3,3,3] with rank 23. This is complex. Let's use schoolbook for now: rank 27.
    # If we use schoolbook for [3,3,3], total rank = 6 + 17 + 27 = 50 for [3,3,5], then + 15 for [3,1,5] = 65.
    # That's worse than 60. So we need a better [3,3,3] decomposition.
    
    # Let's just use schoolbook for [3,3,3] for now and see if we can find a better overall strategy.
    # Actually, let's try a different partition that avoids [3,3,3].
    
    # Alternative: rank([3,4,5]) <= rank([3,2,5]) + rank([3,2,5]) = 2 * rank([3,2,5])
    # rank([3,2,5]) <= rank([3,2,2]) + rank([3,2,3])
    # rank([3,2,2]) <= rank([1,2,2]) + rank([2,2,2]) = 4 + 7 = 11
    # rank([3,2,3]) <= rank([1,2,3]) + rank([2,2,3]) = 6 + rank([2,2,3])
    # rank([2,2,3]) <= rank([2,2,1]) + rank([2,2,2]) = 4 + 7 = 11
    # So rank([3,2,3]) <= 6 + 11 = 17
    # rank([3,2,5]) <= 11 + 17 = 28
    # Total: 28 + 28 = 56.
    
    # Let's construct rank-56.
    
    # rank([1,2,2]) = 4
    U122 = np.zeros((2, 4))
    V122 = np.zeros((4, 4))
    W122 = np.zeros((2, 4))
    for c in range(4):
        j = c // 2
        k = c % 2
        U122[j, c] = 1
        V122[j*2+k, c] = 1
        W122[k, c] = 1
    
    # rank([2,2,2]) = 7 (Strassen)
    # Already have U222, V222, W222
    
    # Combine to get [3,2,2]:
    U322 = np.zeros((6, 11))
    U322[0:2, 0:4] = U122
    U322[2:6, 4:11] = U222
    V322 = np.hstack([V122, V222])  # (4, 11)
    W322 = np.hstack([W122, W222])  # (2, 11)
    
    # rank([1,2,3]) = 6
    U123 = np.zeros((2, 6))
    V123 = np.zeros((6, 6))
    W123 = np.zeros((3, 6))
    for c in range(6):
        j = c // 3
        k = c % 3
        U123[j, c] = 1
        V123[j*3+k, c] = 1
        W123[k, c] = 1
    
    # rank([2,2,1]) = 4
    U221 = np.zeros((4, 4))
    V221 = np.zeros((2, 4))
    W221 = np.zeros((2, 4))
    for c in range(4):
        i = c // 2
        j = c % 2
        U221[i*2+j, c] = 1
        V221[j, c] = 1
        W221[i, c] = 1
    
    # Combine to get [2,2,3]:
    U223 = np.hstack([U221, U222])  # (4, 11)
    V223 = np.hstack([V221, V222])  # (2, 11)
    W223 = np.vstack([W221, W222])  # (4, 11)... wait
    # W for [2,2,1] has shape (p*n, rank) = (1*2, 4) = (2, 4)
    # W for [2,2,2] has shape (p*n, rank) = (2*2, 7) = (4, 7)
    # For [2,2,3]: p=3, n=2, so W has shape (3*2, rank) = (6, rank)
    # The first 2 rows correspond to k=0 (from [2,2,1]), the next 4 rows correspond to k=1,2 (from [2,2,2])
    W223 = np.zeros((6, 11))
    W223[0:2, 0:4] = W221
    W223[2:6, 4:11] = W222
    
    # Combine [1,2,3] and [2,2,3] to get [3,2,3]:
    U323 = np.zeros((6, 17))
    U323[0:2, 0:6] = U123
    U323[2:6, 6:17] = U223
    V323 = np.hstack([V123, V223])  # (6, 17)
    W323 = np.hstack([W123, W223])  # (3, 17)
    
    # Combine [3,2,2] and [3,2,3] to get [3,2,5]:
    U325 = np.hstack([U322, U323])  # (6, 28)
    V325 = np.hstack([V322, V323])  # (4, 28)
    W325 = np.hstack([W322, W323])  # (5, 28)
    
    # Verify [3,2,5]
    if not _verify(U325, V325, W325, 3, 2, 5):
        pass
    
    # Now combine two [3,2,5] to get [3,4,5]:
    U345 = np.hstack([U325, U325])  # (6, 56)
    V345 = np.vstack([V325, V325])  # (8, 56)
    W345 = np.hstack([W325, W325])  # (5, 56)
    
    # Verify [3,4,5]
    if not _verify(U345, V345, W345, 3, 4, 5):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time
    if time.time() > deadline:
        # Return schoolbook if we're running out of time
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U345.tolist(), "V": V345.tolist(), "W": W345.tolist()}
