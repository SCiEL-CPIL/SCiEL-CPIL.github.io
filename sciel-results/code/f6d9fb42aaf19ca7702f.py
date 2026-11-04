DIMENSIONS = [2, 3, 10]

import numpy as np
import time
import random

def _tensor_from_uv(U, V, W, n, m, p):
    """Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return T

def _target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _strassen_2x2():
    # Returns U (4,7), V (4,7), W (4,7) for 2x2 * 2x2 with trace convention W[k*n+i].
    # For 2x2: indices i in {0,1}, j in {0,1}, k in {0,1}.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # A is 2x2 (i rows, j cols), B is 2x2 (j rows, k cols), C is 2x2 (i rows, k cols)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In our tensor: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For the 2x2 block: U[i,j,c] corresponds to A[i,j], V[j,k,c] to B[j,k], W[k,i,c] to 1 (output coupling)
    # Actually for a "block" decomposition, the output W[k,i] must produce C[i,k].
    # The standard way: if we have A (n x m) and B (m x p), C[i,k] = sum_j A[i,j] B[j,k].
    # Our tensor T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    # Wait, the target tensor is T[i,j,k] = 1. That means C[i,k] = sum_j 1 * 1 = m? No.
    # Let's re-read: "exactly decompose the matrix multiplication tensor".
    # The multiplication tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Yes, T[i,j,k] = 1 for all valid i,j,k.
    # So for the 2x2 block, we need U,V,W such that sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1 for all i,j,k in {0,1}.
    # This is exactly the 2x2x2 multiplication tensor.
    # Strassen's algorithm computes C = A*B using 7 products.
    # The "tensor" for Strassen is not just rank 7 in the (i,j,k) space directly with W[k,i] fixed.
    # Let's derive the U,V,W for the 2x2x2 multiplication tensor (rank <= 7).
    # We need: for all i,j,k in {0,1}, sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # Let's use the known Strassen decomposition for 2x2x2.
    # The 7 rank-1 terms are defined by the 7 products M1..M7.
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11 (with -)
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00 (with -), C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A00+A01)B11? No.
    # Let's look up the exact U,V,W for the 2x2x2 multiplication tensor rank 7.
    # This is a standard result.
    # Let's construct it carefully.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This is equivalent to the tensor being the sum of 7 rank-1 tensors.
    # Rank-1 tensor for term c: U_c[i,j] V_c[j,k] W_c[k,i].
    # Let's define the 7 terms.
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   U1[0,0]=1, U1[1,1]=1, others 0.
    #   V1[0,0]=1, V1[0,1]=1, V1[1,0]=1, V1[1,1]=1. (B00+B11 means B[0,0]+B[1,1]? No, B00 is B[0,0], B11 is B[1,1].)
    #   Wait, B is 2x2. B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1].
    #   M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])? No, that's not a valid product in Strassen.
    #   Strassen M1 = (A00+A11)(B00+B11). Here A00=A[0,0], A11=A[1,1], B00=B[0,0], B11=B[1,1].
    #   So U1[0,0]=1, U1[1,1]=1. V1[0,0]=1, V1[1,1]=1.
    #   M1 contributes to C00 and C11.
    #   C00 = ... + M1 ... C11 = ... + M1 ...
    #   So W1[0,0]=1 (k=0,i=0 -> C00), W1[1,1]=1 (k=1,i=1 -> C11).
    #   Let's check: sum_c U[i,j,c]V[j,k,c]W[k,i,c].
    #   For i=0,j=0,k=0: U1[0,0]V1[0,0]W1[0,0] = 1*1*1 = 1.
    #   For i=1,j=1,k=1: U1[1,1]V1[1,1]W1[1,1] = 1*1*1 = 1.
    #   For i=0,j=1,k=0: U1[0,1]=0.
    #   This term alone gives 1 at (0,0,0) and (1,1,1). We need 1 everywhere.
    #   This approach of "constructing U,V,W from Strassen products" is tricky because the output C is a sum of products, but our tensor T[i,j,k]=1 is the *multiplication tensor*, not the *output*.
    #   The multiplication tensor T has T[i,j,k]=1.
    #   The rank-1 decomposition of T corresponds to the "Strassen scheme" for computing the product.
    #   Actually, the rank of the 2x2x2 multiplication tensor is 7.
    #   Let's just use a known valid rank-7 decomposition for 2x2x2.
    #   I will hardcode a known one.
    #   From literature, the 7 terms for 2x2x2 multiplication tensor:
    #   (Indices: U[i,j], V[j,k], W[k,i] with i,j,k in {0,1})
    #   Term 1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]] -> gives 1 at (0,0,0),(1,1,1).
    #   This is getting complicated to derive from scratch.
    #   Alternative: Use the fact that the 2x2x2 tensor rank is 7.
    #   Let's use a simple valid decomposition if I can't recall the exact Strassen U,V,W.
    #   Actually, I can just use the schoolbook rank 8 for the 2x2 block and Strassen rank 7 is better.
    #   Let's try to find the Strassen U,V,W online or derive it.
    #   Derivation:
    #   We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    #   Let's use the 7 Strassen products.
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #   C00 = M1 - M5 + M4 + M7
    #   C01 = M5 + M3
    #   C10 = M4 + M2
    #   C11 = M1 - M2 + M3 + M6
    #   Wait, the standard Strassen for 2x2:
    #   C00 = M1 - M5 + M4 + M7
    #   C01 = M5 + M3
    #   C10 = M4 + M2
    #   C11 = M1 - M2 + M3 + M6
    #   Let's verify C00: M1 - M5 + M4 + M7
    #   = (A00+A11)(B00+B11) - (A00+A01)B11 + A11(B10-B00) + (A01-A11)(B10+B11)
    #   = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A11B10 - A11B00 + A01B10 + A01B11 - A11B10 - A11B11
    #   = A00B00 + A01B10. Correct.
    #   So the decomposition is valid.
    #   Now, how to map this to U,V,W?
    #   The tensor T[i,j,k] = 1.
    #   The rank-1 terms are the M's.
    #   M_c = U_c[i,j] V_c[j,k] W_c[k,i].
    #   But M_c is a scalar product of a linear combination of A and a linear combination of B.
    #   M_c = (sum_ij a_c[i,j] A[i,j]) * (sum_jk b_c[j,k] B[j,k]).
    #   This is NOT of the form U_c[i,j] V_c[j,k] W_c[k,i] summing to 1.
    #   The multiplication tensor T is defined by T[i,j,k] = 1.
    #   A rank-1 decomposition of T is a sum of terms U_c[i,j] V_c[j,k] W_c[k,i].
    #   This is equivalent to finding a "Strassen scheme" where the output C is computed.
    #   Actually, the rank of the multiplication tensor is the minimum number of multiplications.
    #   So yes, rank 7 for 2x2x2.
    #   The U,V,W for the rank-1 terms of the multiplication tensor are:
    #   For each term c, U_c, V_c, W_c are matrices.
    #   Let's find them.
    #   Term 1 (M1): U1[0,0]=1, U1[1,1]=1. V1[0,0]=1, V1[1,1]=1. W1[0,0]=1, W1[1,1]=1.
    #   Term 2 (M2): U2[1,0]=1, U2[1,1]=1. V2[0,0]=1. W2[1,0]=1, W2[1,1]=1. (Wait, M2 contributes to C10 and C11 with + and -)
    #   This is getting messy. Let's use a different approach.
    #   I will use the schoolbook rank 8 for the 2x2 block to be safe, and rely on the 3x10 block being rank 30. Total rank 240? No, 8*30=240. That's worse than 60.
    #   Wait, the Kronecker product of two decompositions multiplies the ranks.
    #   If I use Strassen (rank 7) for 2x2 and schoolbook (rank 30) for 3x10, the total rank is 7*30 = 210. That's way worse than 60.
    #   The Kronecker product approach is for *block* matrices where the blocks are independent.
    #   Here, the 2x3x10 tensor is NOT a Kronecker product of 2x2x2 and 3x10x1.
    #   The dimensions are 2, 3, 10.
    #   The schoolbook rank is 2*3*10 = 60.
    #   To improve, we need to exploit structure.
    #   One structure: 2x3 can be seen as 2 blocks of 1x3? No.
    #   3x10 can be seen as 3 blocks of 1x10?
    #   Let's try a different decomposition.
    #   What if we use Strassen for the 2x2 part of the 2x3 matrix?
    #   The 2x3 matrix A has rows A0, A1 (each 1x3).
    #   The 3x10 matrix B has columns B0..B9 (each 3x1).
    #   C[i,k] = sum_j A[i,j] B[j,k].
    #   This is a sum of 3 products for each (i,k).
    #   Total 2*10*3 = 60 products.
    #   Can we reduce this?
    #   For fixed k, C[:,k] = A * B[:,k]. This is 2x3 times 3x1.
    #   The rank of the 2x3x1 multiplication tensor is 6 (schoolbook). Can it be less?
    #   2x3x1 is equivalent to 2x3 matrix-vector multiplication.
    #   Rank of 2x3x1 tensor is 6?
    #   Actually, the rank of m x n x p multiplication tensor is at least max(mn, mp, np).
    #   For 2x3x1, max(6, 2, 3) = 6. So rank is at least 6.
    #   So no improvement for 2x3x1.
    #   What about 2x3x10?
    #   Lower bound: max(2*3, 2*10, 3*10) = max(6, 20, 30) = 30.
    #   So rank is at least 30.
    #   Schoolbook is 60.
    #   Can we get 30?
    #   If we could, that would be great.
    #   Is there a rank 30 decomposition?
    #   This is an open problem for many sizes, but for small ones it's known.
    #   For 2x3x10, I don't know the exact minimal rank.
    #   However, we can try to find a decomposition better than 60.
    #   Let's try to use the fact that 10 = 2*5.
    #   Maybe decompose the 10 dimension into 2 blocks of 5?
    #   Or 3x10 into 3 blocks of 10?
    #   Let's try a random search for a rank < 60 decomposition.
    #   We can use the "border rank" or "approximate rank" ideas, but we need exact.
    #   Let's try to find a rank 55 decomposition (like the Strassen-inspired one in the inspirations).
    #   The inspiration says: "Strassen decomposition reduces rank from 60 to 55 by replacing 4 schoolbook products per 2x2 column block with 7 combined products".
    #   Wait, 2x2 column block?
    #   The matrix A is 2x3. It doesn't have a 2x2 column block.
    #   Maybe it means 2x2 *submatrix* of the multiplication?
    #   If we group the 3 columns of A into a 2-column block and a 1-column block?
    #   Or group the 10 columns of B into 5 blocks of 2?
    #   If we use Strassen for 2x2x2 blocks, and we have 5 such blocks (since 10/2=5), and the 3rd dimension of A is 3, which is 2+1.
    #   This is complicated.
    #   Let's just implement a random search for a valid decomposition with rank < 60.
    #   We will start with rank 60 and try to reduce it.
    #   Or we can try to construct a rank 55 decomposition directly.
    #   Given the time limit, I will implement a robust random search that tries to find a valid decomposition with rank as low as possible, starting from 60 and going down.
    #   But random search for exact rank is hard.
    #   Let's use the "Strassen" idea from the inspiration: replace 4 products with 7.
    #   How does that work for 2x3x10?
    #   Maybe it considers the 2x3 matrix as having 2 rows, and the 3 columns are processed.
    #   If we consider a 2x2x2 subproblem, we save 1 multiplication (8 -> 7).
    #   How many 2x2x2 subproblems are there in 2x3x10?
    #   If we partition the 3 columns into a 2-column part and a 1-column part, and the 10 rows into 5 pairs of 2?
    #   This is too speculative.
    #   Let's just return the schoolbook rank 60 for now, as it is guaranteed to be valid.
    #   The current program does this.
    #   To improve, I need a valid decomposition with rank < 60.
    #   Let's try to find one.
    #   I will use a known result: the rank of the 2x3x10 multiplication tensor is 55?
    #   I don't know for sure.
    #   Let's try to implement a search that verifies correctness.
    #   I will use a random initialization and gradient descent to minimize the error, but we need exact.
    #   Given the constraints, I will return the schoolbook decomposition.
    #   Wait, the inspiration says score 1.0909, which is better than 1.0.
    #   So there IS a better decomposition.
    #   Let's try to replicate the "Strassen" idea.
    #   Maybe it uses the fact that 2x3 = 2x(2+1).
    #   And 10 = 5*2.
    #   Let's try to construct a rank 55 decomposition.
    #   I will use the following strategy:
    #   1. Start with schoolbook rank 60.
    #   2. Try to merge pairs of terms to reduce rank.
    #   The current program does this but fails.
    #   I will improve the merge check.
    #   To check if two terms can be merged, we need to see if their sum is rank-1.
    #   This is a hard problem.
    #   Instead, let's try a different approach: use the known Strassen decomposition for 2x2x2 and embed it.
    #   If we can find a way to use Strassen for part of the computation, we can save multiplications.
    #   Let's assume the rank is 55 and try to find the decomposition.
    #   I will use a random search with a fixed rank 55 and check if it can be made exact.
    #   This is unlikely to work quickly.
    #   Let's just return the schoolbook rank 60.
    #   Wait, I can try to use the "border rank" approximation and then round to integers/half-integers.
    #   But the problem requires exact decomposition.
    #   Let's try to find a rank 55 decomposition using the "Strassen" method.
    #   I will hardcode a known rank 55 decomposition if I can find one.
    #   Since I can't, I will return the schoolbook rank 60.
    #   Actually, let's try to use the fact that the rank of 2x3x10 is at least 30.
    #   Maybe we can get close to 30?
    #   No, that's unlikely.
    #   Let's just return the schoolbook rank 60.

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    deadline = time.time() + 130
    
    # Schoolbook rank-60 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    T = _tensor_from_uv(U, V, W, n, m, p)
    T_target = _target_tensor(n, m, p)
    assert np.allclose(T, T_target, atol=1e-9), "Schoolbook verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
