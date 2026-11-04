"""Matrix multiplication 3x3 * 3x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The tensor T[i,j,k] for matrix multiplication satisfies:
    sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_{i,k} C[i,k] (A@B)[i,k] for all A,B,C
    This implies T[i,j,k] = 1 for all i,j,k (the all-ones tensor).
    
    Wait, that's not right. Let me reconsider.
    
    Actually, the bilinear form for matrix multiplication is:
    f(A, B) = A @ B, which is a matrix. The trilinear form is:
    sum_{i,k} C[i,k] * (A@B)[i,k] = sum_{i,k} C[i,k] * sum_j A[i,j] B[j,k]
    = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    
    So the tensor T[i,j,k] must satisfy:
    sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_{i,k} C[i,k] (A@B)[i,k] for all A, B, C
    
    This means T[i,j,k] is not a fixed tensor - it depends on C. That can't be right.
    
    Let me reconsider the tensor decomposition framework. In the standard framework:
    A tensor T in U ⊗ V ⊗ W has rank R if T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    where u_r ∈ U, v_r ∈ V, w_r ∈ W.
    
    For matrix multiplication, the "tensor" is the multiplication map:
    (A, B) -> A@B, which can be viewed as a bilinear map from (R^n × R^m) ⊗ (R^m × R^p) -> R^(n × p).
    
    The tensor associated with this bilinear map is in (R^(n*m))^* ⊗ (R^(m*p))^* ⊗ R^(n*p).
    Its entries are T[i,j,k] where i indexes (row, col) of A, j indexes (row, col) of B, k indexes (row, col) of C.
    
    Specifically, T[(i1,j1),(j2,k2),(i3,k3)] = 1 if j1=j2, i1=i3, k1=k3, else 0.
    Wait, that's also not quite right.
    
    Let me use the convention from the problem:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    U[i*m+j, r] - so row index is i*m+j, where i is row of A, j is col of A / row of B.
    V[j*p+k, r] - row index is j*p+k, where j is row of B, k is col of B.
    W[k*n+i, r] - row index is k*n+i, where k is col of B, i is row of C.
    
    The tensor decomposition says:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T should be the matrix multiplication tensor. What is T[i,j,k]?
    For the bilinear form: sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] should equal sum_{i,k} C[i,k] (A@B)[i,k] when C is arbitrary... no, that's not how it works.
    
    Actually, I think the correct interpretation is:
    The tensor T[i,j,k] is defined such that for any A, B, C:
    sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} A[i,j] U[i*m+j, r]) (sum_{j,k} B[j,k] V[j*p+k, r]) (sum_{i,k} C[i,k] W[k*n+i, r])
    
    This must hold for all A, B, C. By linearity, this means:
    T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    And T[i,j,k] is the matrix multiplication tensor, which is:
    T[i,j,k] = 1 if the indices are "compatible" in the right way.
    
    For matrix multiplication (A@B)[i,k] = sum_j A[i,j] B[j,k], the tensor is:
    T[i,j,k] = 1 for all i, j, k where 0<=i<n, 0<=j<m, 0<=k<p.
    
    Wait, that gives rank 1! That can't be right.
    
    Let me look at this more carefully. The bilinear form is:
    f(A,B,C) = sum_{i,k} C[i,k] (A@B)[i,k] = sum_{i,k} C[i,k] sum_j A[i,j] B[j,k]
    = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    
    So f(A,B,C) = sum_{i,j,k} A[i,j] B[j,k] C[i,k] * 1
    This means T[i,j,k] = 1 for all i,j,k. And indeed, the all-ones tensor has rank 1.
    
    But that contradicts the known result that matrix multiplication has rank at least n*m*p... wait, no, the rank of the tensor is not the same as the number of multiplications.
    
    I think I'm confusing two different concepts. Let me look at this from the Strassen algorithm perspective.
    
    Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications instead of 8. The tensor for 2x2x2x2 matrix multiplication has rank 7.
    
    The tensor is T in (R^4)^* ⊗ (R^4)^* ⊗ R^4, where the 4 dimensions correspond to the 4 entries of each 2x2 matrix.
    
    For 3x3 x 3x7, the tensor is in (R^9)^* ⊗ (R^21)^* ⊗ R^21.
    
    The tensor entries are: T[a, b, c] where a indexes entries of A (0..8), b indexes entries of B (0..20), c indexes entries of C (0..20).
    
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i, j, k. Otherwise T[a,b,c] = 0.
    
    So for a given (i,j,k): a = i*m+j, b = j*p+k, c = k*n+i. There's exactly one (a,b,c) triple for each (i,j,k).
    
    So T has 1s at positions (i*m+j, j*p+k, k*n+i) for all i,j,k, and 0s elsewhere.
    
    The rank of this tensor is what we're trying to minimize.
    
    Now, for the schoolbook algorithm, we use n*m*p = 3*3*7 = 63 multiplications. Each multiplication corresponds to one (i,j,k) triple.
    
    The schoolbook decomposition has rank 63, with U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col corresponding to (i,j,k).
    
    Can we do better? Yes, potentially. The best known bound for 3x3x3x3 is rank 19 (or 20 with some schemes). For 3x3x3x7, we might be able to do better than 63.
    
    Let me try a structured approach:
    - Split the 7 columns of B into groups: 3+3+1
    - For each 3x3 x 3x3 block, use a rank-19 or rank-20 decomposition
    - For the 3x3 x 3x1 block, use rank-3 (schoolbook) or possibly less
    
    Total rank: 19 + 19 + 3 = 41 (using rank-19 for 3x3x3x3)
    Or: 20 + 20 + 3 = 43
    
    Actually, I can try to find a rank-19 decomposition for 3x3x3x3. Let me think about known algorithms.
    
    For 3x3 matrix multiplication, the best known rank is 19 (achieved by various schemes). Let me try to construct one.
    
    Actually, let me just try to use a simple approach: for each of the 7 columns, compute A @ B[:,k] using 3 multiplications (one per row of A). That gives rank 7*3 = 21.
    
    Wait, that's not quite right. For each column k, we need to compute 3 entries of C[:,k], each requiring 3 multiplications. So that's 7*3*3 = 63 multiplications total. That's the schoolbook.
    
    To do better, we need to share multiplications across columns.
    
    Let me try a different approach. For each row i of A and each column k of B, C[i,k] = sum_j A[i,j] B[j,k].
    
    If I fix i and vary k, I'm computing a linear combination of the rows of B. If I fix k and vary i, I'm computing a linear combination of the columns of A.
    
    One idea: use a fast 3x3x3x3 multiplication for the first 3 columns, then another for the next 3, then 3 separate multiplications for the last column.
    
    Let me look up or derive a rank-19 scheme for 3x3x3x3.
    
    Actually, let me just try the simple rank-21 approach first and see if it's valid, then try to improve.
    
    For rank 21: for each k (0..6) and each j (0..2), we have one multiplication. U[i*m+j, r] = 1 if r corresponds to (k,j) and i is anything... no, that doesn't work.
    
    Let me think about this differently. The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    
    For a fixed j, the entries where b = j*p+k (for any k) and a = i*m+j (for any i) and c = k*n+i (for any i,k) form a sub-tensor.
    
    For fixed j, this sub-tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,k. That's a 3x7 all-ones tensor, which has rank 1 (in the appropriate sense).
    
    So for each j, we can use 1 "multiplication" to handle all (i,k) pairs for that j. That gives rank 3 total!
    
    Wait, is that right? Let me verify.
    
    For j=0: U[i*m+0, r0] = 1 for all i (where r0 is the rank element for j=0). V[0*p+k, r0] = 1 for all k. W[k*n+i, r0] = 1 for all i,k.
    
    Then T[i*m+0, 0*p+k, k*n+i] = sum_r U[i*m+0, r] V[0*p+k, r] W[k*n+i, r] = U[i*m+0, r0] V[0*p+k, r0] W[k*n+i, r0] = 1*1*1 = 1.
    
    And for other (a,b,c) not matching this pattern, the contribution from r0 is 0.
    
    So yes! For each j, we can use one rank element. Total rank = 3.
    
    Let me verify this more carefully. The tensor T[a,b,c] should be 1 iff there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i.
    
    With rank 3, using r=j for j=0,1,2:
    T[a,b,c] = sum_{j=0}^2 U[a, j] V[b, j] W[c, j]
    
    U[a, j] = 1 iff a = i*m+j for some i (i.e., a mod m == j)
    V[b, j] = 1 iff b = j*p+k for some k (i.e., b mod p == j*p, or b div p == j)
    W[c, j] = 1 iff c = k*n+i for some i,k (but we need consistency with a and b)
    
    Wait, W[c, j] should be 1 iff c corresponds to some (i,k) such that the triple (a=i*m+j, b=j*p+k, c=k*n+i) is valid.
    
    For a fixed j, the valid (i,k) pairs give c = k*n+i. So W[c, j] = 1 iff c = k*n+i for some i in [0,n) and k in [0,p).
    
    But c ranges over [0, n*p) = [0, 21). And k*n+i ranges over [0, 21) for k in [0,7) and i in [0,3). Every c in [0,21) can be written as k*n+i uniquely.
    
    So W[c, j] = 1 for all c and all j? That can't be right.
    
    Let me reconsider. For the rank element j, we want:
    T[a,b,c] += U[a,j] V[b,j] W[c,j]
    
    And we want this sum to be 1 iff (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,k.
    
    For a given j, the set of valid (a,b,c) is {(i*m+j, j*p+k, k*n+i) : i in [0,3), k in [0,7)}.
    
    U[a,j] = 1 iff a = i*m+j for some i, i.e., a mod m == j.
    V[b,j] = 1 iff b = j*p+k for some k, i.e., b div p == j (and b mod p < p).
    W[c,j] = 1 iff c = k*n+i for some i,k consistent with a and b.
    
    But i is determined by a: i = (a - j) / m (if a mod m == j).
    k is determined by b: k = b - j*p (if b div p == j).
    Then c should be k*n+i.
    
    So W[c,j] should be 1 iff c = (b - j*p)*n + (a - j)/m, but W doesn't depend on a or b.
    
    This approach doesn't work with rank 3. The issue is that W[c,j] needs to know both i and k, but it only sees c.
    
    Let me go back to the schoolbook approach and try to find a better structured decomposition.
    
    For now, let me just return the schoolbook solution with rank 63, which is known to be valid.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition: rank = n*m*p = 63
    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
