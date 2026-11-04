"""Matrix multiplication 2x3x9 tensor decomposition.

We aim for a rank-minimal exact decomposition.  The 2x3x9 tensor is the
Kronecker product of the 2x3x3 matrix multiplication tensor with a 3-dim
identity tensor (the 3x9 matrix B viewed as three independent 3x3 blocks).

If we know a rank-r decomposition of the 2x3x3 tensor, we can get a rank 3r
decomposition of the 2x3x9 tensor by tensoring with the identity (the 3
Kronecker factors for the three blocks).

The known rank of the 2x3x3 matrix multiplication tensor is 7 (same as 3x3x3
is 23, 2x2x2 is 7, 2x3x3 should be at most 5*3=15 trivially, but actually...).
Actually, the rank of the 2x3x3 matrix multiplication tensor: this is the rank
of the tensor T[i,j,k] = delta_{i,j} * delta_{j,k} type structure... no.

Let me reconsider. The 2x3x9 matrix multiplication tensor has entries
T[i, j, k] where i in {0,1}, j in {0,1,2}, k in {0,...,8}.
The entry (i,j,k) is 1 if we multiply A[i,j] * B[j,k].
So T[i, j, k] = 1 for all valid (i,j,k) — wait, no. The tensor entry at
(index_U, index_V, index_W) where index_U = i*m+j, index_V = j*p+k,
index_W = k*n+i. The tensor value is 1 when the indices are consistent,
i.e., the j in U matches the j in V, and the k in V matches the k in W,
and the i in U matches the i in W.

Actually the matrix multiplication tensor M has entries M[a,b,c] where
a = (i,j), b = (j,k), c = (k,i), and M = 1 if the indices are consistent,
0 otherwise. But actually for matrix multiplication, EVERY entry where the
indices are valid is 1. The tensor is:
M[(i,j), (j,k), (k,i)] = 1 for all i in [n], j in [m], k in [p].
And M = 0 for all other index combinations.

So the non-zero entries are exactly n*m*p of them, and each is 1. This is
the "all-ones on the valid slice" tensor.

The rank of this tensor is the matrix multiplication rank. For n=2, m=3, p=9:
This is equivalent to multiplying a 2x3 matrix by a 3x9 matrix.

We can view B as a 3x3 block diagonal (three 3x3 blocks). Then A*B =
[A*block0, A*block1, A*block2]. Each A*block_t is a 2x3 matrix product of a
2x3 and 3x3 matrix. The tensor for 2x3x3 has some rank r_233, and the
Kronecker with a 3-dim identity gives rank 3*r_233 for the 2x3x9 tensor.

The rank of the 2x3x3 matrix multiplication tensor: by symmetry with 3x2x3
and 3x3x2, and known results... The bilinear complexity of 2x3 matrix
multiplication (i.e., 2x3 * 3x3 -> 2x3) is known. Actually, 2x3 * 3x3
is the same as 3x2 * 3x3 by transposition, which has the same rank as
3x3 * 3x2, which is the same as 3x3 * 2x3 by transposition. All of these
are related to the rank of 3x3 * 3x2.

The rank of 3x3 * 3x2 matrix multiplication: this is a known quantity.
It's the same as the rank of the 3x2x3 tensor, which equals the rank of
the 2x3x3 tensor. 

Known: rank(2x2x2) = 7. rank(3x3x3) = 23 (best known upper bound, exact is
unknown but at least 19 and at most 23... actually for 3x3x3 the exact rank
is not known; best known upper bound is 23).

For 2x3x3: We can embed this in 3x3x3, so rank(2x3x3) <= rank(3x3x3) = 23.
Also, we can use the schoolbook: 2*3*3 = 18.
Can we do better than 18 for 2x3x3?

Actually, there's a known result: the rank of the 2x3x3 matrix multiplication
tensor is at most 14, and possibly 13 or 14. But I'm not 100% sure.

Let me think about this differently. I'll implement a search that:
1. Tries to find a low-rank decomposition using a numerical approach.
2. Falls back to schoolbook if it can't find one quickly.

Actually, given the 150s time limit, let me try a more targeted approach:
- The 2x3x9 tensor can be decomposed using the Kronecker structure.
- If I can find ANY decomposition of the 2x3x3 tensor with rank < 18,
  I can scale it up to 2x3x9 with rank = 3 * rank_233.

Let me try to find a rank-14 or rank-15 decomposition of 2x3x3.

For 2x3x3, the schoolbook is rank 18. Let me see if I can find a rank < 18
decomposition.

The tensor T for 2x3x3 has entries T[(i,j),(j,k),(k,i)] = 1 for i in {0,1},
j in {0,1,2}, k in {0,1,2}. That's 18 entries.

I'll try a random search / optimization approach. But exactness is required,
so I need rational/integer entries.

Alternative: use the known 7-rank decomposition of 2x2x2 and extend.

Actually, let me just try to construct a rank-15 decomposition of 2x3x9
directly. The key insight: if 2x3x3 has rank r, then 2x3x9 has rank 3r.
So if I can find 2x3x3 with rank 15, I get 2x3x9 with rank 45 (vs 54).

But wait — can I do even better? The 2x3x9 tensor might have a structure
that allows rank < 3 * rank(2x3x3) because the three blocks share the
same left matrix A.

Actually, the Kronecker structure is: T_239 = T_233 ⊗ I_3 (tensor product
with the 3-dim identity in the third mode). The rank of a tensor product
T ⊗ S is at most rank(T) * rank(S), and the identity tensor I_3 has rank 3
(it's a sum of 3 rank-1 tensors). So rank(T_239) <= rank(T_233) * 3.

But could it be strictly less? In general, rank(T ⊗ S) can be less than
rank(T)*rank(S), but for these specific tensors it's typically equal or
close.

Let me just implement a practical approach:
1. Try to find a low-rank decomposition of 2x3x3 via random search.
2. If found with rank r, construct 2x3x9 with rank 3r.
3. Otherwise fall back to schoolbook rank 54.

For the 2x3x3 decomposition, I'll use the fact that the tensor has 18
non-zero entries and try to find a rank-r decomposition where r < 18.

A rank-r decomposition means: for each (i,j,k) in the valid set,
sum_{t=0}^{r-1} U[i*3+j, t] * V[j*3+k, t] * W[k*2+i, t] = 1,
and for all other (a,b,c), the sum = 0.

This is a system of polynomial equations. I can set up a linear system
if I fix U and V, then solve for W.

Approach:
- Randomly initialize U (6 x r) and V (9 x r) with small integer entries.
- For each W index (k*2+i), the constraint is a linear equation in W.
- Actually, each entry of the tensor gives one equation. There are
  (2*3)*(3*3)*(3*2) = 6*9*6 = 324 entries, of which 18 are 1 and 306 are 0.
- If I fix U and V, each of the 324 entries gives a linear equation in the
  6r entries of W. But W has only 6r entries total (6 rows, r columns).
- So I have 324 linear equations in 6r unknowns. For a solution to exist,
  the system must be consistent.

This is underdetermined when 6r > 324... no wait, 6r is the number of
unknowns and 324 is the number of equations. For r=18, 6*18=108 < 324,
so it's overdetermined. For r=54, 6*54=324, just determined.

Actually I made an error. Let me recount. For 2x3x3:
- U has shape (2*3, r) = (6, r), so 6r entries.
- V has shape (3*3, r) = (9, r), so 9r entries.
- W has shape (3*2, r) = (6, r), so 6r entries.
- Total unknowns: 21r.
- Number of tensor entries: 6 * 9 * 6 = 324.
- Of these, 18 are 1 and 306 are 0.

If I fix U and V, then each of the 324 entries gives a linear equation in
the 6r entries of W. So I need the 324x6r system to be consistent.

For this to have a solution, the 324 equations must be consistent with
6r unknowns. This means the rank of the coefficient matrix (324 x 6r) must
equal the rank of the augmented matrix, AND the system must be consistent.

This is a very constrained problem. For random U, V, the probability that
a solution exists for W is low unless r is large enough.

Let me try a different approach: fix U, V, W all randomly and check if the
tensor matches. This is unlikely to work for small r.

Better approach: Use the structure of the problem. The 2x3x3 tensor can
be seen as: for each j (the middle index), we have a 2x3 * 3x3 = 2x3
multiplication. The three j-slices are independent in the sense that U
and V entries with different j don't interact with each other in the
tensor product structure... actually they do interact because W depends
on both i and k.

Let me try yet another approach: directly search for a rank-r decomposition
of the 2x3x9 tensor using a random initialization and Newton's method,
checking for exactness at the end.

Actually, given the time constraint and the difficulty of finding exact
decompositions, let me try a more pragmatic approach:

1. Use the known rank-7 decomposition of 2x2x2 (Strassen-like).
2. Extend it to 2x3x3 by padding.
3. Then extend to 2x3x9 by the Kronecker structure.

The 2x2x2 rank-7 decomposition is well-known. For 2x3x3, I can use a
combination of the 2x2x2 decomposition (for the top-left 2x2 block of A
and corresponding blocks of B) and schoolbook for the remaining parts.

Actually, let me think about this more carefully. 

For 2x3 * 3x3:
A = [[a00, a01, a02], [a10, a11, a12]]
B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]

The bilinear forms are:
C[0,0] = a00*b00 + a01*b10 + a02*b20
C[0,1] = a00*b01 + a01*b11 + a02*b21
C[0,2] = a00*b02 + a01*b12 + a02*b22
C[1,0] = a10*b00 + a11*b10 + a12*b20
C[1,1] = a10*b01 + a11*b11 + a12*b21
C[1,2] = a10*b02 + a11*b12 + a12*b22

Each C[i,k] is a sum of 3 products. The question is whether we can compute
all 6 outputs using fewer than 18 bilinear multiplications.

Using Strassen-like techniques for the 2x2 * 2x2 part (using columns 0,1
of A and rows 0,1 of B), we can compute 4 products with 7 bilinear
multiplications. Then we need to handle the third column/row.

For the 2x2 * 2x2 part (A[:, 0:2] * B[0:2, :]):
Using Strassen: 7 multiplications for 4 outputs.

For the remaining terms involving column 2 of A and row 2 of B:
C[0,0] += a02*b20
C[0,1] += a02*b21
C[0,2] += a02*b22
C[1,0] += a12*b20
C[1,1] += a12*b21
C[1,2] += a12*b22

And also terms from A[:, 2] * B[0:2, :] are already handled... wait, no.
Let me redo this.

If I split B into B1 = B[0:2, :] (2x3) and B2 = B[2, :] (1x3), then:
C = A[:, 0:2] * B1 + A[:, 2:3] * B2

A[:, 0:2] is 2x2, B1 is 2x3. A[:, 2:3] is 2x1, B2 is 1x3.

A[:, 2:3] * B2: this is (2x1) * (1x3) = 2x3. Each entry is a product:
a02*b2k and a12*b2k for k=0,1,2. That's 6 products, but they share
the a02 and a12 factors. So:
C[0,k] gets a02*b2k, C[1,k] gets a12*b2k.
We can compute a02*b20, a02*b21, a02*b22, a12*b20, a12*b21, a12*b22.
That's 6 bilinear multiplications.

A[:, 0:2] * B1: 2x2 * 2x3. This is like two 2x2 * 2x1 multiplications
side by side. The rank of 2x2 * 2x3 is... we can use Strassen for the
2x2 * 2x2 part (7 mults for 4 outputs) plus additional for the third column.

Actually, 2x2 * 2x3: C = A2x2 * B2x3. C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k].
For k=0: C[0,0]=a00*b00+a01*b10, C[1,0]=a10*b00+a11*b10.
For k=1: C[0,1]=a00*b01+a01*b11, C[1,1]=a10*b01+a11*b11.
For k=2: C[0,2]=a00*b02+a01*b12, C[1,2]=a10*b02+a11*b12.

This is like three 2x2 * 2x1 multiplications. The rank of 2x2 * 2x1 is 4
(schoolbook: 4 products). But can we share? The 2x2 * 2x3 tensor has
6 entries. Schoolbook is 2*2*3 = 12. But we can do better.

Actually, the bilinear complexity of 2x2 * 2x3 matrix multiplication is
known to be at most 10 (and possibly less). Let me think...

For 2x2 * 2x3, we have 6 bilinear forms. Each involves 2 products.
The total schoolbook is 12. Can we share computations?

Let me try to find a decomposition. The tensor has entries:
T[(i,j),(j,k),(k,i)] where i in {0,1}, j in {0,1}, k in {0,1,2}.
That's 2*2*3 = 12 non-zero entries, each equal to 1.

A rank-r decomposition: sum_t U[i*2+j,t] * V[j*3+k,t] * W[k*2+i,t] = T[i,j,k].

I'll just implement a random search for small ranks and see if I can find
anything. Given the time constraints, I'll set a deadline.

Let me write the code now.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _construct_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if val != 1:
                    return False
    # Check that all other entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    if i2 == i:
                        continue
                    # This check is incomplete; let's do it properly
    # More thorough check:
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = 0
                for t in range(rank):
                    val += U[a][t] * V[b][t] * W[c][t]
                # Determine expected value
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    expected = 1
                else:
                    expected = 0
                if val != expected:
                    return False
    return True


def _construct_kronecker(U233, V233, W233):
    """Given a rank-r decomposition of 2x3x3, construct rank-3r for 2x3x9."""
    n, m, p = 2, 3, 3
    r = len(U233[0])
    # The 2x3x9 tensor is T_233 ⊗ I_3 in the k-mode.
    # U has shape (n*m, r) = (6, r). For 2x3x9, U has shape (6, 3r).
    # V has shape (m*p3, r) = (9, r). For 2x3x9, V has shape (m*p9, 3r) = (27, 3r).
    # W has shape (p3*n, r) = (6, r). For 2x3x9, W has shape (p9*n, 3r) = (18, 3r).
    
    # The Kronecker structure: B is 3x9 = three 3x3 blocks.
    # Block t (t=0,1,2) uses V233 rows and W233 entries with an extra factor
    # that selects the block.
    
    # For block t: U is the same, V[j*3+k, t'] -> V233[j*3+k, t'],
    # W[k*2+i, t'] -> W233[k*2+i, t'], but we need to select block t.
    # The selection is done by: in V, the k index goes 0..8, so k = 3*t + k_local.
    # In W, the k index also goes 0..8, so k = 3*t + k_local.
    
    # So for the Kronecker product with identity in the block index:
    # U_239[i*3+j, t*3 + s] = U233[i*3+j, s] for t=0,1,2
    # V_239[j*9+k, t*3+s] = V233[j*3+k_local, s] if k = 3*t + k_local, else 0
    # W_239[k*2+i, t*3+s] = W233[k_local*2+i, s] if k = 3*t + k_local, else 0
    
    U239 = [[0] * (3 * r) for _ in range(6)]
    V239 = [[0] * (3 * r) for _ in range(27)]
    W239 = [[0] * (3 * r) for _ in range(18)]
    
    for i in range(2):
        for j in range(3):
            for s in range(r):
                for t in range(3):
                    U239[i * 3 + j][t * 3 + s] = U233[i * 3 + j][s]
    
    for j in range(3):
        for k in range(9):
            t = k // 3
            k_local = k % 3
            for s in range(r):
                for t2 in range(3):
                    if t2 == t:
                        V239[j * 9 + k][t2 * 3 + s] = V233[j * 3 + k_local][s]
    
    for k in range(9):
        for i in range(2):
            t = k // 3
            k_local = k % 3
            for s in range(r):
                for t2 in range(3):
                    if t2 == t:
                        W239[k * 2 + i][t2 * 3 + s] = W233[k_local * 2 + i][s]
    
    return {"U": U239, "V": V239, "W": W239}


def _try_find_233_decomposition(max_rank=17, time_limit=30):
    """Try to find a rank-r decomposition of the 2x3x3 tensor for r < 18."""
    deadline = time.time() + time_limit
    n, m, p = 2, 3, 3
    # Tensor entries: T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    # We need: sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*2+i, t] = 1 for all i,j,k
    # And = 0 for all other (a,b,c) combinations.
    
    # Strategy: fix U and V randomly, solve for W linearly.
    # For each (a,b,c) where a=i*3+j, b=j*3+k, c=k*2+i:
    #   sum_t U[a,t]*V[b,t]*W[c,t] = expected(a,b,c)
    # This is linear in W.
    
    for r in range(5, max_rank + 1):
        if time.time() > deadline:
            break
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline:
                break
            # Random U, V with entries from {-1, 0, 1} or small integers
            U = np.random.randint(-2, 3, size=(6, r))
            V = np.random.randint(-2, 3, size=(9, r))
            
            # Build linear system for W
            # Number of constraints: 6*9*6 = 324
            # Unknowns: 6*r
            # A[c, :] * w_col = expected_val for each (a,b,c)
            # But the system is: for each (a,b), sum_t U[a,t]*V[b,t]*W[c,t] = expected
            # This couples all (a,b,c) through the same W.
            
            # Actually, for a fixed (a,b), the equation for each c is:
            # sum_t U[a,t]*V[b,t]*W[c,t] = expected(a,b,c)
            # The coefficient of W[c,t] in the equation for (a,b,c') is
            # U[a,t]*V[b,t] if c' == c, else 0.
            # So each (a,b) gives a set of 6 equations (one for each c) that
            # only involve W[c, :] for c=0..5.
            
            # For a fixed (a,b), the 6 equations are:
            # sum_t U[a,t]*V[b,t]*W[c,t] = expected(a,b,c) for c=0..5
            # This is a 6x(6r) linear system for W.
            
            # For the system to have a solution, for each (a,b), the 6x(6r)
            # system must be consistent.
            
            # Let's build the full system:
            # Variables: W[0,0], W[0,1], ..., W[0,r-1], W[1,0], ..., W[5,r-1]
            # Total: 6r variables.
            # Equations: for each (a,b,c), sum_t U[a,t]*V[b,t]*W[c,t] = expected(a,b,c)
            # That's 324 equations.
            
            # Coefficient matrix: 324 x (6r)
            # Row for (a,b,c): has U[a,t]*V[b,t] in position (c, t) for t=0..r-1,
            # and 0 elsewhere.
            
            A_mat = np.zeros((324, 6 * r))
            b_vec = np.zeros(324)
            row = 0
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        c = k * 2 + i
                        expected = 1
                        for t in range(r):
                            A_mat[row, c * r + t] = U[a, t] * V[b, t]
                        b_vec[row] = expected
                        row += 1
            # Now add the zero constraints for all other (a,b,c)
            for a in range(6):
                for b in range(9):
                    for c in range(6):
                        i = a // 3
                        j = a % 3
                        j2 = b // 3
                        k = b % 3
                        k2 = c // 2
                        i2 = c % 2
                        if j == j2 and k == k2 and i == i2:
                            continue  # Already handled
                        for t in range(r):
                            A_mat[row, c * r + t] = U[a, t] * V[b, t]
                        b_vec[row] = 0
                        row += 1
            
            # Solve the least-squares problem and check if residual is ~0
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    W = W_flat.reshape(6, r)
                    # Verify
                    valid = True
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i])
                                if abs(val - 1) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        # Check zero entries
                        for a in range(6):
                            for b in range(9):
                                for c in range(6):
                                    i = a // 3
                                    j = a % 3
                                    j2 = b // 3
                                    k = b % 3
                                    k2 = c // 2
                                    i2 = c % 2
                                    if j == j2 and k == k2 and i == i2:
                                        continue
                                    val = np.sum(U[a] * V[b] * W[c])
                                    if abs(val) > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            U_list = U.tolist()
                            V_list = V.tolist()
                            W_list = W.tolist()
                            # Convert to int if close
                            U_list = [[round(x) for x in row] for row in U_list]
                            V_list = [[round(x) for x in row] for row in V_list]
                            W_list = [[round(x) for x in row] for row in W_list]
                            return U_list, V_list, W_list, r
            except Exception:
                continue
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a better decomposition
    result = None
    
    # First, try to find a 2x3x3 decomposition and Kronecker-expand
    try:
        dec233 = _try_find_233_decomposition(max_rank=17, time_limit=40)
        if dec233 is not None and time.time() < deadline:
            U233, V233, W233, r233 = dec233
            result239 = _construct_kronecker(U233, V233, W233)
            # Verify
            if _verify(result239["U"], result239["V"], result239["W"], 2, 3, 9):
                result = result239
                # Score would be 54 / (3*r233)
                print(f"Found 2x3x3 rank {r233}, 2x3x9 rank {3*r233}")
    except Exception as e:
        print(f"Error in Kronecker approach: {e}")
    
    if result is None and time.time() < deadline:
        # Try direct random search for 2x3x9
        try:
            n, m, p = 2, 3, 9
            # Try ranks from 45 down to 53
            for r in range(45, 54):
                if time.time() > deadline:
                    break
                for trial in range(50):
                    if time.time() > deadline:
                        break
                    U = np.random.randint(-2, 3, size=(6, r))
                    V = np.random.randint(-2, 3, size=(27, r))
                    # Build linear system for W
                    A_mat = np.zeros((6 * 27 * 18, 18 * r))
                    b_vec = np.zeros(6 * 27 * 18)
                    row = 0
                    for a in range(6):
                        for b in range(27):
                            for c in range(18):
                                i = a // 3
                                j = a % 3
                                j2 = b // 9
                                k = b % 9
                                k2 = c // 2
                                i2 = c % 2
                                if j == j2 and k == k2 and i == i2:
                                    expected = 1
                                else:
                                    expected = 0
                                for t in range(r):
                                    A_mat[row, c * r + t] = U[a, t] * V[b, t]
                                b_vec[row] = expected
                                row += 1
                    try:
                        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        if len(residuals) > 0 and residuals[0] < 1e-8:
                            W = W_flat.reshape(18, r)
                            U_list = [[round(x) for x in row] for row in U.tolist()]
                            V_list = [[round(x) for x in row] for row in V.tolist()]
                            W_list = [[round(x) for x in row] for row in W.tolist()]
                            if _verify(U_list, V_list, W_list, 2, 3, 9):
                                result = {"U": U_list, "V": V_list, "W": W_list}
                                print(f"Found direct 2x3x9 rank {r}")
                                break
                    except Exception:
                        continue
                if result is not None:
                    break
        except Exception as e:
            print(f"Error in direct search: {e}")
    
    if result is None:
        result = _construct_schoolbook()
    
    return result

# EVOLVE-BLOCK-END
