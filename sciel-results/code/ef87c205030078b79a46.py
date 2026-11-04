"""Matrix multiplication 3,3,8 rank search via rank factorization + determinant method."""
DIMENSIONS = [3, 3, 8]
import itertools
import random
import time
from numpy import array, zeros, dot, outer

n, m, p = 3, 3, 8

def build_matrices(iA, iB, j):
    A = [iA if i == j else (i + 1) % n + 1 for i in range(n)]
    B = [iB if k == j else (k + 1) % p + 1 for k in range(p)]
    return A, B

def mat_mult(A, B, C, i, j, k):
    return A[i][j] * B[j][k] == C[i][k]

def search_rank(rank, timeout, rng):
    start = time.time()
    T = rank  # tensor rank to test

    # Check: for each i-pattern and j, need rank condition
    # Build the reduced 4D tensor using rank factorization of B
    # For each j, A_{i,j} is n×n, B_{i,j} is m×p, C_{i,j} = A_{i,j} @ B_{i,j}
    # Tensor M[k, i, j, l] = C[i,j][k,l]
    # Using rank factorization: C = (A⊗I) * (B^T) effectively
    
    # Build 4D tensor T4[k, i, j, l] = sum_m A[i][m] * B[j][m] * C[k][i] ... 
    # Actually: M[k, i, j, l] = (A_{i,j} * B_{i,j})[k,l]
    # = sum_m A_{i,j}[k,m] * B_{i,j}[m,l]
    
    # For a rank-r decomposition:
    # M = sum_s U_s[k,i] * V_s[j,l]
    # where U_s[k,i] = a_s[k]*b_s[i], V_s[j,l] = c_s[j]*d_s[l]
    # (separable due to structure)
    
    # Lower bound: for fixed j, define M_j[k, i, l] = M[k,i,j,l]
    # This is a 3×3×8 tensor. Its rank ≥ max over i of rank(M_j[·,i,·]) (n×p)
    # and ≥ max over k of rank(M_j[k,·,·]) (n×p)
    # and ≥ max over l of rank(M_j[·,·,l]) (n×n)
    
    # For the FULL tensor M[k,i,j,l] (8×3×3×8), rank ≥
    # max over (i1,i2,i3) of 3 distinct i-values: rank of 3×3×3×8 tensor,
    # which ≥ max_j rank of (matrix from flattening)
    
    # Practical: for each triple of i-values {i1,i2,i3} and each j,
    # M_{k,(i1,i2,i3),j,l} is 8×3×8. 
    # The 3×3×8 tensor for fixed j and 3 i's has rank ≥
    # max over i in triple: rank of 8×8 matrix (k,l -> M[k,i,j,l])
    
    # So global LB = max over {i1,i2,i3}, j of rank(M[k,(i1,i2,i3),j,l])
    # where M[k,(i1,i2,i3),j,l] = stack of M[k,i,j,l] for i in triple -> 8×3×8
    # rank of 8×3×8 tensor ≥ max over 3 i's: rank(8×8 slice)
    
    # Let's compute LB
    LB = 0
    for triple in itertools.combinations(range(n), 3):
        for j in range(m):
            # 8×3×8 tensor
            max_slice_rank = 0
            for i in triple:
                A_i_j, B_i_j = build_matrices(i, j, j)
                # M[k, i, j, l] = sum_m A[i][m] * B[j][m] for k=i, l=j
                # Wait, A_i_j is A evaluated at (i,j), so it's a fixed n×n matrix
                # M[k, i, j, l] = (A_{i,j} @ B_{i,j})[k,l]
                # For fixed (i,j): it's n×p = 3×8 matrix
                # But we need the tensor across k and l for FIXED i and j
                # M[k, i, j, l] = A_{i,j}[k, m'] * B_{i,j}[m', l] (summed over m')
                # This is just the product matrix, which is 3×8
                # So for fixed (i,j), the "slice" is just a 3×8 matrix, rank ≤ 3
                
                # I think I'm overcomplicating. Let me use a different approach.
                pass
    
    # Simpler approach: just search with random factorization and check
    # For rank r, try random integer matrices U (9×r), V (24×r), W (24×r)
    # and check if they satisfy the multiplicative constraint.
    
    # Actually, let's use the structural property:
    # U[i*m+j, s] can be written as u1[i,j,s] (separable in i,j for small entries)
    # V[j*p+k, s] = u2[j,k,s]
    # W[k*n+i, s] = u3[k,i,s]
    
    # For integer entries, try small values
    # The key constraint: for all (i,j,k), sum_s U[i*m+j,s] * V[j*p+k,s] = W[k*n+i,s]... no
    # sum_s U[i*m+j,s] * V[j*p+k,s] * W... 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor equation: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = [i,j,k] indicator... 
    
    # No, the standard form: T_{ijk} = sum_s U_{i,s} * V_{j,s} * W_{k,s}
    # Here indices: i is (row of A = row of C), j is (col of A = row of B), k is (col of B = col of C)
    # T_{i,j,k} = δ (the multiplication tensor)
    # With our flattening: U has rows indexed by (i,j) -> i*m+j
    # V has rows indexed by (j,k) -> j*p+k
    # W has rows indexed by (k,i) -> k*n+i
    
    # So the constraint is: for all (i,j,k):
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 if this corresponds to a valid term
    # Actually no - the tensor T_{i,j,k} for matrix mult (n,m,p) is:
    # T[i,j,k] = 1 (it's always 1, representing that A[i,j]*B[j,k] contributes to C[i,k])
    # And we need: sum_s U_{(i,j),s} * V_{(j,k),s} * W_{(k,i),s} = T[i,j,k] = 1 for all valid (i,j,k)
    # But wait, that's not right either. Let me re-examine.
    
    # The matrix multiplication tensor T ∈ R^{n × m × p} (or rather the structure is different)
    # Standard: C = A @ B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor form: T[i, j, k] with the property that
    # <T, A ⊗ B> = C means T is the 3-tensor with T[i,j,k] such that
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]... this doesn't work directly.
    
    # Correct formulation: The bilinear map (A, B) -> C has tensor T where
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" is T with components indexed by (input pair, output pair):
    # T[(i,j),(j,k),(i,k)] = 1
    # In our flattened form with U indexed by (i,j), V by (j,k), W by (k,i):
    # sum_s U[(i,j),s] V[(j,k),s] W[(k,i),s] should give the coefficient for term (i,j,k)
    # which is 1 for each (i,j,k) triple.
    # So: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for ALL (i,j,k) in ranges.
    
    # That's the constraint! Let me verify with schoolbook:
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k'), V[j*p+k, c] = 1 if c corresponds to (i',j,k),
    # W[k*n+i, c] = 1 if c corresponds to (k'',j,i). For c = (i0,j0,k0):
    # U[i*m+j, c] = 1 iff i=i0 and j=j0
    # V[j*p+k, c] = 1 iff j=j0 and k=k0  
    # W[k*n+i, c] = 1 iff k=k0 and i=i0
    # Product is 1 iff i=i0, j=j0, k=k0, so sum over c gives 1 iff there exists c with all three = 1
    # which is exactly when (i,j,k) = (i0,j0,k0) for some c, i.e., always 1. Correct!
    
    # So constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
    
    # Now search for small rank solutions with integer/half-integer entries.
    
    def check(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True
    
    # Try various approaches:
    # 1. Start from known constructions (e.g., for square case, fast matrix mult)
    # 2. Random search with small integer/half-integer values
    
    # For 3x3x3, the Coppersmith-Winograd gives rank 23 vs 27
    # For 3x3x8, we might do better than 72
    
    # Let's try a systematic approach: fix U and V, solve for W (or vice versa)
    # This is a bilinear system.
    
    # Alternative: use the fact that for fixed j, the problem decomposes
    # For each j, define A_j (n×n effective), B_j (m×p effective)... 
    
    # Let me just do a computational search with random initialization and local refinement.
    
    # Key observation: entries can be half-integers (multiples of 0.5)
    # Let's search over a limited set of values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    best_found = None
    for r in range(20, min(73, 50)):  # try ranks from 20 upward
        if time.time() - start > timeout - 5:
            break
        # For small r, try to find solutions
        # Use structured random search
        found = False
        for attempt in range(10000):
            if time.time() - start > timeout - 3:
                break
            U = [[rng.choice(vals) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.choice(vals) for _ in range(r)] for _ in range(m*p)]
            W = [[rng.choice(vals) for _ in range(r)] for _ in range(p*n)]
            if check(U, V, W, r):
                found = True
                best_found = (r, U, V, W)
                break
        if found:
            break
    
    if best_found:
        return best_found
    
    # If nothing found with random search, try the schoolbook
    return (72, None, None, None)

def construct():
    start = time.time()
    deadline = start + 120
    rng = random.Random(42)
    
    # First, try to find a rank lower than 72
    # Use a more structured search
    
    # For now, return the schoolbook if no better found quickly
    # Let's do a limited structured search
    
    n, m, p = DIMENSIONS
    
    # Try rank 48 (reduction factor 3/4) with structured patterns
    # Pattern: use only 3/4 of the schoolbook terms with clever cancellation
    
    # Actually, let's try a different approach: 
    # For 3x3 matrices, if we can exploit symmetry
    # A common trick: if A and B are symmetric, rank reduces
    # But we need it to work for ALL A, B, so this doesn't help directly.
    
    # Let me just do a careful computational search with a time limit
    timeout = min(deadline - time.time() - 10, 60)
    
    result = search_rank(72, timeout, rng)
    
    if result[0] < 72:
        r, U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
