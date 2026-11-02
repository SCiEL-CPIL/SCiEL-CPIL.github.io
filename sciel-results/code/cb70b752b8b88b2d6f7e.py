"""Program for rank-minimal tensor decomposition of 2x3x9 matrix multiplication."""
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p

    # We search for structured low-rank decompositions.
    # The tensor T_{i,j,k} = e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (with W index k*n+i)
    # We look for decompositions of the form:
    #   T = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
    # where U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).

    # Strategy: exploit that n=2 is small.
    # For each rank r, we try to find U, V, W with integer entries.
    # We use a constructive approach based on "factoring" the multiplication.

    # Approach: Use the fact that multiplying a 2x3 by 3x9 can be seen as
    # 9 independent 2x3-by-1x3 products, but with shared structure.
    # Actually, output is 2x9. Each output (i,k) = sum_j A[i,j]*B[j,k].
    # So we need to compute 18 dot products of length 3.
    # A rank-r decomposition corresponds to r "trilinear forms" that span the multiplication.

    # Lower bound: at least 18 (one per output entry) but structure allows less?
    # Actually, the bilinear complexity of 2x3 by 3x9:
    # This is the same as computing 2x9 = 2x3 * 3x9.
    # The tensor rank of this specific multiplication.

    # Let's try a constructive decomposition using "block" structure.
    # Since p=9 is large, we can use a "sliding" decomposition.

    # Try ranks from a lower bound upward, using random structured search.
    # Lower bound: max(n*m, m*p, p*n) is not a valid lower bound for rank,
    # but the rank is at least the "slice rank" which is at least max(n*m, m*p, p*n) / min(n,m,p) ... 
    # Actually, a simple lower bound: rank >= n*m*p / max(n,m,p) ... no.
    # The tensor rank is at least the maximum over all slices.
    # For our tensor, fixing i gives an m*p x p*n matrix of rank at most m.
    # So rank >= m = 3. That's weak.

    # Better: use the "sub-tensor" lower bound.
    # Let's just search from a reasonable lower bound.

    # Known: for m x n by n x p, the rank is at least max(m,n,p) * something.
    # For 2x3x9, let's try ranks from 18 upward (since we have 18 outputs).

    # Constructive approach: 
    # We can write the multiplication as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (sum over patterns) ...
    
    # Let's use a "Kronecker product" style decomposition.
    # Decompose the 3x9 block of B into 3x3 blocks: B = [B0, B1, B2] where Bj is 3x3.
    # Then C[i, 3c + k'] = sum_j A[i,j] B[j, 3c+k'] = sum_j A[i,j] (Bc)[j,k']
    # So for each c in {0,1,2}, we compute a 2x3 * 3x3 multiplication, giving 2x3 outputs.
    # Total: 3 separate 2x3x3 multiplications.
    # Rank of 2x3x3 multiplication: at most 2*3*3 = 18, but likely less.
    # Actually, 2x3x3 is the same as 3x3x2 by symmetry.
    # The rank of 2x3x3 tensor: we can compute it.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, let's estimate: at most 18, but we can do better.
    # Actually, we can use the fact that 2x3x3 can be embedded in 3x3x3.
    # Rank of 3x3x3 is at most 23 (or less). But we only need 2x3x3.

    # Let's try a direct constructive decomposition for 2x3x9.
    # We'll use a "repeated block" approach.
    # For each of the 3 blocks of B (each 3x3), we do a 2x3x3 multiplication.
    # If we can do 2x3x3 with rank r0, then total rank is 3*r0.
    # But we can share some terms across blocks if the A matrix is the same.
    # Since A is the same for all blocks, we can potentially share.

    # Let's try to find a decomposition for 2x3x3 first, then extend.
    # For 2x3x3: C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is 6 outputs. 
    # A rank-r decomposition: C[i,k] = sum_{c=1}^r alpha_c(i) beta_c(j) gamma_c(k) * (A dot alpha_c) * (B dot beta_c) ... no, that's not right.
    # The tensor form: T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ... wait, the indices are different.

    # Let me re-read the contract.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition is: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T_{i,j,k} is 1 if we're computing the (i,k) entry of C = A*B, which is sum_j A[i,j] B[j,k].
    # So the tensor T has T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    # The multiplication tensor T^{n,m,p} has T_{(i,j),(j,k),(k,i)} = 1, i.e., T_{i,j,k} in a flattened sense.
    # More precisely, the tensor is: T_{a,b,c} where a indexes (i,j), b indexes (j,k), c indexes (k,i).
    # T_{(i,j),(j,k),(k,i)} = 1, and 0 otherwise.
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i.
    # So T_{i*m+j, j*p+k, k*n+i} = 1.

    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations of (a, b, c) that don't correspond to valid (i,j,k) with matching j and k, the sum should be 0.
    # Wait, the tensor is only non-zero on the "diagonal" where the indices match.
    # Actually, the tensor T is defined on {0,...,n*m-1} x {0,...,m*p-1} x {0,...,p*n-1}.
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # So we need: sum_c U[a,c] V[b,c] W[c,c] ... no, sum_c U[a,c] V[b,c] W[c_row,c] where c_row = c index in W.
    # Let me denote W row index as d = k*n+i.
    # So: sum_c U[a,c] V[b,c] W[d,c] = 1 if (a,b,d) corresponds to valid (i,j,k), else 0.

    # OK so the condition is: for all (a,b,d) in the domain,
    #   sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
    # where T[a,b,d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k, else 0.

    # This is a system of equations. The number of equations is n*m * m*p * p*n = (n*m)*(m*p)*(p*n).
    # For n=2,m=3,p=9: 6*27*18 = 2916 equations.
    # Unknowns: 3 * rank * (n*m + m*p + p*n) ... no, rank * (n*m + m*p + p*n).
    # For rank r: r * (6 + 27 + 18) = 51r unknowns.
    # We need 51r >= 2916, so r >= 57.2, so r >= 58? That can't be right because schoolbook is 54.
    # Wait, I'm confusing myself. The tensor is sparse. Most entries are 0.
    # The non-zero entries are exactly n*m*p = 54.
    # So we have 54 equations that must equal 1, and 2916-54 = 2862 equations that must equal 0.
    # Total 2916 equations, 54 of which are "active".

    # So we need rank r such that 51r >= 2916? No, the system is not full rank.
    # The tensor has a lot of structure, so the effective number of independent constraints is less.

    # Let's just try a random search for small ranks and verify.

    def verify(U, V, W, n, m, p):
        """Check if U, V, W exactly decompose the tensor."""
        rank = len(U[0])
        # Check all active entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[d][c]
                    if s != 1:
                        return False
        # Check some inactive entries (sample)
        for _ in range(200):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            d = random.randint(0, p*n - 1)
            # Check if this is an active entry
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k1 = d // n
            i2 = d % n
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue  # active, already checked
            s = 0
            for c in range(rank):
                s += U[a][c] * V[b][c] * W[d][c]
            if s != 0:
                return False
        return True

    # Try structured decompositions.
    # Idea: use "shift" patterns in k.
    # For n=2, m=3, p=9.
    # Let's try a decomposition where we group k into 3 groups of 3.
    # For group g in {0,1,2}, k = 3*g + k', k' in {0,1,2}.
    # Then for each group, we do a 2x3x3 multiplication.
    # If we use the same U for all groups (since A is the same), and V_g, W_g for each group.
    # Then total rank = rank_of_2x3x3 * 3, but U is shared so it's more efficient.
    # Actually, U is the same, so the total rank is rank_2x3x3 * 3 if we don't share, 
    # but since U is shared, we can have rank = rank_2x3x3 * 3 still, because V and W are different per group.
    # Wait, no. If U is shared, then for each group g, we have U (shared), V_g, W_g.
    # The total rank is sum_g rank_g. If each group uses rank r0, total is 3*r0.
    # But U is the same matrix, so we're not saving on U's rank, just on the number of distinct U matrices.
    # The total rank is still 3*r0.

    # Let's try to find a low-rank decomposition for 2x3x3.
    # For 2x3x3, n=2, m=3, p=3.
    # Active entries: 2*3*3 = 18.
    # Let's try rank 9, 10, 11, etc.

    # Actually, let's try a direct constructive approach for 2x3x9.
    # Use the fact that 9 = 3*3, and use a "Kronecker" decomposition.
    # Decompose the 3x9 matrix B as B = B1 \otimes I_3 + ... no.

    # Let's try a random search with integer entries in {-1, 0, 1}.
    best_rank = schoolbook_rank
    best_result = None

    # Try ranks from a lower bound to schoolbook_rank
    # Lower bound: at least n*m*p / max(n,m,p) = 54/9 = 6? No, that's not a valid lower bound.
    # A valid lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) ... 
    # Actually, the "slice rank" lower bound: for any fixed i, the matrix T_i (m*p x p*n) has rank at most m.
    # So the slice rank is at most m, and the rank is at least the slice rank.
    # But this doesn't give a strong lower bound.

    # Let's just try ranks from 20 to 54.
    random.seed(42)
    for r in range(20, schoolbook_rank + 1):
        # Try a few random structured decompositions
        for attempt in range(50):
            # Construct U, V, W with random integer entries in {-1, 0, 1}
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(r)] for _ in range(p * n)]
            if verify(U, V, W, n, m, p):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break
        if best_result is not None:
            break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
