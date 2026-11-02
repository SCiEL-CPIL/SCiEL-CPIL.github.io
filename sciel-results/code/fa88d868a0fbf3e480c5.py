"""Tensor decomposition search for 4x4 * 4x6 matrix multiplication."""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 6]
n, m, p = DIMENSIONS

# Time budget
DEADLINE = 120.0
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The tensor T[i,j,k] for multiplying A[i,:] * B[:,k] should be 1 if the
    output C[i,k] = sum_j A[i,j]*B[j,k].
    
    The decomposition says: C[i,k] = sum_r U[i*m + ?... 
    Actually, let me think about this more carefully.
    
    For matrix multiplication (n x m) * (m x p) -> (n x p):
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = delta (it's 1 for all valid i,j,k in the standard basis).
    
    A rank-r decomposition: T[i,j,k] = sum_{r=1}^R U[i,j,r] * V[j,k,r] * W[r]
    
    But the task says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    So the convention must be:
    C[i, k] = sum_r U[i*m + j... no wait, U has n*m rows.
    
    Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    And "W[k*n+i,r]" is the trace convention.
    
    So the decomposition is:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    where a indexes into (i,j) pairs: a = i*m + j (for U, n*m entries)
    b indexes into (j,k) pairs: b = j*p + k (for V, m*p entries)  
    c indexes into (k,i) pairs: c = k*n + i (for W, p*n entries)
    
    And T[i,j,k] = 1 (the standard matrix multiplication tensor).
    
    So we need: for all i, j, k:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for any (a,b,c) that doesn't correspond to valid (i,j,k) with matching indices,
    the sum should be 0.
    
    Actually wait - the tensor is defined on all triples (i,j,k) with 0<=i<n, 0<=j<m, 0<=k<p.
    Each such triple maps to (a=i*m+j, b=j*p+k, c=k*n+i).
    
    For the decomposition to be valid, we need:
    - For valid (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    - For any (a,b,c) where the implied indices don't match (i.e., the j in a doesn't match j in b, or k in b doesn't match k in c, or i in c doesn't match i in a): sum = 0
    
    But actually, since we're summing over r, and each term U[a,r]*V[b,r]*W[c,r] is nonzero only for specific (a,b,c) combinations determined by r, we need to check consistency.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-10:
                    return False, f"Valid ({i},{j},{k}): got {val}, expected 1.0"
    
    # Check that invalid combinations give 0
    # An invalid combination is one where (a,b,c) doesn't correspond to a consistent (i,j,k)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                ia = a // m
                ja = a % m
                jb = b // p
                kb = b % p
                kc = c // n
                ic = c % n
                
                # Valid if ja == jb and kb == kc and ia == ic
                if ja != jb or kb != kc or ia != ic:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-10:
                        return False, f"Invalid ({ia},{ja},{jb},{kb},{kc},{ic}): got {val}, expected 0"
    
    return True, "OK"

def schoolbook():
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def random_decomposition(rank, seed=42):
    """Generate a random rank-r decomposition and verify."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Scale to try to match the tensor
    # This is just a random initialization; won't be correct
    # Return as-is for testing purposes
    return U.tolist(), V.tolist(), W.tolist()

def search_lower_rank(max_rank_to_try=95, time_budget=100):
    """Try to find a decomposition with rank less than schoolbook.
    
    Uses a simple approach: try various structured decompositions.
    """
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    school = schoolbook()
    best = school
    best_rank = n * m * p
    
    # Strassen-based block decomposition: rank 84
    # Split 4x4*4x6 into 2x2 blocks
    # 12 independent 2x2*2x2->2x2 multiplications, each with rank 7 (Strassen)
    # Total rank = 12 * 7 = 84
    
    if time_left() < 10:
        return best, best_rank
    
    def strassen_terms():
        """Return the 7 Strassen terms for 2x2*2x2->2x2."""
        terms = []
        # M1 = (a00+a11)(b00+b11) -> C00(+1), C10(+1)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1}
        ))
        # M2 = (a10+a11)b00 -> C00(-1), C10(+1)
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): -1, (1,0): 1}
        ))
        # M3 = a00(b01-b11) -> C01(+1), C11(+1)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ))
        # M4 = a11(b10-b00) -> C00(+1), C01(+1)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # M5 = (a00+a01)b11 -> C11(-1)
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,1): -1}
        ))
        # M6 = (a10-a00)(b00+b01) -> C11(+1)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M7 = (a01-a11)(b10+b11) -> C00(-1), C11(-1)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (1,1): -1}
        ))
        return terms
    
    st = strassen_terms()
    rank = 12 * 7  # 84
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(2):
        for K in range(3):
            for J in range(2):
                for (u_dict, v_dict, w_dict) in st:
                    for (i_loc, j_loc), u_coeff in u_dict.items():
                        global_i = I * 2 + i_loc
                        global_j = J * 2 + j_loc
                        U[global_i * m + global_j][col] += u_coeff
                    for (j_loc, k_loc), v_coeff in v_dict.items():
                        global_j = J * 2 + j_loc
                        global_k = K * 2 + k_loc
                        V[global_j * p + global_k][col] += v_coeff
                    for (i_loc, k_loc), w_coeff in w_dict.items():
                        global_k = K * 2 + k_loc
                        global_i = I * 2 + i_loc
                        W[global_k * n + global_i][col] += w_coeff
                    col += 1
    
    valid, msg = verify_decomposition(U, V, W)
    if valid:
        return {"U": U, "V": V, "W": W}, rank
    else:
        print(f"Strassen-based decomposition failed: {msg}")
        return best, best_rank
    
    # Try to use Strassen-like ideas for sub-blocks
    # For 4x4 * 4x6, we can partition into 2x2 blocks
    # Left: 2x2 blocks of 2x2
    # Right: 2x3 blocks of 2x2
    
    # Actually, let me try a different approach: use the fact that
    # M(4,4,6) can be decomposed using smaller tensor products.
    
    # One approach: use rank-49 for 4x4x4 (if we had it) and rank-8 for 4x4x2
    # But we don't have a better-than-schoolbook 4x4x4.
    
    # Let me try a direct search with small rank values using structured matrices.
    # For each candidate rank R, try to find U, V, W with entries in {0, 1, -1} or {0, 1/2, -1/2, 1, -1}
    
    # This is a very hard problem. Let me try a few specific constructions.
    
    # Construction 1: Partition into blocks and use Strassen for 2x2 parts
    # 4x4 * 4x6 = (2x2 block matrix) * (2x3 block matrix)
    # Each block is 2x2
    # Left has 2x2=4 blocks, right has 2x3=6 blocks
    # C = A*B where A is 2x2 blocks, B is 2x3 blocks
    # C[i,k] (block) = sum_j A[i,j] * B[j,k]
    # Each block multiplication is 2x2 * 2x2 -> 2x2
    # We can use Strassen for each: rank 7 instead of 8
    # But the blocks are multiplied independently, so total rank = 4 * 7 = 28?
    # No wait, that's not right either. The blocks don't interact.
    
    # Actually, if we partition:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C has 2x3 blocks, each 2x2
    # C[i,k] block = A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This is a 2x2 * 2x2 + 2x2 * 2x2 = sum of two 2x2 multiplications
    # Each 2x2*2x2->2x2 has rank 8 (schoolbook) or 7 (Strassen)
    # But here we're summing two such products, so the tensor is T = T1 + T2
    # where T1 involves A[*,1]*B[1,*] and T2 involves A[*,2]*B[2,*]
    # These use different parts of A and B, so they're independent
    # Total rank = 2 * 8 = 16 for each output block, but we have 2*3=6 output blocks
    # Hmm, this doesn't directly give a total rank.
    
    # Let me think differently. The full tensor M(4,4,6) has rank at most 96.
    # Known upper bounds: M(4,4,6) <= 4*4*6 = 96 (schoolbook)
    # Can we do better? 
    
    # One idea: use the decomposition M(n,m,p) = M(n,m,p1) + M(n,m,p2) if p = p1+p2
    # But this just gives rank M(4,4,3) + M(4,4,3) which is 48+48=96, no improvement.
    
    # Another idea: M(4,4,6) uses the same left matrix A. 
    # C = A * B where B is 4x6.
    # We can write B = [B1 | B2] where B1 is 4x4, B2 is 4x2
    # C = [A*B1 | A*B2]
    # rank M(4,4,6) <= rank M(4,4,4) + rank M(4,4,2)
    # = 64 + 32 = 96 again (schoolbook for each)
    
    # The key insight needed is to find a shared structure.
    
    # Let me try a computational search with small rank using integer/half-integer entries.
    # For rank R, we have R columns in U, V, W.
    # Each column of U has n*m=16 entries, each column of V has m*p=24 entries, each column of W has p*n=24 entries.
    # Total unknowns per column: 16+24+24 = 64
    # For R columns: 64*R unknowns
    # Constraints: (n*m)*(m*p)*(p*n) = 16*24*24 = 9216 equations (but many are redundant)
    # Actually, the non-zero constraints are only for valid (i,j,k): 4*4*6 = 96 constraints = 1
    # And the zero constraints for invalid triples.
    
    # Number of valid triples: 4*4*6 = 96, each gives sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # Number of invalid triples: total - valid = 16*24*24 - 96 = 9216 - 96 = 9120, each gives sum = 0
    # But many of these are automatically satisfied if the decomposition is "local" (each r only touches one (i,j,k) triple).
    
    # For the schoolbook, each r corresponds to exactly one (i,j,k), so only one triple is nonzero.
    # To reduce rank, we need to have some r that contributes to multiple triples, but cancels out for invalid ones.
    
    # This is essentially finding a rank-r tensor factorization, which is NP-hard in general.
    # For small ranks, we might be able to search.
    
    # Let me try rank 95, 94, etc. with a greedy approach.
    # Start from schoolbook and try to merge pairs of terms.
    
    # Two terms r1, r2 with:
    # U[a1], V[b1], W[c1] and U[a2], V[b2], W[c2]
    # Can we replace them with fewer terms?
    
    # Actually, merging two terms into one is not straightforward.
    
    # Let me try a different tactic: use known small tensor decompositions.
    # The 2x2x2 Strassen decomposition has rank 7.
    # Can I build a 4x4x6 decomposition from Strassen blocks?
    
    # Partition A into 2x2 blocks (2x2 grid of 2x2 blocks)
    # Partition B into 2x2 blocks (2x3 grid of 2x2 blocks) 
    # C = A*B, C is 2x3 grid of 2x2 blocks
    # C[i,k] = sum_j A[i,j] B[j,k] for block indices
    # = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Each A[i,j]*B[j,k] is a 2x2*2x2->2x2 multiplication
    # Using Strassen: rank 7 per such product
    # But we have 2*2*3 = 12 such products, each of rank 7
    # However, they share the same Strassen "intermediate" computations?
    # No, each product is independent.
    # Total rank = 12 * 7 = 84? But wait, the outputs C[i,k] for different (i,k) are different blocks.
    # And the inputs A[i,j] and B[j,k] are shared across products that use the same A[i,j] or B[j,k].
    
    # Hmm, this is getting complicated. The Strassen decomposition works for a single 2x2*2x2 multiplication.
    # When we have multiple such multiplications sharing input blocks, we can't simply add the ranks
    # because the intermediate values might be shared.
    
    # Actually, in the block decomposition:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For fixed (i,k), this involves 2 matrix multiplications.
    # For all (i,k) in 2x3 grid, we have 6 output blocks.
    # The total computation is:
    # For each j in {0,1}:
    #   For each i in {0,1}, k in {0,1,2}:
    #     Compute A[i,j] * B[j,k]
    # Then sum over j.
    
    # The Strassen decomposition computes X1 = (A00+A11)*(B00+B11), etc.
    # But here A[i,j] and B[j,k] are different for different (i,k).
    # So we can't share Strassen intermediates across different (i,k) pairs.
    
    # Therefore, total rank = (number of 2x2*2x2 multiplications) * 7 = 12 * 7 = 84.
    # Wait, that's better than 96!
    
    # Let me verify: we have a 2x2 block grid for A (so 4 blocks: A00, A01, A10, A11)
    # and a 2x3 block grid for B (so 6 blocks: B00, B01, B02, B10, B11, B12)
    # C has 2x3 blocks: C00, C01, C02, C10, C11, C12
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Number of 2x2*2x2 multiplications: for each (i,k) pair, we have 2 multiplications (j=0 and j=1)
    # Total: 6 * 2 = 12 multiplications
    # Each has rank 7 (Strassen), so total rank = 12 * 7 = 84
    
    # But wait, I need to make sure the Strassen decomposition applies correctly.
    # In Strassen, we compute 7 products of 2x2 matrices to get the 4 products needed.
    # But here, each "multiplication" A[i,j]*B[j,k] produces a 2x2 result.
    # The Strassen decomposition gives us that 2x2 result using 7 scalar-level 2x2 products.
    # Actually no - Strassen reduces the number of recursive calls. At the top level,
    # instead of 8 multiplications of 2x2 matrices, we do 7.
    # But each "multiplication" here is of 2x2 matrices, which at the scalar level is 8 scalar mults.
    # So Strassen gives 7 * (scalar mults per 2x2 mult) ... no, that's not right either.
    
    # Let me reconsider. The tensor M(2,2,2) has rank 7 (Strassen).
    # This means: for 2x2 * 2x2 -> 2x2, we can do it with 7 "trilinear forms" instead of 8.
    # Each trilinear form corresponds to one entry in U, V, W.
    
    # Now, in our block decomposition, we have 12 independent 2x2*2x2->2x2 multiplications.
    # Each can be done with rank 7.
    # Since they're independent (different input/output blocks), the total rank is 12*7 = 84.
    
    # Let me implement this!
    
    if time_left() < 10:
        return best, best_rank
    
    # Strassen decomposition of 2x2*2x2->2x2
    # C00 = M1 - M2 + M4 - M7
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M3 - M5 + M6 - M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # In tensor notation for 2x2*2x2->2x2:
    # Indices: i,j in {0,1} for A, j,k in {0,1} for B, i,k in {0,1} for C
    # Strassen gives 7 terms.
    
    # For our problem, let me define the mapping.
    # Global indices: i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Block indices: I = i//2, J = j//2, K = k//2
    # Local indices: i' = i%2, j' = j%2, k' = k%2
    # A[i,j] is in block A[I,J] at position (i', j')
    # B[j,k] is in block B[J,K] at position (j', k')
    # C[i,k] is in block C[I,K] at position (i', k')
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{J=0,1} sum_{j'=0,1} A[I*2+i', J*2+j'] * B[J*2+j', K*2+k']
    # = sum_{J=0,1} (A[I,J] * B[J,K])[i', k']
    
    # For each (I, K), C[I,K] = A[I,0]*B[0,K] + A[I,1]*B[1,K]
    # Each product A[I,J]*B[J,K] is a 2x2*2x2->2x2 multiplication.
    
    # Using Strassen for each:
    # Let's denote the 7 Strassen products for A[I,J]*B[J,K] as S1...S7
    # where each S is a 2x2 matrix product.
    
    # Actually, Strassen computes 7 products of 2x2 matrices (not scalar products).
    # Wait no. Strassen for 2x2*2x2->2x2 uses 7 multiplications of 2x2 matrices?
    # No, that's for larger matrices. For 2x2, Strassen uses 7 scalar multiplications of 2x2 "blocks"
    # where the blocks are scalars... 
    # 
    # I'm confusing myself. Let me be precise.
    #
    # M(2,2,2) is the tensor for multiplying two 2x2 matrices to get a 2x2 matrix.
    # The schoolbook rank is 8 (one for each (i,j,k) triple).
    # Strassen shows rank 7.
    #
    # The 7 terms are:
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 - M2 + M4 - M7
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M3 - M5 + M6 - M7
    #
    # Each M is a product of two linear combinations of entries of A and B.
    # So in tensor decomposition terms, each M corresponds to one term in U, V, W
    # where U, V, W have entries that are coefficients of the linear combinations.
    #
    # For M1 = (a+d)(e+h):
    # U contributes: a->1, d->1 (so U[0,0]=1, U[1,1]=1 in the 2x2 sense)
    # V contributes: e->1, h->1 (so V[0,0]=1, V[1,1]=1)
    # W contributes to C00 with +1, C10 with +1 (W[0,0]=1, W[1,0]=1)
    #
    # OK so let me just enumerate the 7 Strassen terms for 2x2*2x2->2x2.
    
    # For a single 2x2*2x2->2x2 multiplication with A having entries (i',j') and B having entries (j',k'):
    # Strassen terms (using 0-based indexing for local indices):
    
    def strassen_terms():
        """Return the 7 Strassen terms for 2x2*2x2->2x2.
        
        Each term is (u_dict, v_dict, w_dict) where:
        u_dict maps (i',j') -> coeff
        v_dict maps (j',k') -> coeff
        w_dict maps (i',k') -> coeff
        """
        terms = []
        
        # M1 = (a00+a11)(b00+b11) -> contributes to C00 (+1), C10 (+1)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1}
        ))
        
        # M2 = (a10+a11)b00 -> contributes to C00 (-1), C10 (+1)
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): -1, (1,0): 1}
        ))
        
        # M3 = a00(b01-b11) -> contributes to C01 (+1), C11 (+1)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ))
        
        # M4 = a11(b10-b00) -> contributes to C00 (+1), C01 (+1)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        
        # M5 = (a00+a01)b11 -> contributes to C11 (-1)
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,1): -1}
        ))
        
        # M6 = (a10-a00)(b00+b01) -> contributes to C11 (+1)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        
        # M7 = (a01-a11)(b10+b11) -> contributes to C00 (-1), C11 (-1)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (1,1): -1}
        ))
        
        return terms
    
    # Verify Strassen terms
    st = strassen_terms()
    for i_prime in range(2):
        for k_prime in range(2):
            total = 0
            for (u_dict, v_dict, w_dict) in st:
                for j_prime in range(2):
                    u_coeff = u_dict.get((i_prime, j_prime), 0)
                    v_coeff = v_dict.get((j_prime, k_prime), 0)
                    w_coeff = w_dict.get((i_prime, k_prime), 0)
                    total += u_coeff * v_coeff * w_coeff
            expected = 1
            if abs(total - expected) > 1e-10:
                print(f"Strassen verification failed for C({i_prime},{k_prime}): got {total}")
    
    # Now build the full 4x4*4x6 -> 4x6 decomposition
    # Total rank = 12 * 7 = 84
    
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for I in range(2):  # block row of A (and C)
        for K in range(3):  # block col of B (and C)
            for J in range(2):  # block col of A, block row of B
                # This is the multiplication A[I,J] * B[J,K] -> contributes to C[I,K]
                for (u_dict, v_dict, w_dict) in st:
                    # u_dict: (i',j') -> coeff, where i' is local row in A[I,J], j' is local col
                    # v_dict: (j',k') -> coeff, where j' is local row in B[J,K], k' is local col
                    # w_dict: (i',k') -> coeff, where i' is local row in C[I,K], k' is local col
                    for (i_loc, j_loc), u_coeff in u_dict.items():
                        global_i = I * 2 + i_loc
                        global_j = J * 2 + j_loc
                        U[global_i * m + global_j][col] += u_coeff
                    
                    for (j_loc, k_loc), v_coeff in v_dict.items():
                        global_j = J * 2 + j_loc
                        global_k = K * 2 + k_loc
                        V[global_j * p + global_k][col] += v_coeff
                    
                    for (i_loc, k_loc), w_coeff in w_dict.items():
                        global_k = K * 2 + k_loc
                        global_i = I * 2 + i_loc
                        W[global_k * n + global_i][col] += w_coeff
                    
                    col += 1
    
    # Verify
    valid, msg = verify_decomposition(U, V, W)
    if valid:
        print(f"Strassen-based decomposition works! Rank {rank}")
        return {"U": U, "V": V, "W": W}, rank
    else:
        print(f"Strassen-based decomposition failed: {msg}")
        # Fall back to schoolbook
        return best, best_rank

def construct():
    """Construct a tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    if time_left() < 5:
        return schoolbook()
    
    # Try to find a better-than-schoolbook decomposition
    result, rank = search_lower_rank()
    
    if time_left() < 5:
        return schoolbook()
    
    return result

# EVOLVE-BLOCK-END
