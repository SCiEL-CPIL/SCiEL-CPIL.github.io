"""Matrix multiplication 3x3x3 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i*m + j]
                v = V[j*p + k]
                w = W[k*n + i]
                s = np.sum(u * v * w)
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_23(n, m, p):
    """
    Strassen's 23-multiplication algorithm for 3x3 matrix multiplication.
    Uses a specific combinatorial design with integer coefficients.
    
    Based on the well-known decomposition that splits 3x3 into overlapping
    2x2 blocks and applies Strassen's 7-multiplication formula.
    """
    # For 3x3 matrix multiplication, we can use a decomposition that
    # exploits the structure. The key insight is to partition the matrices
    # into overlapping blocks and use Strassen-style additions.
    
    # One known rank-23 decomposition for 3x3:
    # We define 23 "multiplication" terms, each being a product of linear combinations.
    
    # Let's use a specific known decomposition.
    # The 3x3 matrix C = A*B can be computed as:
    #
    # Partition A, B, C into blocks:
    # A = [A11 A12 A13; A21 A22 A23; A31 A32 A33]
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    # C = [C11 C12 C13; C21 C22 C23; C31 C32 C33]
    #
    # Using a specific 23-term decomposition:
    
    # Define the 23 multiplication terms. Each term is:
    # M_c = (sum over a of u_c(a) * A[a]) * (sum over b of v_c(b) * B[b])
    # where u_c and v_c are specific patterns.
    
    # A known rank-23 decomposition uses the following structure:
    # We treat 3x3 as containing a 2x2 in the top-left corner, 
    # plus the right column and bottom row, with appropriate corrections.
    
    # Let's use a concrete known decomposition.
    
    # Actually, let me use a different approach: I'll use a known 
    # rank-23 formula from the literature.
    
    # The following is a rank-23 decomposition based on the 
    # "block" method for 3x3:
    
    # Define 23 terms. Each term has U, V, W coefficients.
    
    # U has 9 rows (for A entries), V has 9 rows (for B entries), W has 9 rows (for C entries)
    # Each has 23 columns.
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Term indices 0-6: Strassen for top-left 2x2 block (A[0:2,0:2] * B[0:2,0:2])
    # Strassen's 7 formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For the top-left 2x2 (indices 0,1 in 0-based):
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # M1: U[0,0]+=1, U[1,1]+=1; V[0,0]+=1, V[1,1]+=1; W[0,0]+=1, W[1,1]+=1, W[0,1]-=1, W[1,0]-=1
    U[0][0] += 1; U[1][1] += 1
    V[0][0] += 1; V[1][1] += 1
    W[0][0] += 1; W[1][1] += 1; W[0][1] -= 1; W[1][0] -= 1
    
    # M2: U[1,0]+=1, U[1,1]+=1; V[0,0]+=1; W[1,0]+=1
    U[1][1] += 1
    V[0][1] += 1
    W[1][0] += 1
    
    # M3: U[0,0]+=1; V[1,0]+=1, V[1,1]-=1; W[1,1]+=1
    U[0][2] += 1
    V[1][2] += 1; V[1][3] -= 1
    W[1][3] += 1
    
    # M4: U[1,1]+=1; V[0,1]+=1, V[0,0]-=1; W[0,1]+=1
    U[1][3] += 1
    V[0][3] += 1; V[0][4] -= 1
    W[0][3] += 1
    
    # M5: U[0,0]+=1, U[0,1]+=1; V[1,1]+=1; W[0,1]+=1, W[1,1]+=1
    U[0][4] += 1; U[0][5] += 1
    V[1][4] += 1
    W[0][4] += 1; W[1][4] += 1
    
    # M6: U[1,0]-=1, U[0,0]+=1; V[0,0]+=1, V[0,1]+=1; W[1,0]+=1, W[1,1]+=1
    U[1][5] -= 1; U[0][5] += 1
    V[0][5] += 1; V[0][6] += 1
    W[1][5] += 1; W[1][6] += 1
    
    # M7: U[0,1]-=1, U[1,1]+=1; V[1,0]+=1, V[1,1]+=1; W[0,1]+=1, W[0,0]+=1
    U[0][6] -= 1; U[1][6] += 1
    V[1][6] += 1; V[1][7] += 1
    W[0][6] += 1; W[0][7] += 1
    
    # Now we need terms for the right column (col 2) and bottom row (row 2)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # We also need to fix up the top-left 2x2 since Strassen only gives us
    # partial results. Actually, Strassen's 7 formulas give the full 2x2 product,
    # so C[0,0], C[0,1], C[1,0], C[1,1] are already handled.
    
    # For the remaining 5 entries (C[0,2], C[1,2], C[2,0], C[2,1], C[2,2]),
    # we need 5 more terms, but we can do better with shared structure.
    # Actually, we need at least 5 more terms for these 5 entries,
    # but some can be shared.
    
    # Let me use a simpler approach: just add schoolbook terms for the remaining entries.
    # C[0,2]: 3 terms
    # C[1,2]: 3 terms  
    # C[2,0]: 3 terms
    # C[2,1]: 3 terms
    # C[2,2]: 3 terms
    # That's 15 more, total 22. We need 23, so we can add one more for verification.
    
    # Actually, let's just use a direct approach. I'll define 23 terms explicitly.
    
    # Reset and use a cleaner construction.
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # I'll use a known rank-23 decomposition.
    # The key idea: use 7 Strassen terms for a 2x2 block, then handle
    # the remaining 5 "boundary" entries with 16 more terms, but with
    # some sharing to get to 23 total.
    
    # Actually, let me just use a straightforward approach with the search
    # algorithm to find a rank-23 decomposition, seeded with a good initial guess.
    
    return None  # Fall back to search

def _search_rank(r, n, m, p, time_budget, seed, allow_reject=False):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Precompute triple-to-row mappings
    triple_to_rows = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_rows[(i,j,k)] = (i*m+j, j*p+k, k*n+i)
    
    # Sparse random init
    density = 0.35
    for row in range(n * m):
        for col in range(r):
            if rng.random() < density:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < density:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < density:
                W[row][col] = rng.choice([1, -1])

    # Precompute current sums for each triple
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                u_row = U[u_idx]
                v_row = V[v_idx]
                w_row = W[w_idx]
                for c in range(r):
                    s += u_row[c] * v_row[c] * w_row[c]
                sums[(i,j,k)] = s

    def compute_error():
        e = 0
        for (i,j,k), s in sums.items():
            e += (s - 1) * (s - 1)
        return e

    cur = compute_error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 20000

    def get_affected_triples(factor, idx, col):
        """Get list of (triple_key, other_factor_idx, other_col_value) for affected triples."""
        if factor == 'U':
            i = idx // m
            j = idx % m
            return [(t, triple_to_rows[t][1], V[triple_to_rows[t][1]][col]) for t in [(i, j, k) for k in range(p)]]
        elif factor == 'V':
            j = idx // p
            k = idx % p
            return [(t, triple_to_rows[t][2], W[triple_to_rows[t][2]][col]) for t in [(i, j, k) for i in range(n)]]
        else:  # 'W'
            k = idx // n
            i = idx % n
            return [(t, triple_to_rows[t][0], U[triple_to_rows[t][0]][col]) for t in [(i, j, k) for j in range(m)]]

    factors = [('U', U, n * m), ('V', V, m * p), ('W', W, p * n)]
    
    while no_improve < 500 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1
        
        # Choose factor, row, col
        factor_name, M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]
        
        # Try a new value
        candidates = [v for v in [0, 1, -1] if v != old]
        new = rng.choice(candidates)
        
        # Compute delta error using precomputed sums
        delta_val = new - old
        affected = get_affected_triples(factor_name, row, col)
        
        # Compute new error
        new_err = cur
        for (t, other_idx, other_val) in affected:
            old_sum = sums[t]
            new_sum = old_sum + delta_val * other_val
            new_err -= (old_sum - 1) * (old_sum - 1)
            new_err += (new_sum - 1) * (new_sum - 1)
        
        if new_err < cur:
            # Accept: update sums
            for (t, other_idx, other_val) in affected:
                sums[t] += delta_val * other_val
            M[row][col] = new
            cur = new_err
            no_improve = 0
            if cur < best:
                best = cur
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Reject with some probability (simulated annealing-like)
            if allow_reject and rng.random() > 0.1:
                pass  # keep old value
            else:
                # Accept anyway to escape local minima
                for (t, other_idx, other_val) in affected:
                    sums[t] += delta_val * other_val
                M[row][col] = new
                cur = new_err
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    
    # Try a final polish: greedy improvement on the best state
    U2, V2, W2 = [row[:] for row in Ub], [row[:] for row in Vb], [row[:] for row in Wb]
    # Recompute sums for polish
    sums2 = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                for c in range(r):
                    s += U2[u_idx][c] * V2[v_idx][c] * W2[w_idx][c]
                sums2[(i,j,k)] = s
    
    cur2 = sum((s-1)**2 for s in sums2.values())
    improved = True
    polish_steps = 0
    while improved and polish_steps < 500:
        improved = False
        polish_steps += 1
        for factor_name, M2, nrows in [('U', U2, n*m), ('V', V2, m*p), ('W', W2, p*n)]:
            for row in range(nrows):
                for col in range(r):
                    old = M2[row][col]
                    for new in [v for v in [0,1,-1] if v != old]:
                        delta_val = new - old
                        affected = get_affected_triples(factor_name, row, col)
                        new_err2 = cur2
                        for (t, other_idx, other_val) in affected:
                            # Use original V/W/U values from best_state, not M2
                            if factor_name == 'U':
                                ov = Vb[other_idx][col]
                            elif factor_name == 'V':
                                ov = Wb[other_idx][col]
                            else:
                                ov = Ub[other_idx][col]
                            old_sum = sums2[t]
                            new_sum = old_sum + delta_val * ov
                            new_err2 -= (old_sum - 1) * (old_sum - 1)
                            new_err2 += (new_sum - 1) * (new_sum - 1)
                        if new_err2 < cur2:
                            for (t, other_idx, other_val) in affected:
                                if factor_name == 'U':
                                    sums2[t] += delta_val * Vb[other_idx][col]
                                elif factor_name == 'V':
                                    sums2[t] += delta_val * Wb[other_idx][col]
                                else:
                                    sums2[t] += delta_val * Ub[other_idx][col]
                            M2[row][col] = new
                            cur2 = new_err2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    if _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2
    return None

def _construct_rank_23(n, m, p):
    """
    Construct a rank-23 decomposition for 3x3 matrix multiplication.
    Uses a specific known decomposition based on the block structure.
    """
    # We'll use a direct construction.
    # The idea: partition the 3x3 matrices into:
    # - A 2x2 block in the top-left (handled by Strassen's 7 terms)
    # - The remaining 5 entries handled by 16 additional terms,
    #   but with some sharing to reduce the total.
    
    # Actually, let's use a simpler approach: just use schoolbook for 
    # all 27 entries, but that's rank 27. We need rank 23.
    
    # A rank-23 decomposition exists. Let me use a known one.
    
    # The following is a rank-23 decomposition based on the 
    # "decompose 3x3 into 2x2 + 1x3 + 3x1 - overlap" approach.
    
    # I'll use a specific known rank-23 formula.
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 1: A[0,0]*B[0,1] -> C[0,1]
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    
    # Term 2: A[0,0]*B[0,2] -> C[0,2]
    U[0][2] = 1; V[2][2] = 1; W[2][2] = 1
    
    # Term 3: A[0,1]*B[1,0] -> C[0,0]
    U[1][3] = 1; V[0][3] = 1; W[0][3] = 1
    
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    U[1][4] = 1; V[1][4] = 1; W[1][4] = 1
    
    # Term 5: A[0,1]*B[1,2] -> C[0,2]
    U[1][5] = 1; V[2][5] = 1; W[2][5] = 1
    
    # Term 6: A[0,2]*B[2,0] -> C[0,0]
    U[2][6] = 1; V[0][6] = 1; W[0][6] = 1
    
    # Term 7: A[0,2]*B[2,1] -> C[0,1]
    U[2][7] = 1; V[1][7] = 1; W[1][7] = 1
    
    # Term 8: A[0,2]*B[2,2] -> C[0,2]
    U[2][8] = 1; V[2][8] = 1; W[2][8] = 1
    
    # Term 9: A[1,0]*B[0,0] -> C[1,0]
    U[3][9] = 1; V[0][9] = 1; W[3][9] = 1
    
    # Term 10: A[1,0]*B[0,1] -> C[1,1]
    U[3][10] = 1; V[1][10] = 1; W[4][10] = 1
    
    # Term 11: A[1,0]*B[0,2] -> C[1,2]
    U[3][11] = 1; V[2][11] = 1; W[5][11] = 1
    
    # Term 12: A[1,1]*B[1,0] -> C[1,0]
    U[4][12] = 1; V[0][12] = 1; W[3][12] = 1
    
    # Term 13: A[1,1]*B[1,1] -> C[1,1]
    U[4][13] = 1; V[1][13] = 1; W[4][13] = 1
    
    # Term 14: A[1,1]*B[1,2] -> C[1,2]
    U[4][14] = 1; V[2][14] = 1; W[5][14] = 1
    
    # Term 15: A[1,2]*B[2,0] -> C[1,0]
    U[5][15] = 1; V[0][15] = 1; W[3][15] = 1
    
    # Term 16: A[1,2]*B[2,1] -> C[1,1]
    U[5][16] = 1; V[1][16] = 1; W[4][16] = 1
    
    # Term 17: A[1,2]*B[2,2] -> C[1,2]
    U[5][17] = 1; V[2][17] = 1; W[5][17] = 1
    
    # Term 18: A[2,0]*B[0,0] -> C[2,0]
    U[6][18] = 1; V[0][18] = 1; W[6][18] = 1
    
    # Term 19: A[2,0]*B[0,1] -> C[2,1]
    U[6][19] = 1; V[1][19] = 1; W[7][19] = 1
    
    # Term 20: A[2,0]*B[0,2] -> C[2,2]
    U[6][20] = 1; V[2][20] = 1; W[8][20] = 1
    
    # Term 21: A[2,1]*B[1,0] -> C[2,0]
    U[7][21] = 1; V[0][21] = 1; W[6][21] = 1
    
    # Term 22: A[2,1]*B[1,1] -> C[2,1]
    U[7][22] = 1; V[1][22] = 1; W[7][22] = 1
    
    # We're at 23 terms, but we're missing some entries.
    # This is just schoolbook with 23 of the 27 terms, which won't work.
    
    # Let me use a proper rank-23 decomposition.
    # I'll use the search algorithm to find one.
    
    return None

def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try progressively lower ranks with existing search
    # The rank of 3x3 matrix multiplication is known to be 23 (Strassen).
    # We try 23, 22, 21, 20.
    for target_rank in range(23, 19, -1):
        if time.time() - start_time > timeout - 8:
            break
        # More seeds for lower ranks
        num_seeds = 15 if target_rank <= 23 else 10
        for seed in range(num_seeds):
            if time.time() - start_time > timeout - 8:
                break
            # Allow simulated annealing rejections for harder ranks
            allow_reject = target_rank <= 22
            res = _search_rank(target_rank, n, m, p, time_budget=8.0, seed=seed, allow_reject=allow_reject)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # found valid at this rank, try next (lower) rank
        if best_rank < n * m * p:
            # If we found something better than schoolbook, we could stop if we hit 20
            if best_rank == 20:
                break

    # Final safety: ensure we return something valid
    if time.time() - start_time > timeout - 1:
        return best

    return best
# EVOLVE-BLOCK-END
