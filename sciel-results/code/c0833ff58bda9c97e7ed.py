import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with schoolbook: rank = n*m*p = 75
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for each (i,j,k)
    
    # We'll represent the decomposition as a list of rank-1 terms
    # Each term: (U_entries, V_entries, W_entries) where each is a dict {index: coeff}
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    
    # Schoolbook terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                terms.append((u, v, w))
    
    # Greedy reduction: try to find pairs of terms that can be merged
    # A pair of rank-1 terms (u1,v1,w1) and (u2,v2,w2) can be replaced by
    # a single rank-1 term (u,v,w) if:
    #   u1*v1*w1 + u2*v2*w2 = u*v*w
    # This means the sum tensor must be rank 1.
    # 
    # More generally, we can try replacing k terms with 1 term.
    # For k=2: the sum of two rank-1 tensors is rank 1 iff they're proportional
    # (which they aren't in schoolbook), so simple pair merging won't work directly.
    #
    # Instead, use the Strassen approach: introduce a NEW rank-1 term that has
    # support on multiple indices, and REMOVE several schoolbook terms.
    #
    # Strassen 2x2x2: replaces 4 schoolbook terms with 7 Strassen terms (net +3 per block)
    # Wait, that's wrong. Strassen replaces 4 multiplications with 7, but we're counting
    # the rank of the tensor decomposition, not the number of multiplications.
    #
    # Actually, the matrix multiplication tensor of size n×m×p has rank ≤ n*m*p (schoolbook).
    # Strassen for 2×2×2 gives rank 7 instead of 8. So it saves 1 per 2×2×2 block.
    #
    # For our 3×5×5 case: schoolbook rank = 75.
    # We can place non-overlapping 2×2×2 Strassen blocks.
    # In a 3×5×5 grid, how many non-overlapping 2×2×2 blocks can we place?
    # Along i: floor(3/2) = 1 (positions 0)
    # Along j: floor(5/2) = 2 (positions 0, 2)
    # Along k: floor(5/2) = 2 (positions 0, 2)
    # Total: 1*2*2 = 4 blocks, saving 4*1 = 4, giving rank 71.
    #
    # But we can also overlap blocks in the i-dimension! Since i only goes 0..2,
    # we can place blocks at i=0 (covering i=0,1) and i=1 (covering i=1,2).
    # These overlap at i=1. We need to be careful about double-counting.
    #
    # Let me use a different approach: systematic search for rank-1 terms that
    # can replace subsets of schoolbook terms.
    
    # Alternative structural approach: use the fact that for 3×5×5, we can
    # decompose into:
    # - A 3×3×3 sub-problem (rank 23 via known decomposition) 
    # - Plus the remaining terms
    #
    # 3×3×3 covers i∈{0,1,2}, j∈{0,1,2}, k∈{0,1,2}: 27 terms
    # Remaining: 75 - 27 = 48 terms
    # Total: 23 + 48 = 71 (same as before)
    #
    # But wait - the 3×3×3 decomposition might share structure with adjacent terms.
    #
    # Let me try a completely different structural approach:
    # Use the "bilinear complexity" perspective. The rank of the 3×5×5 matmul
    # tensor is known to be at most 75 (schoolbook) and at least 3*5 = 15 (trivial bound).
    # Best known upper bounds from literature...
    #
    # For now, let me implement a greedy local search:
    # 1. Start with schoolbook (rank 75)
    # 2. Repeatedly try to find a set of ≤ r schoolbook terms that can be
    #    replaced by s < r new rank-1 terms
    # 3. The new terms can have arbitrary (half-integer) coefficients
    
    # Actually, let me try the most promising structural improvement:
    # Use overlapping Strassen blocks in the i-dimension.
    # Block at i=0: covers (i,j,k) for i∈{0,1}, j∈{j0,j0+1}, k∈{k0,k0+1}
    # Block at i=1: covers (i,j,k) for i∈{1,2}, j∈{j0,j0+1}, k∈{k0,k0+1}
    # These share the row i=1. We need to handle this carefully.
    #
    # The issue: if we apply Strassen at i=0 for a (j,k) block, it introduces
    # terms involving A[0,j], A[1,j], etc. If we also apply at i=1, we get
    # terms involving A[1,j], A[2,j]. The A[1,j] terms appear in both.
    #
    # This gets complicated. Let me instead try a direct ILP-style search
    # for small rank reductions.
    
    # Practical approach: try all possible 2×2×2 Strassen placements (including
    # overlapping in i) and pick the combination that minimizes total rank.
    #
    # A Strassen block at (i0, j0, k0) covers entries:
    #   (i0,j0,k0), (i0,j0+1,k0+1), (i0+1,j0,k0+1), (i0+1,j0+1,k0)
    #   and also: (i0,j0+1,k0), (i0+1,j0,k0), (i0,j0,k0+1), (i0+1,j0+1,k0+1)
    #   Wait, no. A 2×2×2 Strassen block covers ALL 8 entries in the 2×2×2 subcube:
    #   i∈{i0,i0+1}, j∈{j0,j0+1}, k∈{k0,k0+1}
    #   It uses 7 rank-1 terms instead of 8.
    
    # All possible 2×2×2 block positions:
    # i0 ∈ {0, 1} (since n=3, i0 can be 0 or 1)
    # j0 ∈ {0, 1, 2, 3} (since m=5, j0 can be 0..3)
    # k0 ∈ {0, 1, 2, 3} (since p=5, k0 can be 0..3)
    
    block_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                block_positions.append((i0, j0, k0))
    
    # For each block, define which 8 (i,j,k) entries it covers
    def block_entries(i0, j0, k0):
        entries = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    entries.add((i0+di, j0+dj, k0+dk))
        return entries
    
    # The key question: can we select a set of blocks such that the total rank is minimized?
    # If blocks don't overlap: each saves 1, total rank = 75 - (#blocks)
    # If blocks overlap: the savings are more complex because shared entries
    # are "covered" by multiple blocks.
    #
    # When two Strassen blocks overlap, the combined decomposition is not simply
    # the union. We need to verify correctness.
    #
    # For safety, let me first try non-overlapping blocks (which we know work),
    # then try to add overlapping ones and verify.
    
    # Non-overlapping: i0 must be distinct (only i0=0 works without overlap in i)
    # Actually i0=0 and i0=1 overlap in the i-dimension (share i=1).
    # For non-overlap, we need i0 values that don't share rows.
    # With n=3: i0=0 uses rows {0,1}, i0=1 uses rows {1,2}. They share row 1.
    # So no two blocks can have different i0 values without overlap.
    # Max non-overlapping blocks: all with same i0, non-overlapping in j and k.
    # i0=0: j0 ∈ {0,2}, k0 ∈ {0,2} → 4 blocks
    # i0=1: j0 ∈ {0,2}, k0 ∈ {0,2} → 4 blocks
    # But i0=0 and i0=1 blocks overlap, so we can only pick one set.
    # Best: 4 blocks, rank = 75 - 4 = 71.
    
    # Now let's try overlapping blocks and see if we can do better.
    # The idea: if we use blocks at i0=0 AND i0=1, the shared row i=1
    # means some entries are "double-covered." But the Strassen decomposition
    # for each block is independent - it just introduces 7 new rank-1 terms
    # and removes 8 schoolbook terms. If an entry is in two blocks, we'd
    # be removing it twice, which is wrong.
    #
    # So overlapping blocks don't directly compose. We need a different approach.
    #
    # NEW STRUCTURAL IDEA: Instead of Strassen blocks, use a global rank-1
    # term search. Find rank-1 tensors that have support on many entries
    # and can "explain" multiple schoolbook terms at once.
    #
    # A rank-1 term (u, v, w) with u supported on S_u ⊂ {0..14}, v on S_v ⊂ {0..24},
    # w on S_w ⊂ {0..14} can "explain" the entry (i,j,k) if:
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] ≠ 0
    # For it to exactly decompose, we need the sum of all rank-1 terms to
    # equal the identity tensor.
    #
    # This is a hard combinatorial problem. Let me try a heuristic:
    # 1. Start with schoolbook (75 terms)
    # 2. At each step, find the "best" replacement: a set of new rank-1 terms
    #    that can replace a larger set of existing terms
    # 3. Use local search / simulated annealing
    
    # For time efficiency, let me try a specific structured approach:
    # Decompose the 3×5×5 tensor as:
    #   T = T_3x3x3 + T_remaining
    # where T_3x3x3 is the sub-tensor for i∈{0,1,2}, j∈{0,1,2}, k∈{0,1,2}
    # and T_remaining has all other entries.
    #
    # Use rank-23 decomposition for T_3x3x3, schoolbook for T_remaining.
    # T_remaining has 75 - 27 = 48 entries.
    # Total rank: 23 + 48 = 71.
    #
    # Can we do better by finding a decomposition of T_3x3x3 that shares
    # structure with T_remaining? Unlikely to help directly.
    #
    # Let me try yet another approach: exploit the rectangular structure.
    # The 3×5×5 tensor can be seen as 3 copies of 1×5×5 "slices" (one per i).
    # Or 5 copies of 3×1×5 slices (one per j).
    # Or 5 copies of 3×5×1 slices (one per k).
    #
    # A 1×5×5 tensor has rank 5 (it's just a row of the B matrix times... wait).
    # Actually, the matmul tensor T[i,j,k] = 1 for the "identity" structure.
    # The rank of the full 3×5×5 matmul tensor is what we're computing.
    #
    # Known results: rank of 3×n×n matmul tensor.
    # For n=1: rank 3
    # For n=2: rank ? 
    # For n=5: unknown exactly, but ≤ 75
    #
    # Let me just implement the best known constructive bound and verify it.
    # The 4-block Strassen gives 71. Let me verify this is correct and then
    # try to push further.
    
    # Build the 4-block Strassen decomposition (rank 71)
    U = [[0.0] * 71 for _ in range(n * m)]
    V = [[0.0] * 71 for _ in range(m * p)]
    W = [[0.0] * 71 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen block starting at (i0,j0,k0)"""
        # M0 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M1 = (A[i0+1,j0]) * (B[j0,k0] + B[j0+1,k0+1])
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A[i0,j0] + A[i0,j0+1]) * (B[j0+1,k0+1] - B[j0,k0])
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[j0*p + k0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + i0][c] += -1
        
        # M3 = (A[i0+1,j0+1] - A[i0,j0]) * (B[j0,k0] + B[j0+1,k0])
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[k0*n + i0][c] += -1
        
        # M4 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0+1] + B[j0+1,k0])
        c = col_offset + 4
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + i0][c] += -1
        
        # M5 = (A[i0+1,j0] - A[i0,j0]) * (B[j0+1,k0] - B[j0,k0])
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += -1
        
        # M6 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0+1,k0+1] - B[j0,k0])
        # Wait, I need to use the standard Strassen formulas. Let me be more careful.
        # Standard 2x2x2 Strassen:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)(B00)
        # M2 = (A00)(B10-B11)
        # M3 = (A00-A11)(B01+B11)
        # M4 = (A10+A11)(B01)
        # M5 = (A01-A10)(B10)
        # M6 = (A01+A11)(B11)
        # C00 = M0 + M2 - M3 - M6
        # C01 = M3 + M4
        # C10 = M1 + M5
        # C11 = M0 - M1 + M5 + M6
        
        pass  # This is getting complicated; let me use the verified version
    
    # Let me use the verified Strassen from the current program
    def add_strassen_block_v2(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # Place 4 non-overlapping blocks
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    for i0, j0, k0 in blocks:
        add_strassen_block_v2(i0, j0, k0, col)
        col += 7
    
    # Mark covered entries
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all (i,j,k)
    # Build the tensor
    tensor = np.zeros((n, m, p))
    for r in range(U_arr.shape[1]):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        for i in range(n):
            for j in range(m):
                if u[i*m+j] != 0:
                    for k in range(p):
                        if v[j*p+k] != 0:
                            if w[k*n+i] != 0:
                                tensor[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-10):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
