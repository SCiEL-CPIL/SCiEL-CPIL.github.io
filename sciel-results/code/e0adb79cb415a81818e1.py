import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Aims for Rank 97 by finding 8 disjoint 2x2x2 Strassen blocks.
    Falls back to Rank 98 if 8 blocks cannot be found.
    """
    start_time = time.time()
    deadline = start_time + 112.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total = n * m * p  # 105

    # Precompute all blocks: each = (params, frozenset of (i,j,k), bitmask)
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)
    # Overlap matrix as bitmasks
    ov = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                bits |= (1 << b)
        ov[a] = bits

    # Heuristic: Sort blocks by number of conflicts (descending) to prune faster
    # A block that conflicts with many others is "tighter"
    conflict_counts = [bin(ov[i]).count('1') for i in range(B)]
    block_indices = list(range(B))
    block_indices.sort(key=lambda x: conflict_counts[x], reverse=True)
    
    # Map new index to old index
    sorted_blocks = [blocks[i] for i in block_indices]
    sorted_ov = [ov[i] for i in block_indices]
    # Re-compute ov for sorted indices? No, we can just use the old masks.
    # But for the search, it's easier to work with the sorted list.
    # Let's re-compute conflict masks in the sorted space for simplicity of bit ops?
    # Or just use the original masks.
    
    # We will use a recursive backtracking search.
    # State: available_mask (bitmask of triples), current_count, best_count
    # To speed up, we can also maintain a list of "candidate" blocks that don't conflict with chosen ones.
    
    best_packing = []
    best_count = 0
    
    def search(available_mask, current_packing, current_count):
        nonlocal best_packing, best_count
        
        # Pruning
        if current_count > best_count:
            best_count = current_count
            best_packing = list(current_packing)
            if best_count == 8: # Max possible? 105/8 = 13.12. But grid limits it.
                # Actually, can we fit more than 8?
                # i-dim 3 -> max 1 block per i-pair? No.
                # (0,1) and (1,2) overlap at i=1.
                # So at most 1 block can use i=1? No, they can be disjoint in j,k.
                # But i=1 is in both. So any block using (0,1) and any block using (1,2) share i=1.
                # Do they share a triple?
                # Block A: i in {0,1}, j in {j1,j2}, k in {k1,k2}
                # Block B: i in {1,2}, j in {j3,j4}, k in {k3,k4}
                # They share triple (1, j, k) if j in {j1,j2} intersect {j3,j4} AND k in {k1,k2} intersect {k3,k4}.
                # If j ranges are disjoint and k ranges are disjoint, they don't share triples.
                # So we can have multiple blocks.
                # Max i-blocks: 2.
                # Max j-blocks: 2 (e.g., 0-1, 2-3, 3-4? 0-1 and 2-3 are disjoint. 2-3 and 3-4 overlap at 3).
                # Max j-disjoint pairs: 2 (e.g., (0,1) and (2,3) or (3,4)).
                # Max k-disjoint pairs: 3 (e.g., (0,1), (2,3), (4,5) or (5,6)).
                # So max blocks $\approx 2 \times 2 \times 3 = 12$?
                # But they must be disjoint in 3D.
                # Let's let the search find the max.
                pass
        
        # Upper bound estimation
        # Count available triples
        # A simple bound: available_triples // 8
        # A better bound: Max matching in conflict graph of remaining blocks?
        # For now, use count of available blocks that fit in available_mask
        # This is expensive.
        
        # Heuristic: If time is running out, stop
        if time.time() > deadline:
            return
            
        # Find candidate blocks
        # We iterate through all blocks, but we can skip those that conflict with current_packing?
        # Actually, `available_mask` already reflects the used triples.
        # A block is a candidate if (block_mask & ~available_mask) == 0, i.e., block_mask & available_mask == block_mask?
        # No, block_mask must be a subset of available_mask?
        # Wait, `available_mask` starts as all 1s (105 bits).
        # When we pick a block, we do `available_mask &= ~block_mask`.
        # So a block is available if `block_mask & available_mask == block_mask`.
        
        candidates = []
        for i in range(B):
            if blocks[i][2] & available_mask == blocks[i][2]:
                candidates.append(i)
                
        if not candidates:
            return
            
        # Order candidates: pick the one with the most conflicts among remaining candidates?
        # Or just pick the first one?
        # To find MAX, we should explore all.
        # To prune, we should pick the most constrained first.
        
        # Sort candidates by number of conflicts with other candidates
        # This is O(C^2) per node. C <= 48. 48^2 = 2304. Acceptable.
        
        # Optimization: If we just want to find ANY packing of size 8, we can use a greedy heuristic first.
        # But we want the MAX.
        
        # Let's use a simple recursion.
        # To avoid deep recursion overhead, we can use a stack, but recursion is fine for depth < 15.
        
        # Pruning: If current_count + len(candidates) <= best_count, return?
        # No, because not all candidates can be picked.
        # Better: current_count + (available_triples // 8) <= best_count.
        
        # Calculate available triples count
        # bin(available_mask).count('1')
        
        if current_count + (bin(available_mask).count('1') // 8) <= best_count:
            return
            
        # Try each candidate
        for i in candidates:
            # Pick block i
            current_packing.append(i)
            new_available = available_mask & ~blocks[i][2]
            search(new_available, current_packing, current_count + 1)
            current_packing.pop()
            
            # Symmetry breaking?
            # If we pick i, we don't need to try any block j that is "equivalent" to i?
            # Hard to define equivalence here.
            
    # Initial available mask: all 1s for 105 bits
    initial_mask = (1 << total) - 1
    search(initial_mask, [], 0)
    
    # If best_count is 0, something went wrong.
    # If best_count is 7, we use the previous logic.
    # If best_count is 8, we use the new packing.
    
    # Build matrices
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best_packing:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    # Schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
