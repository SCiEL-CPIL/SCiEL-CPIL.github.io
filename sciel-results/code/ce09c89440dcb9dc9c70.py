"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute 5-subsets of 0..11
# Total C(12, 5) = 792
FIVE_SUBSETS = list(combinations(range(12), 5))
FIVE_TO_IDX = {s: i for i, s in enumerate(FIVE_SUBSETS)}

# Precompute 6-subsets of 0..11
# Total C(12, 6) = 924
SIX_SUBSETS = list(combinations(range(12), 6))
SIX_TO_FIVE_IDX = []
for b in SIX_SUBSETS:
    idxs = [FIVE_TO_IDX[s] for s in combinations(b, 5)]
    SIX_TO_FIVE_IDX.append(idxs)

# Precompute the known Witt design S(5,6,12) blocks.
# The Witt design S(5,6,12) is a Steiner system. We can construct it
# using the known orbit structure under the group PSL(2,11) or simply
# use the known list. Since hardcoding 132 blocks is tedious, we can
# use a known construction method or a small subset search that is very fast.
# However, the most robust "structural" change is to use a known exact algorithm
# or a very targeted search that finds the Witt design quickly.
# Actually, the most efficient way to get 132 blocks is to use the fact that
# the Witt design is unique (up to isomorphism). We can construct it via
# the "small Witt design" S(3,6,22) and extend, but that's complex.
# Instead, we can use a very fast greedy + backtracking that is guided by
# the structure, or we can simply hardcode the 132 blocks.
# Hardcoding is the most reliable way to ensure 132 blocks are returned instantly.
# Let's generate the 132 blocks of the Witt design S(5,6,12).
# One standard construction uses the group PSL(2,11) acting on 12 points.
# The points can be identified with the projective line over GF(11).
# The blocks are the images of a single block under the group.
# A base block is {0, 1, 3, 4, 5, 9} (or similar).
# Let's verify a known base block and its orbits.
# The Witt design S(5,6,12) can be constructed from the Steiner system S(3,6,22)
# by "doubling" or "extending" the point at infinity.
# A simpler approach: Use the known list of 132 blocks.
# Since I cannot easily generate the full list here in my head, I will use a
# fast search that is very likely to find the 132 blocks quickly, or I will
# use a known construction.
# Actually, the task says "Maximize the number of blocks; the optimum 132 is attained
# exactly by the Steiner system S(5,6,12)."
# The current program's greedy approach often finds ~100-120 blocks but not always 132.
# A structural change is to use a **backtracking search with constraint propagation**
# or **local search (simulated annealing)** that is much more effective than random greedy.
# However, the most "structural" change that guarantees 132 is to **hardcode** the
# Witt design blocks. This is a valid approach for a fixed, small problem.
# Let's generate the 132 blocks of the Witt design S(5,6,12) using a known method.
# The 12 points are {0, 1, ..., 11}.
# The group PSL(2,11) has order 660. The stabilizer of a block has order 5 (since 660/132 = 5).
# The block orbits under PSL(2,11) are of size 132 (if the block is not fixed by a subgroup of order > 5).
# A base block is {0, 1, 3, 4, 5, 9}.
# We can generate the 132 blocks by applying all elements of PSL(2,11) to this base block.
# But implementing PSL(2,11) is non-trivial.
# Instead, we can use a **fast local search** (e.g., hill climbing with random restarts)
# that optimizes the number of 5-subsets covered. This is a different optimization method.
# Or, we can use a **backtracking search** that is guided by the most constrained 5-subsets.
# Let's try a **backtracking search** with heuristic ordering.
# This is a structural change from random greedy to systematic backtracking.

def _backtracking_search():
    """Attempt to find a packing of 132 blocks using backtracking."""
    # Order the 5-subsets by their frequency of appearing in 6-subsets
    # (all are equal, 6 each, so no help there).
    # Instead, order the 6-subsets by some heuristic.
    # We'll use a simple backtracking that tries to add blocks one by one,
    # checking for conflicts.
    # This is likely too slow for 132 blocks without strong pruning.
    
    # A better structural change: **Local Search (Simulated Annealing)**.
    # Start with a random valid packing (or the greedy result).
    # Try to improve it by swapping blocks or adding/removing blocks.
    # The objective is to maximize the number of 5-subsets covered.
    # This is a different optimization paradigm.
    
    # However, the most reliable way to get 132 is to **hardcode** the blocks.
    # Let's generate them once and store them.
    # I will use a known list of the 132 blocks of the Witt design S(5,6,12).
    # Since I don't have the list memorized, I will use a **fast construction**
    # based on the **Z_11 + infinity** model.
    
    # Points: {inf, 0, 1, ..., 10}
    # The group is the affine group AG(1,11) plus some additional symmetries.
    # The base block is {inf, 0, 1, 3, 4, 5} (example).
    # We need to generate the 132 blocks.
    # This is still complex.
    
    # Let's stick to a **fast local search** that is likely to find 132 quickly.
    # Start from the greedy solution and perform hill climbing.
    pass

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_blocks = []
    best_count = 0
    
    # Initial deterministic greedy to have a baseline
    covered = bytearray(792)
    current_blocks = []
    
    for idx in range(len(SIX_SUBSETS)):
        idxs = SIX_TO_FIVE_IDX[idx]
        if not any(covered[i] for i in idxs):
            current_blocks.append(list(SIX_SUBSETS[idx]))
            for i in idxs:
                covered[i] = 1
                
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    if best_count == 132:
        return best_blocks
    
    # Structural change: Use **local search (simulated annealing)** to improve the packing.
    # State: set of blocks (represented by their indices).
    # Objective: maximize the number of covered 5-subsets.
    # Moves: add a block, remove a block, swap a block.
    # This is a different optimization method from random greedy.
    
    def get_covered_count(blocks):
        covered = bytearray(792)
        for b_idx in blocks:
            for i in SIX_TO_FIVE_IDX[b_idx]:
                covered[i] = 1
        return sum(covered)
    
    def is_valid(blocks):
        covered = bytearray(792)
        for b_idx in blocks:
            for i in SIX_TO_FIVE_IDX[b_idx]:
                if covered[i]:
                    return False
                covered[i] = 1
        return True
    
    # Start from the greedy solution
    current_blocks_idx = [i for i in range(len(SIX_SUBSETS)) if i in [id2idx for id2idx, b in enumerate(SIX_SUBSETS) if b in best_blocks]]
    # This is inefficient. Let's just use the indices from the greedy step.
    # We need to track which indices were used.
    # Let's restart the greedy step but track indices.
    
    covered = bytearray(792)
    current_blocks_idx = []
    for idx in range(len(SIX_SUBSETS)):
        idxs = SIX_TO_FIVE_IDX[idx]
        if not any(covered[i] for i in idxs):
            current_blocks_idx.append(idx)
            for i in idxs:
                covered[i] = 1
    
    best_count = len(current_blocks_idx)
    best_blocks = [list(SIX_SUBSETS[i]) for i in current_blocks_idx]
    
    if best_count == 132:
        return best_blocks
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline and T > T_min:
        # Try a random move
        if random.random() < 0.3 and len(current_blocks_idx) > 0:
            # Remove a random block
            i = random.randrange(len(current_blocks_idx))
            removed = current_blocks_idx[i]
            del current_blocks_idx[i]
            # Check if we can add a random block
            j = random.randrange(len(SIX_SUBSETS))
            if j not in current_blocks_idx:
                idxs = SIX_TO_FIVE_IDX[j]
                covered = bytearray(792)
                valid = True
                for b_idx in current_blocks_idx:
                    for k in SIX_TO_FIVE_IDX[b_idx]:
                        covered[k] = 1
                for k in idxs:
                    if covered[k]:
                        valid = False
                        break
                if valid:
                    current_blocks_idx.append(j)
                    for k in idxs:
                        covered[k] = 1
                    new_count = len(current_blocks_idx)
                    if new_count > best_count:
                        best_count = new_count
                        best_blocks = [list(SIX_SUBSETS[i]) for i in current_blocks_idx]
                        if best_count == 132:
                            return best_blocks
                else:
                    # Reject
                    pass
            else:
                # Put it back
                current_blocks_idx.insert(i, removed)
        else:
            # Swap a random block with a random other block
            i = random.randrange(len(current_blocks_idx))
            j = random.randrange(len(SIX_SUBSETS))
            if j not in current_blocks_idx:
                old = current_blocks_idx[i]
                current_blocks_idx[i] = j
                # Check validity
                covered = bytearray(792)
                valid = True
                for b_idx in current_blocks_idx:
                    for k in SIX_TO_FIVE_IDX[b_idx]:
                        if covered[k]:
                            valid = False
                            break
                        covered[k] = 1
                    if not valid:
                        break
                if valid:
                    new_count = len(current_blocks_idx)
                    if new_count > best_count:
                        best_count = new_count
                        best_blocks = [list(SIX_SUBSETS[i]) for i in current_blocks_idx]
                        if best_count == 132:
                            return best_blocks
                else:
                    # Reject
                    current_blocks_idx[i] = old
        T *= cooling_rate
    
    return best_blocks
# EVOLVE-BLOCK-END
