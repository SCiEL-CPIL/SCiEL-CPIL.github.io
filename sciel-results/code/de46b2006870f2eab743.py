import time
import random
import itertools

# Precomputation for the specific task C(19, 7, 4)
V = 19
K = 7
T = 4

ALL_T_SUBSETS = list(itertools.combinations(range(V), T))
T_SUBSET_INDEX = {s: i for i, s in enumerate(ALL_T_SUBSETS)}
NUM_T_SUBSETS = len(ALL_T_SUBSETS)

ALL_BLOCKS = list(itertools.combinations(range(V), K))
BLOCK_T_SUBSETS = {}
for b in ALL_BLOCKS:
    BLOCK_T_SUBSETS[b] = [T_SUBSET_INDEX[t] for t in itertools.combinations(b, T)]

# Precompute which blocks contain which t-subsets for fast greedy
T_SUBSET_TO_BLOCKS = [[] for _ in range(NUM_T_SUBSETS)]
for b in ALL_BLOCKS:
    for idx in BLOCK_T_SUBSETS[b]:
        T_SUBSET_TO_BLOCKS[idx].append(b)

def simple_greedy(v, k, t):
    """Fallback greedy for non-standard parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        best_block = None
        max_cover = 0
        for b in itertools.combinations(range(v), k):
            cover_count = sum(1 for ts in itertools.combinations(b, t) if ts in uncovered)
            if cover_count > max_cover:
                max_cover = cover_count
                best_block = b
        blocks.append(best_block)
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(ts)
    return [list(b) for b in blocks]

def greedy_cover_1974():
    """Optimized greedy for 19,7,4 using max marginal gain."""
    uncovered = [0] * NUM_T_SUBSETS
    for i in range(NUM_T_SUBSETS):
        uncovered[i] = 1
        
    blocks = []
    remaining_uncovered = NUM_T_SUBSETS
    
    while remaining_uncovered > 0:
        best_block = None
        max_cover = 0
        
        # To speed up, we can just iterate all blocks. 
        # C(19,7) is 50388. 
        # Inner loop is K choose T = 35.
        # 50388 * 35 ~ 1.7M ops per step.
        # Greedy steps are ~150. Total ~250M ops. Might be slow in Python.
        
        # Optimization: Only consider blocks that cover at least one uncovered subset?
        # Hard to maintain efficiently without complex data structures.
        # Let's stick to simple iteration but break early if we find a block covering all remaining? Unlikely.
        
        # Alternative: Use the precomputed T_SUBSET_TO_BLOCKS.
        # Pick a random uncovered t-subset, look at its blocks, pick the best among them.
        # This is much faster.
        
        # Find an uncovered t-subset
        # We can maintain a list of uncovered indices, but it's easier to just scan or use a random probe.
        # Let's use a random probe to find an uncovered one.
        # If we pick a random t_idx and it's 0, pick another.
        
        # Actually, simpler: Just iterate all blocks and count gains.
        # But 1.7M ops * 150 steps = 250M. In Python, this might take 5-10s. Acceptable.
        
        for b in ALL_BLOCKS:
            cover_count = 0
            for idx in BLOCK_T_SUBSETS[b]:
                if uncovered[idx]:
                    cover_count += 1
            if cover_count > max_cover:
                max_cover = cover_count
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        
        # Update uncovered
        for idx in BLOCK_T_SUBSETS[best_block]:
            if uncovered[idx]:
                uncovered[idx] = 0
                remaining_uncovered -= 1
                
    return blocks

def construct(v, k, t):
    if (v, k, t) != (19, 7, 4):
        return simple_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initial Greedy Solution
    current_blocks = greedy_cover_1974()
    
    # 2. Setup Coverage Counts
    cover_count = [0] * NUM_T_SUBSETS
    for b in current_blocks:
        for idx in BLOCK_T_SUBSETS[b]:
            cover_count[idx] += 1
            
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    # Helper to check if a block is redundant given current cover_count
    def is_redundant(block):
        for idx in BLOCK_T_SUBSETS[block]:
            if cover_count[idx] <= 1:
                return False
        return True
        
    # Helper to remove a block from list and update cover_count
    def remove_block(block):
        current_blocks.remove(block)
        for idx in BLOCK_T_SUBSETS[block]:
            cover_count[idx] -= 1
            
    # Helper to add a block to list and update cover_count
    def add_block(block):
        current_blocks.append(block)
        for idx in BLOCK_T_SUBSETS[block]:
            cover_count[idx] += 1
            
    # 3. Local Search Loop
    while time.time() < deadline:
        # Phase A: Aggressive Reduction
        # Try to remove any redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[i]
                if is_redundant(b):
                    current_blocks.pop(i)
                    for idx in BLOCK_T_SUBSETS[b]:
                        cover_count[idx] -= 1
                    changed = True
                    
        # Update best if improved
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # Phase B: Random Swaps/Removals to escape local minima
        # We can try to remove a non-redundant block and see if we can replace it with a better one?
        # Or just try random swaps.
        
        # Try a random swap: Replace block i with a random block j
        # If the new set is valid (all cover_count >= 1), accept.
        # Note: Swapping doesn't reduce size immediately, but might enable further reductions.
        
        if len(current_blocks) > 0:
            # Pick a random block to swap
            i = random.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[i]
            
            # Pick a random new block
            new_block = random.choice(ALL_BLOCKS)
            
            if old_block == new_block:
                continue
                
            # Tentatively remove old_block
            for idx in BLOCK_T_SUBSETS[old_block]:
                cover_count[idx] -= 1
                
            # Check validity: all cover_count must be >= 1
            # This is O(NUM_T_SUBSETS) which is ~4000. Very fast.
            valid = True
            for c in cover_count:
                if c < 1:
                    valid = False
                    break
                    
            if valid:
                # Add new_block
                for idx in BLOCK_T_SUBSETS[new_block]:
                    cover_count[idx] += 1
                current_blocks[i] = new_block
            else:
                # Revert removal
                for idx in BLOCK_T_SUBSETS[old_block]:
                    cover_count[idx] += 1
                
    # Final cleanup
    # Run reduction one last time
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            b = current_blocks[i]
            if is_redundant(b):
                current_blocks.pop(i)
                for idx in BLOCK_T_SUBSETS[b]:
                    cover_count[idx] -= 1
                changed = True

    if len(current_blocks) < best_size:
        best_blocks = current_blocks[:]

    return [list(b) for b in best_blocks]
