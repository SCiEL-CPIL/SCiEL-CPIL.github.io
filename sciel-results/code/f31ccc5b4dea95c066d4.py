import itertools
import time
import random

# Precompute triple IDs for v=24, t=3
# Map each combination of 3 points to a unique integer 0..2023
TRIPLE_TO_ID = {}
ID_TO_TRIPLE = []
def init_triples(v, t):
    global TRIPLE_TO_ID, ID_TO_TRIPLE
    idx = 0
    for c in itertools.combinations(range(v), t):
        TRIPLE_TO_ID[c] = idx
        ID_TO_TRIPLE.append(c)
        idx += 1

# Initialize for the specific task parameters if they match, 
# but the function signature implies v, k, t are passed.
# However, the task description says "The evaluator passes this task's parameters v, k and t."
# and specifically mentions C(24,7,3).
# To be safe and general, we should handle the specific case efficiently but 
# the prompt implies we are rewriting for THIS task.
# I will assume v=24, k=7, t=3 for optimization but write it to work generally 
# if the overhead is small. Given the 150s limit, general code might be too slow 
# if not optimized for 24. I will hardcode optimizations for v=24, k=7, t=3 
# but keep the function signature.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    if v != 24 or k != 7 or t != 3:
        # Fallback to simple greedy for other parameters if ever called
        return _simple_greedy(v, k, t)
        
    # 1. Precompute triple mapping for v=24, t=3
    TRIPLE_TO_ID = {}
    ID_TO_TRIPLE = []
    idx = 0
    for c in itertools.combinations(range(24), 3):
        TRIPLE_TO_ID[c] = idx
        ID_TO_TRIPLE.append(c)
        idx += 1
    num_triples = 2024
    
    # Helper to get triple IDs for a block
    def get_block_triples(block):
        return [TRIPLE_TO_ID[tup] for tup in itertools.combinations(block, 3)]
    
    # 2. Greedy Construction
    coverage = [0] * num_triples
    blocks = []
    block_triples_list = [] # list of lists of triple IDs for each block
    
    # Find initial uncovered triples
    uncovered_count = num_triples
    
    # Greedy loop
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find a triple with coverage 0
        # Optimization: keep a list of uncovered triple IDs? 
        # Scanning 2024 ints is fast.
        target_triple_id = -1
        for i in range(num_triples):
            if coverage[i] == 0:
                target_triple_id = i
                break
        
        if target_triple_id == -1:
            break # Should not happen if uncovered_count > 0
            
        target_triple = ID_TO_TRIPLE[target_triple_id]
        
        best_block = None
        best_gain = -1
        
        # Candidate blocks must contain the target triple
        # Points in target triple: p0, p1, p2
        # We need to choose 4 more points from the remaining 21 points
        remaining_points = [p for p in range(24) if p not in target_triple]
        
        # Iterate through combinations of 4 points from remaining
        # C(21, 4) = 5985 combinations. This is manageable per step.
        # But we do this for each new block.
        # Total blocks ~ 76. 76 * 6000 = 456,000 evaluations.
        # Each evaluation involves checking 35 triples.
        # 456,000 * 35 = 16M operations. Fast enough.
        
        for combo in itertools.combinations(remaining_points, 4):
            candidate_block = tuple(sorted(target_triple + combo))
            
            # Calculate gain: number of triples in candidate_block with coverage 0
            gain = 0
            for tup in itertools.combinations(candidate_block, 3):
                tid = TRIPLE_TO_ID[tup]
                if coverage[tid] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = list(candidate_block)
                
                # Early exit if we covered all 35 triples of this block? 
                # No, gain is max 35. If gain == 35, it's perfect.
                if gain == 35:
                    break
                    
        if best_block is None:
            break # Error
            
        # Add best_block
        blocks.append(best_block)
        b_triples = get_block_triples(best_block)
        block_triples_list.append(b_triples)
        
        for tid in b_triples:
            if coverage[tid] == 0:
                uncovered_count -= 1
            coverage[tid] += 1
            
    # 3. Local Search Optimization
    # State: blocks, block_triples_list, coverage
    # Objective: Minimize len(blocks)
    
    best_blocks = [b[:] for b in blocks]
    best_count = len(blocks)
    
    # Helper to check if a block is redundant
    def is_redundant(block_idx):
        b_triples = block_triples_list[block_idx]
        for tid in b_triples:
            if coverage[tid] <= 1:
                return False
        return True
        
    # Helper to remove a block
    def remove_block(block_idx):
        nonlocal coverage, blocks, block_triples_list
        b_triples = block_triples_list[block_idx]
        for tid in b_triples:
            coverage[tid] -= 1
        # Remove from lists
        # Using pop is O(N), but N is small (<100)
        del blocks[block_idx]
        del block_triples_list[block_idx]
        
    # Main optimization loop
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        removed_any = False
        # Check all blocks
        redundant_indices = []
        for i in range(len(blocks)):
            if is_redundant(i):
                redundant_indices.append(i)
                
        if redundant_indices:
            # Remove one (or all)
            # Removing one at a time allows us to update coverage carefully
            # But if multiple are redundant, they might depend on each other?
            # No, "redundant" means covered by OTHERS. 
            # If A is redundant w.r.t B, and B is redundant w.r.t A, 
            # it means A and B cover the same unique triples? 
            # If coverage[tid] == 2 for all tid in A, and coverage[tid] == 2 for all tid in B,
            # and A and B share no unique triples, then removing A leaves coverage >= 1.
            # So we can remove all identified redundant blocks in one go?
            # Let's be safe and remove one by one, or just one.
            # Actually, if we remove A, coverage drops. B might no longer be redundant.
            # So we should re-check.
            # For efficiency, let's just remove one random redundant block.
            idx_to_remove = random.choice(redundant_indices)
            remove_block(idx_to_remove)
            removed_any = True
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
        if not removed_any:
            # Phase 2: Perturbation (Swap)
            if len(blocks) == 0:
                break
                
            # Pick a random block
            i = random.randint(0, len(blocks) - 1)
            block = blocks[i]
            
            # Pick a point to remove and a point to add
            x = random.choice(block)
            y = random.choice([p for p in range(24) if p not in block])
            
            new_block = tuple(sorted((set(block) - {x}) | {y}))
            
            # Check validity
            old_triples = set(block_triples_list[i])
            new_triples = set(get_block_triples(new_block))
            
            # Triples lost: in old, not in new
            lost = old_triples - new_triples
            # Triples gained: in new, not in old
            gained = new_triples - old_triples
            
            valid = True
            for tid in lost:
                if coverage[tid] <= 1:
                    valid = False
                    break
                    
            if valid:
                # Apply swap
                # Update coverage
                for tid in lost:
                    coverage[tid] -= 1
                for tid in gained:
                    coverage[tid] += 1
                    
                # Update block
                blocks[i] = list(new_block)
                block_triples_list[i] = list(new_triples)
                
                # Note: We don't immediately check for redundancy here, 
                # it will be caught in the next iteration of the loop.
                
    return best_blocks

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
