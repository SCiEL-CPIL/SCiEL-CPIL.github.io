import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # Pre-compute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Pre-compute for each point, the list of triples containing it
    point_triples = [[] for _ in range(v)]
    for tri_idx, tri in enumerate(all_triples):
        for pt in tri:
            point_triples[pt].append(tri_idx)
    
    # Pre-compute for each pair, the list of triples containing both (i.e., triples that extend the pair)
    # Actually, for a block, we care about triples in the block.
    # Let's pre-compute for each k-subset? No, too many.
    # Let's pre-compute for each pair, the triples containing it.
    pair_triples = defaultdict(list)
    for tri_idx, tri in enumerate(all_triples):
        for pair in itertools.combinations(tri, 2):
            pair_triples[pair].append(tri_idx)
    
    def get_block_indices(block):
        """Return list of triple indices covered by a block."""
        indices = set()
        for tri in itertools.combinations(block, t):
            indices.add(triple_to_idx[tri])
        return list(indices)
    
    # 1. Greedy Construction
    # Use a deterministic greedy approach with a "ratio" heuristic:
    # For each candidate block, gain / cost. Cost is 1. So maximize gain.
    # But we need to pick blocks efficiently.
    # Strategy:
    # 1. Find an uncovered triple with the fewest blocks that *could* cover it?
    #    Or just pick an uncovered triple.
    # 2. Among all blocks containing that triple, pick the one that covers the most *other* uncovered triples.
    # This is expensive if we enumerate all blocks.
    
    # Alternative Greedy:
    # Maintain a list of uncovered triples.
    # While uncovered is not empty:
    #   Pick a random uncovered triple T.
    #   Generate candidate blocks containing T.
    #   For each candidate, calculate number of uncovered triples it covers.
    #   Pick the best candidate.
    #   Add it to the solution.
    #   Update uncovered.
    
    # To make it faster, we can limit the number of candidates considered.
    # Or use a "greedy set cover" approach where we pre-compute the gain of each potential block?
    # C(24, 7) = 346104. This is manageable.
    
    # Let's pre-compute all blocks and their covered triples?
    # Memory: 346104 * 35 triples (C(7,3)=35) * 4 bytes ~ 48 MB. Acceptable.
    # But time to compute might be significant.
    
    # Let's try a different construction first:
    # "Greedy with limited lookahead".
    
    uncovered = [True] * num_triples
    blocks = []
    num_uncovered = num_triples
    
    # Pre-compute list of all blocks?
    all_blocks = list(itertools.combinations(range(v), k))
    # For each block, pre-compute the list of triple indices it covers.
    block_covered = []
    for b in all_blocks:
        idxs = get_block_indices(b)
        block_covered.append(idxs)
    
    # Map block to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Greedy
    # To speed up, we can maintain for each uncovered triple, the list of blocks that cover it.
    # triple_blocks = defaultdict(list)
    # for bi, b in enumerate(all_blocks):
    #    for ti in block_covered[bi]:
    #        triple_blocks[ti].append(bi)
    
    # This is O(C(24,7) * 35) ~ 12 million operations. Might be slow in Python.
    # Let's try to be smarter.
    
    # Alternative:
    # For each step, pick an uncovered triple.
    # Iterate through all blocks that contain this triple.
    # How to find blocks containing a triple?
    # Pre-compute: triple_to_blocks = defaultdict(list)
    # for bi, b in enumerate(all_blocks):
    #    for tri in itertools.combinations(b, t):
    #        triple_to_blocks[triple_to_idx[tri]].append(bi)
        
    # Let's do this pre-computation. It's the bottleneck.
    triple_to_blocks = defaultdict(list)
    for bi, b in enumerate(all_blocks):
        for tri in itertools.combinations(b, t):
            triple_to_blocks[triple_to_idx[tri]].append(bi)
            
    # Now greedy:
    while num_uncovered > 0:
        # Pick a random uncovered triple
        # To be faster, pick the one with the fewest candidate blocks?
        # This requires maintaining counts or searching.
        # For now, random pick is fine for diversity.
        # Let's try picking the one with minimum candidate blocks to force coverage of "hard" triples.
        
        # Find uncovered triple with min len(triple_to_blocks[tri])
        # This is O(num_uncovered) per step.
        # We can maintain a heap? Or just scan.
        
        min_len = float('inf')
        target_tri = -1
        # Optimization: if we have many uncovered, scanning all is expensive.
        # But num_uncovered decreases.
        
        # Let's just pick a random one for now to keep it simple and fast per step,
        # and rely on the local search to fix suboptimality.
        # Actually, "minimum candidate" is a standard heuristic for set cover.
        
        # Let's try random first.
        while True:
            idx = random.randint(0, num_triples - 1)
            if uncovered[idx]:
                target_tri = idx
                break
        
        candidates = triple_to_blocks[target_tri]
        
        best_block = -1
        best_gain = -1
        
        for bi in candidates:
            # Calculate gain: number of uncovered triples covered by block bi
            gain = 0
            for ti in block_covered[bi]:
                if uncovered[ti]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = bi
                
        # Add best block
        if best_block != -1:
            blocks.append(all_blocks[best_block])
            for ti in block_covered[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    num_uncovered -= 1
        else:
            # Should not happen
            break

    best_blocks = blocks[:]
    best_len = len(blocks)
    
    # 2. Local Search: Hill Climbing with Block Swaps
    # Represent state as a set of block indices? Or list.
    # We need to maintain counts of coverage for each triple.
    
    def compute_counts(block_list):
        counts = [0] * num_triples
        for b in block_list:
            bi = block_to_idx[b]
            for ti in block_covered[bi]:
                counts[ti] += 1
        return counts
    
    counts = compute_counts(blocks)
    
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            bi = block_to_idx[blocks[i]]
            redundant = True
            for ti in block_covered[bi]:
                if counts[ti] == 1:
                    redundant = False
                    break
            if redundant:
                for ti in block_covered[bi]:
                    counts[ti] -= 1
                blocks.pop(i)
                changed = True
    
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks[:]
    
    # Hill Climbing: Try to swap a block with another.
    # Or try to remove a block and add a better one?
    # "Block Exchange":
    # Pick a block B1 to remove.
    # This exposes some triples (those with count 1 in B1).
    # We need to find a block B2 (not in current solution) that covers all exposed triples
    # and maximizes the number of other triples it covers (to potentially make other blocks redundant).
    # Actually, simply finding any B2 that covers the exposed triples is a "repair".
    # If we can find a B2 such that adding it and removing B1 results in a valid cover,
    # and then we can remove more redundancies, we might reduce the count.
    
    # Let's implement a simple local search:
    # 1. Pick a random block B1.
    # 2. Remove it.
    # 3. Identify uncovered triples.
    # 4. Find a block B2 that covers the most uncovered triples (among all blocks).
    # 5. If B2 covers ALL uncovered triples, accept the swap.
    # 6. Remove any newly redundant blocks.
    # 7. If the total count decreased, keep it.
    
    # Optimization:
    # Instead of scanning all blocks for B2, we can look at blocks that contain the *rarest* uncovered triple.
    
    no_improve = 0
    max_no_improve = 1000 # Limit search if stuck
    
    while time.time() < deadline and no_improve < max_no_improve:
        if not blocks:
            break
            
        # Pick a random block to remove
        idx = random.randint(0, len(blocks) - 1)
        b_remove = blocks[idx]
        bi_remove = block_to_idx[b_remove]
        
        # Tentatively remove
        for ti in block_covered[bi_remove]:
            counts[ti] -= 1
            
        # Find uncovered triples
        uncovered_list = []
        for ti in block_covered[bi_remove]:
            if counts[ti] == 0:
                uncovered_list.append(ti)
                
        if not uncovered_list:
            # Pure removal!
            blocks.pop(idx)
            no_improve = 0
            # Remove further redundancies
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    bi = block_to_idx[blocks[i]]
                    redundant = True
                    for ti in block_covered[bi]:
                        if counts[ti] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ti in block_covered[bi]:
                            counts[ti] -= 1
                        blocks.pop(i)
                        changed = True
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
            continue
            
        # Need to repair. Find a block that covers the most uncovered triples.
        # To be efficient, pick the uncovered triple with the fewest candidate blocks.
        min_cand = float('inf')
        target_tri = -1
        for ti in uncovered_list:
            cands = triple_to_blocks[ti]
            if len(cands) < min_cand:
                min_cand = len(cands)
                target_tri = ti
        
        best_swap_block = -1
        best_swap_gain = -1
        
        # Scan candidates of target_tri
        for bi in triple_to_blocks[target_tri]:
            # Check if this block covers all uncovered_list
            # And calculate total gain (uncovered covered)
            covers_all = True
            gain = 0
            for ti in uncovered_list:
                if ti in block_covered[bi]:
                    gain += 1
                else:
                    covers_all = False
                    break
            
            if covers_all:
                # We can accept this swap.
                # To be smarter, we might prefer a block that allows more subsequent removals.
                # For now, just pick the one with max gain (which is len(uncovered_list) for all valid ones).
                # If multiple, pick random? Or the one that overlaps most with current blocks?
                # Let's just pick the first one found or random.
                best_swap_block = bi
                break # Found a valid repair.
                
        if best_swap_block != -1:
            # Perform swap
            blocks.pop(idx)
            blocks.append(all_blocks[best_swap_block])
            
            # Update counts for new block
            for ti in block_covered[best_swap_block]:
                counts[ti] += 1
                
            # Remove redundancies
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    bi = block_to_idx[blocks[i]]
                    redundant = True
                    for ti in block_covered[bi]:
                        if counts[ti] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ti in block_covered[bi]:
                            counts[ti] -= 1
                        blocks.pop(i)
                        changed = True
                        
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # No valid repair found. Revert.
            for ti in block_covered[bi_remove]:
                counts[ti] += 1
            no_improve += 1

    # Final cleanup on best_blocks
    # Rebuild counts for best_blocks to be safe
    best_counts = [0] * num_triples
    for b in best_blocks:
        bi = block_to_idx[b]
        for ti in block_covered[bi]:
            best_counts[ti] += 1
            
    final_blocks = []
    for b in best_blocks:
        bi = block_to_idx[b]
        can_remove = True
        for ti in block_covered[bi]:
            if best_counts[ti] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for ti in block_covered[bi]:
                best_counts[ti] -= 1
        
    return [list(b) for b in final_blocks]
