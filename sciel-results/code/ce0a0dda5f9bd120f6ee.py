import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Map triple to list of blocks covering it? 
    # No, generating blocks on the fly is faster for memory.
    # But we need to find blocks covering a specific triple quickly.
    # For a triple (a,b,c), a block is (a,b,c, x,y,z,w) where {x,y,z,w} is a subset of {0..v-1}\{a,b,c}.
    # We can generate these on the fly.
    
    uncovered = set(triples)
    selected_blocks = []
    
    # Greedy Set Cover
    # To speed up, we can randomize the order of uncovered triples or just pick one.
    # Picking the triple with the fewest covering blocks is optimal but hard to compute on the fly.
    # Picking a random uncovered triple and the best block for it is a good heuristic.
    
    rng = random.Random(42)
    
    while uncovered and time.time() < deadline - 30.0:
        # Pick a random uncovered triple
        tri = rng.choice(list(uncovered))
        
        # Generate all blocks covering this triple
        # Remaining points
        remaining = [p for p in range(v) if p not in tri]
        # We need to choose k-t points from remaining
        # k=7, t=3 -> choose 4 from 21
        best_block = None
        max_gain = -1
        
        # Optimization: Don't check all 5985 blocks if it's too slow.
        # But 5985 is small. Let's try all.
        for combo in itertools.combinations(remaining, k - t):
            block = tuple(sorted(tri + list(combo)))
            # Calculate gain: how many uncovered triples does this block cover?
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        selected_blocks.append(best_block)
        
        # Update uncovered
        for sub in itertools.combinations(best_block, t):
            if sub in uncovered:
                uncovered.remove(sub)
                
    # If greedy didn't finish or failed, we might have uncovered triples.
    # But greedy set cover is guaranteed to finish if we keep going.
    # The time limit might stop us. If so, we have an invalid cover.
    # We must ensure validity. If time runs out, we should return the best valid cover found?
    # No, the task requires a *valid* cover.
    # If we run out of time, we can't guarantee validity.
    # However, the greedy loop above is fast. 76 blocks * 5985 blocks * 35 triples/block * check = ~16M ops.
    # This should finish well under 115s.
    
    # Redundancy removal
    def remove_redundant(blocks, v, t, triple_to_idx, deadline):
        if not blocks:
            return []
            
        n_blocks = len(blocks)
        # Precompute triples for each block
        block_triples = []
        for b in blocks:
            tris = []
            for tri in itertools.combinations(b, t):
                tris.append(tri)
            block_triples.append(tris)
        
        # Count coverage
        cov_count = defaultdict(int)
        for tris in block_triples:
            for ti in tris:
                cov_count[ti] += 1
        
        active = [True] * n_blocks
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            for i in range(n_blocks):
                if not active[i]:
                    continue
                # Check if block i is redundant
                redundant = True
                for ti in block_triples[i]:
                    if cov_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    changed = True
                    for ti in block_triples[i]:
                        cov_count[ti] -= 1
        
        return [blocks[i] for i in range(n_blocks) if active[i]]
    
    selected_blocks = remove_redundant(selected_blocks, v, t, triple_to_idx, deadline - 10.0)
    
    # Local search with point swaps
    def local_search_swaps(blocks, v, k, t, triple_to_idx, deadline):
        blocks = [list(b) for b in blocks]
        rng = random.Random(42)
        
        def compute_cov_count(blks):
            cov = defaultdict(int)
            for b in blks:
                for tri in itertools.combinations(b, t):
                    cov[tri] += 1
            return cov
        
        cov_count = compute_cov_count(blocks)
        improved = True
        iters = 0
        
        while improved and time.time() < deadline:
            improved = False
            iters += 1
            if iters > 100: # Limit iterations
                break
            
            # First, remove any redundant blocks
            blocks = remove_redundant(blocks, v, t, triple_to_idx, deadline)
            cov_count = compute_cov_count(blocks)
            
            # Try point swaps
            for _ in range(50):
                if time.time() >= deadline or len(blocks) == 0:
                    break
                    
                idx = rng.randint(0, len(blocks) - 1)
                old_x = rng.choice(blocks[idx])
                new_x = rng.randint(0, v - 1)
                
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                # Compute effect of swap
                old_block = blocks[idx]
                new_block = old_block.copy()
                new_block.remove(old_x)
                new_block.append(new_x)
                new_block.sort()
                
                # Get triples covered by old and new blocks
                old_tris = set(itertools.combinations(old_block, t))
                new_tris = set(itertools.combinations(new_block, t))
                
                # Check if any triple would become uncovered
                valid_swap = True
                for ti in old_tris:
                    if ti not in new_tris and cov_count[ti] == 1:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
                
                # Apply swap
                blocks[idx] = new_block
                for ti in old_tris:
                    cov_count[ti] -= 1
                for ti in new_tris:
                    cov_count[ti] += 1
                improved = True
        
        return blocks
    
    if time.time() < deadline - 5.0:
        selected_blocks = local_search_swaps(selected_blocks, v, k, t, triple_to_idx, deadline - 5.0)
    
    # Final redundancy removal
    selected_blocks = remove_redundant(selected_blocks, v, k, t, triple_to_idx, deadline)
    
    # Verify validity
    if not selected_blocks:
        # Fallback: Return a valid cover constructed simply
        # This should not happen, but better than returning []
        # Simple valid cover: All blocks containing 0
        # This is huge, but valid.
        # Or just return the greedy result if it was valid.
        # Since we can't be sure, let's try to construct a minimal valid cover quickly.
        # Actually, if selected_blocks is empty, the greedy failed.
        # Let's just return the greedy result before redundancy removal if it was valid?
        # We don't have it saved.
        # Let's assume the greedy worked. If not, we have a problem.
        # For now, return [] is invalid.
        # Let's return a single block? No, must cover all triples.
        # The error message "return 1..2024 blocks" implies the list length must be between 1 and 2024.
        # If we return [], it fails.
        # We must ensure we return a valid cover.
        pass

    # Check if valid
    covered = set()
    for b in selected_blocks:
        for tri in itertools.combinations(b, t):
            covered.add(tri)
            
    if covered != set(triples):
        # Invalid cover. This is bad.
        # Try to repair by adding blocks greedily for uncovered triples
        uncovered = set(triples) - covered
        while uncovered and time.time() < deadline - 1.0:
            tri = rng.choice(list(uncovered))
            remaining = [p for p in range(v) if p not in tri]
            best_block = None
            max_gain = -1
            for combo in itertools.combinations(remaining, k - t):
                block = tuple(sorted(tri + list(combo)))
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = block
            if best_block:
                selected_blocks.append(best_block)
                for sub in itertools.combinations(best_block, t):
                    if sub in uncovered:
                        uncovered.remove(sub)
            else:
                break

    return [list(b) for b in selected_blocks]
