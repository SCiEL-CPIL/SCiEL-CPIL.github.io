import time
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # cover_count[i] is the number of blocks covering triple i
    cover_count = np.zeros(n_triples, dtype=np.int32)
    blocks = []
    
    def add_block(block):
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            cover_count[triple_to_idx[tr]] += 1

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            cover_count[triple_to_idx[tr]] -= 1

    def get_uncovered():
        return np.where(cover_count == 0)[0]

    def gain_of_block(block):
        """Calculate how many currently uncovered triples this block covers."""
        gain = 0
        for tr in itertools.combinations(block, t):
            if cover_count[triple_to_idx[tr]] == 0:
                gain += 1
        return gain

    def greedy_construct():
        """Greedy construction: add block with max gain."""
        nonlocal blocks, cover_count
        
        while True:
            if time.time() > deadline - 10:
                break
            
            uncovered_indices = get_uncovered()
            if len(uncovered_indices) == 0:
                break
            
            # We need to find a block that covers many uncovered triples.
            # Strategy: Sample some uncovered triples, build candidate blocks around them.
            # Since k=7, t=3, we can pick a triple and extend it.
            
            best_block = None
            best_gain = -1
            
            # Sample up to 50 uncovered triples to try as seeds
            # If there are few, take all
            n_samples = min(50, len(uncovered_indices))
            # Use deterministic sampling based on time or fixed seed for reproducibility in this run?
            # The task asks for a rewrite, determinism is good for debugging but randomization helps escape local minima.
            # We will use a simple random state.
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            
            # Shuffle uncovered indices to pick random samples
            sample_indices = rng.choice(uncovered_indices, size=n_samples, replace=False)
            
            for idx in sample_indices:
                seed_triple = all_triples[idx]
                
                # Build a block containing seed_triple
                # Start with the 3 points
                block = list(seed_triple)
                remaining_points = set(range(v))
                for p in seed_triple:
                    remaining_points.discard(p)
                
                # Greedily add 4 more points
                for _ in range(k - t):
                    best_cand = None
                    best_local_gain = -1
                    
                    # Evaluate each candidate point
                    # To speed this up, we can just check the gain of the resulting partial block
                    # But we need to check the final gain.
                    # Approximation: pick point that maximizes overlap with uncovered triples involving current block points.
                    
                    # For simplicity and speed, we check the actual gain of the full block if we were to finish it?
                    # No, that's expensive.
                    
                    # Heuristic: Pick point that maximizes number of uncovered triples formed with 2 points from current block.
                    # Let current block be B. For candidate c, new block B' = B U {c}.
                    # New triples are (c, x, y) where x, y in B.
                    # There are C(|B|, 2) such triples.
                    # Count how many are uncovered.
                    
                    for cand in remaining_points:
                        # Check triples (cand, b1, b2)
                        local_gain = 0
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                tr = (block[i], block[j], cand)
                                tr = tuple(sorted(tr))
                                if cover_count[triple_to_idx[tr]] == 0:
                                    local_gain += 1
                        
                        if local_gain > best_local_gain:
                            best_local_gain = local_gain
                            best_cand = cand
                    
                    if best_cand is not None:
                        block.append(best_cand)
                        remaining_points.discard(best_cand)
                    else:
                        # Fallback: pick random
                        best_cand = list(remaining_points)[0]
                        block.append(best_cand)
                        remaining_points.discard(best_cand)
                        
                block = tuple(sorted(block))
                gain = gain_of_block(block)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None or best_gain == 0:
                # If we can't find a block with positive gain, we might be stuck or covered
                # If covered, break. If stuck, we might need a different strategy or accept failure.
                # But since it's a covering problem, there should always be a block covering an uncovered triple.
                # If best_gain == 0, it means the sampled seeds didn't lead to good blocks, or all are covered.
                if len(uncovered_indices) > 0 and best_gain == 0:
                     # Try one more time with a specific uncovered triple
                     idx = uncovered_indices[0]
                     seed_triple = all_triples[idx]
                     block = list(seed_triple)
                     remaining_points = set(range(v))
                     for p in seed_triple:
                         remaining_points.discard(p)
                     for _ in range(k - t):
                         best_cand = list(remaining_points)[0]
                         block.append(best_cand)
                         remaining_points.discard(best_cand)
                     best_block = tuple(sorted(block))
                     best_gain = gain_of_block(best_block)
                     
                if best_gain == 0:
                     break
                
                add_block(best_block)
            else:
                add_block(best_block)

    # Phase 1: Greedy Construction
    greedy_construct()
    
    # Phase 2: Local Search / Improvement
    # 1. Redundancy Removal
    # 2. Block Replacement
    
    for iteration in range(20): # Limit iterations
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved:
            improved = False
            
            # --- Redundancy Check ---
            # Check if any block is redundant
            redundant_idx = -1
            for i, block in enumerate(blocks):
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove
                remove_block(block)
                if np.any(cover_count == 0):
                    # Restore
                    add_block(block)
                else:
                    # Redundant!
                    del blocks[i]
                    redundant_idx = i
                    improved = True
                    break # Restart scan as indices changed
            
            if redundant_idx != -1:
                continue # Restart while loop
        
        # --- Replacement Search ---
        # Try to replace blocks with better ones
        # A block is "worse" if it covers fewer unique triples than another potential block
        # Strategy: For each block, try to find a replacement that covers all its unique triples
        # and maximizes total coverage.
        
        if len(blocks) == 0:
            break
            
        # Sample some blocks to try replacing
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        n_replace_attempts = min(20, len(blocks))
        indices = rng.choice(len(blocks), size=n_replace_attempts, replace=False)
        
        for bi in indices:
            if time.time() > deadline - 5:
                break
                
            old_block = blocks[bi]
            
            # Identify triples uniquely covered by old_block
            unique_triples = []
            for tr in itertools.combinations(old_block, t):
                if cover_count[triple_to_idx[tr]] == 1:
                    unique_triples.append(tr)
            
            # If no unique triples, we can try to replace with any block that covers at least as many total triples?
            # Actually, if no unique triples, removing it is safe (handled by redundancy check above, unless it became redundant in a previous pass).
            # Wait, redundancy check runs first. So if we are here, the block has at least one unique triple OR it was just restored.
            # If it has unique triples, the replacement MUST cover them.
            
            # Generate candidate replacement
            # Heuristic: Pick one of the unique triples as a seed, or if none, pick random uncovered/low-cover triple.
            
            seed_triple = None
            if unique_triples:
                seed_triple = unique_triples[0]
            else:
                # Pick a triple with low count
                min_count = np.min(cover_count)
                candidates = np.where(cover_count == min_count)[0]
                seed_triple = all_triples[rng.choice(candidates)]
            
            # Build candidate block around seed_triple
            block = list(seed_triple)
            remaining_points = set(range(v))
            for p in seed_triple:
                remaining_points.discard(p)
            
            for _ in range(k - t):
                best_cand = None
                best_local_gain = -1
                for cand in remaining_points:
                    local_gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tr = (block[i], block[j], cand)
                            tr = tuple(sorted(tr))
                            # We want to cover uncovered triples, and ideally keep cover count low for others?
                            # Just maximizing uncovered coverage is a good heuristic.
                            if cover_count[triple_to_idx[tr]] == 0:
                                local_gain += 1
                            # Also, if this triple is one of the unique ones we MUST cover, we should prioritize it?
                            # The seed ensures we cover at least one unique triple (if it was a seed).
                            # The greedy extension should cover more.
                            
                    if local_gain > best_local_gain:
                        best_local_gain = local_gain
                        best_cand = cand
                
                if best_cand is not None:
                    block.append(best_cand)
                    remaining_points.discard(best_cand)
                else:
                    best_cand = list(remaining_points)[0]
                    block.append(best_cand)
                    remaining_points.discard(best_cand)
                    
            new_block = tuple(sorted(block))
            
            # Check if new_block covers all unique_triples of old_block
            if unique_triples:
                new_block_set = set(itertools.combinations(new_block, t))
                valid = all(tr in new_block_set for tr in unique_triples)
                if not valid:
                    continue # Invalid replacement
            
            # Calculate gain/loss
            # Remove old, add new, check if any count drops to 0
            remove_block(old_block)
            # Temporarily add new
            for tr in itertools.combinations(new_block, t):
                cover_count[triple_to_idx[tr]] += 1
            
            if np.any(cover_count == 0):
                # Invalid replacement, restore
                for tr in itertools.combinations(new_block, t):
                    cover_count[triple_to_idx[tr]] -= 1
                add_block(old_block)
            else:
                # Valid replacement!
                # Does it improve "quality"? 
                # Since we just need a valid covering, any valid replacement is fine.
                # However, to reduce future redundancy, we might prefer blocks that cover more "shared" triples?
                # Or simply accept it and let redundancy check clean up.
                blocks[bi] = new_block
                improved = True

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline - 2:
                break
            block = blocks[i]
            remove_block(block)
            if np.any(cover_count == 0):
                add_block(block)
            else:
                del blocks[i]
                changed = True
                break

    # Return list of lists
    return [list(b) for b in blocks]

# Ensure the function signature matches
# The task says: construct(v, k, t) -> list of blocks
# The current program has: def construct(v, k, t):
# So we are good.

if __name__ == "__main__":
    # Quick test
    # v, k, t = 24, 7, 3
    # res = construct(v, k, t)
    # print(f"Blocks: {len(res)}")
    pass
