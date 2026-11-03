import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute triple indices
    # Map (i, j, k) to a unique index 0..C(v,t)-1
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # uncovered[i] is True if triple i is not yet covered
    uncovered = [True] * num_triples
    blocks = []
    
    # Helper to update uncovered status for a block
    def mark_covered(block):
        for comb in itertools.combinations(block, t):
            idx = triple_to_idx[comb]
            uncovered[idx] = False
            
    # Greedy Construction
    while any(uncovered):
        if time.time() > deadline - 5: # Leave 5s for cleanup
            break
            
        # Find first uncovered triple
        # Optimization: Maintain a list of uncovered indices?
        # For now, scan. 2024 is small.
        first_uncovered_idx = -1
        for i in range(num_triples):
            if uncovered[i]:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break
            
        start_triple = triples[first_uncovered_idx]
        block = list(start_triple)
        
        # Grow block to size k
        while len(block) < k:
            best_gain = -1
            best_point = -1
            # Candidates are all points not in block
            # We can filter candidates to those that appear in uncovered triples?
            # For now, try all.
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # New triples formed by adding x to block:
                # Must contain x and t-1 elements from block
                for rest in itertools.combinations(block, t - 1):
                    # rest is a tuple of t-1 elements
                    # Form triple
                    # Sort to match key in triple_to_idx
                    # Since t=3, rest has 2 elements.
                    # triple = (x, rest[0], rest[1]) sorted
                    temp = [x] + list(rest)
                    temp.sort()
                    key = tuple(temp)
                    idx = triple_to_idx.get(key, -1)
                    if idx != -1 and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    # Early stop if gain is max possible?
                    # Max gain for a point is C(len(block), t-1)
                    # If gain == C(len(block), t-1), it's perfect for this step.
                    # But we still need to check other points? No, if one is perfect, it's as good as it gets.
                    # However, "perfect" might be rare.
            
            if best_point == -1:
                # Should not happen if v > k
                break
                
            block.append(best_point)
            
        block.sort()
        
        # Check if block is already in blocks? 
        # Greedy usually produces distinct blocks because we start with a new uncovered triple.
        # But it's possible to generate a duplicate block if the logic is flawed.
        # The seed program didn't check, but distinctness is required.
        # If a block is duplicate, it adds no new coverage. We should skip it.
        is_dup = False
        for b in blocks:
            if b == block:
                is_dup = True
                break
        
        if not is_dup:
            blocks.append(block)
            mark_covered(block)
            
    # If we timed out during construction, we might have uncovered triples.
    # We must return a valid covering.
    # If we broke early, we need to cover the rest.
    # Simple fallback: Add blocks for remaining uncovered triples greedily, even if slow, 
    # but we are close to deadline. 
    # Actually, if we break, we should just finish the current block and stop.
    # But the result must be VALID.
    # So we must ensure all triples are covered.
    
    # Check validity
    covered_final = [False] * num_triples
    for b in blocks:
        for comb in itertools.combinations(b, t):
            covered_final[triple_to_idx[comb]] = True
            
    if not all(covered_final):
        # Emergency fix: Add blocks for uncovered triples
        # This is slow, but better than invalid.
        remaining_uncovered = [i for i in range(num_triples) if not covered_final[i]]
        for idx in remaining_uncovered:
            if time.time() > deadline - 1:
                break
            tr = triples[idx]
            # Create a block containing tr
            # Just pick random points to fill it
            block = list(tr)
            while len(block) < k:
                # Pick random point not in block
                while True:
                    p = random.randint(0, v-1)
                    if p not in block:
                        block.append(p)
                        break
            block.sort()
            # Check dup
            if block not in blocks:
                blocks.append(block)
                # Update covered_final
                for comb in itertools.combinations(block, t):
                    covered_final[triple_to_idx[comb]] = True

    # Post-processing: Remove redundant blocks
    # Calculate coverage counts
    coverage_count = [0] * num_triples
    for b in blocks:
        for comb in itertools.combinations(b, t):
            coverage_count[triple_to_idx[comb]] += 1
            
    # Try to remove blocks
    # A block is redundant if all its triples are covered by other blocks.
    # i.e., for all T in C(B, t), coverage_count[T] > 1.
    
    removed_any = True
    while removed_any and time.time() < deadline - 2:
        removed_any = False
        for i in range(len(blocks)):
            b = blocks[i]
            is_redundant = True
            for comb in itertools.combinations(b, t):
                if coverage_count[triple_to_idx[comb]] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                # Decrement counts
                for comb in itertools.combinations(b, t):
                    coverage_count[triple_to_idx[comb]] -= 1
                blocks.pop(i)
                removed_any = True
                break # Restart loop as indices changed

    return blocks
