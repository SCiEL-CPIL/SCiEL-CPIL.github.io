"""Covering design C(24,7,3) with greedy + local search."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Precompute all t-subsets as bitmasks
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    # Map each triple to an index for fast lookup
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Represent each triple as a bitmask of its elements
    # But we'll track coverage by index instead
    
    # Precompute for each k-subset (block), which triples it covers
    # This is too many (C(24,7) = 346104), so we'll compute on the fly
    
    # Greedy set cover
    uncovered = set(range(n_triples))
    blocks = []
    
    # For efficiency, precompute which triples each point-pair-triple covers
    # Actually, let's use a different approach: track uncovered triples as a set
    # and for each candidate block, count how many uncovered triples it covers
    
    # Precompute: for each element, which triples contain it
    triples_by_element = [[] for _ in range(v)]
    for i, tr in enumerate(all_triples):
        for e in tr:
            triples_by_element[e].append(i)
    
    def time_check():
        return time.time() > deadline
    
    # Greedy phase
    while uncovered and not time_check():
        # Find the block that covers the most uncovered triples
        best_gain = -1
        best_block = None
        
        # Start with the first uncovered triple
        first_triple = min(uncovered)
        first_triple_elems = all_triples[first_triple]
        
        # Try blocks containing this triple
        # A block must contain all t elements of some uncovered triple
        # To be efficient, start from the uncovered triple's elements
        
        # Generate candidate blocks: start with the triple, add k-t more elements
        base = set(first_triple_elems)
        remaining = [x for x in range(v) if x not in base]
        
        # Greedily build a block
        current = list(base)
        
        # Try to find a good completion
        # Use a heuristic: at each step, add the element that covers most uncovered triples
        while len(current) < k and not time_check():
            best_elem = -1
            best_gain_elem = -1
            for x in remaining:
                if x in current:
                    continue
                # Count how many uncovered triples would be covered by adding x
                # A triple is covered if all its elements are in current + {x}
                # So we need triples where x is in the triple and the other t-1 elements are in current
                gain = 0
                for tr_idx in triples_by_element[x]:
                    if tr_idx not in uncovered:
                        continue
                    tr = all_triples[tr_idx]
                    # Check if all other elements of tr are in current
                    others = [e for e in tr if e != x]
                    if all(e in current for e in others):
                        gain += 1
                if gain > best_gain_elem:
                    best_gain_elem = gain
                    best_elem = x
            if best_elem >= 0:
                current.append(best_elem)
                remaining.remove(best_elem)
            else:
                # No good element found, break
                break
        
        if len(current) == k:
            # Compute actual gain
            current_set = set(current)
            gain = 0
            newly_covered = []
            for tr_idx in uncovered:
                tr = all_triples[tr_idx]
                if all(e in current_set for e in tr):
                    gain += 1
                    newly_covered.append(tr_idx)
            
            if gain > 0:
                current.sort()
                blocks.append(current)
                uncovered.difference_update(newly_covered)
        
        # If we couldn't make progress, force add a block
        if len(current) < k:
            # Just pick arbitrary remaining elements
            for x in remaining:
                if len(current) >= k:
                    break
                if x not in current:
                    current.append(x)
            current.sort()
            blocks.append(current)
            current_set = set(current)
            for tr_idx in list(uncovered):
                tr = all_triples[tr_idx]
                if all(e in current_set for e in tr):
                    uncovered.discard(tr_idx)
    
    if not uncovered:
        return blocks
    
    # Local search: try to replace blocks to reduce count
    # Strategy: for each pair of blocks, see if we can remove one
    # Or: try to find a block that covers all triples covered by another block
    
    # Better: iterative improvement - try removing each block and see if coverage is maintained
    # If not, try to find a replacement block that covers the missing triples
    
    # First, let's try a simple local search: randomly shuffle and re-greedy
    best_blocks = blocks[:]
    best_count = len(blocks)
    
    # Try multiple restarts with different random seeds
    rng = random.Random(42)
    
    for restart in range(20):
        if time_check():
            break
        
        # Create a perturbed version
        # Randomly remove some blocks and re-greedy
        num_remove = max(1, len(best_blocks) // 10)
        
        # Track coverage with bitmasks
        # For speed, use a list of sets
        covered = [False] * n_triples
        for block in best_blocks:
            block_set = set(block)
            for tr_idx, tr in enumerate(all_triples):
                if all(e in block_set for e in tr):
                    covered[tr_idx] = True
        
        # Remove some random blocks
        indices_to_remove = rng.sample(range(len(best_blocks)), min(num_remove, len(best_blocks)))
        for idx in indices_to_remove:
            block = best_blocks[idx]
            block_set = set(block)
            for tr_idx, tr in enumerate(all_triples):
                if all(e in block_set for e in tr):
                    # Check if this triple is still covered by other blocks
                    still_covered = False
                    for other_idx, other_block in enumerate(best_blocks):
                        if other_idx == idx:
                            continue
                        other_set = set(other_block)
                        if all(e in other_set for e in tr):
                            still_covered = True
                            break
                    if not still_covered:
                        covered[tr_idx] = False
        
        # Now re-greedy to cover remaining uncovered triples
        uncovered = set(i for i in range(n_triples) if not covered[i])
        
        # Build new blocks list
        new_blocks = [b for i, b in enumerate(best_blocks) if i not in indices_to_remove]
        
        while uncovered and not time_check():
            first_triple = min(uncovered)
            first_triple_elems = all_triples[first_triple]
            base = set(first_triple_elems)
            remaining = [x for x in range(v) if x not in base]
            current = list(base)
            
            while len(current) < k and not time_check():
                best_elem = -1
                best_gain_elem = -1
                for x in remaining:
                    if x in current:
                        continue
                    gain = 0
                    for tr_idx in triples_by_element[x]:
                        if tr_idx not in uncovered:
                            continue
                        tr = all_triples[tr_idx]
                        others = [e for e in tr if e != x]
                        if all(e in current for e in others):
                            gain += 1
                    if gain > best_gain_elem:
                        best_gain_elem = gain
                        best_elem = x
                if best_elem >= 0:
                    current.append(best_elem)
                    remaining.remove(best_elem)
                else:
                    break
            
            if len(current) < k:
                for x in remaining:
                    if len(current) >= k:
                        break
                    if x not in current:
                        current.append(x)
            
            current.sort()
            new_blocks.append(current)
            current_set = set(current)
            newly_covered = []
            for tr_idx in uncovered:
                tr = all_triples[tr_idx]
                if all(e in current_set for e in tr):
                    newly_covered.append(tr_idx)
            uncovered.difference_update(newly_covered)
        
        if len(new_blocks) < best_count:
            best_blocks = new_blocks
            best_count = len(new_blocks)
    
    return best_blocks
