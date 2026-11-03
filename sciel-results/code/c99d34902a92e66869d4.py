"""Improved covering design C(24,7,3) via triple-based greedy + local search."""
import time
import random
import itertools
from collections import defaultdict

# Deadline guard: evaluator kills at 150s wall-clock; use 115s safety margin
_DEADLINE = 115.0
_start_time = None

def _elapsed():
    return time.time() - _start_time

def construct(v, k, t):
    global _start_time
    _start_time = time.time()
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Map each triple to an index for fast lookup
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # For a k-subset (block), precompute which triples it covers
    # A block covers all t-subsets of its k elements
    
    def block_triples(block):
        """Return set of triple-indices covered by this block."""
        return set(triple_to_idx[tr] for tr in itertools.combinations(block, t))
    
    # Greedy construction: start with uncovered triples, build blocks
    # Key insight: for C(24,7,3), we can greedily pick blocks that cover
    # many uncovered triples.
    
    def greedy_construct(random_seed=0):
        """Triple-based greedy: open block on first uncovered triple,
        then add points that cover the most uncovered triples."""
        rng = random.Random(random_seed)
        
        uncovered = set(range(n_triples))  # set of triple indices
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to seed the block (or min for determinism)
            if random_seed == 0:
                seed_triple = min(uncovered)
            else:
                seed_triple = rng.choice(list(uncovered))
            
            block = list(all_triples[seed_triple])
            block_set = set(block)
            
            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Gain: number of uncovered triples that would be covered
                    # by adding x to the current block
                    # A triple is covered if all t elements are in block+{x}
                    # So we need triples where x is one element and the other
                    # t-1 elements are already in block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tr = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(tr)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    # Pick any remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining) if random_seed else remaining[0]
                    else:
                        break
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered triples
            covered = block_triples(block)
            uncovered -= covered
            
            # Check deadline
            if _elapsed() > _DEADLINE - 5:
                break
        
        return blocks
    
    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        covered = set()
        for block in blocks:
            covered |= block_triples(block)
        return len(covered) == n_triples
    
    # Try multiple random seeds for greedy construction
    best_blocks = None
    best_count = float('inf')
    
    # First try deterministic greedy
    blocks0 = greedy_construct(0)
    if is_valid(blocks0):
        best_blocks = blocks0
        best_count = len(blocks0)
    
    # Try randomized greedy with different seeds
    seed = 1
    while _elapsed() < _DEADLINE - 30:
        blocks = greedy_construct(seed)
        if is_valid(blocks):
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        # Fallback: use the last greedy result even if not fully valid
        # (shouldn't happen, but safety)
        best_blocks = blocks0
    
    # Local search: try to swap out blocks that cover few unique triples
    # or try replacing pairs of blocks with a single better block
    if _elapsed() < _DEADLINE - 15:
        best_blocks = local_search(best_blocks, v, k, t, all_triples, triple_to_idx, block_triples)
    
    return best_blocks


def local_search(blocks, v, k, t, all_triples, triple_to_idx, block_triples):
    """Try to reduce block count by finding redundant or replaceable blocks."""
    n_triples = len(all_triples)
    
    def is_valid(blks):
        covered = set()
        for b in blks:
            covered |= block_triples(b)
        return len(covered) == n_triples
    
    best = list(blocks)
    best_len = len(best)
    
    # Try removing each block and checking validity
    for i in range(len(best)):
        if _elapsed() > _DEADLINE - 10:
            break
        candidate = best[:i] + best[i+1:]
        if is_valid(candidate):
            best = candidate
            best_len -= 1
            # Restart removal check
            i = 0
    
    # Try replacing pairs of blocks with a single block
    for i in range(len(best)):
        if _elapsed() > _DEADLINE - 10:
            break
        for j in range(i + 1, len(best)):
            if _elapsed() > _DEADLINE - 10:
                break
            # Try to find a single block that covers all triples covered by blocks i and j
            # that aren't covered by other blocks
            other_blocks = [best[x] for x in range(len(best)) if x != i and x != j]
            other_covered = set()
            for b in other_blocks:
                other_covered |= block_triples(b)
            
            # Triples that must be covered by the new block
            needed = (block_triples(best[i]) | block_triples(best[j])) - other_covered
            
            if not needed:
                # Both blocks are redundant
                candidate = [best[x] for x in range(len(best)) if x != i and x != j]
                best = candidate
                continue
            
            # Try to find a k-subset that covers all 'needed' triples
            # This is a set cover subproblem; use greedy
            # The new block must contain all elements that appear in any needed triple
            # Actually, the block must contain each needed triple as a subset
            # So for each needed triple, all 3 elements must be in the block
            required_elements = set()
            for idx in needed:
                required_elements.update(all_triples[idx])
            
            if len(required_elements) > k:
                continue
            
            # Build a block containing required_elements plus filler
            block = list(required_elements)
            remaining = [x for x in range(v) if x not in required_elements]
            while len(block) < k and remaining:
                block.append(remaining.pop())
            block.sort()
            
            candidate = [best[x] for x in range(len(best)) if x != i and x != j] + [block]
            if is_valid(candidate):
                best = candidate
    
    return best


def construct(v, k, t):
    global _start_time
    _start_time = time.time()
    
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    def block_triples(block):
        return set(triple_to_idx[tr] for tr in itertools.combinations(block, t))
    
    def is_valid(blocks):
        covered = set()
        for block in blocks:
            covered |= block_triples(block)
        return len(covered) == n_triples
    
    def greedy_construct(random_seed=0):
        rng = random.Random(random_seed)
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            if random_seed == 0:
                seed_triple = min(uncovered)
            else:
                seed_triple = rng.choice(list(uncovered))
            
            block = list(all_triples[seed_triple])
            block_set = set(block)
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tr = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(tr)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining) if random_seed else remaining[0]
                    else:
                        break
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            uncovered -= block_triples(block)
            
            if _elapsed() > _DEADLINE - 5:
                break
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    blocks0 = greedy_construct(0)
    if is_valid(blocks0):
        best_blocks = blocks0
        best_count = len(blocks0)
    
    seed = 1
    while _elapsed() < _DEADLINE - 30:
        blocks = greedy_construct(seed)
        if is_valid(blocks):
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        best_blocks = blocks0
    
    if _elapsed() < _DEADLINE - 15:
        best_blocks = local_search(best_blocks, v, k, t, all_triples, triple_to_idx, block_triples)
    
    return best_blocks
