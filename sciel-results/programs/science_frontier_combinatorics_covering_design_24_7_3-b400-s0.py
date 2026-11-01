"""Covering design C(24,7,3) via greedy with efficient gain computation."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsets)
    blocks = []
    
    # For efficiency, maintain a count of how many times each t-set is "covered" by current blocks
    # But we only need to know if it's covered or not, so we track uncovered set
    
    # For gain computation: when we have a partial block B and consider adding x,
    # gain = number of uncovered t-sets that are subsets of B ∪ {x} and contain x
    # i.e., t-sets S where x ∈ S ⊆ B ∪ {x}
    # Equivalently: S = {x} ∪ R where R ⊆ B and |R| = t-1
    
    # Precompute for each pair of points, the t-sets containing both? No, that's too big.
    # Instead, for gain computation with partial block B (size b) and candidate x:
    # gain(x) = |{R ⊆ B, |R|=t-1 : {x} ∪ R is uncovered}|
    
    # We can precompute for each (t-1)-subset R, which t-sets contain R
    # But let's just compute it on the fly with the uncovered set
    
    while uncovered and time.time() < deadline:
        # Pick the first uncovered t-set to seed a new block
        seed_tset = next(iter(uncovered))
        block = list(seed_tset)
        
        while len(block) < k and time.time() < deadline:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = frozenset(list(R) + [x])
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point adds coverage; pick randomly
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            block.append(best_x)
        
        block.sort()
        block_frozen = frozenset(block)
        
        # Remove all covered t-sets
        covered = set()
        for c in itertools.combinations(block, t):
            covered.add(frozenset(c))
        uncovered -= covered
        blocks.append(block)
    
    # Verify coverage - if not complete, we need to add more blocks
    # But if we ran out of time, return what we have (may be invalid)
    # Let's check if we have full coverage
    if uncovered:
        # Try to fill in the remaining uncovered t-sets with additional blocks
        # Use a simpler greedy: for each uncovered t-set, create a block containing it
        # This may not be optimal but ensures validity
        for ts in list(uncovered):
            if time.time() >= deadline - 5:
                break
            block = list(ts)
            # Fill the rest with random points
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
    
    return blocks
