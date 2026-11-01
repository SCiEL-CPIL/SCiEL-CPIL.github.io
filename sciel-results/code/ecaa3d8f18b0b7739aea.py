import itertools
import time
import random
from collections import defaultdict
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {s: i for i, s in enumerate(all_tsubsets)}

    # uncovered: bitarray-like using a set for now, but we'll use a list of bools for speed
    uncovered = [True] * n_tsubsets
    uncovered_count = n_tsubsets

    blocks = []

    # Phase 1: Randomized greedy (Rödl nibble style)
    # We pick a random starting t-subset, then greedily build a block
    # weighted by coverage, with some randomization.

    # For efficiency, we'll maintain for each point x, the count of uncovered t-subsets
    # that contain x. But updating this incrementally is complex.
    # Instead, we'll use a simpler randomized greedy:
    #   1. Pick a random uncovered t-subset as the "seed"
    #   2. Build a block starting with that seed, adding points that cover the most uncovered subsets
    #      but with some randomization (sample from top candidates)

    random.seed(42)

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Pick a random uncovered t-subset
        # Find an uncovered index
        # For speed, we can maintain a list of uncovered indices, but that's expensive to update.
        # Let's just find one by scanning (or we can keep a list).
        # Better: maintain a list of uncovered indices.
        # Actually, let's use a different approach: keep a list of uncovered t-subsets.
        
        # Let me restructure: use a list of uncovered t-subsets for random access.
        # But removing from a list is O(n). 
        # Alternative: use a set of indices.
        
        # For simplicity and given the small size (C(19,5) = 11628), let's use a set.
        # Actually, let's go back to using a set of tuples for uncovered.
        
        break  # This approach is getting complicated. Let me restart with a cleaner design.

    # Let me rewrite with a cleaner approach using sets.
    uncovered = set(all_tsubsets)
    blocks = []
    
    # Precompute: for each point x, the list of t-subsets containing x
    # This helps in fast gain computation
    point_tsubsets = defaultdict(list)
    for ts in all_tsubsets:
        for x in ts:
            point_tsubsets[x].append(ts)

    # For each pair of points (x,y), the list of t-subsets containing both
    # Actually, for gain computation when building a block, we need:
    # given current block B, for candidate x not in B, gain(x) = number of uncovered t-subsets
    # that are subsets of B ∪ {x} but not subsets of B.
    # A t-subset S is covered by B ∪ {x} but not by B iff:
    #   S ⊆ B ∪ {x} and S ⊄ B
    # This means S contains x and S \ {x} ⊆ B, i.e., S ∩ B = S \ {x} and x ∉ B.
    # So gain(x) = |{S ∈ uncovered : x ∈ S and S \ {x} ⊆ B}|

    # For efficiency, let's precompute for each t-subset S and each x ∈ S, 
    # the key (S \ {x}) which is a (t-1)-subset.
    
    # Actually, let's just do the greedy with a time limit and randomization.
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start
        seed_subset = random.choice(list(uncovered))
        block = list(seed_subset)
        
        while len(block) < k and time.time() < deadline:
            # Compute gain for each candidate
            best_gain = -1
            best_candidates = []
            for x in range(v):
                if x in block:
                    continue
                # Count uncovered t-subsets S where x ∈ S and S \ {x} ⊆ block
                gain = 0
                for S in point_tsubsets[x]:
                    if S in uncovered:
                        # Check if S \ {x} ⊆ block
                        rest = tuple(p for p in S if p != x)
                        if all(p in block for p in rest):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            
            if not best_candidates or best_gain <= 0:
                # No good candidate, pick random
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                x = random.choice(candidates)
            else:
                # Pick from top candidates with some randomness
                x = random.choice(best_candidates)
            
            block.append(x)
        
        block.sort()
        block_tuple = tuple(block)
        
        # Add all t-subsets of this block to covered
        new_covered = set(itertools.combinations(block, t))
        uncovered.difference_update(new_covered)
        blocks.append(block)
    
    # If we ran out of time, we might not have a complete cover.
    # But the problem requires a valid cover. Let's check.
    # If not complete, we need to finish it with pure greedy (no randomization)
    # to ensure validity, even if it's not optimal.
    
    if uncovered and time.time() < deadline - 5:
        # Finish with deterministic greedy
        while uncovered and time.time() < deadline - 2:
            seed_subset = min(uncovered)  # deterministic
            block = list(seed_subset)
            while len(block) < k:
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for S in point_tsubsets[x]:
                        if S in uncovered:
                            rest = tuple(p for p in S if p != x)
                            if all(p in block for p in rest):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            new_covered = set(itertools.combinations(block, t))
            uncovered.difference_update(new_covered)
            blocks.append(block)
    
    return blocks
