import itertools
import time
import random
from collections import Counter
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    n_tsub = len(all_tsubsets_set)

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial)
        if blocks is not None:
            # Phase 1: Remove redundant blocks (greedy pass)
            blocks = _remove_redundant(blocks, v, k, t, deadline)

            # Phase 2: Local search / refinement
            blocks = _local_search(blocks, v, k, t, all_tsubsets_set, deadline, trial)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        # Early termination if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]


def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy set cover with ratio-based heuristic.
    Score for candidate x = marginal_gain / (1 + overlap_penalty)
    where overlap_penalty measures how many of the new t-subsets would be
    duplicates (already covered). This explicitly penalizes redundancy.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))
    
    # Point usage counts for symmetry breaking
    point_usage = [0] * v
    usage_weight = 0.3
    
    while uncovered:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as the seed
        seed_tuple = tuple(rng.choice(list(uncovered)))
        current_block_set = set(seed_tuple)

        while len(current_block_set) < k:
            if time.time() > deadline:
                return None

            best_point = -1
            best_score = -1.0
            best_gain = -1

            # Pre-compute (t-1) subsets of current block
            sorted_block = sorted(current_block_set)
            t_minus_1_subsets = list(itertools.combinations(sorted_block, t - 1))
            
            # For each candidate x, compute gain and overlap
            for x in points:
                if x in current_block_set:
                    continue

                gain = 0
                overlap = 0
                for sub in t_minus_1_subsets:
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered:
                        gain += 1
                    else:
                        overlap += 1

                if gain == 0:
                    continue

                # Ratio: gain / (1 + overlap_penalty * overlap)
                # Also penalize over-used points
                usage_penalty = 1.0 + usage_weight * (point_usage[x] + sum(point_usage[p] for p in current_block_set) / len(current_block_set))
                score = gain / (1.0 + overlap * 0.5 + usage_penalty * 0.3)

                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_point = x

            if best_point == -1:
                # No point adds new coverage; fill randomly
                available = [x for x in points if x not in current_block_set]
                if not available:
                    return None
                best_point = rng.choice(available)

            current_block_set.add(best_point)

        block = tuple(sorted(current_block_set))
        blocks.append(block)

        # Update point usage
        for p in block:
            point_usage[p] += 1

        # Remove all t-subsets covered by this new block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant (every t-subset in them is covered by others).
    """
    if not blocks:
        return blocks

    cover_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    changed = True

    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Optimized local search with merge moves.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)
    
    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(itertools.combinations(b, t)))
    
    cover_count = Counter()
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] += 1
    
    points = list(range(v))
    no_progress_count = 0
    max_no_progress = 15
    
    while time.time() < deadline and no_progress_count < max_no_progress:
        
        # 1. Check for and remove redundant blocks
        redundant_found = False
        n = len(improved)
        for i in range(n):
            is_redundant = True
            for sub in block_tsubs[i]:
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                del improved[i]
                del block_tsubs[i]
                redundant_found = True
                break
        
        if redundant_found:
            no_progress_count = 0
            continue
        
        # 2. Try merge moves
        n = len(improved)
        if n < 2:
            break
            
        unique_subs = [None] * n
        for i in range(n):
            unique_subs[i] = [s for s in block_tsubs[i] if cover_count[s] == 1]
            
        indices = list(range(n))
        rng.shuffle(indices)
        
        move_made = False
        for b_idx in indices:
            if time.time() > deadline:
                break
            if len(unique_subs[b_idx]) == 0:
                continue
                
            b_unique = unique_subs[b_idx]
            
            for a_idx in indices:
                if a_idx == b_idx:
                    continue
                if time.time() > deadline:
                    break
                    
                a_unique = unique_subs[a_idx]
                
                union_points = set()
                for s in a_unique:
                    union_points.update(s)
                for s in b_unique:
                    union_points.update(s)
                    
                if len(union_points) <= k:
                    a_prime = list(union_points)
                    while len(a_prime) < k:
                        p = rng.randint(0, v - 1)
                        if p not in a_prime:
                            a_prime.append(p)
                    a_prime = tuple(sorted(a_prime))
                    
                    is_distinct = True
                    for j in range(len(improved)):
                        if j != a_idx and improved[j] == a_prime:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    # Perform replacement
                    for sub in block_tsubs[a_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]
                            
                    new_tsubs = list(itertools.combinations(a_prime, t))
                    for sub in new_tsubs:
                        cover_count[sub] = cover_count.get(sub, 0) + 1
                    
                    improved[a_idx] = a_prime
                    block_tsubs[a_idx] = new_tsubs
                    
                    b_redundant = True
                    for sub in block_tsubs[b_idx]:
                        if cover_count.get(sub, 0) <= 1:
                            b_redundant = False
                            break
                            
                    if b_redundant:
                        for sub in block_tsubs[b_idx]:
                            cover_count[sub] -= 1
                            if cover_count[sub] == 0:
                                del cover_count[sub]
                        del improved[b_idx]
                        del block_tsubs[b_idx]
                        move_made = True
                        break
                    else:
                        move_made = True
                        break
                        
            if move_made:
                no_progress_count = 0
                break
                
        if not move_made:
            no_progress_count += 1
            
    return improved
