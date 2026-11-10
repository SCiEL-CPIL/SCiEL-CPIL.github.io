"""Rewritten program: construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

# Precompute all triples and their bitmasks
_ALL_TRIPLES = list(itertools.combinations(range(16), 3))
_ALL_TRIPLE_MASKS = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in _ALL_TRIPLES]
_TRIPLE_TO_MASK = {t: m for t, m in zip(_ALL_TRIPLES, _ALL_TRIPLE_MASKS)}

# Cache for independence number computations
_indep_cache = {}


def _independence_number(triples):
    """Return the size of the largest subset of {0..15} containing no triple.
    Uses branch-and-bound with bitmasks for speed."""
    # Cache key: frozenset of triple masks
    key = tuple(sorted(_TRIPLE_TO_MASK[t] for t in triples))
    if key in _indep_cache:
        return _indep_cache[key]
    
    n = 16
    triple_masks = list(key)
    T = len(triple_masks)
    
    if T == 0:
        _indep_cache[key] = n
        return n
    
    # For each point, bitmask of triples containing it
    point_to_triples = [0] * n
    for i, tm in enumerate(triple_masks):
        for p in range(n):
            if tm & (1 << p):
                point_to_triples[p] |= (1 << i)
    
    # Branch and bound: find largest set with no triple fully contained.
    # We process points 0..15. At each step, decide include or exclude.
    # Prune if current chosen set already contains a triple.
    # Bound: current_size + remaining_points <= best -> prune.
    
    best = [0]
    
    def _solve(point_idx, chosen_mask, chosen_count):
        if point_idx == n:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        # Upper bound
        if chosen_count + (n - point_idx) <= best[0]:
            return
        
        # Try including point_idx
        p_bit = 1 << point_idx
        new_mask = chosen_mask | p_bit
        # Check if adding this point completes any triple
        # A triple is completed if all its points are in new_mask
        # We can check: for each triple containing point_idx, see if other two points are in new_mask
        completes = False
        for i in range(T):
            if point_to_triples[point_idx] & (1 << i):
                tm = triple_masks[i]
                if (new_mask & tm) == tm:
                    completes = True
                    break
        if not completes:
            _solve(point_idx + 1, new_mask, chosen_count + 1)
        
        # Try excluding point_idx
        _solve(point_idx + 1, chosen_mask, chosen_count)
    
    _solve(0, 0, 0)
    result = best[0]
    _indep_cache[key] = result
    return result


def _are_compatible(triples, new_triple):
    """Check if new_triple can be added to the existing list without violating linearity."""
    mask = _TRIPLE_TO_MASK[new_triple]
    for t in triples:
        existing_mask = _TRIPLE_TO_MASK[t]
        common = mask & existing_mask
        if bin(common).count('1') > 1:
            return False
    return True


def _greedy_construct(seed, time_limit=120):
    """Build a linear hypergraph greedily with randomized ordering, then try to improve."""
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    
    best_triples = None
    best_indep = 16  # worst case
    
    num_attempts = 0
    
    while time.time() < deadline - 2:
        num_attempts += 1
        
        # Random shuffle of all triples
        shuffled = _ALL_TRIPLES[:]
        rng.shuffle(shuffled)
        
        triples = []
        # Track pair usage for fast compatibility check
        pair_used = set()
        
        for t in shuffled:
            a, b, c = t
            pairs = ((a, b), (a, c), (b, c))
            # Check if any pair is already used
            conflict = False
            for p in pairs:
                if p in pair_used:
                    conflict = True
                    break
            if not conflict:
                triples.append(t)
                pair_used.update(pairs)
            if len(triples) >= 40:
                break
        
        if not triples:
            continue
        
        # Compute independence number
        indep = _independence_number(triples)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = triples[:]
        
        # If we've achieved a very good score, we can stop
        if best_indep <= 5:
            break
    
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_triples


def _local_search(triples, deadline, rng):
    """Improve a linear hypergraph via targeted local search."""
    current = triples[:]
    current_indep = _independence_number(current)
    best_triples = current[:]
    best_indep = current_indep
    
    # Precompute for fast compatibility
    def _try_add_remove(current_list, remove_idx, new_triple):
        """Check if replacing current_list[remove_idx] with new_triple maintains linearity."""
        rm = _TRIPLE_TO_MASK[current_list[remove_idx]]
        nm = _TRIPLE_TO_MASK[new_triple]
        # Check new_triple against all other triples
        for i, t in enumerate(current_list):
            if i == remove_idx:
                continue
            em = _TRIPLE_TO_MASK[t]
            if bin(nm & em).count('1') > 1:
                return False
        return True
    
    all_triples = _ALL_TRIPLES[:]
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try all possible single swaps
        for remove_idx in range(len(current)):
            if time.time() > deadline - 2:
                break
            
            removed = current[remove_idx]
            
            # Generate candidate replacements: triples that share at most 1 point with each other triple
            # But we need to check against all except the one being removed
            candidates = []
            for t in all_triples:
                if t == removed:
                    continue
                if _try_add_remove(current, remove_idx, t):
                    candidates.append(t)
            
            if not candidates:
                continue
            
            # Shuffle and try a subset
            rng.shuffle(candidates)
            max_try = min(len(candidates), 200)
            
            for new_t in candidates[:max_try]:
                candidate = current[:remove_idx] + current[remove_idx+1:] + [new_t]
                cand_indep = _independence_number(candidate)
                if cand_indep < current_indep:
                    current = candidate
                    current_indep = cand_indep
                    if cand_indep < best_indep:
                        best_indep = cand_indep
                        best_triples = current[:]
                    improved = True
                    break
                if time.time() > deadline - 2:
                    break
            
            if improved:
                break
        
        if not improved:
            # Try adding a new triple (if we haven't hit max)
            if len(current) < 40:
                candidates = []
                for t in all_triples:
                    if t not in current and _are_compatible(current, t):
                        candidates.append(t)
                if candidates:
                    rng.shuffle(candidates)
                    for new_t in candidates[:100]:
                        candidate = current + [new_t]
                        cand_indep = _independence_number(candidate)
                        if cand_indep < current_indep:
                            current = candidate
                            current_indep = cand_indep
                            if cand_indep < best_indep:
                                best_indep = cand_indep
                                best_triples = current[:]
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            # Try removing a triple
            if len(current) > 5:
                for remove_idx in range(len(current)):
                    candidate = current[:remove_idx] + current[remove_idx+1:]
                    cand_indep = _independence_number(candidate)
                    if cand_indep < current_indep:
                        current = candidate
                        current_indep = cand_indep
                        if cand_indep < best_indep:
                            best_indep = cand_indep
                            best_triples = current[:]
                        improved = True
                        break
                    if time.time() > deadline - 2:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_triples


def construct():
    """Construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
    start = time.time()
    deadline = start + 115  # safety margin
    
    # Clear cache for fresh run
    _indep_cache.clear()
    
    best_triples = None
    best_indep = 16
    
    # Phase 1: Multiple greedy constructions with different seeds
    num_greedy = 30
    for seed in range(num_greedy):
        if time.time() > deadline - 60:
            break
        triples = _greedy_construct(seed, time_limit=5)
        if triples:
            indep = _independence_number(triples)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples[:]
        
        if best_indep <= 5:
            break
    
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rng = random.Random(42)
        best_triples = _local_search(best_triples, deadline, rng)
    
    return best_triples

# EVOLVE-BLOCK-END
