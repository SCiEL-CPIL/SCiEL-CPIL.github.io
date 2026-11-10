"""Rewritten program: construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
import time
import random
import itertools
import sys
import math

# EVOLVE-BLOCK-START

# Precompute all triples and their bitmasks
_ALL_TRIPLES = list(itertools.combinations(range(16), 3))
_ALL_TRIPLE_MASKS = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in _ALL_TRIPLES]
_TRIPLE_TO_MASK = {t: m for t, m in zip(_ALL_TRIPLES, _ALL_TRIPLE_MASKS)}
_MASK_TO_TRIPLE = {m: t for t, m in zip(_ALL_TRIPLES, _ALL_TRIPLE_MASKS)}

# Cache for MVC (and thus independence number) computations
_mvc_cache = {}

def _min_vertex_cover_size(triple_masks):
    """
    Find the size of the minimum vertex cover for a hypergraph defined by triple_masks.
    Uses Branch and Bound.
    """
    # Cache key: tuple of sorted masks
    key = tuple(sorted(triple_masks))
    if key in _mvc_cache:
        return _mvc_cache[key]
    
    if not triple_masks:
        _mvc_cache[key] = 0
        return 0
        
    # Precompute for each vertex, which triples it covers
    # We will use a recursive B&B.
    # State: set of remaining triples (indices or masks), current cover size.
    # To make it fast, we represent remaining triples as a bitmask of indices?
    # Or just a list. Since T is small (< 100), list is fine.
    
    # Optimization: If there is a triple of size 1 (not possible here, size 3), 
    # or if we can find a small cover quickly.
    
    best_cover_size = 16 # Upper bound
    
    # Sort triples by size? All size 3.
    # Sort by frequency of vertices?
    
    def _solve(remaining_indices, current_size):
        nonlocal best_cover_size
        
        if current_size >= best_cover_size:
            return
            
        if not remaining_indices:
            best_cover_size = current_size
            return
            
        # Heuristic: Pick a triple that appears in the most other remaining triples?
        # Or just the first one.
        # Let's pick the first one for simplicity, or one with high overlap.
        # For speed, just pick the first one.
        target_idx = remaining_indices[0]
        target_mask = key[target_idx]
        
        # Get vertices in target triple
        vertices = [v for v in range(16) if target_mask & (1 << v)]
        
        for v in vertices:
            v_bit = 1 << v
            # New remaining triples: those not covered by v
            new_remaining = []
            for idx in remaining_indices[1:]: # Skip target_idx as it's covered
                if not (key[idx] & v_bit):
                    new_remaining.append(idx)
            
            _solve(new_remaining, current_size + 1)
            
            # Pruning: If we found a cover of size current_size + 1, 
            # we might prune other branches if they can't beat it?
            # The recursion handles this via best_cover_size.
            
    _solve(list(range(len(key))), 0)
    
    _mvc_cache[key] = best_cover_size
    return best_cover_size

def _independence_number(triples):
    """
    Return the size of the largest independent set.
    alpha(H) = |V| - tau(H), where tau(H) is min vertex cover size.
    """
    if not triples:
        return 16
        
    masks = [_TRIPLE_TO_MASK[t] for t in triples]
    tau = _min_vertex_cover_size(masks)
    return 16 - tau

def _are_compatible(triples, new_triple):
    """Check if new_triple can be added to the existing list without violating linearity."""
    mask = _TRIPLE_TO_MASK[new_triple]
    for t in triples:
        existing_mask = _TRIPLE_TO_MASK[t]
        common = mask & existing_mask
        if bin(common).count('1') > 1:
            return False
    return True

def _construct_greedy(seed):
    """
    Construct a dense linear hypergraph using greedy random ordering.
    """
    rng = random.Random(seed)
    shuffled = _ALL_TRIPLES[:]
    rng.shuffle(shuffled)
    
    triples = []
    pair_used = set()
    
    for t in shuffled:
        a, b, c = t
        pairs = ((a, b), (a, c), (b, c))
        if not any(p in pair_used for p in pairs):
            triples.append(t)
            pair_used.update(pairs)
            
    return triples

def _simulated_annealing(initial_triples, deadline):
    """
    Improve the hypergraph using Simulated Annealing.
    """
    current = initial_triples[:]
    current_alpha = _independence_number(current)
    best = current[:]
    best_alpha = current_alpha
    
    rng = random.Random(123)
    T = 5.0
    cooling = 0.999
    
    # Precompute valid swaps? Too expensive.
    
    while time.time() < deadline - 5:
        T *= cooling
        if T < 0.01:
            T = 0.01 # Keep some noise
            
        # Generate a move
        move_type = rng.random()
        
        new_triples = None
        
        if move_type < 0.3:
            # Add a triple
            candidates = []
            for t in _ALL_TRIPLES:
                if t not in current and _are_compatible(current, t):
                    candidates.append(t)
            if candidates:
                new_triples = current + [rng.choice(candidates)]
                
        elif move_type < 0.6:
            # Remove a triple
            if current:
                idx = rng.randint(0, len(current) - 1)
                new_triples = current[:idx] + current[idx+1:]
                
        else:
            # Swap a triple
            if current:
                idx = rng.randint(0, len(current) - 1)
                old_t = current[idx]
                candidates = []
                for t in _ALL_TRIPLES:
                    if t == old_t:
                        continue
                    # Check compatibility with all other triples
                    # We remove old_t, add t.
                    # t must be compatible with current \ {old_t}
                    if _are_compatible([x for i, x in enumerate(current) if i != idx], t):
                        candidates.append(t)
                if candidates:
                    new_t = rng.choice(candidates)
                    new_triples = current[:idx] + [new_t] + current[idx+1:]
                    
        if new_triples is None:
            continue
            
        new_alpha = _independence_number(new_triples)
        
        if new_alpha < current_alpha:
            current = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best = current[:]
        else:
            # Accept with probability
            delta = new_alpha - current_alpha
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current = new_triples
                current_alpha = new_alpha
                
    return best

def construct():
    """Construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
    start = time.time()
    deadline = start + 115  # safety margin
    
    # Clear cache
    _mvc_cache.clear()
    
    best_triples = None
    best_alpha = 16
    
    # Phase 1: Multiple greedy constructions
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        triples = _construct_greedy(seed)
        if triples:
            alpha = _independence_number(triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples[:]
                
        if best_alpha <= 5:
            break
            
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        
    # Phase 2: Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Run SA for the best solution found
        best_triples = _simulated_annealing(best_triples, deadline)
        
    return best_triples

# EVOLVE-BLOCK-END
