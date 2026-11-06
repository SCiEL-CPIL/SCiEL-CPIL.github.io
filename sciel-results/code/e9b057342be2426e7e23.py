"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _calc_alpha(triples, n=16):
    """Calculate the independence number (largest set with no triple)."""
    # Use bitset representation for speed
    # For each point, track which triples contain it
    triple_masks = [0] * len(triples)
    for i, (a, b, c) in enumerate(triples):
        triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
    
    # Try all subsets of size k from largest to smallest
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            contained = False
            for tm in triple_masks:
                if (tm & mask) == tm:
                    contained = True
                    break
            if not contained:
                return k
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            shared = len(set(triples[i]) & set(triples[j]))
            if shared > 1:
                return False
    return True

def _build_linear_graph_backtrack(n=16, target_alpha=5, time_limit=100.0):
    """Backtracking CSP solver to build linear 3-graph with small independence number."""
    start_time = time.time()
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Sort triples to improve pruning
    all_triples.sort()
    
    best_triples = []
    best_alpha = n + 1
    
    def backtrack(current_triples, used_pairs, remaining_time):
        nonlocal best_triples, best_alpha
        
        if time.time() - start_time > remaining_time:
            return
        
        # Check if current configuration is valid
        if not _is_linear(current_triples):
            return
        
        # Calculate current alpha (only if we have enough triples)
        if len(current_triples) >= 5:
            alpha = _calc_alpha(current_triples, n)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = current_triples[:]
            
            # If we've reached target, stop
            if alpha <= target_alpha:
                return
        
        # Try adding more triples
        for triple in all_triples:
            # Check if this triple can be added
            a, b, c = triple
            pairs = [(a, b), (a, c), (b, c)]
            if any(p in used_pairs for p in pairs):
                continue
            
            # Forward checking: see if adding this triple still allows completion
            new_used_pairs = used_pairs | set(pairs)
            new_triples = current_triples + [list(triple)]
            
            # Heuristic: prefer triples that "cover" more uncovered points
            backtrack(new_triples, new_used_pairs, remaining_time)
            
            if time.time() - start_time > remaining_time:
                return
            if best_alpha <= target_alpha:
                return
    
    # Start with empty configuration
    backtrack([], set(), time_limit)
    
    return best_triples, best_alpha

def _local_search_reduce_alpha(triples, n=16, max_iter=1000, time_limit=50.0):
    """Local search to reduce independence number."""
    start_time = time.time()
    current_alpha = _calc_alpha(triples, n)
    best_triples = triples[:]
    best_alpha = current_alpha
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if best_alpha <= 5:
            break
        
        # Try removing a random triple and replacing with a better one
        if len(triples) == 0:
            break
            
        idx = random.randint(0, len(triples) - 1)
        old_triple = triples[idx]
        
        # Try to find a replacement triple
        all_triples = list(itertools.combinations(range(n), 3))
        random.shuffle(all_triples)
        
        found_replacement = False
        for new_triple in all_triples:
            test_triples = triples[:]
            test_triples[idx] = list(new_triple)
            
            if not _is_linear(test_triples):
                continue
            
            test_alpha = _calc_alpha(test_triples, n)
            if test_alpha < best_alpha:
                best_alpha = test_alpha
                best_triples = test_triples
                triples = test_triples
                found_replacement = True
                break
        
        if not found_replacement:
            # Try adding a new triple
            if len(triples) < 20:
                for new_triple in all_triples:
                    test_triples = triples + [list(new_triple)]
                    if _is_linear(test_triples):
                        test_alpha = _calc_alpha(test_triples, n)
                        if test_alpha <= best_alpha:
                            best_triples = test_triples
                            triples = test_triples
                            break
    
    return best_triples, best_alpha

def construct():
    """Construct a linear 3-graph on 16 points with small independence number."""
    # Use backtracking to find a good initial solution
    triples, alpha = _build_linear_graph_backtrack(n=16, target_alpha=5, time_limit=80.0)
    
    if not triples:
        # Fallback: use a known good construction
        # Based on Steiner triple system S(2,3,7) extended to 16 points
        triples = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 9, 12], [1, 10, 13], [2, 11, 14],
            [3, 9, 15], [4, 10, 15], [5, 11, 15],
            [6, 12, 15], [7, 13, 15], [8, 14, 15]
        ]
        # Filter to ensure linearity
        filtered = []
        for t in triples:
            valid = True
            for ft in filtered:
                if len(set(t) & set(ft)) > 1:
                    valid = False
                    break
            if valid:
                filtered.append(t)
        triples = filtered
        alpha = _calc_alpha(triples, 16)
    
    # Local search to further reduce alpha
    triples, alpha = _local_search_reduce_alpha(triples, n=16, max_iter=500, time_limit=30.0)
    
    return triples
# EVOLVE-BLOCK-END
