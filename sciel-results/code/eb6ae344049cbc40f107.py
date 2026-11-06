import random
import time
from itertools import combinations

def _is_valid_triple(triple, used_pairs):
    a, b, c = triple
    if (a, b) in used_pairs or (b, a) in used_pairs: return False
    if (a, c) in used_pairs or (c, a) in used_pairs: return False
    if (b, c) in used_pairs or (c, b) in used_pairs: return False
    return True

def _add_triple(triple, used_pairs, edges):
    a, b, c = triple
    used_pairs.add((a, b))
    used_pairs.add((b, a))
    used_pairs.add((a, c))
    used_pairs.add((c, a))
    used_pairs.add((b, c))
    used_pairs.add((c, b))
    edges.append(triple)

def _calc_alpha(edges):
    if not edges:
        return 16
    # Bitmask representation
    # 16 vertices -> 16 bits
    edge_masks = []
    for e in edges:
        mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
        edge_masks.append(mask)
    
    # Check subsets from size 16 down to 1?
    # Or just check if size K is possible.
    # Since we know min is 5, check 8, 7, 6, 5.
    
    # Helper to check if a mask is independent
    def is_independent(mask):
        for em in edge_masks:
            if (em & mask) == em:
                return False
        return True

    # Generate all masks of size k?
    # 16 choose 8 is 12870.
    # 16 choose 9 is 11440.
    # It's fast enough to just iterate all 2^16 masks? 65536.
    # For each mask, check independence.
    
    max_alpha = 0
    # Iterate all subsets
    # Optimization: If a set is not independent, its supersets are not independent?
    # No, if a set contains an edge, its supersets contain that edge.
    # So if S is NOT independent, S' superset S is NOT independent.
    # We want the MAX size INDEPENDENT set.
    
    # We can just iterate all masks and keep the max popcount for independent ones.
    # 65536 iterations * len(edges) checks.
    # len(edges) ~ 40. 65536 * 40 = 2.6M ops. Fast in Python? Maybe 1-2 seconds.
    # We can optimize by precomputing "bad" masks?
    
    # Alternative: Backtracking search for Max Independent Set.
    # Given N=16, simple recursion is fine.
    
    # Let's use the bitmask iteration for simplicity and robustness, but optimize the check.
    
    # Precompute edge masks.
    # For a given mask m, it is independent if no edge mask is a subset of m.
    
    # We can speed this up:
    # If we process masks in increasing size, and mark "bad" masks (those containing an edge),
    # we can skip them.
    
    # Actually, just iterating 2^16 is fine if done in a tight loop.
    
    best = 0
    for mask in range(1 << 16):
        # Check if mask is independent
        indep = True
        for em in edge_masks:
            if (em & mask) == em:
                indep = False
                break
        if indep:
            cnt = bin(mask).count('1')
            if cnt > best:
                best = cnt
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_alpha = 16
    
    # Pre-generate all triples
    all_triples = list(combinations(range(16), 3))
    
    while time.time() < deadline:
        # Randomized Greedy Construction
        random.shuffle(all_triples)
        used_pairs = set()
        edges = []
        
        for t in all_triples:
            if _is_valid_triple(t, used_pairs):
                _add_triple(t, used_pairs, edges)
                
        # Calculate Alpha
        alpha = _calc_alpha(edges)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            
            # Local Improvement: Try to break independent sets
            # If alpha > 5, try to add edges inside max independent sets
            if alpha > 5:
                # Find a max independent set
                # We need the set itself, not just the size.
                # Modify _calc_alpha to return the set?
                # Or just re-run the check to find one.
                
                # For efficiency, let's just try to add any valid triple that reduces alpha.
                # This is complex.
                # Simpler: If alpha is high, restart.
                # But we can try a quick improvement:
                # Find all independent sets of size alpha.
                # For each, try to add a triple from it.
                
                # Let's implement a specific improver.
                improved = False
                while alpha > 5 and time.time() < deadline:
                    # Find one max independent set
                    max_set_mask = 0
                    max_set_size = 0
                    for mask in range(1 << 16):
                        indep = True
                        for em in [(1<<e[0])|(1<<e[1])|(1<<e[2]) for e in edges]:
                            if (em & mask) == em:
                                indep = False
                                break
                        if indep:
                            cnt = bin(mask).count('1')
                            if cnt > max_set_size:
                                max_set_size = cnt
                                max_set_mask = mask
                            
                    if max_set_size <= 5:
                        break
                    
                    # Try to add a triple from max_set_mask
                    vertices_in_set = [i for i in range(16) if (max_set_mask >> i) & 1]
                    found_add = False
                    for t in combinations(vertices_in_set, 3):
                        if _is_valid_triple(t, used_pairs):
                            _add_triple(t, used_pairs, edges)
                            found_add = True
                            break
                    
                    if found_add:
                        # Recalculate alpha? Or just continue loop.
                        # The loop will recalculate.
                        continue
                    else:
                        break # Stuck
                    
                if best_alpha > alpha: # alpha might have decreased?
                    # We didn't update alpha in the inner loop.
                    # Let's just re-calc if we added edges.
                    pass # The outer loop will re-calc if we restart?
                    # No, we are in the same iteration.
                    # We should update best_edges if we improved.
                    # But we don't know the new alpha without re-calc.
                    # Let's just assume if we added edges, alpha might be lower.
                    # We'll just update best_edges to the current edges if we added any.
                    # And let the next outer iteration re-evaluate?
                    # No, we are in the same `while` block.
                    # We should update best_alpha and best_edges if we improved.
                    # But we don't know the new alpha.
                    # Let's just set best_edges = edges[:] and let the next iteration handle it?
                    # Or calculate it now.
                    new_alpha = _calc_alpha(edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = edges[:]

    return best_edges
