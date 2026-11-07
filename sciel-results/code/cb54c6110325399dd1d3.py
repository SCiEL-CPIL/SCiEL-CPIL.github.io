"""Cochromatic partition via branch-and-bound over maximal cliques and independent sets."""
import time
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique(vertices, adj):
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if not (adj[u] >> v) & 1:
                return False
    return True

def _is_independent(vertices, adj):
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if (adj[u] >> v) & 1:
                return False
    return True

def _find_maximal_sets(vertices, adj):
    """Find all maximal cliques and maximal independent sets among the given vertices."""
    sets = []
    if not vertices:
        return sets
    
    # Find maximal cliques
    def expand_clique(current, candidates):
        if not candidates:
            if len(current) >= 1:
                sets.append(frozenset(current))
            return
        # Try each candidate
        for i, v in enumerate(candidates):
            new_current = current + [v]
            # New candidates must be neighbors of v and come after i to avoid duplicates
            new_candidates = [c for c in candidates[i+1:] if (adj[v] >> c) & 1]
            expand_clique(new_current, new_candidates)
    
    expand_clique([], list(vertices))
    
    # Find maximal independent sets (cliques in complement)
    def expand_independent(current, candidates):
        if not candidates:
            if len(current) >= 1:
                sets.append(frozenset(current))
            return
        for i, v in enumerate(candidates):
            new_current = current + [v]
            new_candidates = [c for c in candidates[i+1:] if not ((adj[v] >> c) & 1)]
            expand_independent(new_current, new_candidates)
    
    expand_independent([], list(vertices))
    
    # Remove duplicates and keep only maximal sets
    unique = set(sets)
    maximal = []
    for s in unique:
        is_maximal = True
        for t in unique:
            if s != t and s < t:
                is_maximal = False
                break
        if is_maximal:
            maximal.append(frozenset(s))
    
    return maximal

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    adj = _build_graph()
    all_vertices = frozenset(range(n))
    
    # Precompute all maximal homogeneous sets
    all_sets = _find_maximal_sets(list(range(n)), adj)
    
    # Sort by size descending for better pruning
    all_sets.sort(key=lambda s: -len(s))
    
    # Convert to bitmask representation for fast operations
    set_masks = []
    for s in all_sets:
        mask = 0
        for v in s:
            mask |= (1 << v)
        set_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    best_count = n  # Worst case: each vertex alone
    best_partition = None
    
    # Branch and bound
    def solve(remaining_mask, current_sets, count):
        nonlocal best_count, best_partition
        
        if time.time() > deadline:
            return
        
        if count >= best_count:
            return
        
        if remaining_mask == 0:
            best_count = count
            best_partition = list(current_sets)
            return
        
        # Lower bound: at least ceil(popcount(remaining) / max_set_size)
        remaining_count = bin(remaining_mask).count('1')
        if remaining_count == 0:
            return
        
        # Find the vertex with fewest options (most constrained)
        # Try to pick a vertex that appears in fewest sets covering remaining
        best_vertex = -1
        best_options = None
        min_option_count = float('inf')
        
        for v in range(n):
            if (remaining_mask >> v) & 1:
                options = [i for i, m in enumerate(set_masks) if (m & remaining_mask) == m and (m >> v) & 1]
                # Actually, we need sets that are subsets of remaining and contain v
                options = [i for i, m in enumerate(set_masks) if (m & ~remaining_mask) == 0 and (m >> v) & 1]
                if len(options) < min_option_count:
                    min_option_count = len(options)
                    best_vertex = v
                    best_options = options
        
        if best_vertex == -1:
            return
        
        if min_option_count == 0:
            return
        
        # Try each option for best_vertex
        for idx in best_options:
            s = set_masks[idx]
            new_remaining = remaining_mask & ~s
            current_sets.append(idx)
            solve(new_remaining, current_sets, count + 1)
            current_sets.pop()
            
            if time.time() > deadline:
                return
    
    solve(full_mask, [], 0)
    
    if best_partition is None:
        # Fallback: greedy or individual vertices
        return [[v] for v in range(n)]
    
    # Convert back to list of lists
    result = []
    for idx in best_partition:
        s = list(all_sets[idx])
        s.sort()
        result.append(s)
    
    # Ensure each vertex appears exactly once
    seen = set()
    for part in result:
        for v in part:
            if v in seen:
                # Should not happen, but fallback
                return [[v] for v in range(n)]
            seen.add(v)
    
    if len(seen) != n:
        return [[v] for v in range(n)]
    
    return result

# Test
if __name__ == '__main__':
    result = construct()
    print(f"Number of parts: {len(result)}")
    for i, part in enumerate(result):
        print(f"Part {i}: {part}")
