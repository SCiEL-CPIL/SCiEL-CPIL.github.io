# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses maximum clique formulation with greedy coloring upper bounds for pruning.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

    m = 205
    k = 2

    # Forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build the "compatible" graph: edge between a,b if (a-b) mod m is NOT in forbidden
    # We want maximum clique in this graph
    # compatible[i] is a bitmask of vertices compatible with vertex i
    compatible = [0] * m
    for a in range(m):
        for b in range(a + 1, m):
            d = (a - b) % m
            if d not in forbidden:
                compatible[a] |= (1 << b)
                compatible[b] |= (1 << a)

    # Reorder vertices for better search: use degree in compatible graph
    # Higher degree vertices first (they have more connections, so branching on them prunes more)
    degrees = [bin(compatible[i]).count('1') for i in range(m)]
    order = sorted(range(m), key=lambda v: -degrees[v])
    
    # Map original vertex to position in new ordering
    new_to_old = [0] * m
    old_to_new = [0] * m
    for new_idx, old_idx in enumerate(order):
        new_to_old[new_idx] = old_idx
        old_to_new[old_idx] = new_idx
    
    # Rebuild compatible in new ordering
    compat_new = [0] * m
    for i in range(m):
        old_i = new_to_old[i]
        mask = compatible[old_i]
        new_mask = 0
        for j in range(m):
            if mask & (1 << new_to_old[j]):
                new_mask |= (1 << j)
        compat_new[i] = new_mask

    best_clique = []
    best_size = 0

    sys.setrecursionlimit(10000)

    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns (color_count, color_classes) where color_count is an upper bound
        on the clique number, and color_classes is a list of masks for each color.
        """
        if candidates_mask == 0:
            return 0, []
        
        colors = []
        remaining = candidates_mask
        
        while remaining:
            color_mask = 0
            temp = remaining
            while temp:
                # Pick the lowest set bit
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                # Check if v can be added to current color (no edges to color_mask in compatible graph)
                # In compatible graph, we want independent set for coloring
                # v is compatible with vertices in color_mask if compat_new[v] & color_mask != 0
                # For coloring, we need v to NOT be adjacent to any vertex in color_mask
                # i.e., compat_new[v] & color_mask == 0
                if compat_new[v] & color_mask == 0:
                    color_mask |= lsb
                    temp &= ~lsb
                else:
                    temp &= ~lsb
            
            if color_mask:
                colors.append(color_mask)
            remaining &= ~color_mask
        
        return len(colors), colors

    def branch_and_bound(clique_mask, candidates_mask, depth=0):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if bin(clique_mask).count('1') > best_size:
                best_size = bin(clique_mask).count('1')
                best_clique = [i for i in range(m) if clique_mask & (1 << i)]
            return
        
        # Upper bound via greedy coloring
        color_count, color_classes = greedy_color_bound(candidates_mask)
        
        # Pruning: even with all remaining candidates, can we beat best?
        if bin(clique_mask).count('1') + color_count <= best_size:
            return
        
        # Order candidates by color class (larger color classes first for better pruning)
        # Process vertices in order of their color class, starting from highest color
        ordered_candidates = []
        for c in range(color_count - 1, -1, -1):
            cm = color_classes[c]
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                ordered_candidates.append(v)
                temp &= ~lsb
        
        # Branch on each candidate
        for v in ordered_candidates:
            if time.time() > deadline:
                return
            
            # Check if v is still in candidates
            if not (candidates_mask & (1 << v)):
                continue
            
            # Include v in clique
            new_clique_mask = clique_mask | (1 << v)
            # New candidates: those compatible with v and still in candidates
            new_candidates = candidates_mask & compat_new[v]
            
            branch_and_bound(new_clique_mask, new_candidates, depth + 1)
            
            # Remove v from candidates for subsequent branches
            candidates_mask &= ~(1 << v)
            
            # Re-check upper bound after removing v
            remaining_count = bin(candidates_mask).count('1')
            if bin(clique_mask).count('1') + remaining_count <= best_size:
                break

    # Initial call: empty clique, all vertices as candidates
    all_mask = (1 << m) - 1
    branch_and_bound(0, all_mask)

    # Convert back to original vertex labels
    result = [new_to_old[v] for v in best_clique]
    
    # Verify the solution
    if result:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d = (result[i] - result[j]) % m
                if d in forbidden:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result = []

    if not result:
        # Fallback to greedy
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        result = selected
    
    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
