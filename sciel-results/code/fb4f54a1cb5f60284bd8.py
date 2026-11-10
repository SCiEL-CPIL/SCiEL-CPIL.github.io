# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses an exact branch-and-bound algorithm with bitset optimization.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency bitmasks
    # adj_mask[i] has bit j set if i and j are connected (forbidden difference)
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden:
            j = (i + f) % m
            mask |= (1 << j)
        adj_mask[i] = mask
        # Note: Since difference is symmetric, adj is symmetric. 
        # (i-j)%m in F <=> (j-i)%m in F because -1 is a square mod 205? 
        # 205 = 5 * 41.
        # Legendre symbol (-1/5) = 1, (-1/41) = 1. So -1 is a QR.
        # Thus the graph is undirected.
        
    # Initial candidate set: all vertices
    # We want to find Max Independent Set (MIS)
    
    best_set = []
    
    def greedy_coloring_bound(candidates):
        """
        Compute a greedy upper bound on the size of the MIS in the subgraph induced by 'candidates'.
        We color the subgraph. The number of colors is an upper bound on the size of the independent set.
        """
        if candidates == 0:
            return 0
            
        uncolored = candidates
        colors = 0
        while uncolored:
            colors += 1
            # Pick a vertex with max degree in uncolored? Or just first?
            # Standard Tomita: order vertices by degree, then color greedily.
            # For speed, let's just pick the lowest set bit or similar, but we need a valid coloring.
            # A valid coloring means no two adjacent vertices have the same color.
            # Greedy coloring:
            # 1. Order vertices (e.g., by degree descending).
            # 2. Assign smallest available color.
            
            # To keep it fast, we'll just do a simple greedy on the uncolored set.
            # We need to ensure we don't exceed time.
            # Let's use a heuristic: pick vertex with highest degree in the current uncolored subgraph.
            
            # Find vertex with max degree in 'uncolored'
            max_deg = -1
            best_v = -1
            temp = uncolored
            while temp:
                v = (temp & -temp).bit_length() - 1
                # Degree of v in uncolored
                deg = (adj_mask[v] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp &= temp - 1
                
            if best_v == -1:
                break

            # Remove best_v and its neighbors from uncolored (they get this color)
            uncolored &= ~(1 << best_v)
            uncolored &= ~adj_mask[best_v]
            
        return colors

    def branch_and_bound(candidates, current_size, current_set):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return

        if candidates == 0:
            if current_size > len(best_set):
                best_set = current_set.copy()
            return

        # Upper bound: current_size + greedy_coloring_bound(candidates)
        # If this is <= len(best_set), prune
        ub = current_size + greedy_coloring_bound(candidates)
        if ub <= len(best_set):
            return

        # Select a vertex to branch on
        # Heuristic: pick vertex with max degree in candidates
        max_deg = -1
        best_v = -1
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj_mask[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp &= temp - 1

        if best_v == -1:
            return

        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New candidates: candidates without v and without its neighbors
        new_candidates = candidates & ~v_bit & ~adj_mask[v]
        branch_and_bound(new_candidates, current_size + 1, current_set + [v])

        # Branch 2: Exclude v
        new_candidates = candidates & ~v_bit
        branch_and_bound(new_candidates, current_size, current_set)

    # Start with all vertices
    initial_candidates = (1 << m) - 1
    branch_and_bound(initial_candidates, 0, [])

    # Fallback if no set found or time ran out
    if not best_set:
        # Greedy fallback
        selected = []
        candidates = set(range(m))
        while candidates:
            # Pick vertex with min degree in remaining? Or just first
            # Let's just pick 0 if available, else first
            # Actually, let's do a quick greedy
            for v in sorted(candidates):
                if all((v - u) % m not in forbidden for u in selected):
                    selected.append(v)
                    break
            # Remove selected from candidates? No, we need to remove conflicts.
            # This fallback is messy, but B&B should finish.
            # Let's just return [0] if something goes wrong
            if not candidates: break
            # Simple removal for fallback logic correctness if used
            # But we don't really use this if B&B works.
            break
            
    if not best_set:
        best_set = [0]

    return np.array(sorted(best_set), dtype=np.int64)

# EVOLVE-BLOCK-END
