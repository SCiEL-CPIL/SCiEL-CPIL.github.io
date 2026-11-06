import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
    
    def popcount(x):
        """Count bits in integer x."""
        return bin(x).count('1')
    
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic with bit operations."""
        rng = random.Random(seed)
        remaining = valid_mask_all
        chosen = []
        chosen_mask = 0
        
        while remaining:
            # Find node with minimum degree in remaining using bit ops
            min_deg = float('inf')
            candidates_at_min = []
            temp = remaining
            while temp:
                a = (temp & -temp).bit_length() - 1
                if a >= 1:
                    deg = popcount(conflict_mask[a] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        candidates_at_min = [a]
                    elif deg == min_deg:
                        candidates_at_min.append(a)
                temp &= temp - 1
            
            if candidates_at_min:
                min_node = rng.choice(candidates_at_min)
                chosen.append(min_node)
                chosen_mask |= (1 << min_node)
                # Remove min_node and its neighbors
                remaining &= ~(1 << min_node)
                remaining &= ~conflict_mask[min_node]
        
        return chosen_mask, chosen
    
    def greedy_coloring_upper_bound(candidate_mask):
        """
        Greedy coloring upper bound for maximum independent set.
        This gives a tighter bound than simple cardinality.
        Returns the number of colors needed (upper bound on IS size).
        """
        if candidate_mask == 0:
            return 0
        
        # Get list of vertices in candidate_mask
        verts = []
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                verts.append(v)
            temp &= temp - 1
        
        if not verts:
            return 0
        
        # Order by descending degree in the subgraph
        verts.sort(key=lambda v: -popcount(conflict_mask[v] & candidate_mask))
        
        # Greedy coloring
        color_of = {}
        color_classes = []  # list of sets of vertices
        
        for v in verts:
            # Find smallest color class that v doesn't conflict with
            placed = False
            for i, cls in enumerate(color_classes):
                # Check if v conflicts with any vertex in cls
                conflict = False
                for u in cls:
                    if conflict_mask[v] & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    cls.add(v)
                    color_of[v] = i
                    placed = True
                    break
            if not placed:
                color_classes.append({v})
                color_of[v] = len(color_classes) - 1
        
        return len(color_classes)
    
    def local_search_improve(mask, max_iters=2000):
        """Try to improve an independent set by adding vertices or swapping."""
        if time.time() - start_time > deadline - 5:
            return mask
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() - start_time < deadline - 5:
            improved = False
            iterations += 1
            # Try to add any vertex not in mask that doesn't conflict
            candidates = valid_mask_all & ~mask
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                if v >= 1:
                    if (conflict_mask[v] & mask) == 0:
                        mask |= (1 << v)
                        improved = True
                    temp &= temp - 1
            # Try swaps: remove one, add two
            if not improved:
                temp = mask
                while temp:
                    a = (temp & -temp).bit_length() - 1
                    if a >= 1:
                        mask_without_a = mask & ~(1 << a)
                        new_candidates = valid_mask_all & ~mask_without_a
                        new_candidates &= ~conflict_mask[a]
                        temp2 = new_candidates
                        while temp2:
                            b = (temp2 & -temp2).bit_length() - 1
                            if b >= 1 and b != a:
                                if (conflict_mask[b] & mask_without_a) == 0:
                                    mask_with_b = mask_without_a | (1 << b)
                                    temp3 = valid_mask_all & ~mask_with_b
                                    while temp3:
                                        c = (temp3 & -temp3).bit_length() - 1
                                        if c >= 1 and c != a and c != b:
                                            if (conflict_mask[c] & mask_with_b) == 0:
                                                new_mask = mask_with_b | (1 << c)
                                                if popcount(new_mask) > popcount(mask):
                                                    mask = new_mask
                                                    improved = True
                                                    break
                                        temp3 &= temp3 - 1
                                    if improved:
                                        break
                            temp2 &= temp2 - 1
                        if improved:
                            break
                    temp &= temp - 1
        return mask
    
    def branch_and_bound(candidate_mask, current_size, best_ref):
        """
        Branch and bound with greedy coloring upper bound.
        best_ref: list [best_size, best_mask]
        """
        if time.time() - start_time > deadline - 5:
            return
        
        if candidate_mask == 0:
            return
        
        # Upper bound using greedy coloring
        ub = current_size + greedy_coloring_upper_bound(candidate_mask)
        if ub <= best_ref[0]:
            return
        
        # Pick pivot with max degree in candidate subgraph
        pivot = None
        max_deg = -1
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                deg = popcount(conflict_mask[v] & candidate_mask)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            temp &= temp - 1
        
        if pivot is None:
            return
        
        # Branch 1: Include pivot
        new_mask_include = candidate_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        # Track the actual set by reconstructing from include decisions
        # We need to track which vertices are included
        # Use a different approach: pass the included set
        branch_and_bound_include(new_mask_include, current_size + 1, best_ref, pivot)
        
        # Branch 2: Exclude pivot
        new_mask_exclude = candidate_mask & ~(1 << pivot)
        branch_and_bound(new_mask_exclude, current_size, best_ref)
    
    def branch_and_bound_include(candidate_mask, current_size, best_ref, last_included):
        """Helper for the include branch that tracks the set."""
        # This is getting complex; let's use a simpler approach
        # Actually, let's just use the standard B&B and track via the mask
        pass
    
    # Simpler B&B that tracks the solution mask
    def bb_solve(candidate_mask, chosen_mask, current_size, best_ref):
        """Branch and bound with coloring bound, tracking chosen mask."""
        if time.time() - start_time > deadline - 5:
            return
        
        if candidate_mask == 0:
            if current_size > best_ref[0]:
                best_ref[0] = current_size
                best_ref[1] = chosen_mask
            return
        
        # Upper bound using greedy coloring
        ub = current_size + greedy_coloring_upper_bound(candidate_mask)
        if ub <= best_ref[0]:
            return
        
        # Pick pivot with max degree in candidate subgraph
        pivot = None
        max_deg = -1
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                deg = popcount(conflict_mask[v] & candidate_mask)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            temp &= temp - 1
        
        if pivot is None:
            return
        
        # Branch 1: Include pivot
        new_cand = candidate_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        new_chosen = chosen_mask | (1 << pivot)
        bb_solve(new_cand, new_chosen, current_size + 1, best_ref)
        
        # Branch 2: Exclude pivot
        new_cand_ex = candidate_mask & ~(1 << pivot)
        bb_solve(new_cand_ex, chosen_mask, current_size, best_ref)
    
    # First, get a good initial solution using greedy
    init_mask, init_list = greedy_min_degree_init(0)
    best_size = len(init_list)
    best_mask = init_mask
    best_list = list(init_list)
    
    # Improve with local search
    best_mask = local_search_improve(best_mask)
    best_size = popcount(best_mask)
    
    # Try multiple greedy seeds and keep best
    for seed in range(1, 10):
        if time.time() - start_time > deadline - 15:
            break
        gm, gl = greedy_min_degree_init(seed)
        gm = local_search_improve(gm)
        gs = popcount(gm)
        if gs > best_size:
            best_size = gs
            best_mask = gm
    
    # Use B&B with coloring bound
    best_ref = [best_size, best_mask]
    
    if time.time() - start_time < deadline - 10:
        bb_solve(valid_mask_all, 0, 0, best_ref)
    
    best_size = best_ref[0]
    best_mask = best_ref[1]
    
    # Final local search polish
    if time.time() - start_time < deadline - 3:
        best_mask = local_search_improve(best_mask)
        best_size = popcount(best_mask)
    
    # Extract the list from best_mask
    best_list = []
    temp = best_mask
    while temp:
        v = (temp & -temp).bit_length() - 1
        if v >= 1:
            best_list.append(v)
        temp &= temp - 1
    
    # Final verification and cleanup
    result = set(best_list)
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = int(math.isqrt(s))
                if r * r == s:
                    result.discard(b)
                    break
    
    return list(result)
