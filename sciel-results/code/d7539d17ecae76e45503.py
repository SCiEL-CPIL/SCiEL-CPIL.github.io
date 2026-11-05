import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflicts using bitsets for speed
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))  # bit (j-1) represents vertex j
        conflict_mask[i] = mask

    # Helper to get neighbors in remaining set
    def get_neighbors(v, remaining_mask):
        """Returns bitmask of neighbors of v that are in remaining_mask."""
        return conflict_mask[v] & remaining_mask

    # Greedy MIS with min-degree heuristic
    def greedy_mis(vertices_list, remaining_mask, min_degree_order=True):
        """
        Greedy independent set.
        vertices_list: list of vertices to consider
        remaining_mask: bitmask of available vertices
        Returns: (solution_list, final_remaining_mask)
        """
        solution = []
        # Work with a mutable copy of remaining
        rem = remaining_mask
        
        while rem:
            if min_degree_order:
                # Find vertex with minimum degree in the induced subgraph on rem
                min_v = None
                min_deg = float('inf')
                # Iterate through bits set in rem
                temp = rem
                while temp:
                    # Get lowest set bit
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1  # 0-indexed
                    v = v_idx + 1  # vertex number
                    
                    # Count neighbors in rem
                    deg = (conflict_mask[v] & rem).bit_count()
                    
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb  # clear the bit
            else:
                # Just pick the lowest vertex
                lsb = rem & (-rem)
                min_v = lsb.bit_length()  # vertex number
                
            solution.append(min_v)
            
            # Remove min_v and its neighbors from rem
            neighbors = conflict_mask[min_v]
            rem &= ~(1 << (min_v - 1))  # remove min_v
            rem &= ~neighbors  # remove neighbors
            
        return solution, rem

    # Initial greedy solution
    all_mask = (1 << N) - 1  # bits 0..255 set
    best_sol, _ = greedy_mis(vertices, all_mask, min_degree_order=True)
    best_size = len(best_sol)
    best_sol_set = set(best_sol)
    
    # Randomized restarts with different tie-breaking
    best_sol = list(best_sol)
    
    for iteration in range(5000):  # Max restarts
        if time.time() > deadline:
            break
        
        # Randomized greedy: pick from top-k min-degree vertices
        rem = all_mask
        sol = []
        
        while rem:
            # Find candidates with low degree
            temp = rem
            candidates = []
            deg_list = []
            
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                deg = (conflict_mask[v] & rem).bit_count()
                candidates.append(v)
                deg_list.append(deg)
                temp ^= lsb
            
            # Sort by degree
            sorted_pairs = sorted(zip(deg_list, candidates))
            
            # Pick from top 3 (or all if fewer)
            k = min(3, len(sorted_pairs))
            if k == 0:
                break
            chosen = sorted_pairs[random.randrange(k)][1]
            
            sol.append(chosen)
            neighbors = conflict_mask[chosen]
            rem &= ~(1 << (chosen - 1))
            rem &= ~neighbors
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_sol_set = set(sol)
    
    # Local search: try to swap vertices to increase size
    # Since it's an independent set, we can try 2-opt: remove 2, add 3, etc.
    # But for simplicity, let's try 1-for-1 swaps first.
    
    for iteration in range(2000):  # Local search iterations
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a vertex
        for v in vertices:
            if v in best_sol_set:
                continue
            # Check if v can be added
            # v can be added if no neighbor is in best_sol
            conflict_with_sol = 0
            for u in best_sol:
                if conflict_mask[v] & (1 << (u - 1)):
                    conflict_with_sol = 1
                    break
            if conflict_with_sol == 0:
                best_sol.append(v)
                best_sol_set.add(v)
                best_size = len(best_sol)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swap
        for v_out in best_sol:
            # If we remove v_out, which new vertices can we add?
            new_candidates = []
            for v_in in vertices:
                if v_in in best_sol_set and v_in != v_out:
                    continue
                if v_in == v_out:
                    continue
                # Check if v_in conflicts with best_sol - {v_out}
                ok = True
                for u in best_sol:
                    if u == v_out:
                        continue
                    if conflict_mask[v_in] & (1 << (u - 1)):
                        ok = False
                        break
                if ok:
                    new_candidates.append(v_in)
            
            if len(new_candidates) >= 1:
                # Try swapping v_out with a new candidate
                # Only beneficial if we can add more than we remove
                # For 1-for-1, size stays same, so not helpful for maximization
                # But it might enable further improvements
                # Skip for now, focus on size increase
                pass
        
        # Try 2-for-3 or other swaps is complex; skip for time
        
    return best_sol
