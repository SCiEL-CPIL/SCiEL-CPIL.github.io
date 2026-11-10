# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build conflict bitmask: conflict[i] has bit j set if i and j cannot both be in set.
    3. Use multi-start local search with 1-add, 1-swap, and 2-swap moves.
    4. Use branch-and-bound for initial large solution.
    5. Return the best solution found within time budget.
    """
    m = 205
    deadline = time.time() + 590  # 40s safety margin before 630s kill
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute: for each i, conflict[i] is a bitmask of all j that conflict with i
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
    
    def can_add(cand, selected_mask):
        """Check if cand can be added to the set represented by selected_mask."""
        return (conflict[cand] & selected_mask) == 0
    
    def greedy_random(seed):
        """Random greedy construction using bitmask conflict checks."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        selected_list = []
        for cand in candidates:
            if can_add(cand, selected_mask):
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_mask, selected_list
    
    def local_search(selected_mask, selected_list, rng, max_iters=5000, deadline=None):
        """1-add, 1-swap, and 2-swap local search using bitmask conflict checks."""
        best_mask = selected_mask
        best_list = list(selected_list)
        best_size = len(best_list)
        
        no_improve = 0
        iters = 0
        while iters < max_iters and no_improve < 200:
            if deadline and time.time() > deadline:
                break
            iters += 1
            improved = False
            
            # Try 1-add: find a candidate that can be added
            candidates_to_add = [c for c in range(m) if not (best_mask >> c) & 1]
            rng.shuffle(candidates_to_add)
            for cand in candidates_to_add:
                if can_add(cand, best_mask):
                    best_mask |= (1 << cand)
                    best_list.append(cand)
                    best_size += 1
                    improved = True
                    no_improve = 0
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one element, add another
            swap_order = list(range(len(best_list)))
            rng.shuffle(swap_order)
            for idx in swap_order:
                rem = best_list[idx]
                temp_mask = best_mask & ~(1 << rem)
                
                # Find a candidate not in temp_mask that can be added
                candidates_to_add = [c for c in range(m) if not (temp_mask >> c) & 1]
                rng.shuffle(candidates_to_add)
                for cand in candidates_to_add:
                    if can_add(cand, temp_mask):
                        # Perform the swap
                        best_list[idx] = cand
                        best_mask = temp_mask | (1 << cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove two elements, add two different elements
            if len(best_list) >= 2:
                pair_order = list(range(len(best_list)))
                rng.shuffle(pair_order)
                found_swap = False
                for i in range(min(30, len(pair_order))):
                    for j in range(i+1, min(30, len(pair_order))):
                        rem1 = best_list[pair_order[i]]
                        rem2 = best_list[pair_order[j]]
                        temp_mask = best_mask & ~(1 << rem1) & ~(1 << rem2)
                        
                        # Find two candidates that can both be added
                        candidates_to_add = [c for c in range(m) if not (temp_mask >> c) & 1]
                        rng.shuffle(candidates_to_add)
                        
                        # Try to find two compatible additions
                        added = []
                        temp_mask2 = temp_mask
                        for cand in candidates_to_add:
                            if can_add(cand, temp_mask2):
                                added.append(cand)
                                temp_mask2 |= (1 << cand)
                                if len(added) == 2:
                                    break
                        
                        if len(added) == 2:
                            # Perform the 2-swap
                            new_list = [x for x in best_list if x != rem1 and x != rem2]
                            new_list.extend(added)
                            new_mask = temp_mask2
                            best_list = new_list
                            best_mask = new_mask
                            best_size = len(best_list)
                            improved = True
                            no_improve = 0
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            if not improved:
                no_improve += 1
        
        return best_mask, best_list, best_size
    
    def upper_bound(candidates_mask, current_size):
        """
        Compute an upper bound on the size of an independent set from candidates_mask.
        Uses greedy coloring approximation on the conflict graph.
        """
        if candidates_mask == 0:
            return current_size
        
        n_cands = candidates_mask.bit_count()
        if n_cands == 0:
            return current_size
        
        # Greedy coloring: assign colors to vertices, the number of colors is an upper bound
        # on the clique cover, and |V| / max_color_class_size gives a lower bound on chromatic number.
        # For independent set: alpha <= |V| (trivial). Use a tighter bound.
        
        # Simple bound: in any graph, alpha(G) >= |V| / (1 + avg_degree)
        # So upper bound on what we can still add is at most n_cands.
        # Better: use the fact that each selected vertex eliminates its neighbors.
        # Greedy estimate: repeatedly pick a vertex, count it, remove it and neighbors.
        
        temp_cands = candidates_mask
        count = 0
        while temp_cands:
            # Pick vertex with highest degree
            best_v = -1
            best_deg = -1
            tc = temp_cands
            while tc:
                i = (tc & (-tc)).bit_length() - 1
                tc &= tc - 1
                deg = (conflict[i] & temp_cands).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = i
            
            if best_v == -1:
                break
            
            count += 1
            temp_cands &= ~(1 << best_v)
            temp_cands &= ~conflict[best_v]
        
        # count is a lower bound on the number of vertices we MUST exclude
        # So upper bound on independent set size from candidates is n_cands - (something)
        # Actually, count is a lower bound on vertex cover, so alpha <= n_cands
        # This bound is too weak. Use: alpha <= n_cands - floor(count/2) as heuristic?
        # For now, just return n_cands as a safe (but weak) upper bound.
        return current_size + n_cands
    
    def branch_and_bound(candidates_mask, current_size, current_list, best_size_ref, deadline, max_nodes=200000):
        """
        Branch and bound for maximum independent set.
        candidates_mask: bitmask of vertices still available
        current_size: number of vertices in current independent set
        current_list: list of vertices in current independent set
        best_size_ref: [int] reference to best size found
        Returns: (best_mask, best_list, best_size)
        """
        best_mask = 0
        best_list = []
        best_size = 0
        nodes_visited = 0
        
        stack = [(candidates_mask, 0, tuple())]
        
        while stack and time.time() < deadline:
            if nodes_visited > max_nodes:
                break
            
            cands, cur_mask, cur_list = stack.pop()
            nodes_visited += 1
            
            # Compute upper bound
            ub = upper_bound(cands, len(cur_list))
            if ub <= best_size:
                continue
            
            # If no candidates left, check if this is a valid solution
            if cands == 0:
                if len(cur_list) > best_size:
                    best_size = len(cur_list)
                    best_mask = cur_mask
                    best_list = list(cur_list)
                continue
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            temp_cands = cands
            while temp_cands:
                i = (temp_cands & (-temp_cands)).bit_length() - 1
                temp_cands &= temp_cands - 1
                deg = (conflict[i] & cands).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = i
            
            if best_v == -1:
                continue
            
            # Branch 1: include best_v
            new_cands = cands & ~(1 << best_v)
            new_cands &= ~conflict[best_v]
            new_mask = cur_mask | (1 << best_v)
            new_list = cur_list + (best_v,)
            
            ub_include = upper_bound(new_cands, len(new_list))
            if ub_include > best_size:
                stack.append((new_cands, new_mask, new_list))
            
            # Branch 2: exclude best_v
            new_cands_exclude = cands & ~(1 << best_v)
            ub_exclude = upper_bound(new_cands_exclude, len(cur_list))
            if ub_exclude > best_size:
                stack.append((new_cands_exclude, cur_mask, cur_list))
        
        return best_mask, best_list, best_size
    
    # Main search loop
    best_mask = 0
    best_list = []
    best_size = 0
    
    # Start with deterministic greedy (in order)
    sel = []
    sel_mask = 0
    for cand in range(m):
        if can_add(cand, sel_mask):
            sel.append(cand)
            sel_mask |= (1 << cand)
    best_list = sel
    best_size = len(sel)
    best_mask = sel_mask
    
    # Try branch-and-bound with a time limit (use 120s for BnB)
    bnb_deadline = time.time() + 120
    full_cands = (1 << m) - 1
    bnb_mask, bnb_list, bnb_size = branch_and_bound(
        full_cands, 0, tuple(), [best_size], bnb_deadline, max_nodes=200000
    )
    
    if bnb_size > best_size:
        best_mask = bnb_mask
        best_list = bnb_list
        best_size = bnb_size
    
    # Polish with local search
    if time.time() < deadline:
        res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(42), deadline=deadline)
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
    
    # Try multiple random restarts with local search
    seed = 1
    while time.time() < deadline - 5:  # Leave 5s safety margin
        # Random greedy start
        start_mask, start_list = greedy_random(seed)
        
        # Local search with deadline awareness
        res_mask, res_list, res_size = local_search(start_mask, start_list, random.Random(seed), deadline=deadline)
        
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
        
        seed += 1
        
        # Also try local search on the current best every few iterations
        if seed % 5 == 0:
            res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(seed + 10000), deadline=deadline)
            if res_size > best_size:
                best_mask = res_mask
                best_list = res_list
                best_size = res_size
    
    # Final verification: ensure no forbidden pairs
    final_list = sorted(best_list)
    verified = []
    for i, a in enumerate(final_list):
        valid = True
        for j in range(i+1, len(final_list)):
            b = final_list[j]
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                valid = False
                break
        if valid:
            verified.append(a)
    
    # If verification failed, fall back to a simple greedy
    if len(verified) < len(best_list):
        verified = []
        v_mask = 0
        for cand in range(m):
            if can_add(cand, v_mask):
                verified.append(cand)
                v_mask |= (1 << cand)
    
    return np.array(verified, dtype=np.int64)

# EVOLVE-BLOCK-END
