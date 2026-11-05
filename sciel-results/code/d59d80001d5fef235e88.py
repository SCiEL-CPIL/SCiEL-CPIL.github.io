"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    edge_to_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y, i))
                edge_to_index[(x, y)] = idx

    num_edges = len(all_edges)

    # Precompute for each edge the list of C4s it participates in.
    # A C4 is defined by two opposite edges or by the four edges themselves.
    # For an edge e=(u,v,i), any other bit j != i defines a face.
    # The other three edges in that face are:
    # e1: (u^j, v^j)
    # e2: (u, u^j)
    # e3: (v, v^j)
    # If e, e1, e2, e3 are all selected, it's a C4.
    
    # For efficient checking, for each edge e, store the set of "constraints" it imposes.
    # Actually, it's easier to just check the 6 C4s containing a given edge when adding it.
    
    def get_c4s_for_edge(edge):
        x, y, bit_i = edge
        c4s = []
        for j in range(d):
            if j == bit_i:
                continue
            # The other 3 edges
            e1 = (min(x ^ (1 << j), y ^ (1 << j)), max(x ^ (1 << j), y ^ (1 << j)))
            e2 = (min(x, x ^ (1 << j)), max(x, x ^ (1 << j)))
            e3 = (min(y, y ^ (1 << j)), max(y, y ^ (1 << j)))
            c4s.append((e1, e2, e3))
        return c4s

    # Precompute C4s for all edges to save time during search
    edge_c4s = {}
    for e in all_edges:
        edge_c4s[(e[0], e[1])] = get_c4s_for_edge(e)

    def is_valid_add(chosen_set, edge_key):
        """Check if adding edge_key to chosen_set creates a C4."""
        for (e1, e2, e3) in edge_c4s[edge_key]:
            if e1 in chosen_set and e2 in chosen_set and e3 in chosen_set:
                return False
        return True

    def greedy_initial(seed):
        """Run greedy with randomized edge order to get initial solution."""
        rng = random.Random(seed)
        edges = list(all_edges)
        rng.shuffle(edges)

        chosen = set()
        out = []

        for edge in edges:
            x, y, bit_i = edge
            key = (x, y)
            if is_valid_add(chosen, key):
                chosen.add(key)
                out.append([x, y])

        return out, chosen

    def local_search(out, chosen, rng, time_budget):
        """
        Perform local search to improve the solution.
        Strategies:
        1. Try to add any valid edge.
        2. Try to swap one edge for two edges (if possible, increases count).
        3. Try to swap one edge for one edge (if it allows adding more later? complex).
        
        For simplicity and speed, we focus on:
        - Adding valid edges (should be 0 if greedy was thorough, but random order might miss some if we allow re-adding? No, greedy is maximal).
        - So we must rely on swaps.
        
        A better swap: Remove edge e, try to add edge e' such that e' is valid, and maybe e''?
        Actually, a standard move is:
        - Pick a random edge e in the set.
        - Remove e.
        - Try to add random edges until we have added more than removed? 
        - Or just try to add a valid edge after removing one. If we remove 1 and add 1, count stays same. 
        - If we remove 1 and add 2, count increases.
        
        Let's try:
        1. Randomly remove an edge e.
        2. Try to add up to 3 random edges from the unselected set.
        3. If we added > 1 edge, keep the change.
        4. If we added 1 edge, count is same. Might be useful for future moves? 
           Usually we want to increase count. So we prioritize moves that increase count.
        5. If we added 0 edges, revert.
        
        To be efficient, we can limit the number of attempts.
        """
        best_out = [list(e) for e in out]
        best_chosen = set(chosen)
        best_count = len(out)
        
        # We need a list of unselected edges for fast sampling
        # But maintaining a list of unselected edges dynamically is hard.
        # We can just sample from all_edges and check if it's in chosen.
        
        while time.time() - start_time < start_time + time_budget:
            # Check time
            if time.time() - start_time > time_limit - 2.0:
                break
                
            # Try a move
            # Pick a random edge to remove
            if not out:
                break
            idx = rng.randint(0, len(out) - 1)
            rem_key = tuple(out[idx])
            
            # Temporarily remove
            chosen.discard(rem_key)
            
            added_edges = []
            # Try to add edges
            # We want to add at least 2 to improve, or 1 to maintain (for diversity?)
            # Let's aim to add 2.
            for _ in range(50): # Limit attempts per move
                cand = rng.choice(all_edges)
                cand_key = (cand[0], cand[1])
                if cand_key in chosen:
                    continue
                if cand_key == rem_key:
                    continue
                
                if is_valid_add(chosen, cand_key):
                    chosen.add(cand_key)
                    added_edges.append(cand_key)
                    if len(added_edges) >= 2:
                        break
            # End add attempts
            
            # Evaluate
            new_count = len(chosen)
            if new_count > best_count:
                # Update best
                best_count = new_count
                best_chosen = set(chosen)
                # Reconstruct out from best_chosen
                best_out = [[u, v] for u, v in best_chosen]
                
                # Update current state to be the new best
                out = [list(e) for e in best_out]
                chosen = set(best_chosen)
            else:
                # Revert
                for e in added_edges:
                    chosen.discard(e)
                chosen.add(rem_key)
                # out remains unchanged in this branch (it's the previous best or current)
                # Wait, if we didn't improve, we should revert to the state BEFORE the move.
                # The state before the move was `out` and `chosen` (with rem_key removed and added_edges added).
                # So we just remove added_edges and re-add rem_key.
                # `out` list needs to be consistent with `chosen`.
                # Since we didn't update `out` in the "else" branch, it still reflects the state before the move attempt?
                # No, `out` was not modified during the attempt.
                # So `out` is still the list from the start of the loop.
                # `chosen` is now restored to the start of the loop state.
                # So `out` and `chosen` are consistent.
                
            # Occasionally, if no improvement, we might just shuffle or restart?
            # For now, keep going.
            
        return best_out, best_chosen

    # Run multiple iterations
    best_result = None
    best_count = 0

    # Number of restarts depends on time
    # Each restart takes ~0.1s for greedy, and then local search.
    # Let's do 5-10 restarts.
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        seed = i * 7919 + 13
        out, chosen = greedy_initial(seed)
        
        current_count = len(out)
        if current_count > best_count:
            best_count = current_count
            best_result = [list(e) for e in out]
            
        # Allocate remaining time for local search
        remaining_time = time_limit - (time.time() - start_time) - 2.0
        if remaining_time > 0.5:
            rng = random.Random(seed + 1000)
            # Give local search a portion of the time
            # If we have 100s left, give it 90s? 
            # Better to give it a fixed chunk or all remaining?
            # Since we are in a loop of restarts, we should limit local search per restart.
            # Let's say 5 seconds per restart for local search.
            ls_time = min(5.0, remaining_time)
            out, chosen = local_search(out, chosen, rng, ls_time)
            
            if len(out) > best_count:
                best_count = len(out)
                best_result = [list(e) for e in out]

    if best_result is None:
        # Fallback
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            key = (x, y)
            if is_valid_add(chosen, key):
                chosen.add(key)
                best_result.append([x, y])

    return best_result
# EVOLVE-BLOCK-END
