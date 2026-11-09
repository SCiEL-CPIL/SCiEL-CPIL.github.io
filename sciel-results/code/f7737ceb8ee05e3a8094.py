import time
import math
import random

def construct():
    start_time = time.time()
    # 90s limit, keep 10s margin
    deadline = start_time + 80.0
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    nv = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict graph using bitsets
    # conflicts[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
                
    # Precompute degrees for heuristics
    degrees = [bin(conflicts[i]).count('1') for i in range(nv)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_mis(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask

    def local_search(mask, max_iters=1000):
        # Local search with Add, 1-Swap, and Kick
        # We use a simple tabu-like mechanism by keeping a history of last few masks?
        # Or just random kicks.
        
        current_mask = mask
        best_mask = mask
        best_count = popcount(current_mask)
        
        # Precompute list of indices for faster iteration
        all_indices = list(range(nv))
        
        iters = 0
        no_improve_count = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try adding nodes (0-swap)
            # Iterate in random order or low degree first?
            # Low degree first is good for greedy, but for local search, random might help escape.
            # Let's try adding any valid node.
            # Optimization: Only check nodes that are not in mask.
            # We can iterate through all nodes, but skip those in mask.
            
            # To speed up, we can maintain a list of candidates?
            # For nv=250, O(nv) is fast.
            
            # Try adding
            for i in all_indices:
                if current_mask & (1 << i):
                    continue
                if (conflicts[i] & current_mask) == 0:
                    current_mask |= (1 << i)
                    improved = True
                    # Restart inner loop? Or continue?
                    # If we add, we might enable more adds? No, adding only restricts.
                    # But it increases count.
                    # We should continue to try more adds in this pass?
                    # Yes, greedy filling is good.
            
            if improved:
                # We added at least one node.
                # Update best
                c = popcount(current_mask)
                if c > best_count:
                    best_count = c
                    best_mask = current_mask
                no_improve_count = 0
                continue # Restart the pass to see if more can be added? 
                         # Actually, if we added, we should check if we can add more.
                         # The loop above does that. But if we break out of 'for i', 
                         # we need to restart the 'while' to check again?
                         # No, the 'for i' loop checks all i. If it adds one, it continues to next i.
                         # So it does a full pass of greedy filling.
                         # So if 'improved' is True, we have done a full greedy pass.
                         # We should continue to next while iteration to try swaps.
                         # But wait, if we added a node, the state changed.
                         # We should probably try swaps now.
                         # So we don't 'continue' immediately? 
                         # If we 'continue', we go to next while iter, which starts with 'Try adding'.
                         # Since we just did a full pass of adding, no more adds are possible.
                         # So next iter will find no adds.
                         # This is inefficient.
                         # Let's restructure.
            
            if not improved:
                # No adds possible. Try 1-swaps.
                # For each i not in mask, for each r in mask such that i conflicts with r:
                # If i does not conflict with mask \ {r}, then swap r -> i.
                
                # To avoid O(n^2) or O(n^3) complexity in Python, we need care.
                # nv is ~250.
                # Non-chosen nodes: ~250 - best_count.
                # Chosen nodes: best_count.
                # If best_count is ~100, then 150 non-chosen.
                # For each non-chosen i, find conflicts in mask.
                # confs = conflicts[i] & current_mask
                # If confs is 0, we would have added it.
                # If confs has exactly 1 bit set (say r), and i has no other conflicts with mask, 
                # then we can swap r with i.
                # Wait, if confs has 1 bit, that means i ONLY conflicts with r in the mask.
                # So removing r allows adding i.
                
                swap_done = False
                # Randomize order to avoid bias
                indices_not_in_mask = [i for i in all_indices if not (current_mask & (1 << i))]
                random.shuffle(indices_not_in_mask)
                
                for i in indices_not_in_mask:
                    confs = conflicts[i] & current_mask
                    if confs == 0:
                        # Should have been added in previous step, but maybe state changed?
                        # No, we are in 'not improved' branch, so no adds were possible.
                        # So confs != 0.
                        pass
                    
                    # Check if confs is a power of 2 (single bit)
                    if confs and (confs & (confs - 1)) == 0:
                        # Single conflict r
                        r = confs.bit_length() - 1
                        # Swap r with i
                        # This keeps size same.
                        # Why do it? To change the neighborhood and potentially allow future adds.
                        # We should only do this if it leads to a "better" state?
                        # Since size is same, we can't measure improvement by size.
                        # We can use a heuristic: Does the new node i have lower degree?
                        # Or just do it randomly?
                        # If we do it randomly, we might oscillate.
                        # Let's try it and see if it allows an add in the next step.
                        
                        # Perform swap
                        current_mask &= ~(1 << r)
                        current_mask |= (1 << i)
                        swap_done = True
                        break # Try only one swap per iteration to avoid large jumps?
                        
                if swap_done:
                    no_improve_count += 1
                    # Check if this swap allows an add?
                    # We loop back to top, which tries adds.
                    # So we continue.
                    continue
                else:
                    # No 1-swap found.
                    # Try 2-swap? Or Kick?
                    
                    # Kick: Remove 2 random nodes, greedily fill.
                    chosen_indices = [i for i in all_indices if current_mask & (1 << i)]
                    if len(chosen_indices) >= 2:
                        random.shuffle(chosen_indices)
                        r1 = chosen_indices[0]
                        r2 = chosen_indices[1]
                        
                        new_mask = current_mask & ~(1 << r1) & ~(1 << r2)
                        
                        # Greedy fill
                        # Order by degree (ascending)
                        order = sorted(range(nv), key=lambda x: degrees[x])
                        for i in order:
                            if new_mask & (1 << i):
                                continue
                            if (conflicts[i] & new_mask) == 0:
                                new_mask |= (1 << i)
                        
                        new_count = popcount(new_mask)
                        if new_count > best_count:
                            best_count = new_count
                            best_mask = new_mask
                            current_mask = new_mask
                            no_improve_count = 0
                            continue
                        elif new_count == best_count:
                            # Accept with some probability? Or just accept to explore?
                            # Let's accept to explore.
                            current_mask = new_mask
                            no_improve_count += 1
                            continue
                    else:
                        # Cannot kick
                        break
                        
                    # If no kick possible or didn't improve, break?
                    # Let's try a few more kicks?
                    # If we are stuck, break and restart.
                    break

        return best_mask

    best_mask = 0
    best_count = 0
    
    random.seed(42)
    
    # Initial strategies
    # 1. Greedy by degree ascending
    order = sorted(range(nv), key=lambda i: degrees[i])
    mask = greedy_mis(order)
    mask = local_search(mask, max_iters=200)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    # 2. Greedy by degree descending
    order = sorted(range(nv), key=lambda i: degrees[i], reverse=True)
    mask = greedy_mis(order)
    mask = local_search(mask, max_iters=200)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    # 3. Random restarts
    # Budget remaining
    while time.time() < deadline - 5:
        # Random order
        order = list(range(nv))
        random.shuffle(order)
        mask = greedy_mis(order)
        
        # Local search
        # Increase max_iters for random starts?
        mask = local_search(mask, max_iters=500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
            
        # Occasionally, perturb best and search
        if random.random() < 0.2:
            mask = best_mask
            # Remove 2 random nodes
            chosen_indices = [i for i in range(nv) if mask & (1 << i)]
            if len(chosen_indices) >= 2:
                random.shuffle(chosen_indices)
                r1 = chosen_indices[0]
                r2 = chosen_indices[1]
                mask &= ~(1 << r1) & ~(1 << r2)
                # Greedy fill
                order = list(range(nv))
                random.shuffle(order)
                for i in order:
                    if (conflicts[i] & mask) == 0:
                        mask |= (1 << i)
                
                mask = local_search(mask, max_iters=500)
                c = popcount(mask)
                if c > best_count:
                    best_count = c
                    best_mask = mask

    # Final polish
    if time.time() < deadline - 2:
        mask = best_mask
        # Run local search with higher budget
        mask = local_search(mask, max_iters=1000)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask

    # Convert to list
    result = []
    for i in range(nv):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])
            
    if not result:
        result = [valid_nodes[0]]
        
    return result
