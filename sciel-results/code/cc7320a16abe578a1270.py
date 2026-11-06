import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    # We only care about valid nodes (2i not square)
    # But for simplicity, let's build masks for all 1..256 and filter later.
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
            
    # Refined conflict masks for valid nodes only? 
    # Actually, if a is invalid, it's never in the set. 
    # If b is invalid, it's never in the set.
    # So conflicts with invalid nodes don't matter.
    # We can mask out invalid nodes from conflict masks.
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
        
    # Greedy initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = set()
        # Shuffle
        nodes_list = list(remaining)
        rng.shuffle(nodes_list)
        # Sort by degree in remaining? 
        # Let's just do random greedy
        for a in nodes_list:
            # Check if a conflicts with any in chosen
            if conflict_mask[a] & (sum([1<<x for x in chosen]) if chosen else 0) == 0:
                 # Optimization: maintain chosen_mask
                 pass
                 
        # Better Greedy:
        chosen_mask = 0
        chosen_list = []
        candidates = list(valid_nodes)
        rng.shuffle(candidates)
        # Calculate degree for sorting
        def get_deg(node, cand_set):
             m = conflict_mask[node] & cand_set
             return bin(m).count('1')
             
        # Simple greedy: pick random, remove neighbors
        cand_mask = valid_mask_all
        while cand_mask:
             # Pick a node from cand_mask
             # To make it diverse, pick random bit
             # Or pick min degree?
             # Let's pick random for diversity
             bits = []
             temp = cand_mask
             while temp:
                 lsb = temp & -temp
                 idx = lsb.bit_length() - 1
                 bits.append(idx)
                 temp ^= lsb
             if not bits: break
             a = rng.choice(bits)
             chosen_mask |= (1 << a)
             # Remove a and its neighbors from candidates
             cand_mask &= ~(conflict_mask[a] | (1 << a))
             chosen_list.append(a)
        return chosen_mask, chosen_list

    best_mask = 0
    best_size = 0
    best_list = []
    
    def local_search(start_mask, start_list, seed):
        rng = random.Random(seed)
        in_mask = start_mask
        in_set = set(start_list)
        out_set = set(valid_nodes) - in_set
        out_mask = valid_mask_all & ~in_mask
        
        current_size = len(in_set)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 2:
            improved = False
            
            # 1. Try to add 1 node
            # Find nodes in out_set with no conflicts in in_set
            # conflict_mask[a] & in_mask == 0
            candidates_add = []
            temp_out = out_mask
            while temp_out:
                lsb = temp_out & -temp_out
                a = lsb.bit_length() - 1
                if conflict_mask[a] & in_mask == 0:
                    candidates_add.append(a)
                temp_out ^= lsb
                
            if candidates_add:
                # Add one (or all independent ones? Greedy add)
                # Just add one random to keep it simple, or all?
                # If we add all, we might block others. 
                # Let's add one random.
                a = rng.choice(candidates_add)
                in_mask |= (1 << a)
                in_set.add(a)
                out_set.remove(a)
                out_mask &= ~(1 << a)
                # Remove neighbors from out_set? No, out_set is just "not in in_set".
                # But we should update "available" logic? 
                # The check `conflict_mask[a] & in_mask == 0` handles validity.
                current_size += 1
                improved = True
                continue # Restart loop to check for more adds
                
            # 2. Try 1-remove, k-add
            # For each u in in_set, see if removing it allows adding >1 nodes
            # This is expensive to do for all u. 
            # Heuristic: pick a few random u's or the one with highest "potential"
            
            # Potential of u: number of v in out_set such that (conflict_mask[v] & in_mask) == (1 << u)
            # i.e., v conflicts with u and NO OTHER in_set node.
            
            # Let's try a few random u's
            in_list = list(in_set)
            rng.shuffle(in_list)
            
            found_swap = False
            for u in in_list[:5]: # Check top 5 random
                if time.time() - start_time > deadline - 2: break
                
                # Nodes in out_set that ONLY conflict with u in in_set
                # (conflict_mask[v] & in_mask) == (1 << u)
                potential_adds = []
                temp_out = out_mask
                while temp_out:
                    lsb = temp_out & -temp_out
                    v = lsb.bit_length() - 1
                    # v must not conflict with in_set \ {u}
                    # (conflict_mask[v] & (in_mask ^ (1 << u))) == 0
                    # And v must conflict with u? Not necessarily, but if it doesn't conflict with u, 
                    # it would have been added in step 1. 
                    # So we are looking for v that is blocked by u.
                    if (conflict_mask[v] & (in_mask ^ (1 << u))) == 0:
                         # v is compatible with in_set \ {u}
                         # We can add v if we remove u.
                         potential_adds.append(v)
                    temp_out ^= lsb
                         
                if len(potential_adds) > 1:
                    # We can remove u and add k nodes from potential_adds
                    # We need to pick a maximal independent set from potential_adds
                    # Greedy MIS on potential_adds
                    new_in_mask = in_mask ^ (1 << u)
                    new_in_set = in_set - {u}
                    new_out_set = out_set | {u}
                    
                    # Greedy add from potential_adds
                    added_any = False
                    # Sort potential_adds by degree in potential_adds?
                    # Just random greedy
                    cand = list(potential_adds)
                    rng.shuffle(cand)
                    added_count = 0
                    for v in cand:
                         if conflict_mask[v] & new_in_mask == 0:
                             new_in_mask |= (1 << v)
                             new_in_set.add(v)
                             new_out_set.remove(v)
                             added_count += 1
                             
                    if added_count > 1: # Gain > 0
                         in_mask = new_in_mask
                         in_set = new_in_set
                         out_set = new_out_set
                         out_mask = valid_mask_all & ~in_mask
                         current_size = len(in_set)
                         improved = True
                         found_swap = True
                         break
                         
            if found_swap:
                continue # Restart loop
                
            # 3. Try 2-remove, k-add
            # Remove two nodes u1, u2. Add k nodes.
            # Gain if k > 2.
            # This is O(N^2) to check pairs. 256^2 = 65k pairs.
            # For each pair, checking potential adds is O(N). 65k * 256 = 16M ops.
            # Might be slow in Python if done every iteration.
            # Let's limit this to a few random pairs or specific conditions.
            
            if (time.time() - start_time) < deadline - 5:
                in_list = list(in_set)
                # Try a few random pairs
                for _ in range(10):
                    if time.time() - start_time > deadline - 5: break
                    if len(in_list) < 2: break
                    idx1, idx2 = rng.sample(range(len(in_list)), 2)
                    u1 = in_list[idx1]
                    u2 = in_list[idx2]
                    
                    mask_without = in_mask ^ (1 << u1) ^ (1 << u2)
                    
                    potential_adds = []
                    temp_out = out_mask
                    while temp_out:
                        lsb = temp_out & -temp_out
                        v = lsb.bit_length() - 1
                        if (conflict_mask[v] & mask_without) == 0:
                            potential_adds.append(v)
                        temp_out ^= lsb
                            
                    if len(potential_adds) > 2:
                        new_in_mask = mask_without
                        new_in_set = in_set - {u1, u2}
                        new_out_set = out_set | {u1, u2}
                        
                        cand = list(potential_adds)
                        rng.shuffle(cand)
                        added_count = 0
                        for v in cand:
                            if conflict_mask[v] & new_in_mask == 0:
                                new_in_mask |= (1 << v)
                                new_in_set.add(v)
                                new_out_set.remove(v)
                                added_count += 1
                                
                        if added_count > 2:
                            in_mask = new_in_mask
                            in_set = new_in_set
                            out_set = new_out_set
                            out_mask = valid_mask_all & ~in_mask
                            current_size = len(in_set)
                            improved = True
                            break
                            
            if not improved:
                break
                
        return in_mask, in_set

    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        init_mask, init_list = greedy_init(i)
        ls_mask, ls_set = local_search(init_mask, init_list, i)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_mask = ls_mask
            best_list = list(ls_set)
            
    return best_list
