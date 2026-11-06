import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 256
    
    # 1. Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    # We use 0-indexed bits for 1-indexed numbers: bit (i-1) corresponds to number i
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = i + j
            p = i * j
            if p % s == 0:
                conflict_mask[i] |= (1 << (j - 1))
                conflict_mask[j] |= (1 << (i - 1))
                
    # 2. Initialize set with all odds
    S_mask = 0
    for i in range(1, N + 1, 2):
        S_mask |= (1 << (i - 1))
        
    # 3. Greedy add: try to add remaining numbers
    # We iterate until no more can be added
    improved = True
    while improved:
        improved = False
        for v in range(1, N + 1):
            v_bit = 1 << (v - 1)
            if S_mask & v_bit:
                continue
            # Check if v has 0 conflicts in S
            # If (conflict_mask[v] & S_mask) == 0, v is compatible with all in S
            if (conflict_mask[v] & S_mask) == 0:
                S_mask |= v_bit
                improved = True
                
    # 4. Local Search
    # We look for moves that increase cardinality.
    # Primary move: Remove u, add v, w.
    # Condition: 
    #   1. u in S
    #   2. v, w not in S
    #   3. v, w are compatible with S \ {u}
    #   4. v, w are compatible with each other
    #   5. v, w are NOT compatible with u (otherwise they would have been added in greedy, 
    #      unless they were blocked by something else, but if they are compatible with S\{u}, 
    #      their only blocker in S must be u. So conflict_mask[v] & (S_mask ^ (1<<(u-1))) == 0)
    
    # Helper to get list of nodes from mask
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << (i - 1))]
        
    # Helper to check if a node can be added to a mask
    def can_add(v, mask):
        return (conflict_mask[v] & mask) == 0

    # We will run a loop until time runs out
    # To avoid getting stuck in a local minimum where no 1-for-2 exists,
    # we might need 1-for-1 swaps or restarts, but 1-for-2 is the main size increaser.
    
    # Optimization: Precompute "blockers" for nodes not in S?
    # Let's just iterate. With bitmasks, the check is fast.
    
    S_list = mask_to_list(S_mask)
    
    while time.time() < deadline:
        improved = False
        
        # Randomize order to avoid bias
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline:
                break
                
            u_bit = 1 << (u - 1)
            S_without_u = S_mask ^ u_bit
            
            # Find candidates v not in S that are compatible with S_without_u
            # These are the nodes that would become addable if we removed u.
            # Note: They must also conflict with u (otherwise they'd be in S already, 
            # or if they don't conflict with u, they are compatible with S, so they'd be in S).
            # Actually, if v is not in S, it conflicts with at least one node in S.
            # If it is compatible with S \ {u}, then its only conflict in S is u.
            
            cands = []
            for v in range(1, N + 1):
                v_bit = 1 << (v - 1)
                if S_mask & v_bit:
                    continue
                # Check if v is compatible with S \ {u}
                if (conflict_mask[v] & S_without_u) == 0:
                    # Also ensure v conflicts with u? 
                    # If v doesn't conflict with u, then (conflict_mask[v] & S_mask) == 0, 
                    # which means v should have been added in the greedy phase.
                    # So this case shouldn't happen if greedy was complete.
                    # But just in case, we can add it.
                    cands.append(v)
            
            if len(cands) >= 2:
                # We need to find two nodes in cands that don't conflict with each other.
                # Since cands are all compatible with S \ {u}, we just need to check pair conflicts.
                found = False
                c_len = len(cands)
                for i in range(c_len):
                    v1 = cands[i]
                    for j in range(i + 1, c_len):
                        v2 = cands[j]
                        if (conflict_mask[v1] & (1 << (v2 - 1))) == 0:
                            # Valid move: remove u, add v1, v2
                            S_mask ^= u_bit
                            S_mask |= (1 << (v1 - 1))
                            S_mask |= (1 << (v2 - 1))
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    # Update S_list for next iteration? 
                    # We can just break and recompute or update locally.
                    # Breaking is safer and simpler, and recomputing S_list is O(N).
                    break
            
            # If no 1-for-2 found, try a 1-for-1 swap to potentially unblock future 1-for-2s?
            # Or just continue. The random shuffle in the outer loop helps.
            # A 1-for-1 swap keeps size constant. It might be useful to escape local minima 
            # where no 1-for-2 exists but a 1-for-1 followed by a 1-for-2 does.
            # However, 1-for-2 is the direct objective. 
            # Let's try a random 1-for-1 if we have plenty of time and no 1-for-2 found in this pass?
            # For now, let's stick to 1-for-2 to ensure progress.
            
        if not improved:
            # No 1-for-2 move found in this pass.
            # We might be stuck. 
            # Strategy: Try a random 1-for-1 swap to perturb the solution.
            # Find a node u in S and a node v not in S such that 
            # v is compatible with S \ {u} and u is compatible with S \ {v} (trivial) 
            # and v conflicts with u.
            # Actually, simpler: Just pick a random u in S and a random v not in S 
            # that is compatible with S \ {u}. Swap them.
            
            S_list = mask_to_list(S_mask)
            if not S_list:
                break
                
            u = random.choice(S_list)
            u_bit = 1 << (u - 1)
            S_without_u = S_mask ^ u_bit
            
            # Find any v not in S compatible with S_without_u
            candidates_swap = []
            for v in range(1, N + 1):
                v_bit = 1 << (v - 1)
                if S_mask & v_bit:
                    continue
                if (conflict_mask[v] & S_without_u) == 0:
                    candidates_swap.append(v)
            
            if candidates_swap:
                v = random.choice(candidates_swap)
                # Perform swap
                S_mask ^= u_bit
                S_mask |= (1 << (v - 1))
                # Do NOT set improved = True because size didn't change, 
                # but we want to continue searching from this new state.
                # We break the inner loop to restart the outer loop with new S_list
                break

    return mask_to_list(S_mask)

if __name__ == "__main__":
    print(construct())
