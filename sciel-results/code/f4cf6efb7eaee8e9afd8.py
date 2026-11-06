"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N
DEADLINE = 145.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute neighbors for fast lookup
    neighbors = [[] for _ in range(NUM_VERTS)]
    for v in range(NUM_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = [0]
    
    # Helper to check if a path is valid (mostly for initial setup or if we change strategy)
    def is_valid_path(path):
        if not path:
            return True
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        # Check induced condition: non-consecutive differ by >= 2
        # This is O(L^2), only used for final verification or small checks
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    def backtrack(path, forbidden_mask, time_limit):
        # path: list of vertices
        # forbidden_mask: numpy array of bools, True if vertex cannot be added
        # Returns extended path or None if stuck
        
        if time.time() > time_limit:
            return path
        
        cur = path[-1]
        
        # Get potential next vertices
        # A vertex v is valid if:
        # 1. v is a neighbor of cur
        # 2. v is not in forbidden_mask
        # 3. v is not distance 1 from any vertex in path[:-1] (which is already in forbidden_mask if we maintain it correctly)
        
        # We maintain forbidden_mask such that:
        # forbidden_mask[v] is True if v is in path OR v is dist 1 from any v in path[:-1]
        # Note: dist 1 from path[-1] is allowed (that's the extension), but dist 1 from path[-2] is forbidden.
        
        candidates = []
        for nb in neighbors[cur]:
            if not forbidden_mask[nb]:
                candidates.append(nb)
                
        if not candidates:
            return path
        
        # Randomize order to explore different branches
        random.shuffle(candidates)
        
        for nb in candidates:
            # Check if we can extend
            # The new forbidden set will be:
            # old_forbidden | {nb} | {neighbors of path[-2]}
            # Wait, the rule is: non-consecutive vertices must differ by >= 2 bits.
            # So if we add nb to path [..., a, b, nb], then nb must not be dist 1 from a.
            # 'a' is path[-2].
            # So we need to add neighbors of path[-2] to the forbidden set.
            
            # Optimization: 
            # Current forbidden_mask contains:
            # - All vertices in path
            # - All vertices at dist 1 from path[0] to path[-2]
            # Note: path[-1] is in path, so it's in forbidden. 
            # Neighbors of path[-1] are NOT necessarily in forbidden, unless they are dist 1 from earlier nodes.
            
            # When we add nb:
            # 1. nb becomes part of path -> forbidden
            # 2. nb must not be dist 1 from path[-2]. 
            #    Actually, if nb is dist 1 from path[-2], it would have been in forbidden_mask already?
            #    Let's trace:
            #    Step k: path ends in ... u, v.
            #    forbidden contains: path vertices + N(u) + N(prev of u) ...
            #    Specifically, forbidden contains N(path[-2]).
            #    So if nb is in N(path[-2]), forbidden_mask[nb] is True.
            #    Therefore, the check `if not forbidden_mask[nb]` already ensures nb is not dist 1 from path[-2].
            
            # So we just need to update forbidden_mask for the new step.
            # New path: ... u, v, nb
            # New forbidden should include:
            # - Old forbidden
            # - nb
            # - N(v)  (because v becomes path[-2] in the next step, so its neighbors become forbidden for the next-next vertex)
            
            # Wait, let's re-verify the invariant.
            # Invariant: forbidden_mask[x] is True if x cannot be the next vertex.
            # x can be next if:
            # 1. x is not in path
            # 2. dist(x, y) >= 2 for all y in path[:-1]
            # 3. dist(x, path[-1]) == 1 (implied by being a neighbor)
            
            # So forbidden_mask should mark:
            # - All x in path
            # - All x with dist(x, y) == 1 for y in path[:-1]
            
            # Let's check if my update rule holds.
            # Before adding nb (path ends in v, prev u):
            # forbidden has: path + N(path[:-1])
            # Note: v is in path, so v is forbidden.
            # N(u) is in forbidden.
            # N(v) is NOT necessarily in forbidden (unless v's neighbors are in path or N(path[:-2])).
            
            # After adding nb (path ends in nb, prev v):
            # New forbidden should have: path+nb + N(path[:-1]+nb)
            # = (path + nb) + N(path[:-1]) + N(nb)
            # Old forbidden was: path + N(path[:-1])
            # So we need to add: nb + N(v) ??
            
            # Let's look at N(path[:-1] + nb) vs N(path[:-1]).
            # path[:-1] + nb is the new "non-last" part? No.
            # New path: P' = P + [nb]
            # New "non-last" part is P.
            # So new forbidden = P' + N(P)
            # Old forbidden = P + N(P[:-1])
            
            # Difference:
            # New: (P + [nb]) + N(P)
            # Old: P + N(P[:-1])
            
            # N(P) = N(P[:-1]) + N(v) (where v is last of P)
            # So New = P + [nb] + N(P[:-1]) + N(v)
            # Old = P + N(P[:-1])
            
            # So we need to add: [nb] + N(v)
            
            # Is this correct?
            # If we add nb, then v becomes the second-to-last.
            # So the next vertex must not be dist 1 from v.
            # Thus N(v) must be forbidden.
            # And nb itself is now in path, so it is forbidden.
            
            # So the update is:
            # new_forbidden = old_forbidden | {nb} | N(v)
            
            # Let's implement this efficiently.
            
            # We can use a stack of changes to backtrack efficiently, or just copy the array.
            # Given N=9, 512 vertices, a numpy array of bools is small. Copying is fast.
            
            new_forbidden = forbidden_mask.copy()
            new_forbidden[nb] = True
            for n_nb in neighbors[v]:
                new_forbidden[n_nb] = True
                
            # Recurse
            result = backtrack(path + [nb], new_forbidden, time_limit)
            
            if result is not None and len(result) > len(path):
                return result
                
        return path

    # Multi-start search
    # We can start from 0, or from random vertices, or with random bit orders.
    # Since the hypercube is symmetric, starting from 0 is fine, but we can vary the "greediness" or randomness.
    # Actually, backtracking with random shuffling of candidates is sufficient to explore different paths.
    # We will run multiple independent backtracking searches.
    
    num_restarts = 100
    restarts_done = 0
    
    while restarts_done < num_restarts and time.time() < DEADLINE:
        # Reset state
        initial_forbidden = np.zeros(NUM_VERTS, dtype=bool)
        initial_forbidden[0] = True # 0 is in path
        
        # For the first step, path is [0].
        # Forbidden for next step:
        # Path: [0]
        # Non-last: []
        # So forbidden should be: {0} + N([]) = {0}
        # Wait, the invariant says: forbidden = path + N(path[:-1])
        # If path=[0], path[:-1]=[]
        # So forbidden = {0}
        
        # Let's run backtrack
        # We need to be careful with the base case.
        # If path=[0], we look for neighbors of 0 not in forbidden.
        # Forbidden is {0}. Neighbors of 0 are not 0.
        
        # The backtrack function above assumes path is non-empty and handles the extension.
        # It checks `if not forbidden_mask[nb]`.
        
        # However, the update rule in backtrack adds N(v) where v is path[-2].
        # If len(path) == 1, path[-2] is index -1? No, path[:-1] is empty.
        # My update logic: "New forbidden = Old + [nb] + N(v) where v is last of old path".
        # If old path is [0], v=0.
        # New path [0, nb].
        # New forbidden should be {0, nb} + N([0]) = {0, nb} + N(0).
        # Old forbidden was {0}.
        # Update: add nb, add N(0).
        # This matches.
        
        # So the logic holds for len(path) >= 1.
        
        current_path = [0]
        # Initial forbidden mask for the state where path=[0]
        # According to invariant: forbidden = path + N(path[:-1])
        # path=[0], path[:-1]=[]
        # forbidden = {0}
        
        # Wait, the backtrack function I wrote takes `path` and `forbidden_mask`.
        # It assumes `forbidden_mask` is correct for the current `path`.
        
        # Let's refine the backtrack function to be cleaner.
        
        def search(path, forbidden, limit):
            if time.time() > limit:
                return path
            
            cur = path[-1]
            candidates = [nb for nb in neighbors[cur] if not forbidden[nb]]
            
            if not candidates:
                return path
            
            random.shuffle(candidates)
            
            for nb in candidates:
                # Update forbidden
                new_forbidden = forbidden.copy()
                new_forbidden[nb] = True
                if len(path) >= 2:
                    prev_prev = path[-2]
                    for n_pp in neighbors[prev_prev]:
                        new_forbidden[n_pp] = True
                else:
                    # If len(path) == 1, path[:-1] is empty.
                    # But wait, if path=[0], and we add nb, path becomes [0, nb].
                    # The "prev_prev" for the NEXT step will be 0.
                    # So we need to forbid N(0) for the step after nb.
                    # My logic in the previous block:
                    # "New forbidden = Old + [nb] + N(v) where v is last of old path"
                    # If old path is [0], v=0. So we add N(0).
                    # This is correct.
                    
                    # In the code above:
                    # if len(path) >= 2: prev_prev = path[-2] ...
                    # else: prev_prev = path[-1] ?
                    
                    # Let's re-evaluate.
                    # Old path P. Last vertex v = P[-1].
                    # New path P + [nb].
                    # New "second to last" is v.
                    # So we must forbid N(v).
                    # This applies regardless of len(P).
                    
                    # So we should always forbid N(path[-1]).
                    
                    pass # Logic correction needed in code below
            
            return path
        
        # Let's rewrite the search function correctly inside the loop or outside.
        
        def do_search(initial_path, initial_forbidden, limit):
            # Stack-based or recursive
            # Recursive is easier to write, but Python recursion limit might be an issue if path is long?
            # Max path length < 512. Recursion depth ~500. Default limit is 1000. Should be fine.
            
            def rec(path, forbidden):
                if time.time() > limit:
                    return path
                
                cur = path[-1]
                candidates = [nb for nb in neighbors[cur] if not forbidden[nb]]
                
                if not candidates:
                    return path
                
                random.shuffle(candidates)
                
                for nb in candidates:
                    new_forbidden = forbidden.copy()
                    new_forbidden[nb] = True
                    # Forbid neighbors of the vertex that becomes "second to last"
                    # That is path[-1] (the current last vertex before adding nb)
                    v = path[-1]
                    for n_v in neighbors[v]:
                        new_forbidden[n_v] = True
                        
                    result = rec(path + [nb], new_forbidden)
                    if len(result) > len(path):
                        return result
                        
                return path
            
            return rec(initial_path, initial_forbidden)
        
        # Run search
        init_forb = np.zeros(NUM_VERTS, dtype=bool)
        init_forb[0] = True
        
        res = do_search([0], init_forb, DEADLINE)
        
        if len(res) > len(best_path):
            best_path = res
            
        restarts_done += 1
        # Small sleep to avoid tight loop if search is very fast?
        # The search itself takes time.
        
    # Verify best path
    if not is_valid_path(best_path):
        # Fallback to simple greedy if something went wrong
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
        return path
        
    return best_path
# EVOLVE-BLOCK-END
