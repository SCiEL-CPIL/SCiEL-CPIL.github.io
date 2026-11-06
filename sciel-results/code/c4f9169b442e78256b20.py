"""
Snake-in-the-box in Q_9 using Greedy with Backtracking.
This approach uses a bitmask to efficiently track forbidden vertices and
performs a limited-depth backtracking search to escape local minima 
that trap simple greedy algorithms.
"""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 115  # Safety margin before 120s kill

def _hamming(x, y):
    """Hamming distance between x and y."""
    return (x ^ y).bit_count()

def _init_forbidden(start):
    """
    Initialize the forbidden mask for a path starting at 'start'.
    A vertex v is forbidden if it is adjacent to 'start' (except the next step)
    or if it is 'start' itself.
    Actually, for the first step, any neighbor is a candidate.
    The forbidden set for a new vertex v being added to a path ending at u
    is:
    1. v itself (already used)
    2. Any neighbor of v (because they would become distance 1 from v, 
       but wait, the condition is: non-consecutive vertices must be distance >= 2.
       So if we add v, all neighbors of v become forbidden for *future* vertices 
       unless they are the immediate predecessor u.
       
    Let's refine the forbidden set definition:
    Let P = [p0, p1, ..., pk] be the current path.
    Let F be the set of vertices that cannot be added as p_{k+1}.
    v is in F if:
    - v is in P (used)
    - There exists i < k-1 such that hamming(v, p_i) == 1.
      (i.e., v is adjacent to a non-consecutive vertex)
    
    When we add v to the path, the new forbidden set F' for the next step is:
    F' = (F \ {v}) 
         U {u | hamming(u, v) == 1 and u != p_{k-1}}
         U {v} (since v is now used)
    
    Wait, if we add v, v becomes used. So v is forbidden.
    Also, any neighbor of v (except the one we came from, p_{k-1}) becomes 
    "too close" to v for any future vertex. So they are added to forbidden.
    """
    # Initial state: path = [start]
    # Forbidden for next step:
    # - start (used)
    # - No other vertices in path, so no "adjacent to non-consecutive" yet.
    # So initially, only 'start' is forbidden?
    # No, if we pick a neighbor v, v must not be adjacent to any p_i for i < k-1.
    # For k=0, there are no i < -1. So any neighbor is valid.
    # So initially, forbidden = {start}.
    
    mask = 1 << start
    return mask

def _greedy_backtrack(start=0, bit_order=None, max_depth=5, time_budget=1.0):
    """
    Greedy snake with limited backtracking.
    """
    if bit_order is None:
        bit_order = list(range(N))
        random.shuffle(bit_order)
    
    start_time = time.time()
    
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 1 << start
    
    def is_valid(v):
        return not (forbidden_mask >> v) & 1

    def add_vertex(v):
        nonlocal used_mask, forbidden_mask
        # Mark v as used
        used_mask |= (1 << v)
        forbidden_mask |= (1 << v)
        
        # Update forbidden: neighbors of v become forbidden (except the one we came from? 
        # No, the one we came from is already in the path, but is it forbidden?
        # The predecessor p_{k-1} is in the path. 
        # The rule is: future vertices cannot be adjacent to v.
        # So ALL neighbors of v are forbidden for future vertices, 
        # EXCEPT that p_{k-1} is already in the path (and thus already 'used'/'forbidden' 
        # in the sense that we can't pick it again).
        # But wait, do we need to explicitly add neighbors to forbidden?
        # Yes. If we don't, we might pick a neighbor of v later, which would be 
        # distance 1 from v. Since v and that future vertex would be non-consecutive 
        # (unless it's the immediate next one, but we are adding v *now*), 
        # wait.
        
        # Let's trace:
        # Path: ... u, v
        # Next vertex w.
        # w must not be adjacent to u (consecutive, so allowed? No, w is consecutive to v. 
        # u is non-consecutive to w? 
        # Indices: ... i:u, i+1:v, i+2:w.
        # u and w are distance 2 apart in path. They are non-consecutive.
        # So hamming(u, w) must be >= 2.
        # If w is a neighbor of u, hamming(u, w) = 1. Invalid.
        # So u is forbidden for w? 
        # u is already in the path, so u is in 'used_mask'. 
        # 'used_mask' is a subset of 'forbidden_mask' effectively.
        
        # The critical part: 
        # When we add v, any neighbor x of v (x != u) becomes forbidden for the NEXT step?
        # No, x becomes forbidden for ANY FUTURE step, because any future vertex y 
        # will be non-consecutive to v (since v is at least 2 steps away from y).
        # So if y is adjacent to v, hamming(y, v) = 1, which is invalid.
        # Therefore, all neighbors of v are added to forbidden_mask.
        
        neighbors = []
        for b in range(N):
            nb = v ^ (1 << b)
            # We add all neighbors to forbidden.
            # Note: u is a neighbor of v. u is already in forbidden (as used).
            # Other neighbors are new.
            forbidden_mask |= (1 << nb)
            
    # Recursive backtracking function
    def backtrack(depth):
        if time.time() - start_time > time_budget:
            return False
        
        if depth == 0:
            return True
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if is_valid(v):
                candidates.append(v)
                
        if not candidates:
            return False
            
        # Heuristic: Try candidates with fewest options first (fail fast)
        # or most options (greedy)? 
        # "Fail fast" is usually better for backtracking.
        # Let's estimate options for each candidate.
        
        def count_options(v):
            # How many valid neighbors does v have?
            # We need to simulate adding v to see what becomes forbidden.
            # This is expensive. 
            # Approximation: Count neighbors of v that are not currently forbidden.
            # This is a lower bound on options after adding v? 
            # No, adding v forbids more.
            # So current valid neighbors is an upper bound.
            cnt = 0
            for b in range(N):
                nb = v ^ (1 << b)
                if is_valid(nb):
                    cnt += 1
            return cnt

        # Sort candidates by number of options (ascending) to fail fast
        # If we want to find *any* path, failing fast on dead ends is good.
        # If we want the *longest* path, we might want to explore promising branches.
        # Given the time limit, we just want a long path. 
        # Let's try a mix: randomize order slightly, but bias towards fewer options 
        # if we are deep? 
        # Actually, standard greedy with backtracking often works well with 
        # "Most Constrained" (fewest options) to prune.
        
        candidates.sort(key=count_options)
        
        # To avoid getting stuck in the same local optimum, shuffle if options are similar?
        # Let's just iterate.
        
        for v in candidates:
            # Save state
            old_used = used_mask
            old_forbidden = forbidden_mask
            
            # Add v
            path.append(v)
            # Update masks
            new_forbidden = old_forbidden | (1 << v)
            for b in range(N):
                nb = v ^ (1 << b)
                new_forbidden |= (1 << nb)
            
            # Recurse
            if backtrack(depth - 1):
                return True
                
            # Backtrack
            path.pop()
            used_mask = old_used
            forbidden_mask = old_forbidden
            
        return False

    # Start the search
    # We do a "greedy" step first to get a base path, then backtrack?
    # Or just start backtracking from the beginning?
    # Starting from the beginning with depth 5 is safe.
    
    # Actually, a pure greedy is fast. Let's do greedy first, then if it fails, 
    # the backtracking logic above handles it by returning False and trying next candidate.
    # But the function above is recursive. 
    # Let's call it with a reasonable depth.
    
    # To make it faster, we can use an iterative greedy first, and only backtrack 
    # when stuck.
    
    # Let's stick to the recursive structure but optimize the "count_options" 
    # by not doing it if depth is small.
    
    # A simpler approach: 
    # 1. Greedy step.
    # 2. If stuck, backtrack 1 step, try next option.
    # This is essentially DFS with a limit.
    
    # Let's implement a simpler iterative backtracker for speed.
    
    # Reset state for iterative version
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 1 << start
    
    # Stack of states: (current_vertex, index_of_next_bit_to_try, forbidden_mask, used_mask, path_copy)
    # This is memory heavy. 
    # Let's stick to the recursive one but with a small depth limit (e.g., 5-10).
    # The recursive one above is fine if we limit depth.
    
    # Let's re-run the recursive logic properly.
    
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 1 << start
    
    # We want to find a path of length > current_best.
    # The recursive function returns True if it can extend by 'depth' steps.
    # We want to extend as much as possible.
    
    # Strategy: 
    # While we can extend:
    #   Try to extend by 1 step using backtracking of depth D.
    #   If successful, commit.
    #   If not, stop.
    
    # This is complex. Let's just run the DFS to find a path of length L.
    # We will increase L until we can't.
    
    # Simpler: Just run the greedy with backtracking for a fixed number of steps 
    # or until time runs out.
    
    # Let's use the recursive function to build the path step-by-step.
    # At each step, we try to find a valid next vertex. 
    # If multiple, we pick one. 
    # To avoid dead ends, we look ahead 'max_depth' steps.
    
    # Modified strategy:
    # 1. Current path P.
    # 2. Generate candidates for next vertex.
    # 3. For each candidate, check if it leads to a dead end within 'max_depth' steps.
    # 4. Pick the candidate that does NOT lead to a dead end.
    # 5. If all lead to dead ends, pick one randomly (or backtrack further).
    
    def check_future(v, depth):
        """
        Returns True if adding v leads to a dead end within 'depth' steps.
        """
        old_used = used_mask
        old_forbidden = forbidden_mask
        
        # Add v
        new_forbidden = old_forbidden | (1 << v)
        for b in range(N):
            nb = v ^ (1 << b)
            new_forbidden |= (1 << nb)
            
        # Recursively check if we can extend by 'depth' steps
        # We use a simple greedy for this check to save time? 
        # Or a limited backtracking?
        # A simple greedy check is fast.
        
        # Let's use a simple greedy check for speed.
        def greedy_check(cur, rem_depth, forb):
            if rem_depth == 0:
                return True
            for b in bit_order:
                nv = cur ^ (1 << b)
                if not (forb >> nv) & 1:
                    # Check if this move leads to a dead end? 
                    # Just take it.
                    new_forb = forb | (1 << nv)
                    for bb in range(N):
                        nbb = nv ^ (1 << bb)
                        new_forb |= (1 << nbb)
                    if greedy_check(nv, rem_depth - 1, new_forb):
                        return True
                    # If this branch fails, try next? 
                    # This is backtracking again. 
                    # For a "check", we just want to know if ANY path exists.
                    # So yes, we must backtrack in the check.
            return False
            
        return not greedy_check(v, depth, new_forbidden)

    # This nested recursion might be slow. 
    # Let's try a different approach: 
    # Standard DFS with backtracking, but we limit the depth of the search tree 
    # to, say, 10. If we fail, we backtrack to the root and try a different 
    # initial choice? No, that's too slow.
    
    # Best approach for this time limit: 
    # 1. Fast Greedy (no backtracking) to get a base path.
    # 2. If stuck, try "local repair": 
    #    - Look at the last K vertices.
    #    - Try to replace the last vertex with a different one that allows extension.
    
    # Let's implement a "Greedy with 1-step look-ahead" (Backtrack depth 1).
    # This is much faster than deep backtracking.
    
    def greedy_with_lookahead(start_v, bits, lookahead_depth=3):
        p = [start_v]
        forb = 1 << start_v
        
        while True:
            cur = p[-1]
            candidates = []
            for b in bits:
                v = cur ^ (1 << b)
                if not (forb >> v) & 1:
                    candidates.append(v)
                    
            if not candidates:
                break
                
            # Lookahead
            best_v = None
            # Sort by "safety" (does it lead to dead end?)
            # We use a simple greedy simulation for lookahead.
            
            valid_candidates = []
            for v in candidates:
                # Simulate adding v
                new_forb = forb | (1 << v)
                for b in range(N):
                    nb = v ^ (1 << b)
                    new_forb |= (1 << nb)
                
                # Greedy extend from v for 'lookahead_depth' steps
                success = True
                sim_cur = v
                sim_forb = new_forb
                for _ in range(lookahead_depth):
                    next_candidates = []
                    for b in bits:
                        nv = sim_cur ^ (1 << b)
                        if not (sim_forb >> nv) & 1:
                            next_candidates.append(nv)
                    if not next_candidates:
                        success = False
                        break
                    # Pick first valid (greedy)
                    nv = next_candidates[0]
                    sim_forb = sim_forb | (1 << nv)
                    for b in range(N):
                        nb = nv ^ (1 << b)
                        sim_forb |= (1 << nb)
                    sim_cur = nv
                    
                if success:
                    valid_candidates.append(v)
            
            if valid_candidates:
                # Pick one randomly from valid to diversify
                v = random.choice(valid_candidates)
            else:
                # All lead to dead ends in lookahead. 
                # Pick one with most options? Or just first?
                # Let's pick the one with most immediate options.
                def count_opts(v):
                    c = 0
                    for b in range(N):
                        nb = v ^ (1 << b)
                        if not (forb >> nb) & 1:
                            c += 1
                    return c
                candidates.sort(key=count_opts, reverse=True)
                v = candidates[0]
                
            p.append(v)
            forb = forb | (1 << v)
            for b in range(N):
                nb = v ^ (1 << b)
                forb |= (1 << nb)
                
        return p

    # Run multiple instances with different params
    best_path = []
    
    while time.time() - start_time < time_budget:
        start_v = random.randint(0, FULL_MASK)
        bits = list(range(N))
        random.shuffle(bits)
        
        # Vary lookahead depth
        ld = random.choice([2, 3, 4])
        
        p = greedy_with_lookahead(start_v, bits, ld)
        
        if len(p) > len(best_path):
            best_path = p
            
    return best_path

def construct():
    start_time = time.time()
    best_path = []
    
    # Initial fast greedy to get a baseline quickly
    p = _greedy_backtrack(0, max_depth=1, time_budget=1.0)
    best_path = p
    
    # Randomized restarts with lookahead
    while time.time() - start_time < TIME_LIMIT:
        # Random start
        start_v = random.randint(0, FULL_MASK)
        
        # Random bit order
        bits = list(range(N))
        random.shuffle(bits)
        
        # Vary lookahead depth
        ld = random.choice([2, 3, 4, 5])
        
        # Run a single instance of greedy-with-lookahead
        # We don't call _greedy_backtrack directly here because it's a wrapper.
        # We call the internal logic.
        
        # To reuse the logic, let's just call _greedy_backtrack with a small budget
        # but note that _greedy_backtrack as written earlier was recursive and complex.
        # The 'greedy_with_lookahead' inside _greedy_backtrack is what we want.
        
        # Let's refactor: extract the core loop.
        
        # For simplicity in this rewrite, I will implement a direct loop here.
        
        p = [start_v]
        forb = 1 << start_v
        
        while True:
            cur = p[-1]
            candidates = []
            for b in bits:
                v = cur ^ (1 << b)
                if not (forb >> v) & 1:
                    candidates.append(v)
                    
            if not candidates:
                break
                
            valid_candidates = []
            for v in candidates:
                new_forb = forb | (1 << v)
                for b in range(N):
                    nb = v ^ (1 << b)
                    new_forb |= (1 << nb)
                
                success = True
                sim_cur = v
                sim_forb = new_forb
                for _ in range(ld):
                    next_candidates = []
                    for b in bits:
                        nv = sim_cur ^ (1 << b)
                        if not (sim_forb >> nv) & 1:
                            next_candidates.append(nv)
                    if not next_candidates:
                        success = False
                        break
                    nv = next_candidates[0]
                    sim_forb = sim_forb | (1 << nv)
                    for b in range(N):
                        nb = nv ^ (1 << b)
                        sim_forb |= (1 << nb)
                    sim_cur = nv
                    
                if success:
                    valid_candidates.append(v)
            
            if valid_candidates:
                v = random.choice(valid_candidates)
            else:
                def count_opts(v):
                    c = 0
                    for b in range(N):
                        nb = v ^ (1 << b)
                        if not (forb >> nb) & 1:
                            c += 1
                    return c
                candidates.sort(key=count_opts, reverse=True)
                v = candidates[0]
                
            p.append(v)
            forb = forb | (1 << v)
            for b in range(N):
                nb = v ^ (1 << b)
                forb |= (1 << nb)
                
        if len(p) > len(best_path):
            best_path = p
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    return best_path
