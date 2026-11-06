import time
import random

N = 9
MAX_TIME = 110.0 # Safety margin for 120s limit

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Precompute neighbors for quick access? 
    # v ^ (1 << b) is fast enough.
    
    # We will try multiple restarts with randomized tie-breaking
    # and a backtracking search with MRV heuristic.
    
    # To avoid infinite loops or excessive backtracking, we limit the depth of backtracking
    # or use a beam search. Given Q_9 size (512 vertices), a full search is impossible.
    # A greedy approach with "look-ahead" and backtracking on dead ends is effective.
    
    def get_valid_neighbors(v, path_set, path):
        """
        Returns list of (neighbor, constraint_score) for valid moves from v.
        constraint_score is the number of valid neighbors the neighbor has.
        Lower score = more constrained.
        """
        candidates = []
        for b in range(N):
            u = v ^ (1 << b)
            if u in path_set:
                continue
            
            # Check if u is adjacent to any non-consecutive vertex in path
            # Non-consecutive means all vertices in path except the last one (v)
            # Actually, the condition is: u must not be adjacent to any vertex in path[:-1]
            # path[-1] is v, and u is adjacent to v by definition.
            # We need to check path[0] to path[-2]
            
            valid = True
            # Optimization: Only check recent vertices? No, induced path requires global check.
            # However, if the path is long, this is O(N * L).
            # 512 vertices max, so L <= 512. N=9. 9*512 = 4608 ops per neighbor check.
            # 9 neighbors per step. ~40k ops per step. 512 steps -> 20M ops. Python might be slow.
            
            # Let's optimize the check.
            # We can maintain a set of "blocked" vertices?
            # A vertex u is blocked if it is in path_set OR adjacent to any vertex in path_set \ {v}.
            
            # For now, simple check. If too slow, we will optimize.
            for p in path[:-1]:
                if (u ^ p).bit_count() == 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate constraint score for u
            # How many valid neighbors does u have?
            score = 0
            for b2 in range(N):
                w = u ^ (1 << b2)
                if w in path_set:
                    continue
                # Check if w is adjacent to path[:-1] (excluding u itself which is not in path yet)
                # Wait, if we move to u, the new path is path + [u].
                # The new vertex u is adjacent to v.
                # We need to check if w is adjacent to any vertex in path (including v? No, v is in path).
                # Actually, w must not be adjacent to any vertex in path.
                # path contains v. w is adjacent to v? No, w is adjacent to u.
                # If w == v, it's in path_set.
                # If w is adjacent to any p in path, it's invalid.
                
                # Optimization: 
                # w is valid for u if w not in path_set AND w not adjacent to any p in path.
                # Note: u is not in path yet.
                
                # Let's just check w against path.
                is_w_valid = True
                if w in path_set:
                    is_w_valid = False
                else:
                    for p in path:
                        if (w ^ p).bit_count() == 1:
                            is_w_valid = False
                            break
                if is_w_valid:
                    score += 1
            
            candidates.append((score, u))
        
        return candidates

    # To speed up, we can precompute bit counts? No, bit_count is C-level fast.
    # The bottleneck is the Python loop over path.
    # Let's try to optimize the validity check.
    
    # Alternative: Maintain a set of "forbidden" vertices.
    # Forbidden = path_set union {neighbors of path_set}
    # But we need to distinguish between "in path" and "adjacent to path".
    # A vertex u is a valid next step if:
    # 1. u not in path_set
    # 2. u is adjacent to v (current end)
    # 3. u is NOT adjacent to any p in path[:-1]
    
    # This is equivalent to: u is in the neighborhood of v, but u is not in the "exclusion zone" of path[:-1].
    
    # Let's stick to the explicit check but optimize the inner loop by breaking early.
    
    def search(path, path_set, current_depth):
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > MAX_TIME:
            return
        
        # Heuristic: Sort candidates by constraint score (ascending)
        # We want to try the most constrained moves first (fail-first)
        
        # To add randomness/restart diversity, we can shuffle ties or slightly perturb scores
        # But for deterministic best-first, sorting by score is key.
        
        cur = path[-1]
        candidates = get_valid_neighbors(cur, path_set, path)
        
        if not candidates:
            # Dead end
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            return
        
        # Sort by score (ascending), then by vertex value (for determinism in ties, or random?)
        # Using random tie-breaking helps escape local minima in different restarts
        candidates.sort(key=lambda x: (x[0], random.randint(0, 1000)))
        
        for score, nxt in candidates:
            new_path = path + [nxt]
            new_set = path_set | {nxt}
            
            search(new_path, new_set, current_depth + 1)
            
            # Pruning: If we've already found a path longer than possible? 
            # Max length is 512. 
            # Simple pruning: if best_len == 512, stop.
            if best_len == 2**N:
                return
            
            # Check time again inside loop
            if time.time() - start_time > MAX_TIME:
                return

    # Restart strategy
    # The greedy seed is 104 edges (105 vertices).
    # We will run several searches.
    # Since the problem is symmetric, we can fix the first few steps to break symmetry and reduce branching factor.
    # Symmetry breaking: Start at 0. First step must be to 1 (bit 0).
    # Second step: Can be 3 (bit 1) or 2 (bit 1 is same as 2? No. 0->1 (flip 0). 1->3 (flip 1) or 1->2 (flip 1? 1^2=3. 1^1=0. Wait. 1 is 001. Flip bit 1 (2's place) -> 011 (3). Flip bit 2 (4's place) -> 101 (5).
    # So from 1, neighbors are 0, 3, 5, 9, 17...
    # 0 is visited.
    # By symmetry, choosing bit 1, bit 2, etc. is equivalent.
    # So we can fix the first two steps: 0 -> 1 -> 3.
    # This reduces the initial branching.
    
    initial_path = [0, 1, 3]
    initial_set = {0, 1, 3}
    
    # Run multiple restarts to explore different branches
    # We will do a limited number of restarts to ensure we finish in time.
    # Each search is a DFS. If it's too deep, it might take long.
    # We can limit the depth of backtracking or use a beam search.
    # Given 120s, and Python speed, a deep DFS might be slow.
    # Let's try 10 restarts.
    
    for _ in range(20):
        if time.time() - start_time > MAX_TIME - 5:
            break
        
        # Randomize the initial path slightly? 
        # No, symmetry breaking is fixed. 
        # The randomness comes from the tie-breaking in the heuristic.
        
        # We need to clear best_path if we want to compare across restarts? 
        # No, we keep the global best.
        
        search(initial_path, initial_set, 2)
        
        # If we found a very long path, we might want to continue? 
        # The search function updates best_path.
        
    return best_path

# To ensure we don't return an empty list if something goes wrong, 
# and to handle the case where the search is too slow, 
# we can fall back to the greedy seed if best_path is empty or short.

def _greedy_seed():
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

# Wrap construct to ensure we always return at least the seed
def construct():
    try:
        result = _construct_impl()
        if not result or len(result) < 2:
            return _greedy_seed()
        return result
    except Exception:
        return _greedy_seed()

def _construct_impl():
    return construct_impl_inner()

# Refactor to avoid recursion limit issues and allow better time management
# Python recursion limit is usually 1000. Path length can be up to 512.
# 512 depth is fine.

def construct_impl_inner():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    def get_valid_neighbors(v, path_set, path):
        candidates = []
        for b in range(N):
            u = v ^ (1 << b)
            if u in path_set:
                continue
            
            valid = True
            for p in path[:-1]:
                if (u ^ p).bit_count() == 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            score = 0
            for b2 in range(N):
                w = u ^ (1 << b2)
                if w in path_set:
                    continue
                is_w_valid = True
                for p in path:
                    if (w ^ p).bit_count() == 1:
                        is_w_valid = False
                        break
                if is_w_valid:
                    score += 1
            
            candidates.append((score, u))
        
        return candidates

    def search(path, path_set):
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > MAX_TIME:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        candidates = get_valid_neighbors(cur, path_set, path)
        
        if not candidates:
            return
        
        candidates.sort(key=lambda x: (x[0], random.randint(0, 1000)))
        
        for score, nxt in candidates:
            new_path = path + [nxt]
            new_set = path_set | {nxt}
            
            search(new_path, new_set)
            
            if best_len == 2**N:
                return
            if time.time() - start_time > MAX_TIME:
                return

    initial_path = [0, 1, 3]
    initial_set = {0, 1, 3}
    
    # Run restarts
    for _ in range(50):
        if time.time() - start_time > MAX_TIME - 2:
            break
        search(initial_path, initial_set)
        
    if not best_path:
        return _greedy_seed()
    return best_path

# Expose the required function
def construct():
    try:
        return construct_impl_inner()
    except:
        return _greedy_seed()

# To avoid name collision with the inner construct, rename the main one
# The task requires `construct()` to be the entry point.
# I have defined `construct()` at the bottom.
