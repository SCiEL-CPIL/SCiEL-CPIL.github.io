"""Search-based snake construction for Q_9."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# Precompute neighbor masks for O(1) validity checks
# neighbors_mask[v] is an integer bitmask where bit u is set if dist(v, u) == 1
NEIGHBOR_MASKS = np.zeros(MAX_VERTS, dtype=np.uint64)
for v in range(MAX_VERTS):
    mask = 0
    for b in range(N):
        u = v ^ (1 << b)
        mask |= (1 << u)
    NEIGHBOR_MASKS[v] = mask

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Strategy:
      1. Use O(1) validity checks via bitmasking (used_mask, blocked_mask).
      2. Run many fast greedy restarts.
      3. Use limited backtracking on the best candidates.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    best_path = None
    best_len = 0
    
    def is_valid(v, used_mask, blocked_mask):
        """Check if v can be added to the end of path."""
        # v must not be used
        if (used_mask >> v) & 1:
            return False
        # v must not be adjacent to any non-consecutive vertex (handled by blocked_mask)
        if (blocked_mask >> v) & 1:
            return False
        return True

    def greedy_extend(path, used_mask, blocked_mask, rng, bit_order=None):
        """Extend path greedily using the given random number generator."""
        if bit_order is None:
            bit_order = list(range(N))
            rng.shuffle(bit_order)
        
        cur = path[-1]
        while True:
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid(v, used_mask, blocked_mask):
                    path.append(v)
                    used_mask |= (1 << v)
                    # Update blocked_mask: neighbors of the previous last vertex (cur) are now "non-consecutive" for future extensions
                    # Wait, the logic for blocked_mask update:
                    # Old blocked_mask contained neighbors of p_0...p_{L-2}.
                    # We are adding v to p_0...p_{L-1}.
                    # New blocked_mask should contain neighbors of p_0...p_{L-1}.
                    # So we add neighbors of p_{L-1} (which is cur).
                    blocked_mask |= NEIGHBOR_MASKS[cur]
                    cur = v
                    found = True
                    break
            if not found:
                break
        return path, used_mask, blocked_mask

    def try_backtrack_extend(path, used_mask, blocked_mask, rng, max_depth=5):
        """Try to extend path using limited backtracking."""
        if len(path) >= MAX_VERTS:
            return path, used_mask, blocked_mask
        
        cur = path[-1]
        # Find all valid extensions
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if is_valid(v, used_mask, blocked_mask):
                candidates.append((b, v))
        
        if not candidates:
            return path, used_mask, blocked_mask
        
        # Try each candidate with some backtracking
        rng.shuffle(candidates)
        
        def backtrack(current_path, current_used, current_blocked, depth):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return current_path, current_used, current_blocked
            
            if len(current_path) > best_len:
                best_path = current_path[:]
                best_len = len(current_path)
            
            if depth >= max_depth or len(current_path) >= MAX_VERTS:
                return current_path, current_used, current_blocked
            
            cur = current_path[-1]
            candidates_here = []
            for b in range(N):
                v = cur ^ (1 << b)
                if is_valid(v, current_used, current_blocked):
                    candidates_here.append((b, v))
            
            if not candidates_here:
                return current_path, current_used, current_blocked
            
            rng.shuffle(candidates_here)
            
            for b, v in candidates_here:
                if time.time() > deadline:
                    break
                
                new_used = current_used | (1 << v)
                new_blocked = current_blocked | NEIGHBOR_MASKS[cur]
                new_path = current_path + [v]
                
                result_path, result_used, result_blocked = backtrack(new_path, new_used, new_blocked, depth + 1)
                
                if len(result_path) > len(current_path):
                    return result_path, result_used, result_blocked
            
            return current_path, current_used, current_blocked
        
        return backtrack(path, used_mask, blocked_mask, 0)

    # Try multiple strategies
    strategies = [
        ("greedy_low", lambda rng: list(range(N))),
        ("greedy_high", lambda rng: list(range(N-1, -1, -1))),
        ("greedy_random", lambda rng: None),
    ]
    
    # First pass: try basic greedy strategies
    for strategy_name, strategy_func in strategies:
        if time.time() > deadline:
            break
        
        for attempt in range(100): # Increased attempts due to speedup
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 1000 + hash(strategy_name) % 10000)
            bit_order = strategy_func(rng)
            if bit_order is not None and strategy_name == "greedy_random":
                bit_order = None # Let greedy_extend handle shuffling if None passed? 
                # Actually greedy_extend shuffles if None.
                # But for greedy_low/high we pass fixed orders.
            
            # Start from 0
            path = [0]
            used_mask = 1 << 0
            blocked_mask = 0 # No non-consecutive vertices yet
            
            path, used_mask, blocked_mask = greedy_extend(path, used_mask, blocked_mask, rng, bit_order)
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
    
    # Second pass: try to extend the best path with backtracking
    if best_path and time.time() < deadline:
        used_mask = 0
        for v in best_path:
            used_mask |= (1 << v)
        
        # Recalculate blocked_mask for the best path
        blocked_mask = 0
        # blocked_mask contains neighbors of p_0 ... p_{L-2}
        for i in range(len(best_path) - 1):
            blocked_mask |= NEIGHBOR_MASKS[best_path[i]]
            
        extended_path, _, _ = try_backtrack_extend(best_path, used_mask, blocked_mask, random.Random(42), max_depth=4)
        if len(extended_path) > best_len:
            best_path = extended_path
            best_len = len(best_path)
    
    # Third pass: more randomized greedy with different seeds and starts
    if time.time() < deadline:
        for seed in range(200): # Increased attempts
            if time.time() > deadline:
                break
            
            rng = random.Random(seed * 7919 + 1)
            start_v = rng.randint(0, MAX_VERTS - 1)
            
            path = [start_v]
            used_mask = 1 << start_v
            blocked_mask = 0
            
            path, used_mask, blocked_mask = greedy_extend(path, used_mask, blocked_mask, rng, None)
            
            # Try to extend with backtracking if it's promising
            if len(path) > best_len - 5: # Only backtrack if close to best
                extended_path, _, _ = try_backtrack_extend(path, used_mask, blocked_mask, rng, max_depth=3)
                if len(extended_path) > len(path):
                    path = extended_path
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
    
    if best_path is None:
        # Fallback: simple greedy
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
