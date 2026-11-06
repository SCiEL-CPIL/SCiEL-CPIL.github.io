"""Improved snake-in-the-box for Q_9 using backtracking with Warnsdorff's heuristic."""
import sys
import time

N = 9
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using depth-first search
    with backtracking and Warnsdorff's rule (choose the move with the fewest
    future options) to guide the search.
    """
    best_path = []
    best_len = 0
    
    # Time limit for the search
    start_time = time.time()
    time_limit = 115  # Leave buffer for evaluation overhead
    
    # State for backtracking
    # path: list of vertices
    # used: set of vertices in path
    
    def get_valid_next(v, path, used):
        """
        Check if v can be appended to path.
        1. v not in used.
        2. v is not adjacent to any vertex in path[:-1] (non-consecutive).
        """
        if v in used:
            return False
        # Check against all previous vertices except the last one (which is the current end)
        # The current end is path[-1]. v is adjacent to path[-1] by definition of generation.
        # We need to ensure v is NOT adjacent to path[0] ... path[-2].
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def count_future_moves(v, path, used):
        """
        Count how many valid moves are available from v, assuming v is added to path.
        This is the 'degree' in the remaining graph.
        """
        count = 0
        for b in range(N):
            w = v ^ (1 << b)
            if w in used:
                continue
            # Check if w is adjacent to any vertex in path[:-1]
            valid = True
            for i in range(len(path) - 1):
                if (w ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def backtrack(path, used):
        nonlocal best_path, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if get_valid_next(v, path, used):
                # Heuristic: Warnsdorff's rule -> prefer smallest degree
                deg = count_future_moves(v, path, used)
                candidates.append((deg, v))
        
        if not candidates:
            return
        
        # Sort by degree (Warnsdorff: min degree first)
        candidates.sort(key=lambda x: x[0])
        
        for deg, v in candidates:
            # Recurse
            path.append(v)
            used.add(v)
            backtrack(path, used)
            path.pop()
            used.remove(v)
            
            # Early exit if time is up
            if time.time() - start_time > time_limit:
                return

    # Initial setup
    # Try starting from 0. Symmetry suggests all starts are equivalent.
    start_vertex = 0
    path = [start_vertex]
    used = {start_vertex}
    
    backtrack(path, used)
    
    # If the first attempt was fast and didn't find a very long path, try another start
    # to break symmetry or find a different basin.
    if time.time() - start_time < time_limit - 5 and best_len < 200:
        start_vertex = FULL_MASK
        path = [start_vertex]
        used = {start_vertex}
        backtrack(path, used)

    return best_path
# EVOLVE-BLOCK-END
