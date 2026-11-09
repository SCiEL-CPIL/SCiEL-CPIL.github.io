"""Search-based snake construction for Q_9."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Strategy:
      1. Generate many diverse candidate snakes using randomized greedy heuristics.
      2. For each candidate, attempt to extend it further using randomized search.
      3. Keep the best result found before the deadline.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    best_path = None
    best_len = 0
    
    def is_valid_extension(v, path, used_set):
        """Check if v can be added to the end of path."""
        if v in used_set:
            return False
        # Check against all vertices except the last one (which is adjacent by construction)
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used_set, rng, bit_order=None):
        """Extend path greedily using the given random number generator."""
        if bit_order is None:
            bit_order = list(range(N))
            rng.shuffle(bit_order)
        
        cur = path[-1]
        while True:
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used_set):
                    path.append(v)
                    used_set.add(v)
                    cur = v
                    found = True
                    break
            if not found:
                break
        return path
    
    def try_backtrack_extend(path, used_set, rng, max_depth=5):
        """Try to extend path using limited backtracking."""
        if len(path) >= MAX_VERTS:
            return path
        
        cur = path[-1]
        # Find all valid extensions
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if is_valid_extension(v, path, used_set):
                candidates.append((b, v))
        
        if not candidates:
            return path
        
        # Try each candidate with some backtracking
        rng.shuffle(candidates)
        
        def backtrack(current_path, current_used, depth):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return current_path
            
            if len(current_path) > best_len:
                best_path = current_path[:]
                best_len = len(current_path)
            
            if depth >= max_depth or len(current_path) >= MAX_VERTS:
                return current_path
            
            cur = current_path[-1]
            candidates_here = []
            for b in range(N):
                v = cur ^ (1 << b)
                if is_valid_extension(v, current_path, current_used):
                    candidates_here.append((b, v))
            
            if not candidates_here:
                return current_path
            
            rng.shuffle(candidates_here)
            
            for b, v in candidates_here:
                if time.time() > deadline:
                    break
                new_path = current_path + [v]
                new_used = current_used | {v}
                result = backtrack(new_path, new_used, depth + 1)
                if len(result) > len(current_path):
                    return result
            
            return current_path
        
        return backtrack(path, used_set, 0)
    
    def local_search(path, used_set, rng, max_iterations=100):
        """Try to improve path by removing and re-adding vertices."""
        if len(path) >= MAX_VERTS:
            return path
        
        current_path = path[:]
        current_used = used_set.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to insert a vertex at a random position
            if len(current_path) < 2:
                break
            
            # Find a position where we might insert
            pos = rng.randint(1, len(current_path) - 1)
            a, c = current_path[pos - 1], current_path[pos]
            
            # Find vertices that differ from a by 1 bit and from c by 1 bit
            # but are not adjacent to other path vertices
            candidates = []
            for b in range(N):
                v = a ^ (1 << b)
                if v == c:
                    continue
                if v in current_used:
                    continue
                # Check if v is adjacent to c (differ by 1 bit)
                if (v ^ c).bit_count() != 1:
                    continue
                # Check if v is valid to insert
                valid = True
                for p in current_path:
                    if p == a or p == c:
                        continue
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                # Try inserting
                new_path = current_path[:pos] + [v] + current_path[pos:]
                # Verify it's still a valid snake
                is_valid = True
                for i in range(len(new_path) - 1):
                    if (new_path[i] ^ new_path[i+1]).bit_count() != 1:
                        is_valid = False
                        break
                if is_valid:
                    for i in range(len(new_path)):
                        for j in range(i+2, len(new_path)):
                            if (new_path[i] ^ new_path[j]).bit_count() == 1:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                if is_valid:
                    current_path = new_path
                    current_used = set(current_path)
        
        return current_path
    
    # Try multiple strategies
    strategies = [
        ("greedy_low", lambda rng: greedy_extend([0], {0}, rng, list(range(N)))),
        ("greedy_high", lambda rng: greedy_extend([0], {0}, rng, list(range(N-1, -1, -1)))),
        ("greedy_random", lambda rng: greedy_extend([0], {0}, rng, None)),
    ]
    
    # First pass: try basic greedy strategies
    for strategy_name, strategy_func in strategies:
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 1000 + hash(strategy_name) % 10000)
            path = strategy_func(rng)
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
    
    # Second pass: try to extend the best path with backtracking
    if best_path and time.time() < deadline:
        used_set = set(best_path)
        extended = try_backtrack_extend(best_path, used_set, random.Random(42), max_depth=3)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    # Third pass: local search on the best path
    if best_path and time.time() < deadline:
        used_set = set(best_path)
        improved = local_search(best_path, used_set, random.Random(123), max_iterations=50)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
    
    # Fourth pass: more randomized greedy with different seeds
    if time.time() < deadline:
        for seed in range(50):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed * 7919 + 1)
            path = greedy_extend([0], {0}, rng, None)
            
            # Try to extend with backtracking
            used_set = set(path)
            extended = try_backtrack_extend(path, used_set, rng, max_depth=2)
            
            if len(extended) > best_len:
                best_path = extended
                best_len = len(best_path)
    
    # Fifth pass: try starting from different vertices
    if time.time() < deadline:
        for start_v in range(0, MAX_VERTS, MAX_VERTS // 20):  # Sample ~20 starting vertices
            if time.time() > deadline:
                break
            
            rng = random.Random(start_v * 31 + 7)
            path = greedy_extend([start_v], {start_v}, rng, None)
            used_set = set(path)
            extended = try_backtrack_extend(path, used_set, rng, max_depth=2)
            
            if len(extended) > best_len:
                best_path = extended
                best_len = len(best_path)
    
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
