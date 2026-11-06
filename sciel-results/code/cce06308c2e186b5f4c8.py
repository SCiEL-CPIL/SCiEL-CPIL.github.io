"""Snake-in-the-box Q_9: bitmask beam search with time-budgeted restarts and heuristic pruning."""
import time
import random
import heapq

N = 9
SIZE = 1 << N
FULL_MASK = SIZE - 1
DEADLINE = 145.0  # evaluator kills at 150s; keep margin
SAFETY_MARGIN = 5.0  # return 5s before kill

# Precompute bit masks for each dimension
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks for all vertices
# NEIGHBOR_MASKS[v] is a bitmask where bit i is set if vertex i is a neighbor of v
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m


def _greedy_extend(start, rng):
    """Greedy extension from start vertex. Returns path list."""
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0  # vertices adjacent to non-last path vertices
    
    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        
        if candidates == 0:
            break
        
        # Greedy: pick the candidate that minimizes future branching
        # (i.e., the one with the fewest available neighbors for the next step)
        best_v = None
        best_score = SIZE + 1
        
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            
            # Score: number of available neighbors of v (excluding used and forbidden + neighbors of cur)
            new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
            new_used = used_mask | (1 << v)
            next_candidates = NEIGHBOR_MASKS[v] & ~(new_used | new_forbidden)
            score = next_candidates.bit_count()
            
            if score < best_score:
                best_score = score
                best_v = v
        
        if best_v is None:
            break
        
        path.append(best_v)
        used_mask |= (1 << best_v)
        forbidden_mask |= NEIGHBOR_MASKS[cur]
    
    return path


def _beam_search(start, beam_width, max_time, rng):
    """
    Bitmask-based beam search to extend a snake from start.
    Returns list of vertices.
    
    State: (key, last, used_mask, forbidden_mask, path_list)
    """
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0
    
    # Beam: list of (key, last, used_mask, forbidden_mask, path_list)
    # key is a tuple for sorting: (-length, next_options)
    beam = [((-1, 0), start, used_mask, forbidden_mask, path)]
    best_path = list(path)
    best_len = 1
    
    t0 = time.time()
    
    while True:
        if time.time() > max_time:
            break
        
        new_beam = []
        
        for key, cur, cur_used, cur_forbidden, cur_path in beam:
            # Candidates: neighbors of cur not used and not forbidden
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forbidden)
            
            if candidates == 0:
                # Dead end - record if better
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue
            
            # Generate extensions
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                
                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forbidden = cur_forbidden | NEIGHBOR_MASKS[cur]
                
                # Heuristic score: 
                # Prefer longer paths (higher length -> more negative key)
                # Prefer tighter ends (fewer next options)
                next_opts = (NEIGHBOR_MASKS[v] & ~(new_used | new_forbidden)).bit_count()
                new_key = (-len(new_path), next_opts)
                
                new_beam.append((new_key, v, new_used, new_forbidden, new_path))
                
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)
        
        if not new_beam:
            break
        
        # Select top beam_width from new_beam
        # Use heapq.nsmallest for efficiency if new_beam is large
        if len(new_beam) > beam_width:
            beam = heapq.nsmallest(beam_width, new_beam)
        else:
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam
        
        # If beam is empty or all paths are stuck, break
        if not beam:
            break
    
    return best_path


def _backtrack_improve(path, time_budget, rng):
    """
    Try to improve a given path using local search / backtracking.
    This attempts to find a longer path by exploring alternatives at each step.
    """
    if not path:
        return path
    
    best_path = list(path)
    best_len = len(path)
    
    # For each position in the path, try alternative extensions
    # This is a simplified local search approach
    
    t0 = time.time()
    max_time = t0 + time_budget
    
    # Try to extend from the end of the current path using beam search with small width
    start = path[-1]
    # We need to preserve the prefix, so we can't just restart beam search
    # Instead, we do a limited DFS from the end
    
    # Actually, a simpler approach: try different starting points and beam widths
    # The main improvement comes from the beam search itself
    
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    # 1. Greedy seed path
    greedy_path = _greedy_extend(0, rng)
    best_path = greedy_path
    best_len = len(greedy_path)
    
    # 2. Beam search with various parameters
    beam_configs = [
        (50, 8.0),
        (100, 10.0),
        (200, 12.0),
        (500, 15.0),
    ]
    
    for bw, duration in beam_configs:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < duration + 2:
            duration = max(2.0, remaining - 2)
        
        t0 = time.time()
        result = _beam_search(0, bw, t0 + duration, rng)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    # 3. Try multiple random starting points with medium beam width
    if time.time() < deadline - 15:
        starts = list(range(SIZE))
        rng.shuffle(starts)
        
        # Try up to 50 random starts
        for start in starts[:50]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 5:
                break
            
            # Use a smaller beam for random starts to fit more attempts
            bw = 100
            duration = min(3.0, remaining - 2)
            
            t0 = time.time()
            result = _beam_search(start, bw, t0 + duration, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    # 4. Final attempt with large beam if time remains
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 10:
            bw = 1000
            duration = remaining - 5
            t0 = time.time()
            result = _beam_search(0, bw, t0 + duration, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    return best_path


# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Path length: {len(res)}")
    # Validate
    valid = True
    for i in range(len(res)):
        if i > 0:
            if (res[i] ^ res[i-1]).bit_count() != 1:
                valid = False
                print(f"Invalid consecutive pair at index {i}")
                break
        if i > 1:
            for j in range(i-1):
                if (res[i] ^ res[j]).bit_count() == 1:
                    valid = False
                    print(f"Invalid non-consecutive pair at indices {i}, {j}")
                    break
        if not valid:
            break
    print(f"Valid: {valid}")
