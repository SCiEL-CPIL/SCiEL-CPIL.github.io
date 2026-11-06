"""Snake-in-the-box Q_9: beam search with bitmask safety and time-budgeted restarts."""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
DEADLINE = 150.0  # evaluator kills at 150s; keep margin
SAFETY_MARGIN = 5.0  # return 5s before kill

# Precompute bit masks for each dimension
BIT_MASKS = [1 << b for b in range(N)]

def _is_safe(v, path, used_set, path_set):
    """Check if vertex v can be appended to path while keeping induced path property."""
    if v in used_set:
        return False
    # v must differ from all non-consecutive vertices by >= 2 bits
    # Consecutive is path[-1]; non-consecutive are path[0:-1]
    # For efficiency, check against all path vertices except the last
    for i in range(len(path) - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _beam_search(start, beam_width, max_time, rng):
    """Beam search to extend a snake from start. Returns list of vertices."""
    path = [start]
    used = {start}
    # beam: list of (path_tuple, score)
    # We'll use a simple approach: maintain top-k partial paths
    # For speed, we track the path as a tuple and check safety incrementally
    
    # Actually, for Q_9 with up to ~200 vertices, checking safety against all
    # previous vertices is O(n) per candidate, and we have 9 candidates per step.
    # With beam width B and length L, that's B * 9 * L checks per step.
    # For B=20, L=200, that's 36000 checks per step, ~200 steps = 7.2M checks total.
    # Each check is a bit_count which is fast in Python 3.10+.
    
    # Let's use a different approach: maintain for each path in the beam,
    # the set of "forbidden" vertices (those at Hamming distance 1 from any path vertex
    # except the current end). Actually, the constraint is:
    # - v not in used
    # - v not at Hamming distance 1 from any path[i] for i < len(path)-1
    
    # So the forbidden set is: used ∪ {u : dist(u, path[i]) == 1 for some i < len-1}
    # We can maintain this forbidden set incrementally.
    
    beam = [([start], {start}, set())]  # (path, used_set, forbidden_set)
    # forbidden_set: vertices that would violate induced property if added next
    # Initially, forbidden = {} (no non-consecutive vertices yet)
    # When we add vertex v to path, the new forbidden set adds all neighbors of path[-2]
    # (the vertex before the new last), because those neighbors are now at distance 1
    # from a non-consecutive vertex.
    
    # Wait, let me think more carefully.
    # Path: v0, v1, ..., vk. New vertex v_{k+1}.
    # Constraint: dist(v_{k+1}, vi) >= 2 for all i < k (i.e., not adjacent in path).
    # dist(v_{k+1}, v_{k-1}) >= 2, dist(v_{k+1}, v_{k-2}) >= 2, ..., dist(v_{k+1}, v0) >= 2.
    # 
    # When we extend from path ending at vk to v_{k+1}:
    # - v_{k+1} must not be in {v0, ..., vk}
    # - v_{k+1} must not be adjacent to v0, v1, ..., v_{k-1}
    #
    # The set of vertices adjacent to {v0, ..., v_{k-1}} is the "forbidden" set.
    # When we add vk to the path (making it the new last), the vertices adjacent to v_{k-1}
    # become part of the forbidden set (since v_{k-1} is now non-consecutive to future extensions).
    
    # Actually, let me reconsider. If path is [v0, v1, ..., vk], and we want to add v_{k+1}:
    # v_{k+1} must not be adjacent to v0, v1, ..., v_{k-1}.
    # So forbidden = N(v0) ∪ N(v1) ∪ ... ∪ N(v_{k-1})
    #
    # When we extend to [v0, ..., vk, v_{k+1}], the new forbidden set for next extension is:
    # N(v0) ∪ ... ∪ N(v_{k-1}) ∪ N(vk)
    # i.e., we add N(vk) to the forbidden set.
    
    # Let's maintain forbidden as a set of vertices.
    
    best_path = [start]
    best_len = 1
    
    while True:
        if time.time() > max_time:
            break
        new_beam = []
        for path, used_set, forbidden in beam:
            cur = path[-1]
            for b in range(N):
                v = cur ^ BIT_MASKS[b]
                if v in used_set or v in forbidden:
                    continue
                # Valid extension
                new_path = path + [v]
                new_used = used_set | {v}
                # New forbidden: add neighbors of cur (which becomes non-consecutive)
                new_forbidden = forbidden | set(cur ^ BIT_MASKS[i] for i in range(N))
                new_beam.append((new_path, new_used, new_forbidden))
        
        if not new_beam:
            break
        
        # Score: prefer longer paths, then some heuristic
        # For beam search, we want to keep diverse and promising paths
        # Score by length primarily
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Track best
        for p, _, _ in beam:
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
        
        # If all beams are stuck (no extensions), break
        # Actually, if new_beam is empty we already broke.
        # But if all paths in beam can't extend, next iteration will have empty new_beam.
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    # Start with the greedy seed path as initial best
    # Greedy snake
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if v in used:
                continue
            # Check induced property
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    
    best_path = path
    best_len = len(path)
    
    # Now do beam search with various parameters
    beam_widths = [10, 20, 50, 100]
    max_time_per_search = 20.0
    
    for bw in beam_widths:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        t0 = time.time()
        result = _beam_search(0, bw, t0 + max_time_per_search, rng)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    # Try multiple random starting points
    if time.time() < deadline:
        starts = list(range(2**N))
        rng.shuffle(starts)
        for start in starts[:20]:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 10:
                break
            t0 = time.time()
            result = _beam_search(start, 20, t0 + min(10.0, remaining - 2), rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    return best_path
