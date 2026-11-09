"""Snake-in-the-box Q9: multi-start randomized backtracking with bitmask validity."""
import time
import random

N = 9
MASK = (1 << N) - 1

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    best_path = None
    best_len = 0

    # Precompute: for each vertex, its N neighbors
    # neighbors[v] = list of (bit, neighbor)
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    # Fast validity: a candidate vertex v is valid to add after current path
    # if v is not in used AND no vertex in path[:-1] is at distance 1 from v.
    # Distance-1 check: for each bit b, (v ^ (1<<b)) in used_set would mean
    # a path vertex at distance 1. But we need path[:-1], not the last vertex.
    # Actually: non-consecutive means not adjacent in the sequence. So v must not
    # be at distance 1 from any path[i] for i < len(path)-1.
    # The last vertex path[-1] is the one we're extending from, so distance 1 to it is fine.
    # So: v is valid iff v not in used AND for all b in range(N), (v^(1<<b)) not in used_set_excluding_last.
    # Simpler: v not in used AND no neighbor of v is in path[:-1].
    # We can maintain a set of all used vertices. For the distance-1 check,
    # we need to ensure no neighbor of v is in path[:-1]. Since path[-1] is the
    # current vertex (which IS a neighbor of v by construction), we just need
    # to check that v's other neighbors aren't in used. Actually:
    # v is adjacent to path[-1] (that's how we get there). We need v NOT adjacent
    # to any path[i] for i < len(path)-1. So check: for each neighbor u of v,
    # u != path[-1] and u not in used. But u IS in used if u is in path[:-1].
    # So: v is valid iff v not in used AND all neighbors of v except path[-1] are not in used.
    # Equivalently: v not in used AND (count of neighbors of v that are in used) == 1 (only path[-1]).

    def is_valid(v, used_set, last_vertex):
        if v in used_set:
            return False
        # Check that exactly one neighbor of v is in used_set (namely last_vertex)
        # Actually we need: no neighbor of v in used_set other than last_vertex
        # Since last_vertex is a neighbor of v (we're extending from it),
        # we need: for all b, if (v^(1<<b)) != last_vertex, then (v^(1<<b)) not in used_set
        for b in range(N):
            nb = v ^ (1 << b)
            if nb != last_vertex and nb in used_set:
                return False
        return True

    def backtrack(path, used_set, depth, deadline):
        """Try to extend path by one step. Returns (new_path, new_used) or (None, None)."""
        if time.time() > deadline:
            return None, None
        cur = path[-1]
        # Shuffle neighbors for randomization
        cands = list(range(N))
        random.shuffle(cands)
        for b in cands:
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # Check distance-1 constraint with path[:-1]
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            new_path = path + [v]
            new_used = used_set | {v}
            return new_path, new_used
        return None, None

    def extend_greedy_random(path, used_set, max_steps, deadline):
        """Extend path greedily with randomization until stuck or deadline."""
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            result = backtrack(path, used_set, len(path), deadline)
            if result[0] is None:
                break
            path, used_set = result
        return path, used_set

    # Multi-start: try many random starting points and random orderings
    all_vertices = list(range(1 << N))
    random.shuffle(all_vertices)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random start
        start_v = all_vertices[attempts % len(all_vertices)]
        path = [start_v]
        used_set = {start_v}
        
        # Extend with randomized greedy
        path, used_set = extend_greedy_random(path, used_set, 1000, deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # If we've used most of the deadline, do a final check
        if time.time() > deadline - 5.0:
            break
    
    if best_path is None:
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
        best_len = len(path)
    
    return best_path
