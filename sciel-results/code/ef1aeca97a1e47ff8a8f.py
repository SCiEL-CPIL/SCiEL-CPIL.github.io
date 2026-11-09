"""Stronger snake-in-the-box search for Q_9 with deadline guard."""
import time
import random
from collections import deque

N = 9
MASK = (1 << N) - 1
# Precompute bit values for speed
BITS = [1 << b for b in range(N)]

# Precompute popcount for all 9-bit values (0..511)
_POP = [0] * 512
for i in range(1, 512):
    _POP[i] = _POP[i >> 1] + (i & 1)

def _pop(v):
    return _POP[v]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s

    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(512)]
    for v in range(512):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    best_path = None
    best_len = 0

    # Strategy 1: Multiple greedy strategies with different bit orderings
    # Try many random bit orderings and greedy extensions
    rng = random.Random(42)

    def greedy_extend(start, bit_order):
        path = [start]
        used = [False] * 512
        used[start] = True
        cur = start
        while True:
            # For each candidate neighbor in bit_order, check validity
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if used[v]:
                    continue
                # Check that v is not adjacent to any non-consecutive path vertex
                # i.e., v must not be at distance 1 from path[0..len-2]
                ok = True
                for i in range(len(path) - 1):
                    if _POP[v ^ path[i]] == 1:
                        ok = False
                        break
                if ok:
                    path.append(v)
                    used[v] = True
                    cur = v
                    found = True
                    break
            if not found:
                break
        return path

    # Try different starting vertices and bit orderings
    starts = list(range(512))
    rng.shuffle(starts)

    # First, try a few deterministic orderings
    orderings = []
    # Natural order
    orderings.append(list(range(N)))
    # Reverse
    orderings.append(list(range(N - 1, -1, -1)))
    # A few random ones
    for _ in range(20):
        o = list(range(N))
        rng.shuffle(o)
        orderings.append(o)

    for s in starts[:50]:
        if time.time() > DEADLINE:
            break
        for o in orderings[:5]:
            if time.time() > DEADLINE:
                break
            p = greedy_extend(s, o)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p[:]

    # Strategy 2: Backtracking with constraint propagation and time limit
    # Use a smarter backtracking that tries to extend the best path found
    def backtrack_extend(path, used, cur, depth):
        """Try to extend path by one more vertex using backtracking with limited depth."""
        if time.time() > DEADLINE:
            return path
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if used[v]:
                continue
            ok = True
            for i in range(len(path) - 1):
                if _POP[v ^ path[i]] == 1:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        
        if not candidates:
            return path
        
        # Try each candidate, prefer ones that lead to longer extensions
        # Sort by some heuristic: fewer future options might be better (most constrained)
        def heuristic(v):
            # Count how many valid next steps would be available after choosing v
            cnt = 0
            for b in range(N):
                w = v ^ (1 << b)
                if not used[w] and w != cur:
                    # Check if w is adjacent to path[:-1]
                    adj = False
                    for i in range(len(path) - 1):
                        if _POP[w ^ path[i]] == 1:
                            adj = True
                            break
                    if not adj:
                        cnt += 1
            return cnt

        candidates.sort(key=heuristic)
        
        for v in candidates:
            path.append(v)
            used[v] = True
            result = backtrack_extend(path, used, v, depth + 1)
            if len(result) > len(path):
                path = result
            else:
                path.pop()
                used[v] = False
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            if time.time() > DEADLINE:
                return path
        return path

    # Try to improve the best path found so far with limited backtracking
    if best_path and best_len > 104:
        if time.time() < DEADLINE - 5:
            used = [False] * 512
            for v in best_path:
                used[v] = True
            improved = backtrack_extend(best_path[:], used, best_path[-1], 0)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = improved[:]

    # Strategy 3: Try random restarts with greedy + local search
    for _ in range(200):
        if time.time() > DEADLINE:
            break
        s = rng.randint(0, 511)
        o = list(range(N))
        rng.shuffle(o)
        p = greedy_extend(s, o)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Strategy 4: Try to extend from both ends of the best path
    if best_path and time.time() < DEADLINE - 10:
        # Try extending from the start
        def extend_from_start(path, used):
            cur = path[0]
            while True:
                found = False
                for b in range(N):
                    v = cur ^ (1 << b)
                    if used[v]:
                        continue
                    # v must not be adjacent to path[1:]
                    ok = True
                    for i in range(1, len(path)):
                        if _POP[v ^ path[i]] == 1:
                            ok = False
                            break
                    if ok:
                        path.insert(0, v)
                        used[v] = True
                        cur = v
                        found = True
                        break
                if not found:
                    break
            return path

        # Make a copy and try to extend from start
        p_copy = best_path[:]
        used = [False] * 512
        for v in p_copy:
            used[v] = True
        extended = extend_from_start(p_copy, used)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended[:]

    if best_path is None or len(best_path) <= 1:
        return [0]
    return best_path
