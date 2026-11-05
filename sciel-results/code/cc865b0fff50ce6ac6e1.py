"""Snake-in-the-box in Q_9: construct an induced path maximizing its length."""
import time
import random

N = 9
MASK = (1 << N) - 1
DEADLINE = 110.0  # safety margin below 120s

def construct():
    start = time.time()
    
    # Precompute bit positions
    bits = [1 << b for b in range(N)]
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    def greedy_extend(path, used, bit_order):
        """Extend path greedily using given bit order, return new path."""
        p = list(path)
        u = set(used)
        while True:
            cur = p[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ b
                if v in u:
                    continue
                # Check induced property: v must not be at distance 1 from any non-adjacent vertex
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                return p
            p.append(nxt)
            u.add(nxt)
    
    def backtrack_search(path, used, depth, max_depth, bit_order):
        """Backtracking search to extend the path. Returns best extension found."""
        best = list(path)
        if depth >= max_depth:
            return best
        cur = path[-1]
        candidates = []
        for b in bit_order:
            v = cur ^ b
            if v in used:
                continue
            # Check induced property
            ok = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        
        # Try candidates in order of "promisingness" - prefer those with more future options
        # Simple heuristic: prefer flipping lower-order bits first for diversity
        for v in candidates:
            if time_left() < 0.5:
                break
            new_path = path + [v]
            new_used = used | {v}
            result = backtrack_search(new_path, new_used, depth + 1, max_depth, bit_order)
            if len(result) > len(best):
                best = result
            if time_left() < 0.5:
                break
        return best
    
    def local_search(path):
        """Try to improve path by reordering or finding alternative extensions."""
        # Try multiple random bit orderings for greedy extension
        best = list(path)
        for trial in range(50):
            if time_left() < 1.0:
                break
            order = list(range(N))
            random.shuffle(order)
            ext = greedy_extend(path, set(path), [1 << b for b in order])
            if len(ext) > len(best):
                best = ext
        return best
    
    # Phase 1: Get a good initial snake with randomized greedy
    best_path = []
    for trial in range(200):
        if time_left() < 5.0:
            break
        order = list(range(N))
        random.shuffle(order)
        p = [0]
        u = {0}
        while True:
            cur = p[-1]
            nxt = None
            for b in order:
                v = cur ^ (1 << b)
                if v in u:
                    continue
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            p.append(nxt)
            u.add(nxt)
        if len(p) > len(best_path):
            best_path = p
    
    # Phase 2: Local improvement via different bit orderings
    if time_left() > 5.0:
        improved = local_search(best_path)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Phase 3: Backtracking search from the current best
    if time_left() > 3.0:
        # Try backtracking with limited depth
        max_depth = min(20, int(time_left() * 2))
        order = list(range(N))
        random.shuffle(order)
        result = backtrack_search(best_path, set(best_path), 0, max_depth, [1 << b for b in order])
        if len(result) > len(best_path):
            best_path = result
    
    # Phase 4: More random restarts with greedy
    while time_left() > 2.0:
        order = list(range(N))
        random.shuffle(order)
        p = [0]
        u = {0}
        while True:
            cur = p[-1]
            nxt = None
            for b in order:
                v = cur ^ (1 << b)
                if v in u:
                    continue
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            p.append(nxt)
            u.add(nxt)
        if len(p) > len(best_path):
            best_path = p
            # Try to extend further with backtracking
            if time_left() > 5.0:
                max_depth = min(15, int(time_left()))
                order2 = list(range(N))
                random.shuffle(order2)
                result = backtrack_search(best_path, set(best_path), 0, max_depth, [1 << b for b in order2])
                if len(result) > len(best_path):
                    best_path = result
    
    return best_path

# Ensure we always return at least the basic greedy result
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result) - 1}")
