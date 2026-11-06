"""Snake-in-the-box in Q_9: beam search with bitmask validity and randomized restarts."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N
DEADLINE_MARGIN = 12.0  # safety margin before 150s kill

# Precompute bit masks
BIT_MASKS = [1 << b for b in range(N)]


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_path = None
    best_len = 0

    # Precompute popcount for all pairs? No, too large (512^2 = 262144, actually fine)
    # Actually let's just use bit_count() which is fast in Python 3.10+

    def check_valid(v, path, used_set):
        """Check if v can be appended to path (v not in used, and not adjacent to any non-last vertex)."""
        if v in used_set:
            return False
        # v must differ from all vertices except the last by >= 2 bits
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True

    def greedy_extend(path, used_set, bit_order, deadline):
        """Extend path greedily using the given bit order."""
        while time.time() < deadline:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ BIT_MASKS[b]
                if v in used_set:
                    continue
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_set.add(nxt)
        return path, used_set

    def beam_search(start_v, beam_width, deadline, rng):
        """Beam search: maintain a set of candidate paths, extend each, keep top by length."""
        # Each beam entry: (path_list, used_set)
        # We'll represent beams as list of (path, used_set)
        
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        # Initialize with one beam
        beams = [([start_v], {start_v})]
        
        while time.time() < deadline:
            new_beams = []
            for path, used_set in beams:
                cur = path[-1]
                # Find all valid extensions
                extensions = []
                for b in bit_order:
                    v = cur ^ BIT_MASKS[b]
                    if v in used_set:
                        continue
                    valid = True
                    for i in range(len(path) - 1):
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        extensions.append(v)
                
                if not extensions:
                    # Dead end, keep this beam as-is (it's a completed path)
                    new_beams.append((path, used_set))
                else:
                    # Try each extension (or a subset if too many)
                    # For beam search, we try all extensions and keep the best
                    for v in extensions:
                        new_path = path + [v]
                        new_used = used_set | {v}
                        new_beams.append((new_path, new_used))
            
            if not new_beams:
                break
            
            # Sort by length descending, keep top beam_width
            new_beams.sort(key=lambda x: len(x[0]), reverse=True)
            new_beams = new_beams[:beam_width]
            
            # Check if all beams are dead ends
            all_dead = True
            for path, used_set in new_beams:
                cur = path[-1]
                for b in bit_order:
                    v = cur ^ BIT_MASKS[b]
                    if v in used_set:
                        continue
                    valid = True
                    for i in range(len(path) - 1):
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        all_dead = False
                        break
                if not all_dead:
                    break
            if all_dead:
                break
            
            beams = new_beams
        
        return beams

    # Strategy 1: Greedy with fixed bit order
    path, used = [0], {0}
    path, used = greedy_extend(path, used, list(range(N)), deadline)
    best_len = len(path) - 1
    best_path = path[:]

    # Strategy 2: Greedy with various bit orders and starting vertices
    rng = random.Random(12345)
    
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        start_v = rng.randint(0, TOTAL_VERTICES - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        path, used = [start_v], {start_v}
        path, used = greedy_extend(path, used, bit_order, deadline)
        
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
        
        # Small batch of greedy tries per iteration to amortize overhead
        for _ in range(5):
            if time.time() >= deadline:
                break
            start_v = rng.randint(0, TOTAL_VERTICES - 1)
            bit_order = list(range(N))
            rng.shuffle(bit_order)
            
            path, used = [start_v], {start_v}
            path, used = greedy_extend(path, used, bit_order, deadline)
            
            if len(path) - 1 > best_len:
                best_len = len(path) - 1
                best_path = path[:]

    # Strategy 3: Beam search with moderate beam width
    if time.time() < deadline:
        rng2 = random.Random(99999)
        beam_width = 8
        for _ in range(20):
            if time.time() >= deadline:
                break
            start_v = rng2.randint(0, TOTAL_VERTICES - 1)
            beams = beam_search(start_v, beam_width, deadline, rng2)
            for path, used_set in beams:
                if len(path) - 1 > best_len:
                    best_len = len(path) - 1
                    best_path = path[:]

    # Strategy 4: Try to extend the best path found with limited backtracking
    if best_path and time.time() < deadline:
        improved = _try_backtrack_extend(best_path, deadline)
        if improved and len(improved) - 1 > best_len:
            best_len = len(improved) - 1
            best_path = improved

    return best_path


def _try_backtrack_extend(path, deadline):
    """Try to extend path using limited backtracking (depth 6)."""
    path = path[:]
    used = set(path)
    
    def backtrack(depth):
        if depth == 0 or time.time() > deadline:
            return False
        
        cur = path[-1]
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            path.append(v)
            used.add(v)
            
            if backtrack(depth - 1):
                return True
            
            path.pop()
            used.remove(v)
        
        return False
    
    success = backtrack(6)
    if success:
        return path
    return None
