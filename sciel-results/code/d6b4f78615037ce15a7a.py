import time
import random

N = 9
MAX_V = 1 << N
MASK = MAX_V - 1

def construct():
    deadline = time.time() + 115.0  # 5s safety margin before 120s kill
    
    # Precompute bit masks
    bit_masks = [1 << b for b in range(N)]
    
    # Precompute popcount for all possible XOR values (0..511)
    popcount = [0] * MAX_V
    for i in range(1, MAX_V):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    best_path = []
    best_len = 0
    
    def is_valid_extension(v, path, used_mask):
        """Check if v can be appended to path (differs from all non-consecutive by >= 2 bits)."""
        # v must differ from path[-1] by exactly 1 bit (guaranteed by caller)
        # v must differ from path[-2] (if exists) by >= 2 bits
        # v must differ from all earlier path vertices by >= 2 bits
        # Equivalently: v is not adjacent to any vertex in path[:-1]
        # i.e., for all i < len(path)-1: popcount(v ^ path[i]) != 1
        n = len(path)
        if n >= 2:
            prev2 = path[-2]
            if popcount(v ^ prev2) == 1:
                return False
        for i in range(n - 2):
            if popcount(v ^ path[i]) == 1:
                return False
        return True
    
    def get_valid_candidates(cur, path, used_mask):
        """Get all valid next vertices from cur, with their forward degrees."""
        candidates = []
        n = len(path)
        for b in range(N):
            v = cur ^ bit_masks[b]
            if v & used_mask:
                continue
            # Check v is valid extension
            if n >= 2:
                if popcount(v ^ path[-2]) == 1:
                    continue
            for i in range(n - 2):
                if popcount(v ^ path[i]) == 1:
                    break
            else:
                # v is valid, compute its forward degree
                deg = 0
                for b2 in range(N):
                    v2 = v ^ bit_masks[b2]
                    if v2 & used_mask:
                        continue
                    if v2 == cur:
                        continue
                    # Check v2 would be valid after adding v
                    # v2 must not be adjacent to path[-1] (which is cur) -- already checked v2 != cur
                    # v2 must not be adjacent to path[-2] (if exists)
                    if n >= 2:
                        if popcount(v2 ^ path[-2]) == 1:
                            continue
                    for i in range(n - 2):
                        if popcount(v2 ^ path[i]) == 1:
                            break
                    else:
                        deg += 1
                candidates.append((deg, v))
        return candidates
    
    def backtrack(path, used_mask, depth_limit):
        """Backtracking search with Warnsdorff heuristic."""
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        if cur_len >= depth_limit:
            return
        
        cur = path[-1]
        candidates = get_valid_candidates(cur, path, used_mask)
        
        if not candidates:
            return
        
        # Sort by forward degree (Warnsdorff: prefer lower degree)
        candidates.sort(key=lambda x: x[0])
        
        # Add slight randomness among top candidates to explore different branches
        top_k = min(3, len(candidates))
        if top_k > 1 and random.random() < 0.4:
            # Shuffle among top candidates
            random.shuffle(candidates[:top_k])
        
        for deg, v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            used_mask_new = used_mask | (1 << v)
            backtrack(path, used_mask_new, depth_limit)
            path.pop()
            if time.time() > deadline:
                return
    
    def greedy_extend(path, used_mask, bit_order):
        """Greedily extend path using a fixed bit order."""
        while time.time() < deadline:
            cur = path[-1]
            nxt = -1
            for b in bit_order:
                v = cur ^ bit_masks[b]
                if v & used_mask:
                    continue
                n = len(path)
                valid = True
                if n >= 2 and popcount(v ^ path[-2]) == 1:
                    valid = False
                else:
                    for i in range(n - 2):
                        if popcount(v ^ path[i]) == 1:
                            valid = False
                            break
                if valid:
                    nxt = v
                    break
            if nxt < 0:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
        return path
    
    def local_extend(path, used_mask):
        """Try to extend path by trying all valid moves at the end."""
        if time.time() > deadline:
            return path
        cur = path[-1]
        candidates = get_valid_candidates(cur, path, used_mask)
        if not candidates:
            return path
        # Pick the one with lowest degree
        candidates.sort(key=lambda x: x[0])
        v = candidates[0][1]
        path.append(v)
        used_mask |= (1 << v)
        return local_extend(path, used_mask)
    
    # Phase 1: Backtracking search with multiple restarts
    random.seed(12345)
    
    # Use a moderate depth limit to keep backtracking tractable
    depth_limit = 160
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        start = random.randint(0, MAX_V - 1)
        path = [start]
        used_mask = 1 << start
        
        backtrack(path, used_mask, depth_limit)
        
        # Local extend the best path found
        if best_path and time.time() < deadline - 5:
            extended = best_path[:]
            used = 0
            for v in extended:
                used |= (1 << v)
            extended = local_extend(extended, used)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended[:]
    
    # Phase 2: Greedy restarts with random bit orderings (fast, many iterations)
    if time.time() < deadline - 3:
        greedy_count = 0
        while time.time() < deadline - 3:
            bit_order = list(range(N))
            random.shuffle(bit_order)
            start = random.randint(0, MAX_V - 1)
            path = [start]
            used_mask = 1 << start
            path = greedy_extend(path, used_mask, bit_order)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            # Also try local extend on the greedy result
            if time.time() < deadline - 3:
                extended = local_extend(path[:], used_mask)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended[:]
            
            greedy_count += 1
    
    # Final: try local extend on best path one more time
    if best_path and time.time() < deadline - 1:
        used = 0
        for v in best_path:
            used |= (1 << v)
        extended = local_extend(best_path[:], used)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    if not best_path:
        return [0]
    
    return best_path
