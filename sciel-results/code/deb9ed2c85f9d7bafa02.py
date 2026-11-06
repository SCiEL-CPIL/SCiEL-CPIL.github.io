"""Combined snake-in-the-box search: Warnsdorff DFS with bitmask acceleration and structural improvements."""
import time
import random
import sys

N = 9
TIME_LIMIT = 115.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104
FULL_SIZE = 1 << N

# Pre-compute neighbors for all vertices in Q_9
NEIGHBORS = [[] for _ in range(FULL_SIZE)]
for v in range(FULL_SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Pre-compute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
NEIGHBOR_MASK = [0] * FULL_SIZE
for v in range(FULL_SIZE):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = mask

# EVOLVE-BLOCK-START

def _warnsdorff_backtrack_bitmask(start_vertex, deadline, rng, max_backtrack=5):
    """
    Deep backtracking search using Warnsdorff's heuristic and bitmask state.
    Returns the longest path found.
    """
    best_path = [start_vertex]
    best_len = 1
    
    path = [start_vertex]
    # forbidden mask: bit i is 1 if vertex i is forbidden
    # Initially, only the start vertex is "used"
    forbidden = 1 << start_vertex
    current = start_vertex
    
    def backtrack(current, forbidden, path, depth):
        nonlocal best_path, best_len
        
        # Time check
        if time.time() > deadline:
            return False
            
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Find valid neighbors
        valid_neighbors = []
        cur_mask = NEIGHBOR_MASK[current]
        
        for v in NEIGHBORS[current]:
            # Check if v is forbidden
            if (forbidden >> v) & 1:
                continue
            
            # Calculate Warnsdorff score: number of valid neighbors of v
            # new_forbidden if we move to v:
            # forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            new_forbidden = forbidden | cur_mask | (1 << v)
            
            count = 0
            v_mask = NEIGHBOR_MASK[v]
            # Count how many neighbors of v are NOT in new_forbidden
            available = v_mask & ~new_forbidden
            # Count bits in available
            count = available.bit_count()
            
            valid_neighbors.append((count, v))
        
        if not valid_neighbors:
            return False
            
        # Warnsdorff: sort by count ascending (fewest options first)
        valid_neighbors.sort(key=lambda x: x[0])
        
        # Add small random noise to break ties and allow variation
        for i in range(len(valid_neighbors)):
            valid_neighbors[i] = (valid_neighbors[i][0] + rng.random() * 0.05, valid_neighbors[i][1])
        valid_neighbors.sort(key=lambda x: x[0])
        
        # Limit backtracking depth to avoid explosion
        num_to_try = min(len(valid_neighbors), max_backtrack)
        
        for i in range(num_to_try):
            if time.time() > deadline:
                return False
                
            _, v = valid_neighbors[i]
            new_forbidden = forbidden | cur_mask | (1 << v)
            
            path.append(v)
            
            if backtrack(v, new_forbidden, path, depth + 1):
                path.pop()
                return True
                
            path.pop()
            
        return False
    
    backtrack(start_vertex, forbidden, path, 0)
    
    return best_path

def _fast_greedy_bitmask(start_vertex, deadline, rng):
    """
    A fast greedy search using bitmask representation for speed.
    """
    path = [start_vertex]
    used_mask = 1 << start_vertex
    exclusion_mask = 0
    
    while time.time() < deadline:
        cur = path[-1]
        nm = NEIGHBOR_MASK[cur]
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            break
        
        # Extract candidate indices from valid_mask
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Use Warnsdorff-like heuristic: pick candidate with fewest future options
            best_score = float('inf')
            best_cands = []
            for c in candidates:
                # Quick score: count valid neighbors of c
                nm_c = NEIGHBOR_MASK[c]
                valid_c = nm_c & ~(used_mask | (1 << c)) & ~exclusion_mask
                score = valid_c.bit_count()
                if score < best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used_mask |= (1 << v)
        exclusion_mask |= nm
    
    return path

def _validate_path(path):
    """Validate that the path is a valid snake (induced path)."""
    if len(path) < 2:
        return True
    
    # Check for duplicates
    if len(set(path)) != len(path):
        return False
    
    # Check consecutive vertices differ in exactly one bit
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        if diff == 0 or (diff & (diff - 1)) != 0:
            return False
    
    # Check non-consecutive vertices differ in at least two bits
    for i in range(len(path)):
        for j in range(i - 2):
            if j < 0:
                continue
            diff = path[i] ^ path[j]
            if diff == 0 or (diff & (diff - 1)) == 0:
                return False
    
    return True

def _try_splice(path1, path2, deadline):
    """
    Try to splice two paths together to create a longer path.
    """
    set1 = set(path1)
    set2 = set(path2)
    
    # Try to find a valid connection between end of path1 and start of path2,
    # or between end of path2 and start of path1
    
    # Check if we can connect end of path1 to start of path2
    end1 = path1[-1]
    start2 = path2[0]
    
    used = set(path1)
    exclusion = set()
    for p in path1[:-1]:
        for nb in NEIGHBORS[p]:
            exclusion.add(nb)
    
    if start2 in NEIGHBORS[end1] and start2 not in used and start2 not in exclusion:
        # Try to extend path1 with path2
        combined = list(path1) + list(path2[1:])
        # Validate the combined path
        if _validate_path(combined):
            return combined
    
    # Try reverse: end of path2 to start of path1
    end2 = path2[-1]
    start1 = path1[0]
    
    used2 = set(path2)
    exclusion2 = set()
    for p in path2[:-1]:
        for nb in NEIGHBORS[p]:
            exclusion2.add(nb)
    
    if start1 in NEIGHBORS[end2] and start1 not in used2 and start1 not in exclusion2:
        combined = list(path2) + list(path1[1:])
        if _validate_path(combined):
            return combined
    
    # If direct splice doesn't work, try to find a better connection
    # by searching for a valid extension from the end of one path into the other
    for path_a, path_b in [(path1, path2), (path2, path1)]:
        end_a = path_a[-1]
        used_a = set(path_a)
        exclusion_a = set()
        for p in path_a[:-1]:
            for nb in NEIGHBORS[p]:
                exclusion_a.add(nb)
        
        # Try each neighbor of end_a that's in path_b
        for nb in NEIGHBORS[end_a]:
            if nb in used_a or nb in exclusion_a:
                continue
            # Check if nb is in path_b
            if nb in set(path_b):
                # Find position of nb in path_b
                idx = path_b.index(nb)
                # Take the suffix of path_b starting from nb
                suffix = path_b[idx:]
                # Combine path_a with suffix
                combined = list(path_a) + list(suffix[1:])
                if _validate_path(combined):
                    return combined
    
    return None

def _try_extend_bitmask(path, deadline, rng):
    """
    Try to extend an existing path further using greedy lookahead with bitmasks.
    """
    used_mask = 0
    for p in path:
        used_mask |= (1 << p)
    
    # Build exclusion from non-terminal vertices
    exclusion_mask = 0
    for p in path[:-1]:
        exclusion_mask |= NEIGHBOR_MASK[p]
    
    while time.time() < deadline:
        cur = path[-1]
        nm = NEIGHBOR_MASK[cur]
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            break
        
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Use Warnsdorff-like heuristic
            best_score = float('inf')
            best_cands = []
            for c in candidates:
                nm_c = NEIGHBOR_MASK[c]
                valid_c = nm_c & ~(used_mask | (1 << c)) & ~exclusion_mask
                score = valid_c.bit_count()
                if score < best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used_mask |= (1 << v)
        exclusion_mask |= nm
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    sys.setrecursionlimit(10000)
    
    # Strategy 1: Fast randomized greedy with Warnsdorff heuristic - many restarts
    while time.time() < deadline - 30.0:
        rng = random.Random(int(time.time() * 1000) % 1000000)
        start_v = rng.randint(0, FULL_SIZE - 1)
        p = _fast_greedy_bitmask(start_v, deadline - 29.0, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 2: Warnsdorff DFS with backtracking - try a few different starts
    for trial in range(3):
        if time.time() > deadline - 15.0:
            break
        rng = random.Random(42 + trial * 1000)
        start_v = rng.randint(0, FULL_SIZE - 1)
        p = _warnsdorff_backtrack_bitmask(start_v, deadline - 14.0, rng, max_backtrack=5)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 3: Splicing - try to combine two good paths
    if best_path is not None and time.time() < deadline - 10.0:
        # Generate another path to try splicing with
        rng = random.Random(12345)
        start_v = rng.randint(0, FULL_SIZE - 1)
        p2 = _fast_greedy_bitmask(start_v, deadline - 9.0, rng)
        
        if len(p2) > 10:
            spliced = _try_splice(list(best_path), list(p2), deadline - 8.0)
            if spliced is not None and len(spliced) > best_len:
                best_len = len(spliced)
                best_path = spliced
    
    # Final: try to extend the best path found
    if best_path is not None and time.time() < deadline - 2.0:
        rng = random.Random(99999)
        extended = _try_extend_bitmask(list(best_path), deadline - 1.0, rng)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Ensure we have a valid path
    if best_path is None or not _validate_path(best_path):
        # Fallback: return a simple valid path
        best_path = [0, 1, 3, 7, 15, 31, 63, 127, 255]
    
    return best_path

# EVOLVE-BLOCK-END
