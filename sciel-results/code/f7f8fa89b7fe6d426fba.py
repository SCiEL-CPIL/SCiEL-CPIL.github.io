"""Snake-in-the-box in the 9-dimensional hypercube - Bitmask backtracking + local refinement."""
N = 9
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute neighbor masks: neighbor_mask[v] has bits set for all vertices at Hamming distance 1 from v
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            nv = v ^ (1 << b)
            mask |= (1 << nv)
        neighbor_mask[v] = mask

    bit_masks = [1 << b for b in range(N)]

    def _backtrack_search(start_vertex, search_deadline, rng=None, max_depth=None):
        """Backtracking search with Warnsdorff heuristic using bitmask state.
        
        State:
        - path: list of vertices
        - forbidden_mask: int bitmask, bit i set means vertex i cannot be next
          forbidden_mask = used_mask | union_{i=0}^{k-1} neighbor_mask[path[i]]
          (vertices at dist 1 from all non-last vertices in path, plus all used vertices)
        
        When extending by v:
        - v must not be in forbidden_mask
        - new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[path[-1]]
          (path[-1] becomes non-consecutive, so its neighbors become forbidden)
        """
        if rng is None:
            rng = random.Random(42)
        
        best_path = [start_vertex]
        best_len = 1
        
        # Use recursive approach with increased recursion limit
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        def _search_recursive(path, forbidden_mask, depth):
            nonlocal best_path, best_len
            
            if time.time() > search_deadline:
                return
            
            if depth > best_len:
                best_len = depth
                best_path = path[:]
            
            if max_depth is not None and depth >= max_depth:
                return
            
            cur = path[-1]
            
            # Find candidate next vertices
            candidates = []
            for b in range(N):
                v = cur ^ bit_masks[b]
                if forbidden_mask & (1 << v):
                    continue
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                onward = 0
                for b2 in range(N):
                    nv = v ^ bit_masks[b2]
                    if not (new_forbidden & (1 << nv)):
                        onward += 1
                candidates.append((onward, v))
            
            if not candidates:
                return
            
            # Sort by onward moves (Warnsdorff: fewest onward first)
            candidates.sort(key=lambda x: x[0])
            
            # Try each candidate
            for onward, v in candidates:
                if time.time() > search_deadline:
                    return
                new_forbidden = forbidden_mask | (1 << v) | neighbor_mask[cur]
                path.append(v)
                _search_recursive(path, new_forbidden, depth + 1)
                path.pop()
        
        _search_recursive([start_vertex], (1 << start_vertex), 1)
        sys.setrecursionlimit(old_limit)
        
        return best_path

    def _greedy_extend(path, deadline, rng=None):
        """Extend a path greedily from both ends using Warnsdorff."""
        if rng is None:
            rng = random.Random(42)
        
        path = path[:]
        used = set(path)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for end in [0, 1]:
                if end == 0:
                    cur = path[0]
                    # v must differ from path[1:] by >= 2 bits
                    check_list = path[1:]
                else:
                    cur = path[-1]
                    # v must differ from path[:-1] by >= 2 bits
                    check_list = path[:-1]
                
                candidates = []
                for b in range(N):
                    v = cur ^ bit_masks[b]
                    if v in used:
                        continue
                    
                    valid = True
                    for p in check_list:
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Count onward moves
                    onward = 0
                    for b2 in range(N):
                        nv = v ^ bit_masks[b2]
                        if nv in used:
                            continue
                        if end == 0:
                            # nv would be prepended after v. New path: [nv, v, path[0], ...]
                            # nv consecutive to v. Non-consecutive to nv: path[0:]
                            ok = True
                            for p in path:
                                if (nv ^ p).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                onward += 1
                        else:
                            # nv would be appended after v. New path: [..., path[-1], v, nv]
                            # nv consecutive to v. Non-consecutive to nv: path[:-1]
                            ok = True
                            for p in path[:-1]:
                                if (nv ^ p).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                onward += 1
                    
                    candidates.append((onward, b, v))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, b, v = candidates[0]
                    if end == 0:
                        path.insert(0, v)
                        used.add(v)
                    else:
                        path.append(v)
                        used.add(v)
                    improved = True
        
        return path

    def _bidirectional_greedy(start1, start2, rng, deadline):
        """Build two snakes from start1 and start2, try to connect them."""
        path1 = _backtrack_search(start1, deadline, rng, max_depth=60)
        path2 = _backtrack_search(start2, deadline, rng, max_depth=60)
        
        def try_connect(p1, p2):
            """Try to concatenate p1 + p2 where last of p1 is adjacent to first of p2."""
            if not p1 or not p2:
                return []
            last1 = p1[-1]
            first2 = p2[0]
            if (last1 ^ first2).bit_count() != 1:
                return []
            
            # Check that all vertices in p2 are valid against p1[:-1]
            for v in p2:
                for p in p1[:-1]:
                    if (v ^ p).bit_count() == 1:
                        return []
            return p1 + p2
        
        # Try path1 + path2
        connected = try_connect(path1, path2)
        if connected:
            return connected
        
        # Try path1 + reversed path2
        rev_path2 = path2[::-1]
        connected = try_connect(path1, rev_path2)
        if connected:
            return connected
        
        # Try reversed path1 + path2
        rev_path1 = path1[::-1]
        connected = try_connect(rev_path1, path2)
        if connected:
            return connected
        
        # Try reversed path1 + reversed path2
        connected = try_connect(rev_path1, rev_path2)
        if connected:
            return connected
        
        return []

    # Phase 1: Multiple backtracking searches with different starts
    rng = random.Random(12345)
    best_path = None
    best_len = 0
    
    starts = [0, 1, (1 << N) - 1, 0x155, 0xAAA]
    for _ in range(20):
        starts.append(rng.randint(0, (1 << N) - 1))
    
    for sv in starts:
        if time.time() > deadline - 10:
            break
        path = _backtrack_search(sv, deadline - 5, rng, max_depth=80)
        path = _greedy_extend(path, deadline - 5, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: Bidirectional construction
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        sv1 = rng.randint(0, (1 << N) - 1)
        sv2 = rng.randint(0, (1 << N) - 1)
        if sv1 == sv2:
            continue
        connected = _bidirectional_greedy(sv1, sv2, rng, deadline - 2)
        if connected and len(connected) > best_len:
            best_len = len(connected)
            best_path = connected
    
    # Phase 3: More diverse random restarts with local search
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        sv = rng.randint(0, (1 << N) - 1)
        path = _backtrack_search(sv, deadline - 1, rng, max_depth=50)
        path = _greedy_extend(path, deadline - 1, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Final: ensure we have at least a valid path
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        best_len = 2
    
    return best_path
