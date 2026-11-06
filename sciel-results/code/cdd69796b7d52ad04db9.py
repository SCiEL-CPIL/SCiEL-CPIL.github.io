"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
def _build_neighbor_masks():
    """Precompute for each vertex, the bitmask of its N neighbors."""
    masks = []
    for v in range(TOTAL_VERTICES):
        m = 0
        for b in range(N):
            nb = v ^ (1 << b)
            m |= (1 << nb)
        masks.append(m)
    return masks

NEIGHBOR_MASKS = _build_neighbor_masks()

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _backtracking_solve(start_vertex, deadline, max_depth=300):
    """Constraint-based backtracking with heuristic variable ordering.
    
    The key constraint: for any vertex v in the path, no OTHER vertex in the path
    can be a neighbor of v. We track 'blocked' vertices - those that cannot be
    added because they'd be adjacent to a non-consecutive path vertex.
    
    Heuristic: at each step, try candidates that leave the most future options
    (or equivalently, prune the search tree by trying most-constrained first).
    """
    path = [start_vertex]
    used = 1 << start_vertex
    
    # For each position in path, track which vertices are "forbidden"
    # because they're neighbors of path[i] and i is not adjacent to the new position.
    # We maintain a count: forbid_count[v] = number of path vertices that forbid v.
    # A vertex v is forbidden if forbid_count[v] > 0 (it's adjacent to some non-consecutive path vertex).
    
    # Actually, let's think more carefully.
    # When we add vertex v at position k (end of path), v must not be adjacent to path[0..k-2].
    # So v is valid if for all i in [0, k-2], v is not a neighbor of path[i].
    # Equivalently: v is not in the union of neighbor sets of path[0..k-2].
    
    # We'll maintain a bitmask 'blocked_mask' where bit v is set if v is adjacent to 
    # any path vertex at distance >= 2 from the current end.
    # When we extend path by v, we need v not in blocked_mask.
    # After adding v, we update blocked_mask: 
    #   - Remove the constraint from path[k-1] (it's now at distance 1 from new end, so no longer blocks)
    #   - Add v to blocked_mask for future (v will be at distance >= 2 from future extensions)
    
    # Wait, this isn't quite right either. Let me think again.
    # Path: p0, p1, ..., pk
    # When adding p_{k+1}, it must not be adjacent to p0, ..., p_{k-1}.
    # So the "blocked" set is the union of neighbors of p0, ..., p_{k-1}.
    
    # Let's maintain blocked_mask as union of neighbors of p0..p_{k-1} (all but last two).
    # Initially with path = [p0], blocked_mask = 0 (no constraints yet since we need distance >= 2).
    # After adding p1: blocked_mask = 0 still (p0 is at distance 1 from p1, no constraint).
    # After adding p2: blocked_mask = neighbors(p0) (p0 is at distance 2 from p2).
    # After adding p3: blocked_mask = neighbors(p0) | neighbors(p1).
    
    # More precisely: when path has length k+1 (vertices p0..pk), 
    # blocked_mask = union of neighbors(p_i) for i in [0, k-2].
    # A candidate v is valid if: v not in used AND v not in blocked_mask AND v is neighbor of pk.
    
    # When we extend from pk to p_{k+1}:
    #   New path length: k+2
    #   New blocked_mask = old_blocked_mask | neighbors(p_{k-1})
    #   (because p_{k-1} is now at distance 2 from p_{k+1})
    
    blocked_mask = 0  # union of neighbors of p0..p_{k-2}, initially empty
    
    # Precompute neighbor bitmasks for fast union
    # NEIGHBOR_MASKS[v] is already computed above
    
    def extend(path, used, blocked_mask, depth, deadline):
        """Recursively extend path. Returns path if successful, None otherwise."""
        if time.time() > deadline:
            return None
        
        if len(path) >= max_depth:
            return path
        
        cur = path[-1]
        # Candidates: neighbors of cur that are not used and not blocked
        cand_mask = NEIGHBOR_MASKS[cur] & ~used & ~blocked_mask
        
        if cand_mask == 0:
            # Try to extend from the other end? No, let's just return what we have.
            # Actually, for backtracking we should try alternatives.
            return None
        
        # Get candidate list
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length() - 1
            candidates.append(v)
            cm ^= lsb
        
        # Heuristic: sort candidates by how many future options they leave
        # A candidate v that blocks fewer vertices is better.
        # When we add v, new_blocked = blocked_mask | neighbors(path[-2]) (if len >= 2)
        # The "cost" of v is roughly how many of v's neighbors are already in blocked or used.
        # Actually, the key constraint is: after adding v, the new blocked_mask will include
        # neighbors of path[-2] (the second-to-last vertex). v itself doesn't directly affect
        # blocked_mask until later. But v affects what's available: v's neighbors that are 
        # in the new blocked_mask or used will be unavailable.
        
        # Simpler heuristic: try candidates with fewer available next-moves first (most constrained).
        # This is the "fail-first" principle.
        
        # For each candidate v, count how many of v's neighbors are available after adding v.
        # After adding v: new_used = used | (1<<v), new_blocked = blocked_mask | (neighbors of path[-2] if len>=2 else 0)
        # Available next = NEIGHBOR_MASKS[v] & ~new_used & ~new_blocked
        
        new_blocked_base = blocked_mask
        if len(path) >= 2:
            new_blocked_base |= NEIGHBOR_MASKS[path[-2]]
        
        scored = []
        for v in candidates:
            new_used = used | (1 << v)
            available_next = NEIGHBOR_MASKS[v] & ~new_used & ~new_blocked_base
            # Score: number of available next moves (lower = more constrained = try first)
            score = available_next.bit_count()
            scored.append((score, v))
        
        scored.sort()
        
        for score, v in scored:
            new_path = path + [v]
            new_used = used | (1 << v)
            new_blocked = new_blocked_base
            
            result = extend(new_path, new_used, new_blocked, depth + 1, deadline)
            if result is not None:
                return result
            
            if time.time() > deadline:
                return None
        
        return None
    
    return extend(path, used, blocked_mask, 0, deadline)

def _greedy_with_backtrack(start_vertex, deadline, max_bt_depth=50):
    """Hybrid: greedy extension with periodic backtracking to escape local optima."""
    path = [start_vertex]
    used = 1 << start_vertex
    blocked_mask = 0
    
    while time.time() < deadline:
        cur = path[-1]
        cand_mask = NEIGHBOR_MASKS[cur] & ~used & ~blocked_mask
        
        if cand_mask == 0:
            break
        
        # Get candidates
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length() - 1
            candidates.append(v)
            cm ^= lsb
        
        if not candidates:
            break
        
        if len(candidates) == 1:
            v = candidates[0]
            path.append(v)
            used |= (1 << v)
            if len(path) >= 2:
                blocked_mask |= NEIGHBOR_MASKS[path[-2]]
            continue
        
        # Score candidates: prefer those with fewer future options (most constrained first)
        new_blocked_base = blocked_mask
        if len(path) >= 2:
            new_blocked_base |= NEIGHBOR_MASKS[path[-2]]
        
        scored = []
        for v in candidates:
            new_used = used | (1 << v)
            available_next = NEIGHBOR_MASKS[v] & ~new_used & ~new_blocked_base
            score = available_next.bit_count()
            scored.append((score, v))
        
        scored.sort()
        
        # Try the best candidate greedily, but if it leads to a dead end, backtrack
        # For efficiency, just take the best one and use backtracking only when stuck
        v = scored[0][1]
        path.append(v)
        used |= (1 << v)
        if len(path) >= 2:
            blocked_mask |= NEIGHBOR_MASKS[path[-2]]
    
    return path

def _try_extend_both_ends(path, used, blocked_mask, deadline):
    """Try to extend from both ends of the path."""
    # We'll work with the path and try to extend from either end.
    # This requires tracking blocked masks for both ends, which is complex.
    # For simplicity, just extend from the right end.
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_path = []
    best_len = 0
    
    random.seed(42)
    
    # Strategy 1: Multiple greedy runs with different starting vertices
    # Strategy 2: Backtracking from promising starts
    # Strategy 3: Local search improvements
    
    attempt = 0
    while time.time() < deadline - 10:
        attempt += 1
        
        # Choose starting vertex
        if attempt <= 5:
            # Try some structured starts
            start_vertex = random.randint(0, TOTAL_VERTICES - 1)
        else:
            start_vertex = random.randint(0, TOTAL_VERTICES - 1)
        
        # Greedy extension
        path = _greedy_with_backtrack(start_vertex, deadline - 5)
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        # Try to extend from both ends using local search
        if time.time() < deadline - 10 and len(path) > 0:
            # Try extending the beginning
            used = set(path)
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                cur = path[0]
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    # Check if v is adjacent to any path vertex except path[0]
                    ok = True
                    for p in path[1:]:
                        if (v ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        path.insert(0, v)
                        used.add(v)
                        changed = True
                        break
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
        
        # Try backtracking from the end of the greedy path to find longer solutions
        if time.time() < deadline - 20 and len(path) > 20:
            # Try backtracking from various points
            for trim in range(0, min(20, len(path) - 10), 5):
                if time.time() > deadline - 15:
                    break
                sub_path = path[:len(path) - trim]
                # Continue with backtracking from here
                result = _backtracking_solve(
                    sub_path[-1], 
                    deadline - 10, 
                    max_depth=len(sub_path) + 100
                )
                if result is not None and len(result) > best_len:
                    # Prepend the prefix
                    full_path = sub_path[:-1] + result
                    if _is_valid_snake(full_path):
                        best_path = full_path
                        best_len = len(full_path)
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
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
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
# EVOLVE-BLOCK-END
