"""
Snake-in-the-Box solver for Q_9 using bitmask-based Warnsdorff backtracking.
This is a structural change from the seed's pure greedy approach to a
backtracking search with Warnsdorff heuristic, which can undo moves and
explore alternative branches when the greedy path gets stuck.
"""
import time
import random
import sys

N = 9
SIZE = 1 << N  # 512

# Precompute neighbors for each vertex
_NEIGHBORS = []
for v in range(SIZE):
    nbrs = [v ^ (1 << b) for b in range(N)]
    _NEIGHBORS.append(nbrs)

# Precompute adjacency masks: _ADJ_MASK[v] is a 512-bit integer where bit u is 1 iff u is adjacent to v.
_ADJ_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for u in _NEIGHBORS[v]:
        m |= (1 << u)
    _ADJ_MASK[v] = m

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    best_path = [0]
    best_len = 1
    rng = random.Random(42)
    
    def _validate_path(path):
        """Check if path is a valid induced path (snake)."""
        if not path:
            return True
        seen = set()
        for i, v in enumerate(path):
            if v in seen:
                return False
            seen.add(v)
            if i > 0:
                if (v ^ path[i-1]).bit_count() != 1:
                    return False
                for j in range(i-1):
                    if (v ^ path[j]).bit_count() == 1:
                        return False
        return True

    def _backtrack_search(start_v, time_limit, max_depth=200):
        """
        Warnsdorff backtracking search for a long snake starting from start_v.
        Uses bitmask operations for fast validity checks.
        Returns the longest path found during this search.
        """
        best_local = [start_v]
        best_local_len = 1
        
        path = [start_v]
        used_mask = 1 << start_v
        # forbidden: union of neighbors of all vertices in path[:-1]
        # When at path[k], forbidden contains neighbors of path[0..k-1]
        forbidden = 0
        
        def _warnsdorff_order(cur, used_mask, forbidden):
            """
            Order candidates from cur using Warnsdorff heuristic.
            Candidates: neighbors of cur not in used_mask and not in forbidden.
            Score: number of onward options (fewer is better).
            """
            nbrs_mask = _ADJ_MASK[cur]
            valid_mask = nbrs_mask & ~used_mask & ~forbidden
            
            if valid_mask == 0:
                return []
            
            candidates = []
            vm = valid_mask
            while vm:
                lsb = vm & -vm
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                vm ^= lsb
            
            if not candidates:
                return []
            
            # New forbidden after moving to u: forbidden | _ADJ_MASK[cur]
            new_forbidden = forbidden | _ADJ_MASK[cur]
            
            scored = []
            for u in candidates:
                new_used = used_mask | (1 << u)
                nbrs_u = _ADJ_MASK[u]
                valid_next = nbrs_u & ~new_used & ~new_forbidden
                count = valid_next.bit_count()
                scored.append((count, u))
            
            scored.sort(key=lambda x: (x[0], x[1]))
            return [u for _, u in scored]
        
        def _search(depth):
            nonlocal best_local, best_local_len
            
            if time.time() > time_limit:
                return
            
            if len(path) > best_local_len:
                best_local_len = len(path)
                best_local = path[:]
            
            if depth >= max_depth:
                return
            
            cur = path[-1]
            candidates = _warnsdorff_order(cur, used_mask, forbidden)
            
            if not candidates:
                return
            
            # Add some randomness to break ties
            if len(candidates) > 1 and rng.random() < 0.2:
                # Shuffle the first few candidates
                k = min(4, len(candidates))
                subset = candidates[:k]
                rng.shuffle(subset)
                candidates = subset + candidates[k:]
            
            for nxt in candidates:
                # Try this move
                path.append(nxt)
                new_used = used_mask | (1 << nxt)
                new_forbidden = forbidden | _ADJ_MASK[cur]
                
                _search(depth + 1)
                
                # Undo
                path.pop()
                
                # Early termination if we found a very long path
                if best_local_len >= max_depth:
                    return
                
                if time.time() > time_limit:
                    return
        
        _search(0)
        return best_local

    # Main search loop: multiple restarts with backtracking
    attempts = 0
    last_improve_time = time.time()
    
    while time.time() < deadline:
        # Choose a random start vertex
        seed = rng.randint(0, SIZE - 1)
        
        # Run backtracking search with a time budget
        time_budget = min(5.0, deadline - time.time())
        if time_budget < 1.0:
            break
        
        result = _backtrack_search(seed, time.time() + time_budget)
        
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
            last_improve_time = time.time()
        
        attempts += 1
        
        # If we've been stuck for a while, try different strategies
        if time.time() - last_improve_time > 30.0:
            # Try a few quick greedy extensions from random starts
            for _ in range(10):
                s = rng.randint(0, SIZE - 1)
                path = [s]
                used = 1 << s
                forb = 0
                while time.time() < deadline:
                    cur = path[-1]
                    valid = _ADJ_MASK[cur] & ~used & ~forb
                    if valid == 0:
                        break
                    # Pick random valid candidate
                    vm = valid
                    lsb = vm & -vm
                    nxt = lsb.bit_length() - 1
                    path.append(nxt)
                    used |= (1 << nxt)
                    forb = forb | _ADJ_MASK[cur]
                    if time.time() > deadline:
                        break
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
                    last_improve_time = time.time()
        
        if attempts > 500:
            break
    
    # Final validation
    if not _validate_path(best_path):
        # Fallback to a simple valid path
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
