"""
Improved Snake-in-the-Box solver for Q_9.
Uses: Fast Greedy + Splicing + Limited Backtracking.
"""
N = 9
import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def greedy_extend(start_v, rng, time_limit):
        """
        Extends from start_v greedily using block_count optimization.
        Returns (path, used, block_count) at the point where it gets stuck.
        """
        path = [start_v]
        used = [False] * (1 << N)
        used[start_v] = True
        block_count = [0] * (1 << N)
        
        # Initialize block_count for start_v
        for w in neighbors[start_v]:
            block_count[w] += 1
        
        cur = start_v
        while time.time() < time_limit:
            candidates = []
            for v in neighbors[cur]:
                if not used[v] and block_count[v] == 1:
                    candidates.append(v)
            
            if not candidates:
                break
            
            v = rng.choice(candidates)
            path.append(v)
            used[v] = True
            for w in neighbors[v]:
                block_count[w] += 1
            cur = v
        
        return path, used, block_count
    
    def backtrack_extend(path, used, block_count, rng, time_limit, max_depth=15):
        """
        Tries to extend the path by backtracking up to max_depth steps.
        Returns the longest path found.
        """
        best_local_path = list(path)
        best_local_len = len(path)
        
        def dfs(current_path, current_used, current_block_count, depth, time_limit):
            nonlocal best_local_path, best_local_len
            
            if time.time() > time_limit:
                return
            
            if len(current_path) > best_local_len:
                best_local_len = len(current_path)
                best_local_path = list(current_path)
                
            if depth >= max_depth:
                return
                
            cur = current_path[-1]
            
            candidates = []
            for v in neighbors[cur]:
                if not current_used[v] and current_block_count[v] == 1:
                    candidates.append(v)
                    
            if not candidates:
                return
                
            rng.shuffle(candidates)
            
            for v in candidates:
                # Add v
                current_path.append(v)
                current_used[v] = True
                for w in neighbors[v]:
                    current_block_count[w] += 1
                    
                dfs(current_path, current_used, current_block_count, depth + 1, time_limit)
                
                # Backtrack
                for w in neighbors[v]:
                    current_block_count[w] -= 1
                current_used[v] = False
                current_path.pop()
                
                if time.time() > time_limit:
                    return

        dfs(path, used, block_count, 0, time_limit)
        return best_local_path
    
    def try_splice(path1, path2, rng):
        """
        Try to splice path2 onto the end of path1.
        path1 and path2 are lists of vertices.
        We try to find a way to connect them: take a suffix of path1 and prefix of path2
        such that the junction is valid.
        
        Simpler approach: try to extend path1 by continuing from its end,
        but allow jumping into path2's vertices if they're compatible.
        
        Actually, let's try a different splicing: 
        Take path1, then try to find a vertex in path2 that can be appended
        to path1's end (i.e., adjacent to path1[-1] and at distance >= 2 from all of path1[:-1]).
        Then continue from there using path2's remaining vertices as a guide.
        
        Even simpler: just try to greedily extend path1, but if we get stuck,
        try starting from different points in path2.
        
        Let's do: for each vertex v in path2 (in order), check if v can be appended
        to path1. If yes, then greedily extend from v using the remaining context.
        """
        if len(path1) < 2 or len(path2) < 2:
            return None
        
        # Try to find a vertex in path2 that can be appended to path1
        # We need: v is adjacent to path1[-1], and dist(v, u) >= 2 for all u in path1[:-1]
        # And v is not in path1.
        
        used1 = set(path1)
        last = path1[-1]
        
        for idx, v in enumerate(path2):
            if v in used1:
                continue
            # Check if v is adjacent to last
            if (v ^ last).bit_count() != 1:
                continue
            # Check if v is at distance >= 2 from all of path1[:-1]
            valid = True
            for u in path1[:-1]:
                if (v ^ u).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Found a valid splice point! Build new path.
            # New path = path1 + [v] + (remaining of path2 after v, but we need to verify each step)
            # Actually, we should greedily extend from v, not just take path2's suffix blindly.
            new_path = list(path1)
            new_path.append(v)
            used_new = set(new_path)
            
            # Now try to greedily extend from v, but prefer vertices from path2[idx+1:] if valid
            remaining = path2[idx+1:]
            cur = v
            
            # Build a set of remaining for quick lookup
            rem_set = set(remaining)
            
            # We need to track block_count for the new path
            # But that's expensive. Let's just do a simple greedy extension.
            while time.time() < deadline - 1.0:
                # Prefer vertices from remaining if they're valid
                candidates = []
                for w in neighbors[cur]:
                    if w in used_new:
                        continue
                    # Check distance >= 2 from all new_path[:-1]
                    ok = True
                    for u in new_path[:-1]:
                        if (w ^ u).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(w)
                
                if not candidates:
                    break
                
                # Prefer candidates that are in remaining
                pref = [c for c in candidates if c in rem_set]
                if pref:
                    v_next = rng.choice(pref)
                else:
                    v_next = rng.choice(candidates)
                
                new_path.append(v_next)
                used_new.add(v_next)
                if v_next in rem_set:
                    rem_set.discard(v_next)
                cur = v_next
            
            if len(new_path) > len(path1):
                return new_path
        
        return None
    
    # Main Loop
    # Phase 1: Collect many greedy snakes
    # Phase 2: Try splicing and backtracking on the best ones
    
    collected_paths = []
    
    for restart in range(100000):
        # Check time
        if time.time() > deadline - 3.0:
            break
        
        rng = random.Random(restart * 7919 + 1)
        start_v = rng.randint(0, (1 << N) - 1)
        
        # Fast greedy extension
        path, used, block_count = greedy_extend(start_v, rng, deadline - 2.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Keep track of good paths for splicing
        if len(path) >= 50 and len(collected_paths) < 50:
            collected_paths.append(list(path))
        
        # Early exit if we find a very long snake
        if best_len >= 190:
            break
    
    # Phase 2: Try backtracking on the best path
    if best_path and time.time() < deadline - 1.0:
        rng = random.Random(12345)
        bt_time = min(10.0, deadline - time.time() - 1.0)
        if bt_time > 1.0:
            # Reconstruct used and block_count from best_path
            used = [False] * (1 << N)
            block_count = [0] * (1 << N)
            for v in best_path:
                used[v] = True
            for v in best_path:
                for w in neighbors[v]:
                    block_count[w] += 1
            
            improved = backtrack_extend(best_path, used, block_count, rng, deadline - 1.0, max_depth=20)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = list(improved)
    
    # Phase 3: Try splicing between collected paths
    if time.time() < deadline - 1.0:
        rng = random.Random(67890)
        for i in range(len(collected_paths)):
            for j in range(len(collected_paths)):
                if i == j:
                    continue
                if time.time() > deadline - 1.0:
                    break
                
                spliced = try_splice(collected_paths[i], collected_paths[j], rng)
                if spliced and len(spliced) > best_len:
                    best_len = len(spliced)
                    best_path = list(spliced)
                    if best_len >= 190:
                        break
            if best_len >= 190:
                break
    
    if best_path is None:
        return [0]
        
    return best_path
