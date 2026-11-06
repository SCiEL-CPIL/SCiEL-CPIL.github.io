"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = []
    best_len = 0
    
    # Precompute masks for "vertex and its neighbors"
    # mask[v] is a 512-bit integer where bit v and bits of neighbors of v are 1
    # This represents the set of vertices that cannot be visited if v is in the path
    # (because they are either v itself or adjacent to v, violating the induced path property)
    masks = [0] * (1 << N)
    for v in range(1 << N):
        m = 1 << v
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        masks[v] = m
    
    # Precompute neighbors for faster iteration
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def dfs_extend(start, rng, max_time):
        """
        DFS with backtracking and heuristic ordering.
        Returns the longest path found in this specific DFS run.
        """
        path = [start]
        blocked = masks[start]
        current = start
        
        # Local best for this DFS run
        local_best_len = 1
        local_best_path = [start]
        
        # We use a recursive-like loop with an explicit stack to avoid recursion limits,
        # but since depth is max 512, recursion is actually fine and cleaner.
        # However, Python recursion overhead is high. Let's use an iterative approach with backtracking logic
        # or just simple recursion with sys.setrecursionlimit.
        # Given N=9, max depth is 512. Recursion depth 512 is safe.
        
        def _dfs(current, blocked, path):
            nonlocal local_best_len, local_best_path
            
            # Update local best
            if len(path) > local_best_len:
                local_best_len = len(path)
                local_best_path = list(path)
                
            if time.time() > max_time:
                return False # Signal to stop
            
            # Find valid candidates
            candidates = []
            for nb in neighbors[current]:
                if not (blocked >> nb) & 1:
                    candidates.append(nb)
            
            if not candidates:
                return True # Dead end, backtrack
            
            # Heuristic: Sort candidates by "degree" in the remaining graph
            # We want to pick the one with the fewest options? (Warnsdorff) 
            # Or the one with the most options to keep flexibility?
            # In snake problems, usually picking the vertex that constrains the future least is good,
            # but Warnsdorff (fewest options) is often better for Hamiltonian paths to avoid getting stuck in a "loose" end.
            # Let's try: Sort by number of valid next moves from the candidate.
            # Actually, for snakes, we want to avoid isolating vertices.
            # Let's use a simple heuristic: random shuffle if tie, else sort by count of neighbors not in (blocked | masks[candidate])
            
            def get_score(c):
                # Count how many moves are available from c if we go there
                # New blocked would be blocked | masks[current]
                # But we are checking from c, so we check neighbors of c against (blocked | masks[current])
                # Wait, if we move to c, the new blocked set becomes blocked | masks[current].
                # The candidates from c are neighbors of c not in (blocked | masks[current]).
                new_blocked_preview = blocked | masks[current]
                count = 0
                for nb_c in neighbors[c]:
                    if not (new_blocked_preview >> nb_c) & 1:
                        count += 1
                return count

            # Shuffle to break symmetry and randomness
            rng.shuffle(candidates)
            
            # Sort by score (ascending for Warnsdorff-like behavior, or descending?)
            # Let's try ascending (pick the most constrained one first)
            candidates.sort(key=get_score)
            
            for c in candidates:
                path.append(c)
                new_blocked = blocked | masks[current]
                # Note: we must also block c's neighbors? No, the mask logic handles it.
                # When we are at c, the "forbidden" set for the NEXT step is masks[c] union previous forbidden?
                # No. The condition is: next vertex u must not be in path, and not adjacent to any vertex in path.
                # So blocked set should contain all vertices in path AND their neighbors.
                # Initially blocked = masks[start].
                # If we go to c, the new blocked set is (old blocked) | masks[c]?
                # Let's verify.
                # Old blocked contains start and N(start).
                # If we go to c, c is adjacent to start. c is not in old blocked (checked).
                # Now c is in path. So c and N(c) are forbidden.
                # So new_blocked = old_blocked | masks[c].
                # Correct.
                
                if _dfs(c, new_blocked, path):
                    # If time is up, stop
                    if time.time() > max_time:
                        return True
                path.pop()
                
                if time.time() > max_time:
                    return True
            
            return True # All candidates exhausted, backtrack

        _dfs(start, blocked, path)
        
        return local_best_path

    # Multiple restarts
    # Each restart runs a DFS for a short duration
    time_per_restart = 0.2 # 200ms per attempt
    max_restarts = int((deadline - time.time()) / time_per_restart)
    
    # We can also try a few greedy runs first to get a quick baseline
    # But DFS is generally better. Let's just do DFS.
    
    for i in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
            
        rng = random.Random(i * 12345 + 67890)
        start = rng.randint(0, (1 << N) - 1)
        
        # Run DFS with a time limit
        # We pass a deadline for this specific DFS run
        run_deadline = time.time() + time_per_restart
        
        path = dfs_extend(start, rng, run_deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            # Optional: print progress if it's significantly better
            # print(f"Restart {i}: Found length {best_len}")
            
        # If we find a very long snake, we might stop early?
        # The max known is ~190. If we hit 180+, it's good.
        if best_len >= 180:
             # Continue to see if we can go higher, but maybe reduce time per restart?
             # Or just break if we have plenty of time left?
             # Let's keep searching.
             pass

    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
