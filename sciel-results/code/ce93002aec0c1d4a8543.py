"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def greedy_extend(start_v, rng, time_limit):
        """
        Extend path greedily from start_v using block_count for O(1) validity checks.
        Uses Warnsdorff-like heuristic: prefer candidates with fewest onward options.
        """
        path = [start_v]
        used = [False] * (1 << N)
        used[start_v] = True
        # block_count[v] = number of vertices in path (excluding the last one) that are at distance 1 from v
        # A vertex is a valid next step if not used and block_count[v] == 0
        # Wait, let's re-verify the logic.
        # Condition: new vertex v must be at distance >= 2 from ALL previous vertices except the last one.
        # Let path = [p0, p1, ..., pk]. We want to add v.
        # v must not be in {p0, ..., pk}.
        # v must be at distance >= 2 from p0, ..., p_{k-1}.
        # v is at distance 1 from pk (by definition of neighbor).
        
        # So, we need to track for each candidate v, how many of p0, ..., p_{k-1} are at distance 1 from v.
        # If that count is 0, v is valid.
        
        block_count = [0] * (1 << N)
        
        while time.time() < time_limit:
            cur = path[-1]
            
            # Find valid candidates
            candidates = []
            for v in neighbors[cur]:
                if not used[v] and block_count[v] == 0:
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Heuristic: Choose the candidate that leaves the fewest options for the next step.
            # This is the Warnsdorff rule.
            # To avoid computing this for every candidate every time (expensive),
            # we can just pick randomly if we want speed, or compute it for better quality.
            # Let's try a mix: sometimes random, sometimes Warnsdorff.
            # Actually, computing Warnsdorff is O(N^2) per step, which is fine for N=9.
            # But let's keep it simple first: random selection often works well with restarts.
            # However, Warnsdorff is known to be much better for snakes.
            
            # Let's implement a lightweight Warnsdorff:
            # For each candidate v, count how many of v's neighbors are valid (not used, block_count==0).
            # Note: block_count is current state. If we pick v, the state changes.
            # So we approximate: count available neighbors of v in the CURRENT state.
            
            best_v = None
            min_options = float('inf')
            
            # To speed up, if there are very few candidates, just pick one.
            if len(candidates) <= 2:
                v = rng.choice(candidates)
            else:
                # Sample a few candidates to find a good one? Or check all?
                # Checking all is 9 * 9 = 81 ops. Very fast.
                for v in candidates:
                    options = 0
                    for u in neighbors[v]:
                        if not used[u] and block_count[u] == 0:
                            options += 1
                    if options < min_options:
                        min_options = options
                        best_v = v
                    if min_options == 0:
                        # Dead end next step, but we have no choice?
                        # Actually if min_options is 0, it's a bad move.
                        # But if all are 0, we pick the first one.
                        break
            
            if best_v is None:
                best_v = rng.choice(candidates)
            
            v = best_v
            
            # Add v to path
            path.append(v)
            used[v] = True
            
            # Update block_count:
            # The previous vertex 'cur' (which is now path[-2]) is no longer the "last" vertex.
            # So, any vertex at distance 1 from 'cur' should now be counted as blocked by 'cur'.
            # But wait, 'cur' was already in the path.
            # Before adding v: path = [..., cur]. block_count[x] counted neighbors of path[:-1] = path[:-2].
            # 'cur' was the last vertex, so it was NOT counted in block_count.
            # After adding v: path = [..., cur, v].
            # Now 'cur' is in path[:-1]. So we must increment block_count for all neighbors of 'cur'.
            
            for u in neighbors[cur]:
                block_count[u] += 1
            
        return path
    
    def try_backtrack(path, used, block_count, max_depth, rng, time_limit):
        """
        Limited backtracking to escape local minima.
        """
        # This is complex to implement correctly with block_count updates on backtrack.
        # Simpler approach: just do more greedy restarts.
        # Or, trim the path and re-greedy.
        
        # Let's try a "trim and re-greedy" strategy within the restart loop instead.
        pass

    # Main loop: Multiple restarts
    # We can afford many restarts because greedy is fast.
    
    num_restarts = 0
    while time.time() < deadline - 1.0:
        num_restarts += 1
        rng = random.Random(num_restarts * 7919 + 1)
        
        # Random start vertex
        start_v = rng.randint(0, (1 << N) - 1)
        
        # Run greedy extension
        path = greedy_extend(start_v, rng, deadline - 1.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        # Optional: Try to improve by trimming and re-extending?
        # If we have time, take the best path so far, trim the end, and try different choices.
        # This is a form of local search.
        
        if best_len >= 190:
            break
    
    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
