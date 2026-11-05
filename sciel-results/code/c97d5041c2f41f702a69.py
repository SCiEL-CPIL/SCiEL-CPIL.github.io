"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest snake in Q_9.
    Uses a greedy approach with an optimized 'forbidden neighbors' set to ensure O(1) validity checks.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_len = 0
    best_path = []
    
    # Precompute neighbors for all vertices to speed up iteration
    # neighbors[v] is a list of 9 integers
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(512)]
    
    while time.time() < deadline:
        # Start from a random vertex
        start = random.randint(0, 511)
        
        path = [start]
        used = {start}
        
        # forbidden[u] is True if u cannot be the next vertex in the path
        # because it is at distance 1 from some vertex in the path (other than the last one)
        # Initially, all neighbors of start are forbidden? No.
        # The condition is: new vertex v must be dist >= 2 from all path[i] for i < len(path)-1.
        # So, when we add a vertex v to the path, all neighbors of v (except the one we came from)
        # become "forbidden" for future steps.
        
        # Let's maintain a set `blocked` of vertices that cannot be chosen next.
        # If we are at `cur`, we can only choose `nxt` if `nxt` is not in `used` and `nxt` is not in `blocked`.
        # When we move from `cur` to `nxt`:
        # 1. `nxt` becomes the new current.
        # 2. `nxt` is added to `used`.
        # 3. All neighbors of `cur` (except `nxt`) are added to `blocked`. Why?
        #    If a future vertex `z` is a neighbor of `cur`, then dist(z, cur) = 1.
        #    Since `cur` is no longer the last vertex, `z` would be at distance 1 from a non-consecutive vertex `cur`.
        #    Thus `z` is invalid.
        
        blocked = set()
        
        # Initial state: We are at `start`.
        # No previous vertex, so no neighbors of a previous vertex are blocked yet?
        # Wait, the constraint is for non-consecutive vertices.
        # Path: [v0, v1, ...]
        # v_k must be dist >= 2 from v_i for all i < k-1.
        # So v_k cannot be a neighbor of any v_i for i < k-1.
        
        # Let's trace:
        # Path: [v0]
        # Next v1: must be neighbor of v0. No other vertices. So any neighbor is ok.
        # Path: [v0, v1]
        # Next v2: must be neighbor of v1. Must be dist >= 2 from v0.
        # So v2 cannot be a neighbor of v0.
        
        # Algorithm:
        # 1. Start with v0.
        # 2. Candidates for v1: neighbors of v0.
        # 3. Pick v1.
        # 4. Update blocked: All neighbors of v0 except v1 are now blocked.
        #    (Because if we pick a neighbor of v0 later, it would be dist 1 from v0, which is non-consecutive at that point).
        # 5. Candidates for v2: neighbors of v1 that are not in used and not in blocked.
        # 6. Pick v2.
        # 7. Update blocked: All neighbors of v1 except v2 are now blocked.
        
        # Let's implement this.
        
        cur = start
        # We need to handle the first step specially or initialize blocked correctly.
        # For the first step, `blocked` is empty.
        # After moving to `nxt`, we add `neighbors[cur] \ {nxt}` to `blocked`.
        
        # To allow backtracking or different choices, we just run greedy multiple times.
        # To make it more effective, we can randomize the order of neighbors.
        
        # Shuffle neighbors for randomization?
        # Instead of shuffling the list, we can pick a random valid neighbor.
        
        while True:
            if time.time() > deadline:
                break
                
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if nb not in used and nb not in blocked:
                    candidates.append(nb)
            
            if not candidates:
                break
            
            # Pick a candidate
            # Simple greedy: pick first. Better: pick random to explore different paths.
            nxt = random.choice(candidates)
            
            path.append(nxt)
            used.add(nxt)
            
            # Update blocked set
            # All neighbors of the previous current (cur) except the one we just moved to (nxt)
            # become blocked.
            for nb in neighbors[cur]:
                if nb != nxt:
                    blocked.add(nb)
                    
            cur = nxt
            
        # End of this greedy run
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
    return best_path
# EVOLVE-BLOCK-END
