"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
import numpy as np

# Precompute neighbors
neighbors_list = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    for b in range(N):
        u = v ^ (1 << b)
        neighbors_list[v].append(u)

# adj_mask[v] is an integer with bits set for neighbors of v
adj_mask = [0] * TOTAL_VERTICES
for v in range(TOTAL_VERTICES):
    mask = 0
    for u in neighbors_list[v]:
        mask |= (1 << u)
    adj_mask[v] = mask

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

def _greedy_construct(start_vertex=0, random_seed=42):
    """
    Fast greedy construction using a 'most constrained' heuristic.
    This serves as a baseline and for initial bounds.
    """
    rng = random.Random(random_seed)
    path = [start_vertex]
    used = 1 << start_vertex
    forbidden = 0
    
    while True:
        cur = path[-1]
        candidates = []
        for v in neighbors_list[cur]:
            bit = 1 << v
            if (used & bit) == 0 and (forbidden & bit) == 0:
                # Score: number of future options
                new_forbidden = forbidden | adj_mask[cur]
                new_used = used | bit
                options = 0
                for w in neighbors_list[v]:
                    wbit = 1 << w
                    if (new_used & wbit) == 0 and (new_forbidden & wbit) == 0:
                        options += 1
                candidates.append((options, v))
        
        if not candidates:
            break
        
        # Pick the one with fewest options (most constrained)
        candidates.sort(key=lambda x: x[0])
        best_val, best_v = candidates[0]
        
        path.append(best_v)
        used |= (1 << best_v)
        forbidden |= adj_mask[cur]
        
    return path

def _max_clique_solve(deadline):
    """
    Finds the maximum clique in the line graph of the hypercube?
    No, the problem is finding the longest induced path in the hypercube.
    This is equivalent to finding the longest path in the graph where edges exist
    only between vertices at distance 1, subject to the induced constraint.
    
    However, a simpler approach for "longest induced path" is often solved by
    branch and bound on the path extension, which is what the previous code did.
    
    Let's try a different angle: The "snake" is a set of vertices forming a path.
    We can model this as a Maximum Independent Set problem on a specific conflict graph?
    Or just a very optimized Branch and Bound for Longest Path.
    
    Given the 150s limit and N=9 (512 vertices), a generic Max Clique solver on the
    "compatibility graph" (where edges mean "can be consecutive") is essentially
    Longest Path.
    
    Let's implement a highly optimized DFS for Longest Path with:
    1. Bitset representation of available vertices.
    2. Pruning based on upper bound (current_len + max_possible_remaining).
    3. Heuristic ordering of neighbors.
    """
    best_path = []
    best_len = 0
    
    # Precompute neighbor lists sorted by degree or some heuristic?
    # For N=9, degree is always 9.
    
    # We will use a recursive DFS with manual stack to avoid recursion limits
    # and allow for better memory management if needed, but recursion is fine for depth < 512.
    
    # To speed up, we can use bitsets for `used` and `forbidden`.
    # `available` = ~(used | forbidden)
    
    # Upper bound heuristic:
    # A simple upper bound is the number of available vertices.
    # A tighter bound: We can estimate the max path length from the current vertex
    # given the available set. This is expensive.
    # Let's stick to: if current_len + available_count <= best_len, prune.
    
    # We'll try multiple starting vertices.
    # Since the hypercube is symmetric, starting at 0 is WLOG for the *set* of vertices,
    # but the *path* structure matters. However, any longest path can be reversed.
    # So we can fix the start vertex to 0 to reduce search space by factor 512?
    # Wait, a path is a sequence. If a longest path exists, it has two ends.
    # One of the ends could be 0? No.
    # But we can try starting at 0, and also try starting at other vertices if time permits.
    
    # Let's start with 0.
    start_v = 0
    
    # State: (current_v, path_list, used_mask, forbidden_mask)
    # We use a stack to simulate recursion.
    # To improve pruning, we can pre-compute connected components of the available subgraph?
    # That's too slow per step.
    
    # Alternative: Use the "coloring" heuristic from Max Clique solvers.
    # If we map the problem to Max Clique on a graph G' where vertices are the same,
    # and edges in G' exist if (u,v) is an edge in Q_N AND no chord is formed...
    # This is not a static graph. The "forbidden" set changes.
    
    # Let's stick to a robust Branch and Bound for Longest Path.
    
    # Optimization: 
    # 1. Use bit_count() for fast counting.
    # 2. Order neighbors by a heuristic (e.g., degree in the *remaining* graph).
    
    def dfs(current_v, path, used_mask, forbidden_mask):
        nonlocal best_path, best_len
        
        current_len = len(path)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_path = path[:]
            
        # Pruning: 
        # Max possible extension is the number of available vertices.
        # available_mask = ~(used_mask | forbidden_mask) & ((1 << TOTAL_VERTICES) - 1)
        # But we can't just use count, because they must be connected to current_v.
        # A better bound: The size of the connected component of current_v in the available graph.
        # Computing this is O(V+E). For N=9, V=512, E=2304. This is fast enough (~microseconds).
        
        # Let's compute the reachable set from current_v using only available vertices.
        # If reachable_count + current_len - 1 <= best_len, prune.
        # Note: current_v is already in path, so we count neighbors.
        
        available_mask = ~(used_mask | forbidden_mask) & ((1 << TOTAL_VERTICES) - 1)
        
        # BFS/DFS to find reachable vertices from current_v within available_mask
        # We can do this with a bitset flood fill.
        reachable = 0
        frontier = 1 << current_v
        while frontier:
            # Expand frontier
            # For each v in frontier, add neighbors(v) & available_mask
            # This is a bit expensive if done naively.
            
            # Optimized flood fill:
            # neighbors_of_frontier = 0
            # for v in bits(frontier): neighbors_of_frontier |= adj_mask[v]
            # new_frontier = (neighbors_of_frontier & available_mask) & ~reachable
            
            # To iterate bits:
            f = frontier
            nbrs = 0
            while f:
                lsb = f & -f
                idx = lsb.bit_length() - 1
                nbrs |= adj_mask[idx]
                f ^= lsb
            
            new_reach = (nbrs & available_mask) & ~reachable
            if not new_reach:
                break
            
            reachable |= new_reach
            frontier = new_reach
        
        max_possible = current_len + reachable.bit_count()
        if max_possible <= best_len:
            return

        # Generate candidates
        candidates = []
        for v in neighbors_list[current_v]:
            bit = 1 << v
            if (available_mask & bit):
                candidates.append(v)
        
        if not candidates:
            return
        
        # Heuristic: Sort candidates by "degree" in the remaining graph (ascending)
        # i.e., try the most constrained ones first.
        def get_degree(v, avail):
            deg = 0
            for u in neighbors_list[v]:
                if (avail & (1 << u)):
                    deg += 1
            return deg
        
        # To save time, we might not sort every time if the list is small (max 9).
        # Sorting 9 items is cheap.
        candidates.sort(key=lambda v: get_degree(v, available_mask))
        
        for v in candidates:
            new_path = path + [v]
            new_used = used_mask | (1 << v)
            # The forbidden set for the NEXT step includes neighbors of the PREVIOUS vertex (current_v)
            # because they would form a chord with v if visited later?
            # Wait. The induced path condition:
            # Path: v0, v1, ..., vk.
            # vi and vj (|i-j|>1) must not be adjacent.
            # When we extend by v_{k+1}, we must ensure v_{k+1} is not adjacent to v0...v_{k-1}.
            # So forbidden_mask must contain neighbors of v0...v_{k-1}.
            # In our state, forbidden_mask currently contains neighbors of v0...v_{k-1}.
            # When we move to v_k (which is `current_v`), the new forbidden set for the next step
            # should include neighbors of v_{k-1} (which were already in forbidden) AND neighbors of v_k.
            # Wait, let's trace:
            # Step 0: v0. forbidden = 0.
            # Step 1: pick v1. forbidden becomes N(v0). (v1 cannot be adjacent to v0, which is true by edge def, but future nodes cannot be).
            # Step 2: pick v2 from N(v1) \ (used U N(v0)).
            # So, when at `current_v` (which is v_k), `forbidden_mask` contains N(v_0) U ... U N(v_{k-1}).
            # We pick v_{k+1}.
            # New forbidden for next step (v_{k+2}) should be N(v_0) U ... U N(v_{k-1}) U N(v_k).
            # So we update: new_forbidden = forbidden_mask | adj_mask[current_v]
            
            new_forbidden = forbidden_mask | adj_mask[current_v]
            
            dfs(v, new_path, new_used, new_forbidden)
            
            # Check time
            if time.time() > deadline:
                return

    # Start search
    # We can try to improve the initial bound with a greedy pass.
    initial_path = _greedy_construct(0, 42)
    best_len = len(initial_path)
    best_path = initial_path
    
    # Run DFS
    # To avoid recursion depth issues (though 512 is safe in Python if limit is raised),
    # and for speed, we might need to be careful.
    # Let's increase recursion limit just in case.
    sys.setrecursionlimit(10000)
    
    # We can also try starting from other vertices if time allows.
    # But let's focus on making the search from 0 as deep as possible first.
    
    dfs(0, [0], 1, 0)
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Get a good initial solution
    best_path = _greedy_construct(0, 42)
    best_len = len(best_path)
    
    # 2. Try to improve with Branch and Bound
    # The B&B might take a long time. We run it until deadline.
    # We pass the deadline to the inner function.
    
    # We need to modify _max_clique_solve to respect the deadline and return best found.
    # The current _max_clique_solve does this.
    
    # However, the recursion in _max_clique_solve might be too slow in Python for deep searches.
    # Let's check if we can do a simpler local search or VNS.
    
    # Given the previous score of 1.49 (155 vertices) with backtracking, 
    # and the theoretical max is ~200, there is room.
    
    # Let's try a different heuristic for the B&B: 
    # Instead of just "most constrained", try "most central" or "furthest".
    
    # For now, let's rely on the B&B.
    # If it's too slow, we fall back to the greedy.
    
    try:
        # We need a version of the solver that returns the best path found so far
        # and can be interrupted.
        # The _max_clique_solve above does this.
        
        # To allow multiple starts or restarts, we can run it with a shorter time slice
        # and keep the best.
        
        # Let's run it once for the remaining time.
        path = _max_clique_solve(deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    except Exception as e:
        # If B&B fails or errors, keep the greedy result
        pass

    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback
    return _greedy_construct(0, 42)

# EVOLVE-BLOCK-END
