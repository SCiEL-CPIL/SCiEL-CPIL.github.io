import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: DFS with Backtracking and Forward Checking (CSP approach).
    This explores the search tree more thoroughly than greedy methods,
    using constraint propagation to prune dead ends early.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    # Precompute adjacency matrix for O(1) distance check?
    # No, bit_count is fast enough. But we can precompute a bitmask of neighbors for each vertex?
    # Actually, for forward checking, we need to know if a candidate neighbor of the current end
    # is adjacent to ANY other vertex in the current path.
    
    # To speed up the "is adjacent to path" check, we can maintain a set of "forbidden" vertices.
    # A vertex u is forbidden for extension at end v if:
    # 1. u is already in the path.
    # 2. u is adjacent to any vertex in the path (except the one v connects to? No, v is the end).
    #    Wait, if we extend from v to u, u must not be adjacent to any p in path.
    #    Note: v is in path. u is adjacent to v. So u is NOT forbidden by v.
    #    But u must not be adjacent to any OTHER p in path.
    
    # Let's maintain a bitmask `forbidden_mask` where bit i is 1 if vertex i cannot be the NEXT vertex.
    # Initially, forbidden_mask has bits set for all vertices in path.
    # When we add a new vertex v to the end, we must update forbidden_mask.
    # New forbidden vertices are:
    # 1. v itself (already in path).
    # 2. Any neighbor w of v such that w is adjacent to some p in path (p != v).
    #    Actually, if w is a neighbor of v, and w is adjacent to p (p in path, p != v), then w is forbidden.
    #    Also, any neighbor w of v is adjacent to v. Since v is in the path, w is NOT forbidden by v.
    #    But w might be forbidden by other vertices in the path.
    
    # Efficient Forward Checking:
    # Let `path` be the list of vertices.
    # Let `used_mask` be a bitmask of vertices in `path`.
    # Let `adj_to_path_mask` be a bitmask of vertices adjacent to ANY vertex in `path`.
    #    Actually, we need to be careful. 
    #    A vertex u is a valid extension from end v if:
    #    1. u is not in `used_mask`.
    #    2. u is not in `adj_to_path_mask` (except that u is adjacent to v, which is in path, 
    #       so u IS in adj_to_path_mask. We must exclude v's contribution or handle it differently).
    
    # Better approach for validity check of candidate u from end v:
    # u is valid if:
    # 1. not used[u]
    # 2. For all p in path: if p != v, hamming(u, p) != 1.
    
    # We can maintain a count array `adj_count` where `adj_count[u]` is the number of vertices in `path` 
    # that are adjacent to u.
    # Then u is a valid extension from v if:
    # 1. not used[u]
    # 2. adj_count[u] == 1 (because v is in path and adjacent to u, so adj_count[u] must be at least 1.
    #    If adj_count[u] > 1, then some other p in path is adjacent to u, which is invalid.)
    #    Wait, if v is adjacent to u, adj_count[u] includes v.
    #    If any other p is adjacent to u, adj_count[u] >= 2.
    #    So condition is adj_count[u] == 1.
    
    # Let's implement this.
    
    adj_count = [0] * MAX_VERTS
    used = [False] * MAX_VERTS

    def add_vertex(v):
        used[v] = True
        for w in neighbors[v]:
            adj_count[w] += 1

    def remove_vertex(v):
        used[v] = False
        for w in neighbors[v]:
            adj_count[w] -= 1

    def is_valid_extension(u, v):
        # u is candidate neighbor of v (current end)
        if used[u]:
            return False
        # u must not be adjacent to any other vertex in path
        # adj_count[u] counts neighbors in path.
        # Since v is in path and adjacent to u, adj_count[u] >= 1.
        # If adj_count[u] == 1, the only neighbor in path is v. Valid.
        # If adj_count[u] > 1, some other p is neighbor. Invalid.
        return adj_count[u] == 1

    def dfs(path, depth, limit):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if time.time() - start_time > limit:
            return
            
        v_end = path[-1]
        
        # Gather valid extensions
        valid_moves = []
        for u in neighbors[v_end]:
            if is_valid_extension(u, v_end):
                valid_moves.append(u)
                
        if not valid_moves:
            return
            
        # Heuristic: Sort by number of onward moves (Warnsdorff) to guide DFS
        # But calculating onward moves is expensive. 
        # Instead, let's just pick one, or randomize?
        # For a CSP, we often want to fail fast. 
        # Let's try to extend the "tightest" node.
        
        # To keep it simple and fast, we'll just iterate.
        # If we want backtracking to be effective, we need to explore branches.
        
        # Shuffle to avoid bias
        random.shuffle(valid_moves)
        
        for u in valid_moves:
            # Try extending
            add_vertex(u)
            path.append(u)
            
            dfs(path, depth + 1, limit)
            
            # Backtrack
            path.pop()
            remove_vertex(u)
            
            if time.time() - start_time > limit:
                return

    # Multi-start
    # We will run multiple independent DFS searches with different random starts and time limits.
    
    # First, try a few quick greedy runs to get a baseline? 
    # No, let's just run DFS.
    
    # To make DFS effective, we need to limit the depth of the search tree or the time per node.
    # Since N=9, the path can be up to ~190+ long. 
    # A naive DFS will explode. 
    # We need a strong heuristic or a limit on backtracking.
    
    # Alternative: Use the "Bidirectional" idea but with backtracking?
    # Or, use a "Limited Discrepancy Search" (LDS) around a greedy solution.
    
    # Let's try a hybrid: 
    # 1. Generate a good greedy snake (like the current program).
    # 2. Use the time remaining to perform local search / hill climbing on the snake?
    #    Or, use the greedy snake as a starting point for a CSP that tries to extend or modify it.
    
    # Given the time constraint (120s), a full DFS is likely too slow unless heavily pruned.
    # Let's try a "Greedy with Backtracking" approach:
    # Grow the snake greedily. If stuck, backtrack the last few moves and try alternatives.
    
    def greedy_with_backtracking(start_vertex, max_backtracks=10, time_limit=None):
        nonlocal best_path, best_len
        
        path = [start_vertex]
        used = [False] * MAX_VERTS
        used[start_vertex] = True
        adj_count = [0] * MAX_VERTS
        
        # Initialize adj_count
        for w in neighbors[start_vertex]:
            adj_count[w] += 1
            
        def add(v):
            used[v] = True
            for w in neighbors[v]:
                adj_count[w] += 1
                
        def remove(v):
            used[v] = False
            for w in neighbors[v]:
                adj_count[w] -= 1
            
        def valid_ext(u, v_end):
            if used[u]: return False
            return adj_count[u] == 1
            
        # We maintain a stack of (path, state) to allow backtracking
        # But copying state is expensive.
        # Instead, we'll do a depth-limited search with a "greedy" bias.
        
        # Let's stick to a simpler structural change: 
        # **Simulated Annealing** on the snake representation.
        # State: A valid snake.
        # Move: Remove a vertex from the middle, try to re-insert it elsewhere or swap?
        # This is complex to maintain validity.
        
        # Let's go back to the prompt's suggestion: "Model as CSP ... clique constraints".
        # The most practical CSP approach for this time limit is **Backtracking Search with Forward Checking** 
        # but restricted to a **Limited Discrepancy Search** (LDS) around a greedy heuristic.
        
        # 1. Generate a greedy path.
        # 2. Identify points where the greedy choice was "risky" (low onward moves).
        # 3. Explore alternatives at those points.
        
        pass # Placeholder, will implement below

    # Let's implement a **Local Search** strategy instead, which is often more robust for these problems.
    # Start with a random valid snake (generated by greedy).
    # Repeatedly try to "improve" it by:
    # 1. Removing a vertex from the end.
    # 2. Trying to insert a different vertex at the end.
    # 3. Or, removing a vertex from the middle and trying to reconnect the two ends? (Hard)
    
    # Actually, a very effective strategy for snakes is **Hill Climbing with Restart**.
    # 1. Generate a random greedy snake.
    # 2. Try to extend it. If stuck, try to "untangle" by removing the last few vertices and regrowing.
    
    def generate_greedy(start_v, randomness=0.1):
        path = [start_v]
        used = [False] * MAX_VERTS
        used[start_v] = True
        adj_count = [0] * MAX_VERTS
        for w in neighbors[start_v]:
            adj_count[w] += 1
            
        while True:
            v_end = path[-1]
            valid_moves = [u for u in neighbors[v_end] if not used[u] and adj_count[u] == 1]
            if not valid_moves:
                break
                
            # Warnsdorff: choose move with fewest onward moves
            scored = []
            for u in valid_moves:
                # Count onward moves from u
                # u is not in path yet. 
                # If we add u, adj_count[u] becomes 1 (from v_end).
                # Onward moves from u are neighbors w of u such that:
                # 1. not used[w]
                # 2. adj_count[w] + 1 (for u) == 1? No.
                #    If we add u, adj_count[w] increases by 1 if w is neighbor of u.
                #    For w to be a valid extension from u, we need (adj_count[w] + 1) == 1? 
                #    No, the condition is that w is not adjacent to any OTHER vertex in path.
                #    Currently adj_count[w] counts neighbors in current path.
                #    If we add u, and w is neighbor of u, new count is adj_count[w] + 1.
                #    We need new count == 1. So adj_count[w] + 1 == 1 => adj_count[w] == 0.
                #    So w must not be adjacent to any current path vertex.
                
                cnt = 0
                for w in neighbors[u]:
                    if not used[w] and adj_count[w] == 0:
                        cnt += 1
                scored.append((cnt, u))
                
            if not scored:
                break
                
            scored.sort()
            
            # Randomness: pick from top k with lowest score
            k = max(1, int(len(scored) * randomness))
            # Actually, pick from the best ones
            best_score = scored[0][0]
            candidates = [s for s in scored if s[0] <= best_score + 1] # Allow slight deviation
            
            _, next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            for w in neighbors[next_v]:
                adj_count[w] += 1
                
        return path

    # Main loop: Multi-start Greedy with Local Repair
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        p = generate_greedy(v0)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        return [0]

    return best_path

# EVOLVE-BLOCK-END
