"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest (most constrained first)
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)
    
    # Track used edges in K20 (as frozensets of vertex pairs)
    used_edges = set()
    result = [None] * 20
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    for idx in order:
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex, no edges. Pick any available vertex.
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    ek = edge_key(u, w)
                    if ek not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add(ek)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use backtracking to embed the tree
        tree_n = k
        tree_edge_list = tree_edges
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Get BFS order of tree vertices starting from 0
        visited = [False] * tree_n
        order_list = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order_list.append(v)
                    q.append(v)
        
        # Backtracking: assign host vertices to tree vertices in BFS order
        assignment = [None] * tree_n
        # Precompute for each tree vertex in BFS order, which neighbors come before it
        # (i.e., are already assigned when we reach this vertex)
        pos_in_order = [0] * tree_n
        for i, v in enumerate(order_list):
            pos_in_order[v] = i
        
        # Precompute available host vertices for each tree vertex position
        # to enable better ordering and pruning
        def get_available_hosts(tree_v, pos):
            """Get list of valid host vertices for tree_v at position pos, sorted by constraint."""
            candidates = []
            assigned_set = set(assignment)
            for host_v in range(20):
                if host_v in assigned_set:
                    continue
                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in tree_adj[tree_v]:
                    if pos_in_order[nb] < pos:  # nb is already assigned
                        ek = edge_key(assignment[nb], host_v)
                        if ek in used_edges:
                            valid = False
                            break
                if valid:
                    # Score: prefer vertices with fewer unused edges to other unassigned tree vertices
                    # This helps pack edges densely
                    score = 0
                    for nb in tree_adj[tree_v]:
                        if pos_in_order[nb] > pos:  # nb not yet assigned
                            # Count how many of host_v's edges to other unassigned hosts are used
                            pass  # Simplified: just use degree-based heuristic
                    candidates.append((score, host_v))
            # Sort by score (lower score = more constrained = try first)
            candidates.sort()
            return [h for _, h in candidates]
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = order_list[pos]
            # Get available hosts with better ordering
            candidates = get_available_hosts(tree_v, pos)
            
            for host_v in candidates:
                # Tentatively assign
                assignment[tree_v] = host_v
                # Mark edges as temporarily used
                temp_edges = []
                for nb in tree_adj[tree_v]:
                    if pos_in_order[nb] < pos:
                        ek = edge_key(assignment[nb], host_v)
                        temp_edges.append(ek)
                        used_edges.add(ek)
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                for ek in temp_edges:
                    used_edges.discard(ek)
                assignment[tree_v] = None
            
            return False
        
        if backtrack(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
