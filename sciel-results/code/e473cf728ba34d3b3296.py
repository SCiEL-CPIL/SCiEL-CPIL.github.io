"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed); trees = []
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest (most constrained first)
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)
    
    # Track used edges in K20 (as set of (min, max) tuples)
    used_edges = set()
    result = [None] * 20
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    # Precompute all edges in K20 for quick availability checks
    # We can just check (u, v) not in used_edges
    
    def get_host_degree_available(h):
        """Count how many edges incident to h are still available"""
        count = 0
        for v in range(20):
            if v == h: continue
            if edge_key(h, v) not in used_edges:
                count += 1
        return count

    for idx in order:
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex. Pick any vertex.
            # To be safe, pick one with max available edges (though it has no edges)
            # Actually, for T1, any vertex works. Let's pick 0 if available, else next.
            # But T1 consumes no edges. So any vertex is fine.
            # However, we must ensure the vertex is distinct? No, T1 is just a vertex.
            # The problem says "list of k distinct int host vertices".
            # For k=1, it's a list of 1 vertex.
            # We should pick a vertex that doesn't cause issues for others?
            # Since it uses no edges, it doesn't block anything.
            # But the vertex itself is "used" in the sense that it's the image of T1's vertex 0.
            # Wait, the trees are edge-disjoint. Vertices can be shared?
            # "Embedded trees must be edge-disjoint in K20."
            # It does NOT say vertex-disjoint.
            # So T1 can map to any vertex.
            # Let's map T1 to 0.
            result[idx] = [0]
            continue
        
        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            # To preserve flexibility, pick an edge where both endpoints have high remaining degree?
            # Or just any edge.
            # Let's pick an edge that is "central" or just the first available.
            # Better: pick an edge that minimizes the loss of high-degree vertices?
            # Actually, let's just pick the first available edge to save time, 
            # but maybe prioritize edges between high-degree nodes?
            # For now, simple greedy: first available.
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
        
        # For k >= 3, use backtracking
        tree_n = k
        tree_edge_list = tree_edges
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Determine vertex order for embedding
        # Root at 0. BFS order.
        visited = [False] * tree_n
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)
        
        # Parent array based on BFS
        parent = [-1] * tree_n
        for i in range(1, tree_n):
            v = bfs_order[i]
            for nb in tree_adj[v]:
                if nb == 0 or (nb != 0 and parent[v] == -1 and visited[nb] and bfs_order.index(nb) < bfs_order.index(v)):
                     # Simpler: just find the neighbor that appeared earlier in bfs_order
                     pass
            # Let's just do a proper parent tracking during BFS
        # Redo BFS with parent tracking
        parent = [-1] * tree_n
        visited = [False] * tree_n
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    bfs_order.append(v)
                    q.append(v)
        
        # Children list
        children = [[] for _ in range(tree_n)]
        for v in range(1, tree_n):
            children[parent[v]].append(v)
            
        # Subtree sizes (for heuristic)
        subtree_size = [1] * tree_n
        for i in range(tree_n - 1, -1, -1):
            v = bfs_order[i]
            for c in children[v]:
                subtree_size[v] += subtree_size[c]
                
        # Assignment array: tree_vertex -> host_vertex
        assignment = [None] * tree_n
        assigned_count = 0
        
        # Symmetry breaking: Fix tree vertex 0 to host vertex 0
        # This reduces search space significantly.
        assignment[0] = 0
        assigned_count = 1
        
        # Check if host 0 has enough available edges for its neighbors in the tree
        # Neighbors of 0 in tree:
        # If T_k has degree d at 0, we need d available edges from host 0.
        # If not, we can't start with 0->0. But we can try other host vertices for 0?
        # Yes, if 0->0 fails, we should try 0->1, 0->2, etc.
        # However, symmetry breaking is powerful. Let's try 0->0 first.
        # If it fails, we can fall back to trying other hosts for root.
        
        # Let's implement a general backtracking that tries all hosts for the root,
        # but we can optimize by trying 0 first.
        
        def backtrack():
            nonlocal assigned_count
            if time.time() > deadline:
                return False
            if assigned_count == tree_n:
                return True
            
            # Find next unassigned vertex
            # We use BFS order, so we assign in that order.
            # But we can use MRV on the unassigned vertices whose parents are assigned.
            
            best_v = -1
            best_avail = []
            min_avail_count = 21
            
            # Candidates are unassigned vertices whose parent is assigned
            for v in range(1, tree_n):
                if assignment[v] is not None:
                    continue
                if parent[v] is not None and assignment[parent[v]] is None:
                    continue
                
                # Calculate available hosts for v
                avail = []
                parent_host = assignment[parent[v]] if parent[v] is not None else None
                
                for h in range(20):
                    if any(a == h for a in assignment if a is not None):
                        continue
                    
                    valid = True
                    # Check edge to parent
                    if parent_host is not None:
                        ek = edge_key(parent_host, h)
                        if ek in used_edges:
                            valid = False
                    else:
                        # Root case handled by initial assignment, so this shouldn't happen for v>0
                        pass
                    
                    if valid:
                        # Forward checking: check if neighbors of v (in tree) can be placed
                        # This is expensive. Let's do a lighter check first.
                        # Check if there are enough available edges from h for the degree of v in tree?
                        # No, just check if the specific edges to already assigned neighbors are free.
                        # For now, just add to avail.
                        avail.append(h)
                
                if len(avail) < min_avail_count:
                    min_avail_count = len(avail)
                    best_v = v
                    best_avail = avail
                    if min_avail_count == 0:
                        return False
                    if min_avail_count == 1:
                        break
            
            if best_v == -1:
                return False
            
            # Sort hosts by "constraint": hosts with fewer remaining available edges are more constrained
            # We want to try them first? Or last?
            # Usually MRV says pick the value that leaves the least options for others.
            # So try hosts with LOWEST remaining degree first.
            
            def host_constraint(h):
                # Count available edges incident to h
                cnt = 0
                for v2 in range(20):
                    if v2 == h: continue
                    if edge_key(h, v2) not in used_edges:
                        cnt += 1
                return cnt
            
            # Sort best_avail by constraint (ascending: most constrained first)
            best_avail.sort(key=lambda h: host_constraint(h))
            
            for h in best_avail:
                assignment[best_v] = h
                assigned_count += 1
                temp_edges = []
                # Add edges to used_edges
                if parent[best_v] is not None:
                    ph = assignment[parent[best_v]]
                    ek = edge_key(ph, h)
                    used_edges.add(ek)
                    temp_edges.append(ek)
                
                # Check if this assignment leads to a dead end immediately for children?
                # Optional forward check:
                # For each child c of best_v, is there at least one available host?
                # This is strong pruning.
                fail = False
                for c in children[best_v]:
                    if assignment[c] is None:
                        # Check if any host is available for c
                        has_avail = False
                        for hc in range(20):
                            if any(a == hc for a in assignment if a is not None):
                                continue
                            ek_c = edge_key(h, hc)
                            if ek_c not in used_edges:
                                has_avail = True
                                break
                        if not has_avail:
                            fail = True
                            break
                
                if not fail and backtrack():
                    return True
                
                # Undo
                for ek in temp_edges:
                    used_edges.discard(ek)
                assignment[best_v] = None
                assigned_count -= 1
                
            return False
        
        # Try embedding
        # We fixed 0->0. If that fails, we should try other hosts for 0.
        # But fixing 0->0 is a huge symmetry break.
        # Let's try 0->0 first.
        
        # Reset assignment for the try
        assignment = [None] * tree_n
        assignment[0] = 0
        assigned_count = 1
        
        # Check if 0->0 is valid (enough edges for neighbors of 0 in tree)
        # Neighbors of 0 in tree:
        deg_0_tree = len(tree_adj[0])
        avail_edges_0 = 0
        for v in range(1, 20):
            if edge_key(0, v) not in used_edges:
                avail_edges_0 += 1
        
        if avail_edges_0 >= deg_0_tree:
            if backtrack():
                result[idx] = assignment[:]
            else:
                # If 0->0 failed, try other hosts for 0?
                # This is expensive. Let's try a few random ones or 1, 2...
                # But symmetry breaking is key. If 0->0 fails, the tree might not be packable
                # in the remaining graph with that specific structure.
                # Let's try host 1 for tree vertex 0.
                found = False
                for root_host in range(1, 20):
                    if time.time() > deadline: break
                    
                    # Check if root_host has enough edges
                    avail = 0
                    for v in range(20):
                        if v == root_host: continue
                        if edge_key(root_host, v) not in used_edges:
                            avail += 1
                    if avail < deg_0_tree:
                        continue
                    
                    assignment = [None] * tree_n
                    assignment[0] = root_host
                    assigned_count = 1
                    
                    if backtrack():
                        result[idx] = assignment[:]
                        found = True
                        break
                if not found:
                    result[idx] = None
        else:
            # 0 doesn't have enough edges. Try other roots.
            found = False
            for root_host in range(20):
                if time.time() > deadline: break
                avail = 0
                for v in range(20):
                    if v == root_host: continue
                    if edge_key(root_host, v) not in used_edges:
                        avail += 1
                if avail < deg_0_tree:
                    continue
                
                assignment = [None] * tree_n
                assignment[0] = root_host
                assigned_count = 1
                
                if backtrack():
                    result[idx] = assignment[:]
                    found = True
                    break
            if not found:
                result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
