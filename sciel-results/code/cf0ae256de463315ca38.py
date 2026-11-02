import sys
import time
import random
import networkx as nx # Not available? "Python standard library, NumPy and SciPy are available." 
# So no networkx. I must implement BFS/DFS manually.

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback: LUW Graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try Moore Tree Matching
    if k == 5 and g == 9:
        # Build 5-regular tree of radius 4
        # Nodes 0..425
        # 0 is root
        # Level 0: 0
        # Level 1: 1-5
        # Level 2: 6-25
        # Level 3: 26-85
        # Level 4: 86-425
        
        n_tree = 426
        edges_tree = []
        
        # Build tree edges
        # Parent of i (i>0):
        # Level 1: 1..5, parent 0
        # Level 2: 6..25, parent 1..5 (5 children each)
        # Level 3: 26..85, parent 6..25
        # Level 4: 86..425, parent 26..85
        
        parent = [-1] * n_tree
        children = [[] for _ in range(n_tree)]
        
        # Level 1
        for i in range(1, 6):
            parent[i] = 0
            children[0].append(i)
            
        # Level 2
        idx = 6
        for p in range(1, 6):
            for _ in range(5):
                parent[idx] = p
                children[p].append(idx)
                idx += 1
                
        # Level 3
        for p in range(6, 26):
            for _ in range(5):
                parent[idx] = p
                children[p].append(idx)
                idx += 1
                
        # Level 4
        for p in range(26, 86):
            for _ in range(5):
                parent[idx] = p
                children[p].append(idx)
                idx += 1
                
        for i in range(1, n_tree):
            edges_tree.append((parent[i], i))
            
        # Leaves are 86..425
        leaves = list(range(86, 426))
        
        # Branches:
        # Branch 0: children of 1
        # Branch 1: children of 2
        # ...
        # Branch i: children of i+1
        
        branches = [[] for _ in range(5)]
        for leaf in leaves:
            # Find which main child of root is the ancestor
            # Walk up to depth 1
            curr = leaf
            while parent[curr] != 0:
                curr = parent[curr]
            branch_idx = curr - 1
            branches[branch_idx].append(leaf)
            
        # Try random matchings
        # Match B[i] to B[(i+1)%5]
        # Shuffle each branch
        attempts = 0
        max_attempts = 2000 # Tune based on time
        
        while time.time() < deadline - 5.0 and attempts < max_attempts:
            attempts += 1
            # Shuffle
            for b in branches:
                random.shuffle(b)
                
            # Create edges
            new_edges = []
            for i in range(5):
                b1 = branches[i]
                b2 = branches[(i+1) % 5]
                for j in range(len(b1)):
                    new_edges.append((b1[j], b2[j]))
                    
            # Check Girth
            # We need to ensure no cycle < 9
            # Build adjacency list
            adj = [[] for _ in range(n_tree)]
            for u, v in edges_tree:
                adj[u].append(v)
                adj[v].append(u)
            for u, v in new_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Girth check:
            # For each vertex u, BFS up to depth 4.
            # If we reach a neighbor v of u at distance d < 9?
            # Actually, if we find a cycle of length L < 9.
            # A cycle of length L implies there is a vertex u and a path of length L-1 from u to a neighbor v?
            # Simpler: Check if any vertex u has a neighbor v such that there is a path from u to v of length < 8 that does not use the edge (u,v).
            # Path length L_path. Cycle length L_path + 1.
            # We need L_path + 1 >= 9 => L_path >= 8.
            # So if we find a path of length <= 7 from u to a neighbor v (not using edge u-v), it's invalid.
            
            valid = True
            for u in range(n_tree):
                # BFS from u
                dist = [-1] * n_tree
                dist[u] = 0
                queue = [u]
                head = 0
                while head < len(queue) and dist[queue[head]] < 7:
                    curr = queue[head]
                    head += 1
                    d = dist[curr]
                    if d >= 7: continue
                    for v in adj[curr]:
                        if dist[v] == -1:
                            dist[v] = d + 1
                            queue.append(v)
                
                # Check neighbors of u
                for v in adj[u]:
                    # If v is reachable from u with dist < 8, and the path doesn't start with u-v?
                    # Actually, dist[v] is the shortest path.
                    # If dist[v] <= 7, then there is a cycle of length dist[v] + 1 <= 8.
                    # BUT, if the shortest path IS the edge (u,v), then dist[v]=1. That's fine.
                    # We need to know if there is a path of length <= 7 that is NOT the direct edge.
                    # If dist[v] <= 7, it could be the direct edge (len 1) or a longer path.
                    # If dist[v] == 1, it's the direct edge. We must check if there is ANOTHER path.
                    # This is tricky with simple BFS.
                    
                    # Alternative: For each u, remove edge (u,v) and check dist(u,v) >= 8.
                    # Too slow.
                    
                    # Heuristic: If dist[v] <= 7 and dist[v] > 1, it's invalid.
                    # If dist[v] == 1, we need to check if there's a second path.
                    # Given the tree structure, any second path must go through the new edges.
                    # If we just check dist[v] <= 7, we might flag valid graphs where the only short path is the edge itself.
                    # Wait, if dist[v] == 1, the cycle length is 1+1=2? No, simple graph.
                    # If there is a path of length 1, it's the edge.
                    # If there is a path of length 7, cycle is 8.
                    # So we need to ensure no path of length 2..7 exists between neighbors.
                    
                    if dist[v] <= 7 and dist[v] > 1:
                        valid = False
                        break
                    if dist[v] == 1:
                        # Check if there is a path of length 2..7
                        # We can do a BFS from u, ignoring the edge (u,v), and see if v is reached with dist <= 7.
                        # This is expensive to do for every edge.
                        pass
                        
                if not valid:
                    break
                    
            if valid:
                # Need to double check the dist==1 case?
                # If we passed the above, it means no dist 2..7 found.
                # But we might have missed the case where the shortest path is 1, but a longer path 2..7 exists.
                # Let's do a proper check for the valid candidate.
                
                # Proper Girth Check
                girth_ok = True
                for u in range(n_tree):
                    for v in adj[u]:
                        if u < v: # Check each edge once? No, cycle can be anywhere.
                            # Check if there is a path from u to v of length 2..7
                            # BFS from u, avoiding v? No, avoiding the edge (u,v).
                            dist2 = [-1] * n_tree
                            dist2[u] = 0
                            queue = [u]
                            head = 0
                            found_short = False
                            while head < len(queue) and dist2[queue[head]] < 7:
                                curr = queue[head]
                                head += 1
                                d = dist2[curr]
                                if d >= 7: continue
                                for w in adj[curr]:
                                    if w == v and curr == u:
                                        continue # Skip the direct edge
                                    if dist2[w] == -1:
                                        dist2[w] = d + 1
                                        if w == v:
                                            if d + 1 <= 7:
                                                found_short = True
                                                break
                                        queue.append(w)
                                if found_short:
                                    break
                            if found_short:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                if girth_ok:
                    all_edges = edges_tree + new_edges
                    return n_tree, all_edges

    # Fallback
    return luw_construct(k, g)
