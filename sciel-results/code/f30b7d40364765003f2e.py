import numpy as np
import random
import time
import sys

def construct(k, g):
    deadline = time.time() + 110
    
    # If the request is for the specific 4-regular, girth 10 case,
    # we attempt the 260-vertex voltage lift of PG(2,3).
    if k == 4 and g == 10:
        # Construct PG(2,3) incidence graph
        points = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if x == 0 and y == 0 and z == 0: continue
                    if x != 0:
                        inv_x = 1 if x == 1 else 2
                        p = (1, (y*inv_x)%3, (z*inv_x)%3)
                    elif y != 0:
                        inv_y = 1 if y == 1 else 2
                        p = (0, 1, (z*inv_y)%3)
                    else:
                        p = (0, 0, 1)
                    if p not in points:
                        points.append(p)
        
        lines = []
        line_points = []
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if a == 0 and b == 0 and c == 0: continue
                    if a != 0:
                        inv_a = 1 if a == 1 else 2
                        n = (1, (b*inv_a)%3, (c*inv_a)%3)
                    elif b != 0:
                        inv_b = 1 if b == 1 else 2
                        n = (0, 1, (c*inv_b)%3)
                    else:
                        n = (0, 0, 1)
                    
                    if n in [tuple(l) for l in lines]: continue
                    lines.append(n)
                    pts = []
                    for p in points:
                        if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                            pts.append(p)
                    line_points.append(pts)
        
        base_edges = []
        for i, p in enumerate(points):
            for j, lp in enumerate(line_points):
                if p in lp:
                    base_edges.append((i, 13+j))
        
        m = 10
        n_vertices = 26 * m
        
        # Build adjacency for base graph
        adj = {i: [] for i in range(26)}
        edge_index = {}
        for idx, (u, v) in enumerate(base_edges):
            adj[u].append(v)
            adj[v].append(u)
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
        # Find all non-backtracking closed walks of length 6 and 8
        # A non-backtracking walk is a sequence of vertices v0...vk where v_{i-1} != v_{i+1}
        # and v0 = vk.
        
        walks_6 = set()
        walks_8 = set()
        
        # To avoid duplicates, we can normalize the walk by rotating it to start with the smallest vertex
        # and direction. But for constraints, we just need the set of edge indices with signs.
        # Actually, we just need the sum of voltages.
        # For a walk v0->v1->...->vk, the voltage sum is sum(v_{v_i v_{i+1}}).
        # If we reverse the walk, the sum is negated.
        # So we can store the tuple of edge indices in order.
        
        def get_walk_edges(walk):
            edges = []
            for i in range(len(walk)-1):
                u, v = walk[i], walk[i+1]
                edges.append(edge_index[(u, v)])
            return tuple(edges)
            
        # Enumerate non-backtracking walks
        # Start from each vertex, each neighbor
        for start in range(26):
            for first in adj[start]:
                # Walk of length 6: start -> first -> ... -> start
                # Depth first search
                def dfs_nb(current, prev, length, path):
                    if length == 0:
                        if current == start:
                            # Found a closed walk
                            # Normalize: rotate so smallest vertex is first?
                            # For voltage sum, rotation doesn't change the sum.
                            # Reversal changes sign.
                            # We just need to ensure sum != 0.
                            # If we check sum != 0, we don't need to worry about reversal
                            # because if sum(S) != 0, then sum(-S) != 0.
                            # So we just collect the edge tuples.
                            e_tuple = get_walk_edges(path)
                            if length == 6:
                                walks_6.add(e_tuple)
                            elif length == 8:
                                walks_8.add(e_tuple)
                        return
                    
                    for nb in adj[current]:
                        if nb == prev:
                            continue
                        dfs_nb(nb, current, length - 1, path + [nb])
                
                # Length 6
                dfs_nb(first, start, 5, [start, first])
                # Length 8
                dfs_nb(first, start, 7, [start, first])
                
        # Combine constraints
        # We need sum(voltages[e]) != 0 mod 10 for all walks in walks_6 and walks_8.
        # Note: A walk of length 6 in the base might be a simple 6-cycle or a non-simple one.
        # Both are covered by the DFS.
        
        all_walks = list(walks_6) + list(walks_8)
        
        # Backtracking solver
        def solve_voltages():
            voltages = [-1] * len(base_edges)
            
            # Precompute for each edge, which walks it belongs to
            edge_to_walks = {i: [] for i in range(len(base_edges))}
            for i, w in enumerate(all_walks):
                for e in w:
                    edge_to_walks[e].append(i)
            
            walk_edges_list = [list(w) for w in all_walks]
            
            def backtrack(edge_idx):
                if edge_idx == len(base_edges):
                    return True
                
                if time.time() > deadline:
                    return False
                    
                vals = list(range(m))
                random.shuffle(vals)
                
                for val in vals:
                    voltages[edge_idx] = val
                    valid = True
                    for wi in edge_to_walks[edge_idx]:
                        w = walk_edges_list[wi]
                        s = 0
                        complete = True
                        for e in w:
                            if voltages[e] == -1:
                                complete = False
                                break
                            s += voltages[e]
                        if complete and s % m == 0:
                            valid = False
                            break
                    if valid:
                        if backtrack(edge_idx + 1):
                            return True
                    voltages[edge_idx] = -1
                return False
                
            # Try to solve
            if backtrack(0):
                return voltages
            return None

        voltages = None
        random.seed(42)
        for attempt in range(20):
            if time.time() > deadline:
                break
            voltages = solve_voltages()
            if voltages:
                break
        
        if voltages:
            lift_edges = []
            for u, v in base_edges:
                e_idx = edge_index[(u, v)]
                vu = voltages[e_idx]
                for i in range(m):
                    lift_edges.append((u*m + i, v*m + (i + vu) % m))
                    
            # Verify girth
            adj_lift = {i: [] for i in range(n_vertices)}
            for u, v in lift_edges:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                
            min_girth = 999
            for start in range(n_vertices):
                dist = {start: 0}
                parent = {start: None}
                queue = [start]
                while queue:
                    cur = queue.pop(0)
                    for nb in adj_lift[cur]:
                        if nb not in dist:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            queue.append(nb)
                        elif parent[nb] != cur and parent[cur] != nb:
                            cycle_len = dist[nb] + dist[cur] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                
            if min_girth >= g:
                return n_vertices, lift_edges
            # If girth is not sufficient, we can't use this result
            # But we checked walks up to 8. If girth is 8, we missed a walk.
            # If girth is 10, we are good.
            
    # Fallback: Use the LUW construction from the seed, but ensure it's fast
    # This produces a graph, possibly large, but valid.
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
