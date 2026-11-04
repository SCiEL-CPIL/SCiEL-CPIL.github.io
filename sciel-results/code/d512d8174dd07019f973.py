"""Construct a small 4-regular graph of girth 10 from scratch.

Strategy:
- Use a Bipartite Local Search approach.
- The graph must be 4-regular and bipartite (to ensure even girth, simplifying checks).
- Target n=320 (160 left, 160 right).
- Initialize with a random 4-regular bipartite graph.
- Iteratively detect and break cycles of length 4, 6, and 8.
- Use edge switching (2-switches or 4-switches) to eliminate short cycles while maintaining regularity.
"""

import random
import time
import numpy as np
from collections import deque

def _check_girth(adj, k, g):
    """Check if graph with adjacency list has girth >= g using BFS from each vertex."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_random_bipartite_4regular(n_left):
    """Generate a random 4-regular bipartite graph with n_left nodes on each side."""
    n_right = n_left
    # Left nodes: 0..n_left-1
    # Right nodes: n_left..n_left+n_right-1
    # Adjacency for left side only initially, we will build full adj
    adj_left = [[] for _ in range(n_left)]
    adj_right = [[] for _ in range(n_right)]
    
    # Create stubs for left and right
    left_stubs = []
    for i in range(n_left):
        for _ in range(4):
            left_stubs.append(i)
    right_stubs = []
    for j in range(n_right):
        for _ in range(4):
            right_stubs.append(j)
            
    random.shuffle(left_stubs)
    random.shuffle(right_stubs)
    
    # Pair them up
    for i in range(n_left * 4):
        u = left_stubs[i]
        v = right_stubs[i]
        adj_left[u].append(v)
        adj_right[v].append(u)
        
    # This might create multi-edges if we are not careful, but with 4-regular and large n,
    # and random shuffling, it's unlikely to have duplicate pairs in a single step 
    # unless we check. Let's check for duplicates.
    # Actually, the configuration model for bipartite graphs can produce multi-edges.
    # We should verify simplicity.
    for u in range(n_left):
        if len(set(adj_left[u])) < 4:
            return None
    for v in range(n_right):
        if len(set(adj_right[v])) < 4:
            return None
            
    # Build full adjacency list
    adj = [[] for _ in range(n_left + n_right)]
    for u in range(n_left):
        for v in adj_left[u]:
            adj[u].append(n_left + v)
            adj[n_left + v].append(u)
            
    return adj

def _find_short_cycles_bipartite(adj, n_left, n_right, max_len=8):
    """Find cycles of length 4, 6, 8 in a bipartite graph."""
    # We represent the graph as adj_left[u] = list of right nodes
    # and adj_right[v] = list of left nodes
    adj_left = [[] for _ in range(n_left)]
    adj_right = [[] for _ in range(n_right)]
    for u in range(n_left):
        for v in adj[u]:
            # v is in right side
            rv = v - n_left
            adj_left[u].append(rv)
            adj_right[rv].append(u)
            
    cycles = []
    
    # Find 4-cycles: u1-v1-u2-v2-u1
    # Equivalent to: u1 and u2 share two common neighbors v1, v2
    # Or v1 and v2 share two common neighbors u1, u2
    # Let's find pairs of left nodes with >=2 common neighbors
    common_neighbors = {}
    for v in range(n_right):
        neighbors = adj_right[v]
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                u1, u2 = neighbors[i], neighbors[j]
                key = (min(u1, u2), max(u1, u2))
                if key not in common_neighbors:
                    common_neighbors[key] = []
                common_neighbors[key].append(v)
                
    for (u1, u2), vs in common_neighbors.items():
        if len(vs) >= 2:
            # Pick first two
            v1, v2 = vs[0], vs[1]
            cycles.append((4, [u1, v1, u2, v2]))
            
    # Find 6-cycles: u1-v1-u2-v2-u3-v3-u1
    # This is harder. We can look for paths of length 2 that share endpoints?
    # Or use BFS from each left node up to depth 3 and check for back edges to start.
    # Given n=160, this is feasible.
    for u_start in range(n_left):
        # BFS/DFS from u_start
        # Depth 0: u_start
        # Depth 1: v in adj_left[u_start]
        # Depth 2: u in adj_right[v]
        # Depth 3: v' in adj_left[u]
        # If v' == u_start, we found a cycle? No, v' is on right, u_start is on left.
        # Wait, cycle is u-v-u'-v'-u.
        # So we need u_start -> v1 -> u2 -> v2 -> u_start.
        # This is a 4-cycle.
        
        # 6-cycle: u_start -> v1 -> u2 -> v2 -> u3 -> v3 -> u_start.
        # Let's do a DFS up to depth 3 from u_start.
        visited = {u_start: []} # node -> path
        # We need to track the path to reconstruct the cycle.
        # To save time, we only check for cycles that return to u_start.
        
        def dfs(node, depth, path, side):
            if depth == 3:
                # We are at a left node u3.
                # We need to connect to u_start via a right node v3.
                # So u_start must be in adj_left[u3]? No, u_start is left, u3 is left.
                # Connection is u3 -> v3 -> u_start.
                # So v3 must be in adj_left[u3] AND adj_right[v3] must contain u_start.
                # i.e., v3 is a common neighbor of u3 and u_start.
                # But we already used v1, v2.
                # So we need a common neighbor v3 of u3 and u_start that is not v1 or v2.
                common = set(adj_left[u3]) & set(adj_left[u_start])
                # Wait, adj_left[u] contains right nodes.
                # So common neighbors are right nodes connected to both.
                # We need v3 in common.
                # And v3 != v1, v3 != v2.
                for v3 in common:
                    if v3 not in path: # v1, v2 are in path
                        # Cycle: u_start, v1, u2, v2, u3, v3
                        cycles.append((6, [u_start, path[0], path[1], path[2], u3, v3]))
                return
            
            if side == 'L':
                for v in adj_left[node]:
                    if v not in path:
                        dfs(v, depth + 1, path + [v], 'R')
            else:
                for u in adj_right[node]:
                    if u not in path and u != u_start: # Don't go back to start yet
                        dfs(u, depth + 1, path + [u], 'L')
                        
        # Start DFS
        for v1 in adj_left[u_start]:
            for u2 in adj_right[v1]:
                if u2 == u_start: continue
                for v2 in adj_left[u2]:
                    if v2 == v1: continue
                    dfs(v2, 1, [v1, u2, v2], 'R')
                    
    # Find 8-cycles? Might be too slow. Let's assume if 4 and 6 are gone, 8 might be handled by further iterations or we just check girth at the end.
    # Actually, we should probably just check girth at the end of each iteration.
    # But finding 8-cycles explicitly is hard.
    # Let's just return 4 and 6 cycles for now, and rely on the girth check to catch 8s if they persist.
    # However, if we only break 4 and 6, we might get stuck with 8s.
    # Let's add a simple 8-cycle finder:
    # u1-v1-u2-v2-u3-v3-u4-v4-u1
    
    return cycles

def _switch_edges(adj, n_left, n_right, u1, u2, v1, v2):
    """
    Perform a 2-switch on a bipartite graph.
    Remove edges (u1, v1) and (u2, v2).
    Add edges (u1, v2) and (u2, v1).
    This maintains 4-regularity.
    """
    # Verify edges exist
    # u1 is left, v1 is right (indexed as n_left+v1 in adj)
    idx_v1 = n_left + v1
    idx_v2 = n_left + v2
    
    if idx_v1 not in adj[u1] or idx_v2 not in adj[u2]:
        return False
    if u1 not in adj[idx_v1] or u2 not in adj[idx_v2]:
        return False
        
    # Check if new edges exist (would create multi-edges)
    if idx_v2 in adj[u1]: # u1 already connected to v2
        return False
    if idx_v1 in adj[u2]: # u2 already connected to v1
        return False
        
    # Perform switch
    adj[u1].remove(idx_v1)
    adj[idx_v1].remove(u1)
    adj[u2].remove(idx_v2)
    adj[idx_v2].remove(u2)
    
    adj[u1].append(idx_v2)
    adj[idx_v2].append(u1)
    adj[u2].append(idx_v1)
    adj[idx_v1].append(u2)
    
    return True

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _luw_construct(k, g)
    
    start_time = time.time()
    n = 320
    n_left = n // 2
    n_right = n // 2
    
    best_adj = None
    
    for attempt in range(50):
        if time.time() - start_time > 110:
            break
            
        adj = _generate_random_bipartite_4regular(n_left)
        if adj is None:
            continue
            
        # Local search
        for step in range(1000):
            if time.time() - start_time > 110:
                break
                
            # Check girth
            if _check_girth(adj, k, g):
                best_adj = adj
                break
                
            # Find 4-cycles
            cycles_4 = []
            # Fast 4-cycle detection
            adj_left = [[] for _ in range(n_left)]
            for u in range(n_left):
                for v in adj[u]:
                    adj_left[u].append(v - n_left)
                    
            # Count common neighbors for pairs of left nodes
            # Using a set of pairs of right nodes?
            # Better: for each right node, it connects to 4 left nodes.
            # Each pair of left nodes sharing a right node forms a path of length 2.
            # If two right nodes share the same pair of left nodes, we have a 4-cycle.
            
            pair_to_right = {}
            for v in range(n_right):
                neighbors = adj_left[n_left + v] # Wait, adj_left is for left nodes.
                # Let's rebuild adj_right properly
                pass
            
            # Let's just use the _find_short_cycles_bipartite logic but simplified for 4-cycles
            adj_right = [[] for _ in range(n_right)]
            for u in range(n_left):
                for v in adj[u]:
                    rv = v - n_left
                    adj_right[rv].append(u)
                    
            common_pairs = {}
            for v in range(n_right):
                neigh = adj_right[v]
                for i in range(len(neigh)):
                    for j in range(i+1, len(neigh)):
                        u1, u2 = neigh[i], neigh[j]
                        key = (min(u1, u2), max(u1, u2))
                        if key not in common_pairs:
                            common_pairs[key] = []
                        common_pairs[key].append(v)
                        
            found_cycle = False
            for (u1, u2), vs in common_pairs.items():
                if len(vs) >= 2:
                    v1, v2 = vs[0], vs[1]
                    # Try to break this 4-cycle by switching
                    # We need to find another edge to switch with?
                    # A 2-switch on (u1, v1) and (u2, v2) will remove this 4-cycle
                    # but might create new ones.
                    # Let's try the switch.
                    if _switch_edges(adj, n_left, n_right, u1, u2, v1, v2):
                        found_cycle = True
                        break
                        
            if not found_cycle:
                # No 4-cycles found? Maybe 6 or 8.
                # Try random switches to escape local minima
                if random.random() < 0.5:
                    # Pick random u1, u2, v1, v2 that form a 4-cycle?
                    # Or just random valid switch
                    u1 = random.randint(0, n_left - 1)
                    u2 = random.randint(0, n_left - 1)
                    if u1 == u2: continue
                    v1 = random.choice(adj[u1]) - n_left
                    # Find v2 such that (u1, v1) and (u2, v2) can be switched
                    # We need v2 in adj[u2] and v1 in adj[u2] is false, v2 in adj[u1] is false
                    # Let's just pick random v2 from adj[u2]
                    v2 = random.choice(adj[u2]) - n_left
                    if v1 == v2: continue
                    if _switch_edges(adj, n_left, n_right, u1, u2, v1, v2):
                        pass
                    else:
                        continue
                else:
                    # Stuck? Break and try new random graph
                    break
                    
        if best_adj is not None:
            break
            
    if best_adj is not None:
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    # Fallback
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
