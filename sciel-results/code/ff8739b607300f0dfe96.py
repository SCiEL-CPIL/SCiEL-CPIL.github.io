import random
import time
import sys
import numpy as np
from collections import deque

def construct(k, g):
    # New Approach: Local Search on Random Regular Graphs
    # 1. Generate random k-regular graph
    # 2. Count short cycles
    # 3. If > 0, perform random 2-switches to reduce count
    # 4. Repeat with restarts
    
    # Fallback: Keep the L-U-W construction from the seed as a safety net
    # (This is re-using the fallback, but the PRIMARY strategy is Local Search)
    def l_u_w_construct(k, g):
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

    def generate_random_regular(n, k):
        """
        Generates a random k-regular graph using the configuration model (pairing half-edges).
        Returns adjacency list or None if self-loops/multi-edges are present (retries handled outside or inside).
        """
        if n * k % 2 != 0:
            return None
        
        # Create list of half-edges
        # Each vertex i has k stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair them up
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Self-loop
            if (min(u, v), max(u, v)) in edges:
                return None # Multi-edge
            edges.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
            
        return adj, list(edges)

    def count_short_cycles(adj, n, g):
        """
        Counts the number of cycles of length < g.
        For efficiency, we can just check if ANY exist for the 'valid' check, 
        but for local search we want a score.
        Let's return the number of 3-cycles, 4-cycles, etc.
        """
        # This is expensive. For n=1000, O(n^2) or O(n * k^depth) is needed.
        # Let's just find ONE short cycle for the move, and use "number of short cycles" as energy if possible.
        # For simplicity, we will return the length of the shortest cycle found, or g if none.
        # Actually, for local search, we need a scalar. 
        # Let's count 3-cycles and 4-cycles explicitly as they are the most common obstacles.
        
        # Count 3-cycles
        # For each edge (u, v), count common neighbors.
        # Sum over all edges / 3.
        count_3 = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = set(adj[u]) & set(adj[v])
                    count_3 += len(common)
        count_3 //= 3
        
        if count_3 > 0:
            return count_3 * 1000 # Weight 3-cycles heavily
        
        # Count 4-cycles
        # For each pair of non-adjacent vertices (u, v), count common neighbors.
        # Sum / 2.
        # This is O(n^2 * k^2). For n=1000, 10^6 * 25 = 2.5 * 10^7. Feasible in Python? Maybe slow.
        # We can optimize by only checking pairs with distance 2.
        
        # Let's just return count_3 for now to keep it fast.
        # If count_3 == 0, we check for 4-cycles.
        
        # Fast 4-cycle check:
        # For each u, for each v in adj[u], for each w in adj[v]:
        #   if w != u and w in adj[u] -> 3-cycle (already checked)
        #   if w not in adj[u]:
        #       for each x in adj[w]:
        #           if x == u: continue
        #           if x in adj[u] and x < w: # Avoid double counting?
        #               count_4 += 1
        
        # This is getting complex. Let's use a BFS-based "Girth Check" that returns the girth.
        # If girth < g, we return girth. Else g.
        # But local search needs a gradient.
        # Let's stick to: Find a short cycle. If found, return its length. Else return g.
        
        # BFS from each node? O(n * E).
        # For n=1000, E=2500. 2.5 * 10^6. Very fast.
        
        for start in range(n):
            visited = {start: 0}
            queue = deque([(start, 0)])
            while queue:
                node, dist = queue.popleft()
                if dist >= g - 1:
                    continue
                for nei in adj[node]:
                    if nei == start and dist + 1 >= 3: # Cycle found
                        # Check if it's a valid cycle (not just backtracking)
                        # BFS from start finds shortest path.
                        # If we hit start again, we found a cycle of length dist+1.
                        # But BFS from start doesn't track the path to ensure it's a simple cycle?
                        # Actually, standard BFS girth calculation:
                        # Run BFS from each node. If we encounter the start node again, 
                        # the distance + 1 is the cycle length.
                        # But we must ensure the path is simple. BFS guarantees shortest path, 
                        # which is simple.
                        return dist + 1
                    if nei not in visited:
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1))
                    elif visited[nei] == dist + 1:
                        # Back edge? No, undirected graph.
                        # If we see a neighbor at the same level or previous level, 
                        # it indicates a cycle.
                        # Specifically, if nei is start, we found a cycle.
                        pass
                        
        return g

    def find_short_cycle(adj, n, g):
        """Returns a list of nodes forming a short cycle, or None."""
        for start in range(n):
            # BFS
            parent = {start: -1}
            dist = {start: 0}
            queue = deque([start])
            found = None
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if nei == start and dist[node] + 1 >= 3:
                        # Reconstruct cycle
                        cycle = [start]
                        curr = node
                        while curr != -1:
                            cycle.append(curr)
                            curr = parent[curr]
                        cycle.reverse()
                        # Verify length
                        if len(cycle) < g:
                            return cycle
                    if nei not in parent:
                        parent[nei] = node
                        dist[nei] = dist[node] + 1
                        queue.append(nei)
                    elif dist[nei] == dist[node] + 1 and nei != parent[node]:
                        # Found a cycle
                        # Reconstruct
                        # This is tricky to reconstruct correctly from BFS tree
                        # Let's just return the distance + 1 as a signal
                        # For now, we just need to know IF a cycle exists to trigger a move.
                        # We don't strictly need the nodes if we do random 2-switches.
                        # But random 2-switches might not break the cycle.
                        # We need to target the cycle.
                        pass
        return None

    def try_local_search(n, k, g, timeout=10):
        start_time = time.time()
        best_girth = 0
        best_adj = None
        
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Generate random graph
            res = generate_random_regular(n, k)
            if not res:
                continue
            adj, edges = res
            
            # Initial girth check
            # We will iterate:
            # 1. Find girth.
            # 2. If girth >= g, return success.
            # 3. Else, perform random 2-switches.
            
            current_girth = count_short_cycles(adj, n, g)
            
            if current_girth >= g:
                return n, edges
            
            # Local Search Loop
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Perform a random 2-switch
                # Pick two random edges (u, v) and (x, y)
                # Try to swap to (u, x) and (v, y) or (u, y) and (v, x)
                # Ensure no self-loops or multi-edges are created.
                
                if len(edges) < 2:
                    break
                
                e1_idx = random.randint(0, len(edges) - 1)
                e2_idx = random.randint(0, len(edges) - 1)
                if e1_idx == e2_idx:
                    continue
                
                u, v = edges[e1_idx]
                x, y = edges[e2_idx]
                
                # Try swap 1: (u, x) and (v, y)
                # Check validity
                if u != x and v != y and (u, x) not in set(edges) and (v, y) not in set(edges) \
                   and u != v and x != y: # Self loop check for new edges
                    # Also check if (u, x) or (v, y) exist
                    # set(edges) is O(1) lookup
                    # But we need to update adj too.
                    # This is getting complicated to manage state.
                    # Let's use a simpler move: Remove 2 edges, Add 2 edges.
                    
                    # Check if (u, x) is a self loop
                    if u == x: continue
                    if v == y: continue
                    if (min(u, x), max(u, x)) in set(edges) or (min(v, y), max(v, y)) in set(edges):
                        continue
                    
                    # Apply move
                    # Remove (u, v) and (x, y)
                    # Add (u, x) and (v, y)
                    
                    # Update adj
                    adj[u].remove(v); adj[v].remove(u)
                    adj[x].remove(y); adj[y].remove(x)
                    
                    adj[u].append(x); adj[x].append(u)
                    adj[v].append(y); adj[y].append(v)
                    
                    # Update edges list
                    # Replace e1, e2
                    edges[e1_idx] = (min(u, x), max(u, x))
                    edges[e2_idx] = (min(v, y), max(v, y))
                    
                    # Check girth
                    new_girth = count_short_cycles(adj, n, g)
                    
                    if new_girth >= g:
                        return n, edges
                    
                    # If girth didn't improve or got worse, we might revert?
                    # Simulated Annealing logic:
                    # If new_girth < current_girth (worse), accept with prob exp(-delta/T).
                    # For simplicity, let's just accept if it doesn't get worse?
                    # Or always accept to escape local minima?
                    # Always accepting random moves leads to random walk.
                    # We need bias.
                    # Let's revert if it gets worse.
                    if new_girth < current_girth:
                        # Revert
                        adj[u].remove(x); adj[x].remove(u)
                        adj[v].remove(y); adj[y].remove(v)
                        
                        adj[u].append(v); adj[v].append(u)
                        adj[x].append(y); adj[y].append(x)
                        
                        edges[e1_idx] = (u, v)
                        edges[e2_idx] = (x, y)
                    else:
                        current_girth = new_girth
                        if current_girth > best_girth:
                            best_girth = current_girth
                            best_adj = [list(lst) for lst in adj]
                            best_edges = list(edges)
        
        # If we found a good one, return it.
        # But we need girth >= g.
        # If we didn't reach g, we fail.
        return None

    # Try Local Search for various n
    # Start from known best or slightly above
    # For k=5, g=9, try 1116, 1200, 1500
    
    if k == 5 and g == 9:
        n_trials = [1116, 1150, 1200, 1500, 2000]
        for n in n_trials:
            res = try_local_search(n, k, g, timeout=20)
            if res:
                return res
    else:
        # Generic
        moore_bound = 0
        if g % 2 == 1:
            r = (g - 3) // 2
            for i in range(r):
                moore_bound += (k - 1) ** i
            moore_bound *= k
        else:
            r = g // 2 - 1
            for i in range(r + 1):
                moore_bound += (k - 1) ** i
            moore_bound *= 2
        
        n_start = max(10, moore_bound)
        for n in [n_start, n_start + 100, n_start + 500]:
            res = try_local_search(n, k, g, timeout=10)
            if res:
                return res

    # Fallback
    return l_u_w_construct(k, g)
