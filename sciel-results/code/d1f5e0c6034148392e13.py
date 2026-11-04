import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 110
    rng = random.Random(12345)
    
    # Theoretical lower bound: Moore bound for 3-regular graph of girth g
    # For g=16, n_min = 2 * (1 + (k-1) * sum_{i=0}^{g/2-2} (k-1)^i) = 512
    n_min = 2 * (1 + 2 * sum(2**i for i in range(7)))
    
    # We search for the smallest n >= n_min that can form a 3-regular graph of girth g
    # We test n in increments of 2 (since 3-regular graphs must have even number of vertices)
    # To ensure we finish in time, we limit the range of n we test.
    # If we can't find one quickly, we fall back to a known valid construction or the seed.
    
    best_n = None
    best_edges = None
    best_girth = -1
    
    # We try to find a valid graph with n vertices.
    # For each n, we try a limited number of random 3-regular graphs.
    # The non-backtracking check is very fast (O(n*g)), so we can do many trials.
    
    # Let's try a range of n.
    # We start from n_min and go up.
    # We limit the total number of trials to fit within the deadline.
    
    # Helper: check girth using non-backtracking
    def check_girth(edges, n, target_g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Non-backtracking walks:
        # State: (node, prev_node)
        # We want to find the shortest cycle.
        # A cycle of length L corresponds to a non-backtracking walk of length L that returns to the start node.
        # Actually, it's simpler to just do BFS on the directed line graph (non-backtracking graph).
        # Nodes in line graph: (u, v) where u~v.
        # Edges in line graph: (u, v) -> (v, w) where v~w and w != u.
        
        # We can just do a BFS from all "edges" simultaneously?
        # No, we want to detect if there is a cycle of length < g.
        # This is equivalent to checking if there is a non-backtracking walk of length < g that closes a cycle.
        
        # Alternative: For each edge (u, v), do a BFS in the non-backtracking space starting from (u, v).
        # If we reach a state (x, y) where x is the start node of the walk and y is the previous node?
        # Let's be precise.
        # Start at edge (u, v). We are at v, came from u.
        # We want to find a path v -> w1 -> ... -> vk -> u such that no backtracking occurs.
        # This is a path of length k+1 in the original graph.
        # The total cycle length is k+2.
        # We want k+2 < g => k < g-2.
        
        # So we do a BFS in the state space (current_node, prev_node).
        # Start state: (v, u).
        # We want to see if we can reach a state (u, w) where w is a neighbor of u and w != v?
        # No, if we are at u and came from w, that means we traversed w->u.
        # The cycle would be u-v-...-w-u.
        # So we want to reach any state (u, w) where (u, w) is an edge, and w != v (to avoid immediate backtracking u-v-u).
        # Wait, if we start at (v, u), the first step is to a neighbor w of u, w != v.
        # The state becomes (u, w).
        # If we reach a state (x, y) such that x is the start node u and y is the start node v? No.
        
        # Let's use the standard algorithm:
        # For each directed edge (u, v), perform a BFS in the non-backtracking graph.
        # The nodes are directed edges (a, b).
        # Start node: (u, v).
        # We want to find if there is a path from (u, v) to (v, u)? No, that would be a cycle of length 2 (u-v-u), which is backtracking.
        # We want to find a cycle.
        # A cycle u-v-w-x-u corresponds to directed edges (u,v), (v,w), (w,x), (x,u).
        # In the non-backtracking graph, this is a path (u,v) -> (v,w) -> (w,x) -> (x,u).
        # So we are looking for a path from (u,v) to (x,u) where x is a neighbor of u and x != v?
        # Actually, if we reach (y, u) where y is a neighbor of u and y != v, then we have a cycle u-v-...-y-u.
        # The length of the cycle is the number of steps in the non-backtracking walk + 1.
        
        # Let's just do a BFS for each starting edge (u, v) and see if we can return to u via any other neighbor.
        # To be efficient, we can do a multi-source BFS?
        # No, the start node matters for the cycle closure.
        
        # Since g is small (16), the BFS depth is at most 14.
        # The branching factor is at most 2 (since degree is 3, and we can't go back).
        # So the number of states is at most 2^14 = 16384 per edge.
        # With n=512, there are 768 directed edges.
        # 768 * 16384 ~ 12 million operations. This is fast enough in Python if optimized, but might be tight.
        # We can prune if we find a cycle of length < g.
        
        # Optimization: We only need to check if ANY cycle is < g.
        # We can do a BFS from all edges simultaneously?
        # Let's try a simple BFS from each edge, but break early if we find a short cycle.
        
        # Actually, we can just check the girth of the graph directly.
        # The girth is the length of the shortest cycle.
        # We can compute the girth by doing a BFS from each node, but that's O(n^2).
        # The non-backtracking approach is better for large girth.
        
        # Let's implement the non-backtracking check for a specific girth g.
        # We want to know if there is a cycle of length < g.
        
        # For each edge (u, v):
        #   BFS in state space (node, prev_node).
        #   Start: (v, u).
        #   We want to reach a state (u, w) where w is a neighbor of u and w != v.
        #   The distance in the BFS tree is the length of the path from v to u via w.
        #   Let d be the distance. The cycle length is d + 1.
        #   If d + 1 < g, return False.
        
        # We can optimize by doing this for all edges at once?
        # Maybe just iterate and break early.
        
        for u, v in edges:
            # BFS from (v, u)
            # visited: set of (node, prev_node)
            visited = set()
            visited.add((v, u))
            
            # queue: (node, prev_node, depth)
            queue = [(v, u, 1)]
            
            while queue:
                node, prev, depth = queue.pop(0)
                
                # If we are at node u, and prev is not v, we found a cycle of length depth + 1
                # Wait, the start was (v, u). The path is u -> v -> ... -> node.
                # If node == u, the path is u -> v -> ... -> u.
                # The length of the path in the non-backtracking walk is depth.
                # The cycle length is depth + 1?
                # Let's trace:
                # Start: (v, u), depth=1. This represents the edge u-v.
                # Next step: (w, u) where w~u, w!=v. depth=2. Path u-v-w? No, u-v then v-w?
                # My state definition (node, prev) means we are at node, came from prev.
                # Start: at v, came from u.
                # Step 1: go to w~v, w!=u. State (w, v). Depth 2. Path u-v-w.
                # Step 2: go to x~w, x!=v. State (x, w). Depth 3. Path u-v-w-x.
                # ...
                # If we reach state (u, y) where y~u and y!=v.
                # The path is u-v-...-y-u.
                # The number of edges in the path is depth.
                # So the cycle length is depth.
                
                if node == u and prev != v:
                    # Cycle found of length depth
                    if depth < g:
                        return False
                    # If depth >= g, we don't care about this cycle, but we should prune?
                    # No, there might be a shorter cycle from another start.
                    # But for this start, we found a cycle of length depth.
                    # We can stop expanding this branch? No, there might be a shorter cycle?
                    # No, BFS finds the shortest path to (u, y).
                    # So if the shortest path to (u, y) has length depth >= g, then all other paths to (u, y) are longer.
                    # And any cycle closing at u must go through some neighbor y.
                    # So if the shortest cycle closing at u is >= g, we are safe for this start?
                    # Not necessarily, because we might close at a different neighbor?
                    # No, the cycle must close at u.
                    # So if we found a cycle of length depth >= g, we can stop this BFS?
                    # Yes, because BFS explores in increasing depth.
                    # So if we reach node u, the first time we do so is the shortest cycle.
                    # If that shortest cycle is >= g, then all other cycles from this start are longer.
                    # So we can break.
                    break
                
                # Expand
                for neighbor in adj[node]:
                    if neighbor != prev:
                        if (neighbor, node) not in visited:
                            visited.add((neighbor, node))
                            queue.append((neighbor, node, depth + 1))
                            
                            # Optional pruning: if depth + 1 >= g, we don't need to go deeper?
                            # If we are at depth d, and d >= g, then any cycle formed will be length >= d+1 > g.
                            # So we can stop expanding if depth + 1 >= g.
                            if depth + 1 >= g:
                                # We added it, but we don't need to process it?
                                # Actually, if we don't process it, we won't find cycles starting from it.
                                # But any cycle starting from it would be longer.
                                # So we can just not add it to the queue?
                                # But we need to mark it as visited to avoid re-adding.
                                # So we add to visited, but not to queue.
                                pass
                            else:
                                # It's already in the queue if we appended it.
                                # We can't easily remove it.
                                # Let's just let it be processed, but the check `if depth < g` will handle it.
                                pass
            
            # Optimization: if we didn't break, it means no cycle < g was found for this start?
            # Wait, if we break, it means we found a cycle.
            # If we don't break, the queue became empty.
            # This means we explored all reachable states.
            # If we didn't find a cycle < g, we are good for this edge.
            
            # However, the `break` above only breaks if `depth < g`.
            # If `depth >= g`, we don't break.
            # But if we reach `node == u` with `depth >= g`, we know the shortest cycle is >= g.
            # So we can break in that case too.
            # Let's refine the logic.
            
    # The above logic is a bit messy. Let's rewrite `check_girth` cleanly.
    
    def check_girth_v2(edges, n, target_g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for u, v in edges:
            # BFS from (v, u)
            visited = set()
            visited.add((v, u))
            queue = [(v, u, 1)]
            found_cycle = False
            min_cycle_len = float('inf')
            
            while queue:
                node, prev, depth = queue.pop(0)
                
                if node == u:
                    # We reached u. The cycle length is depth.
                    # Since BFS, this is the shortest cycle closing at u starting from v.
                    if depth < target_g:
                        return False
                    else:
                        # Shortest cycle is >= target_g.
                        # We can stop this BFS.
                        break
                        
                for neighbor in adj[node]:
                    if neighbor != prev:
                        if (neighbor, node) not in visited:
                            visited.add((neighbor, node))
                            # Only add to queue if depth+1 < target_g
                            # Because if depth+1 >= target_g, any cycle formed will be >= target_g
                            if depth + 1 < target_g:
                                queue.append((neighbor, node, depth + 1))
        
        return True

    # Now, the search loop
    # We try n = n_min, n_min+2, ...
    # For each n, we try a limited number of random graphs.
    
    # To fit in time, we limit the number of trials per n and the range of n.
    # Let's say we try up to 1000 trials per n, and up to 10 values of n.
    # 1000 * 10 = 10000 trials.
    # Each trial involves generating a graph (O(n)) and checking girth (O(n*2^g)).
    # For n=512, g=16, checking is ~ 512 * 10000 = 5 million ops.
    # 10000 trials * 5 million = 50 billion ops. Too slow.
    
    # We need to be smarter.
    # The check is O(n * 2^(g-2)). For g=16, 2^14 = 16384.
    # n=512. 512 * 16384 = 8 million ops per check.
    # In Python, 8 million ops is ~ 0.1s.
    # So we can do ~1000 checks in 100s.
    # So we can try ~100 random graphs for n=512.
    # If none work, try n=514, etc.
    
    # Let's implement a function to generate a random 3-regular graph.
    def generate_random_3_regular(n):
        # Configuration model
        # 3n stubs
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        rng.shuffle(stubs)
        
        edges = []
        for i in range(0, n, 2):
            u = stubs[2*i]
            v = stubs[2*i+1]
            if u == v:
                return None # Self-loop, invalid
            # Check for multi-edge?
            # We can just add and check later, or use a set.
            edges.append((min(u, v), max(u, v)))
            
        # Check for multi-edges
        edge_set = set()
        for e in edges:
            if e in edge_set:
                return None
            edge_set.add(e)
            
        # Check degree
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            if deg[u] > 3 or deg[v] > 3:
                return None
                
        return edges

    # Search
    current_n = n_min
    max_n = n_min + 200 # Limit search range
    
    trials_per_n = 50
    
    while current_n <= max_n:
        for _ in range(trials_per_n):
            if time.time() > deadline - 5:
                break
                
            edges = generate_random_3_regular(current_n)
            if edges is None:
                continue
                
            if check_girth_v2(edges, current_n, g):
                return current_n, edges
                
        current_n += 2
        
    # Fallback: Return the seed construction or a known valid graph
    # The seed construction is valid but large.
    # We can just return the seed logic if we run out of time.
    
    # Let's reuse the seed logic as a fallback.
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
