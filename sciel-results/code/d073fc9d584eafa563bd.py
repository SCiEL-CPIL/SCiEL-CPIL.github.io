import random
import time
from collections import deque

def _luw_construct(k, g):
    # Original LUW code as fallback
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

def _find_cage_4_10(time_limit):
    N = 320
    L = N // 2
    R = N // 2
    # Adjacency lists: adj[i] is a set of neighbors
    # Left nodes: 0..159, Right nodes: 160..319
    adj = [set() for _ in range(N)]
    
    def init_random():
        for i in range(N):
            adj[i].clear()
        # Create 4 stubs for each left node
        left_stubs = []
        for i in range(L):
            for _ in range(4):
                left_stubs.append(i)
        random.shuffle(left_stubs)
        
        # Create 4 stubs for each right node
        right_stubs = []
        for j in range(R):
            for _ in range(4):
                right_stubs.append(j + L)
        random.shuffle(right_stubs)
        
        # Connect
        for u, v in zip(left_stubs, right_stubs):
            adj[u].add(v)
            adj[v].add(u)
            
    def get_girth():
        # Check for cycles < 10 (4, 6, 8)
        # BFS from each node up to depth 4
        for start in range(N):
            visited = {start: (0, -1)} # node: (dist, parent)
            queue = deque([(start, 0, -1)])
            while queue:
                node, dist, parent = queue.popleft()
                if dist >= 4:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Cycle found
                        # dist(node) + 1 + dist(neighbor)
                        # Actually, if neighbor is visited, we need to check if it's the parent
                        # Standard BFS cycle detection:
                        # If neighbor is in visited and neighbor != parent
                        # Cycle length = dist(node) + 1 + dist(neighbor)
                        # Wait, this logic is flawed for undirected graphs if we don't track the path properly.
                        # Better: Just check if neighbor is visited and neighbor != parent.
                        # If so, cycle length = dist[node] + dist[neighbor] + 1
                        # But we need to ensure it's a simple cycle.
                        # For girth, BFS finds the shortest cycle passing through start?
                        # No, BFS from start finds shortest path from start to all nodes.
                        # A cycle is found if an edge connects two nodes at the same or adjacent levels?
                        # Actually, if we just want to know IF a cycle < 10 exists, 
                        # we can just run BFS and if we find a visited node (not parent) 
                        # with dist[node] + dist[neighbor] + 1 < 10, we return True.
                        d1 = visited[neighbor][0]
                        d2 = dist
                        if d1 + d2 + 1 < 10:
                            return True
                    else:
                        visited[neighbor] = (dist + 1, node)
                        queue.append((neighbor, dist + 1, node))
        return False # No short cycle

    def find_short_cycle_edges():
        # Find a specific 4-cycle to break
        # Check all pairs of left nodes
        for u in range(L):
            for v in range(u + 1, L):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    # 4-cycle u-w-v-x-u
                    w, x = list(common)[:2]
                    return [(u, w), (v, w), (v, x), (u, x)]
        # If no 4-cycle, check 6-cycles?
        # For simplicity, if no 4-cycle, we might be stuck or have girth 6+.
        # We can just return None and rely on random swaps or girth check.
        return None

    def swap_edges(e1, e2):
        u1, v1 = e1
        u2, v2 = e2
        # Ensure e1 and e2 are not the same
        if u1 == u2 and v1 == v2:
            return False
        # Check if new edges exist
        if v2 in adj[u1] or v1 in adj[u2]:
            return False
        # Remove
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        # Add
        adj[u1].add(v2)
        adj[v2].add(u1)
        adj[u2].add(v1)
        adj[v1].add(u2)
        return True

    init_random()
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        if not get_girth():
            # No short cycle
            edges = []
            for i in range(N):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return N, edges
            
        # Try to break a 4-cycle
        cycle = find_short_cycle_edges()
        if cycle:
            # Pick two non-adjacent edges to swap?
            # In a 4-cycle u-w-v-x-u, edges are (u,w), (v,w), (v,x), (u,x).
            # Swap (u,w) and (v,x) -> (u,x) and (v,w).
            # But (u,x) and (v,w) are already in the cycle!
            # This swap just reverses the cycle?
            # No, (u,w) and (v,x) are opposite.
            # Swapping them to (u,x) and (v,w) results in the same set of edges.
            # We need to swap with an edge OUTSIDE the cycle.
            
            # Pick an edge from the cycle, e.g., (u,w).
            # Pick a random edge (a,b) not in the cycle.
            # Try to swap (u,w) and (a,b) -> (u,b) and (w,a).
            
            e_cycle = cycle[0]
            u, w = e_cycle
            # Find a random edge to swap with
            # Iterate through all edges? Too slow.
            # Pick a random neighbor of u, say u2.
            # Pick a random neighbor of u2, say w2.
            # Try swap (u,w) and (u2,w2).
            
            swapped = False
            for _ in range(50): # Try a few times
                u2 = random.choice(list(adj[u]))
                if u2 == w: continue
                w2 = random.choice(list(adj[u2]))
                if w2 == u: continue
                if (u2, w2) in [(a,b) for a,b in cycle] or (w2, u2) in [(a,b) for a,b in cycle]:
                    continue
                if swap_edges((u, w), (u2, w2)):
                    swapped = True
                    break
            if not swapped:
                # Try random swap anywhere
                u_rand = random.randint(0, N-1)
                v_rand = random.choice(list(adj[u_rand]))
                u2_rand = random.randint(0, N-1)
                v2_rand = random.choice(list(adj[u2_rand]))
                if swap_edges((u_rand, v_rand), (u2_rand, v2_rand)):
                    pass
        else:
            # No 4-cycle, but girth < 10. Must be 6 or 8.
            # Random swap to escape local minimum
            u_rand = random.randint(0, N-1)
            v_rand = random.choice(list(adj[u_rand]))
            u2_rand = random.randint(0, N-1)
            v2_rand = random.choice(list(adj[u2_rand]))
            swap_edges((u_rand, v_rand), (u2_rand, v2_rand))

    return None

def construct(k, g):
    if k == 4 and g == 10:
        result = _find_cage_4_10(110)
        if result:
            return result
    return _luw_construct(k, g)
