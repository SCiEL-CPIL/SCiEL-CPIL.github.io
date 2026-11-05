import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Try to find the (4,10)-cage (320 vertices) via Cayley graph search
        # If successful, return it.
        # Otherwise, fall back to LUW.
        
        # Known cage size is 320.
        # Let's try to construct a Cayley graph on Z_2 x Z_2 x Z_80
        n_target = 320
        group_size = n_target
        # Structure: Z_2 x Z_2 x Z_80
        # Index: x * 160 + y * 80 + z
        
        def index_from_coords(x, y, z):
            return x * 160 + y * 80 + z
        
        def coords_from_index(idx):
            x = idx // 160
            rem = idx % 160
            y = rem // 80
            z = rem % 80
            return x, y, z

        def get_neighbors(idx, a, b):
            """
            Returns list of 4 neighbors for vertex idx using generators a, b.
            a, b are tuples (ax, ay, az).
            """
            x, y, z = coords_from_index(idx)
            neighbors = []
            # Add a
            nx = (x + a[0]) % 2
            ny = (y + a[1]) % 2
            nz = (z + a[2]) % 80
            neighbors.append(index_from_coords(nx, ny, nz))
            
            # Add -a (which is (x, y, -z) in Z_2 x Z_2 x Z_80? No, -a is (ax, ay, 80-az) mod 80)
            # In Z_2, -1 = 1. So -ax = ax.
            nx = (x + a[0]) % 2  # -a[0] is same as a[0] in Z_2
            ny = (y + a[1]) % 2
            nz = (z - a[2]) % 80
            neighbors.append(index_from_coords(nx, ny, nz))
            
            # Add b
            nx = (x + b[0]) % 2
            ny = (y + b[1]) % 2
            nz = (z + b[2]) % 80
            neighbors.append(index_from_coords(nx, ny, nz))
            
            # Add -b
            nx = (x + b[0]) % 2
            ny = (y + b[1]) % 2
            nz = (z - b[2]) % 80
            neighbors.append(index_from_coords(nx, ny, nz))
            
            return neighbors

        def check_girth(adj, g):
            """
            Checks if graph has girth >= g.
            Returns True if yes.
            """
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
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True

        # Search for generators
        # We need a, b such that they are not trivial and distinct.
        # a = (ax, ay, az), b = (bx, by, bz)
        # We can fix some components to reduce search space or just randomize.
        # Randomization is safer for finding "good" girth.
        
        found = False
        best_n = None
        best_edges = None
        
        # Try a few random pairs
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Random generators
            # Ensure they are not identity
            ax = random.randint(0, 1)
            ay = random.randint(0, 1)
            az = random.randint(1, 79) # Non-zero to avoid trivial
            a = (ax, ay, az)
            
            bx = random.randint(0, 1)
            by = random.randint(0, 1)
            bz = random.randint(1, 79)
            b = (bx, by, bz)
            
            # Check if a and b are essentially the same or inverses
            # a == b or a == -b
            # -a in Z_2 x Z_2 x Z_80 is (ax, ay, -az % 80)
            neg_a = (ax, ay, (-az) % 80)
            if a == b or a == neg_a:
                continue
            
            # Build adjacency list
            adj = [[] for _ in range(n_target)]
            for i in range(n_target):
                neighs = get_neighbors(i, a, b)
                # Remove duplicates if any (shouldn't happen if a,b distinct and not inverses)
                # But check for self-loops?
                # If a is identity, self loop. We ensured az != 0.
                # If a is (0,0,0) it's identity.
                # If a is (1,0,0) it's order 2.
                adj[i] = neighs
            
            # Quick check: no self loops
            valid = True
            for i in range(n_target):
                if i in adj[i]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check girth
            if check_girth(adj, g):
                # Found a valid graph!
                # Construct edge list
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
                # We return immediately as 320 is the best known.
                
        # If not found, fall back to LUW
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)

def _luw_graph(k, g):
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
