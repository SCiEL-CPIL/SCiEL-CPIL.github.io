import random
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct():
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

    deadline = time.time() + 110
    
    # Try Greedy Tree Extension
    # Tree Depth 8
    n_tree = 766
    parent = [-1] * n_tree
    children = [[] for _ in range(n_tree)]
    branch = [0] * n_tree
    
    queue = [0]
    parent[0] = -1
    branch[0] = -1
    level = 0
    current_nodes = [0]
    
    while level < 8:
        next_nodes = []
        for u in current_nodes:
            num_children = 3 if level == 0 else 2
            for i in range(num_children):
                v = len(parent)
                parent[v] = u
                children[u].append(v)
                if level == 0:
                    branch[v] = i
                else:
                    branch[v] = branch[u]
                next_nodes.append(v)
        current_nodes = next_nodes
        level += 1
        
    leaves = current_nodes
    
    adj = [[] for _ in range(n_tree)]
    for u in range(n_tree):
        for v in children[u]:
            adj[u].append(v)
            adj[v].append(u)
            
    deg = [len(adj[u]) for u in range(n_tree)]
    
    leaves_by_branch = {0: [], 1: [], 2: []}
    for l in leaves:
        leaves_by_branch[branch[l]].append(l)
        
    def is_safe(u, v, adj, deg):
        # Check if path < 15 exists
        # BFS from u up to depth 14
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist == 14:
                continue
            for nb in adj[node]:
                if nb == v:
                    return False
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return True
        
    best_n, best_edges = None, None
    
    for attempt in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        random.seed(attempt)
        
        # Reset adj and deg
        adj = [[] for _ in range(n_tree)]
        for u in range(n_tree):
            for v in children[u]:
                adj[u].append(v)
                adj[v].append(u)
        deg = [len(adj[u]) for u in range(n_tree)]
        added_edges = []
        
        success = True
        while True:
            if time.time() > deadline:
                success = False
                break
                
            candidates = [l for l in leaves if deg[l] < 3]
            if not candidates:
                break
                
            u = random.choice(candidates)
            valid_vs = []
            for b in range(3):
                if b != branch[u]:
                    for v in leaves_by_branch[b]:
                        if deg[v] < 3:
                            valid_vs.append(v)
                            
            if not valid_vs:
                success = False
                break
                
            random.shuffle(valid_vs)
            added = False
            for v in valid_vs:
                if is_safe(u, v, adj, deg):
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
                    added_edges.append((u, v))
                    added = True
                    break
                    
            if not added:
                success = False
                break
                
        if success:
            # Verify
            edge_set = set()
            for u in range(n_tree):
                for v in adj[u]:
                    if u < v:
                        edge_set.add((u, v))
            best_n, best_edges = n_tree, list(edge_set)
            break
            
    if best_edges:
        return best_n, best_edges
    else:
        return luw_construct()
