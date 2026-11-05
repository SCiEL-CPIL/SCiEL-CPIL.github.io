import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    adj = [set() for _ in range(n)]
    # common[i][j] is set of common neighbors of i and j
    common = [[set() for _ in range(n)] for _ in range(n)]
    
    def is_valid_add(u, v):
        if v in adj[u]:
            return False
        
        # Check constraints
        # For any pair (x, y), if v in common[x][y], then len(adj[u] & common[x][y]) must be <= 1
        # If u in common[x][y], then len(adj[v] & common[x][y]) must be <= 1
        
        for x in range(n):
            for y in range(x + 1, n):
                cx_y = common[x][y]
                if v in cx_y:
                    if len(adj[u] & cx_y) > 1:
                        return False
                if u in cx_y:
                    if len(adj[v] & cx_y) > 1:
                        return False
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        
        # Update common
        # v is now in N(u). For any x in N(v), v is in N(u) cap N(x).
        for x in adj[v]:
            common[u][x].add(v)
            common[x][u].add(v)
            
        # u is now in N(v). For any x in N(u), u is in N(v) cap N(x).
        for x in adj[u]:
            common[v][x].add(u)
            common[x][v].add(u)
            
        # common[u][v] is unchanged as reasoned before.

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Update common
        for x in adj[v]:
            if v in common[u][x]:
                common[u][x].remove(v)
                common[x][u].remove(v)
                
        for x in adj[u]:
            if u in common[v][x]:
                common[v][x].remove(u)
                common[x][v].remove(u)

    def edge_count():
        return sum(len(s) for s in adj) // 2

    # Initialization
    edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if is_valid_add(u, v):
            add_edge(u, v)
            
    best_edges = edge_count()
    best_adj = [set(s) for s in adj]
    best_common = [[set(s) for s in row] for row in common]
    
    # SA
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.6:
            # Add
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and v not in adj[u]:
                if is_valid_add(u, v):
                    add_edge(u, v)
                    ec = edge_count()
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [set(s) for s in adj]
                        best_common = [[set(s) for s in row] for row in common]
        elif move_type < 0.85:
            # Remove
            if edge_count() > 0:
                u = random.randint(0, n-1)
                if adj[u]:
                    v = random.choice(list(adj[u]))
                    remove_edge(u, v)
        else:
            # Swap
            u = random.randint(0, n-1)
            if adj[u]:
                v = random.choice(list(adj[u]))
                remove_edge(u, v)
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and y not in adj[x] and not (x==u and y==v) and not (x==v and y==u):
                    if is_valid_add(x, y):
                        add_edge(x, y)
                    else:
                        add_edge(u, v) # Revert
                else:
                    add_edge(u, v)
                    
        T *= cooling

    # Restore best
    adj = [set(s) for s in best_adj]
    common = [[set(s) for s in row] for row in best_common]
    
    # Final Polish
    edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if is_valid_add(u, v):
            add_edge(u, v)

    result = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                result.append([u, v])
    return result
