import time
import random
import sys
import math

N = 20
C = 20
DEADLINE = 150.0  # Evaluator kills at 150s, keep margin
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def all_four_cycles(adj):
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for (x,y,z,w) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            yield (x,y,adj[x][y]), (y,z,adj[y][z]), (z,w,adj[z][w]), (w,x,adj[w][x])

def is_rainbow(col_edges):
    if not col_edges:
        return True
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, u, v, c):
    """Check if edge (u,v) with colour c can be added without violating rainbow 4-cycle constraint."""
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    
    # Check all 4-cycles that would be created by adding edge (u,v,c)
    # A 4-cycle u-x-y-v-u requires: adj[u][x]>=0, adj[x][y]>=0, adj[y][v]>=0
    # with x != v, y != u, x != y
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _try_recolor(adj, u, v, new_c):
    """Check if recoloring edge (u,v) to new_c is valid."""
    if u == v:
        return False
    if adj[u][v] < 0:
        return False
    if adj[u][v] == new_c:
        return True # No change needed, but technically valid
    
    # Check all 4-cycles containing edge (u,v)
    # Cycle: u-x-y-v-u
    # Edges: (u,v), (v,y), (y,x), (x,u)
    # Colours: new_c, adj[v][y], adj[y][x], adj[x][u]
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {new_c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    """Remove edge (u,v) from col_edges and adj."""
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            break
    adj[u][v] = -1
    adj[v][u] = -1

def greedy_construct(seed_val):
    """Run one greedy construction attempt."""
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 10:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges, adj

def simulated_annealing(col_edges, adj, max_iters=50000, T_init=5.0, T_min=0.01, cooling=0.999):
    """
    Simulated annealing allowing add, remove, and recolor moves.
    """
    col_edges = [list(e) for e in col_edges]
    adj = [row[:] for row in adj]
    T = T_init
    best_edges = [list(e) for e in col_edges]
    best_adj = [row[:] for row in adj]
    best_count = len(col_edges)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for it in range(max_iters):
        if elapsed() > DEADLINE - 5:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            u, v = random.choice(all_pairs)
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            added = False
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    added = True
                    break
            if added:
                if len(col_edges) > best_count:
                    best_count = len(col_edges)
                    best_edges = [list(e) for e in col_edges]
                    best_adj = [row[:] for row in adj]
                    
        elif move_type < 0.8:
            # Try to recolor a random edge
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, old_c = col_edges[idx]
            new_c = random.randint(0, C-1)
            if new_c == old_c:
                continue
            
            if _try_recolor(adj, u, v, new_c):
                # Accept recoloring (neutral move for edge count, but changes landscape)
                adj[u][v] = new_c
                adj[v][u] = new_c
                col_edges[idx][2] = new_c
                # Note: We don't update best here unless edge count increased, 
                # but recoloring doesn't change count.
                
        else:
            # Try to remove a random edge
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, c = col_edges[idx]
            
            # Store old state
            old_edges = [list(e) for e in col_edges]
            old_adj = [row[:] for row in adj]
            old_count = len(col_edges)
            
            _remove_edge(col_edges, adj, u, v)
            
            # Metropolis acceptance for removal (delta E = 1)
            if random.random() < math.exp(-1.0 / (T + 1e-10)):
                # Accept removal
                pass
            else:
                # Reject: restore
                col_edges[:] = old_edges
                adj[:] = old_adj
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_edges, best_adj

def local_improve(col_edges, adj):
    """Try to add more edges to the given graph using greedy pass."""
    adj = [row[:] for row in adj] # Work on copy
    col_edges = [list(e) for e in col_edges]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    improved = True
    while improved:
        improved = False
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                return col_edges, adj
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
    return col_edges, adj

def construct():
    best_edges = []
    best_adj = []
    best_count = 0
    
    # 1. Initial Greedy Start
    # Run a few greedy starts to find a good initial seed
    for i in range(10):
        if elapsed() > DEADLINE - 60:
            break
        edges, adj = greedy_construct(i)
        count = len(edges)
        if count > best_count:
            best_edges = edges
            best_adj = adj
            best_count = count
            
    # 2. Local Improvement on Best
    if best_edges:
        best_edges, best_adj = local_improve(best_edges, best_adj)
        best_count = len(best_edges)
        
    # 3. Simulated Annealing
    # Run SA from the best greedy solution
    if best_edges and elapsed() < DEADLINE - 20:
        sa_edges, sa_adj = simulated_annealing(best_edges, best_adj, max_iters=100000, T_init=10.0, T_min=0.01, cooling=0.9995)
        if len(sa_edges) > best_count:
            best_edges = sa_edges
            best_adj = sa_adj
            best_count = len(best_edges)
            
    # 4. Final Local Improvement
    if best_edges and elapsed() < DEADLINE - 5:
        best_edges, best_adj = local_improve(best_edges, best_adj)
        best_count = len(best_edges)

    # Return in required format
    return best_edges

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found {len(res)} edges")
    if is_rainbow(res):
        print("Valid")
    else:
        print("Invalid")
