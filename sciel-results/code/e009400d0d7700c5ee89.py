import time
import random

N = 20
C = 20
DEADLINE = 150.0
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
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, u, v, c):
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

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def greedy_construct(seed_val):
    """Run one greedy construction attempt from scratch with a given seed."""
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 5:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges

def local_improve(col_edges):
    """Try to add more edges to the given graph with multiple passes."""
    adj = make_adj(col_edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for attempt in range(10):
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                break
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    
    return col_edges

def _seed_star():
    return [[0, v, 0] for v in range(1, N)]

def construct():
    best = _seed_star()
    best_count = count_edges(best)
    
    # Run multiple greedy attempts with different seeds
    max_attempts = 50
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 10:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = result
            best_count = count
            
            # Try to improve locally
            if elapsed() < DEADLINE - 15:
                improved = local_improve(list(best))
                improved_count = count_edges(improved)
                if improved_count > best_count:
                    best = improved
                    best_count = improved_count
        
        # Early exit if we've found a very dense graph
        if best_count >= 180:
            break
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback: try a simple star
    star = _seed_star()
    if is_rainbow(star):
        return star
    
    return star
