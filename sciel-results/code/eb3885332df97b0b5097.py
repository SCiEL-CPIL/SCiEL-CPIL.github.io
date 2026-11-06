import time
import random
import sys

N = 20
C = 20
DEADLINE = 120.0
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
    """Try to add more edges to the given graph."""
    adj = make_adj(col_edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for attempt in range(5):
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

def simulated_annealing(col_edges, adj, max_iters=5000, T_init=10.0, T_min=0.01, cooling=0.995):
    """
    Simulated annealing: allow temporary edge removals to escape local optima.
    State: set of edges with colours.
    Moves: add an edge (with valid colour), remove an edge, or recolor an edge.
    Objective: maximize number of edges (minimize -edge_count).
    """
    col_edges = [list(e) for e in col_edges]
    adj = make_adj(col_edges)
    T = T_init
    best_edges = list(col_edges)
    best_adj = [row[:] for row in adj]
    best_count = len(col_edges)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for it in range(max_iters):
        if elapsed() > DEADLINE - 10:
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
            if not added:
                continue
            # Adding always improves (more edges), accept
            if len(col_edges) > best_count:
                best_count = len(col_edges)
                best_edges = [list(e) for e in col_edges]
                best_adj = [row[:] for row in adj]
                
        elif move_type < 0.8:
            # Try to remove a random edge (to escape local optima)
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, c = col_edges[idx]
            # Store old state
            old_edges = [list(e) for e in col_edges]
            old_adj = [row[:] for row in adj]
            old_count = len(col_edges)
            
            _remove_edge(col_edges, adj, u, v)
            
            # After removal, try to add back a different edge or same edge with different colour
            # Actually, let's just accept the removal with some probability based on temperature
            # The idea: removal might allow a denser configuration later
            # For now, just track it; we'll try to add edges in subsequent iterations
            # Accept with probability 1 (always accept removals in SA to explore)
            # But this decreases edge count, so we need to be careful
            # Let's use Metropolis: accept with probability exp(-delta/T) where delta = new_obj - old_obj
            # Objective = -edge_count, so delta = (-new_count) - (-old_count) = old_count - new_count = 1
            delta = 1  # removing one edge increases objective by 1 (worse)
            if random.random() < 2.718281828 / (T + 1e-10):  # exp(-1/T) approx
                # Accept removal
                pass
            else:
                # Reject: restore
                col_edges[:] = old_edges
                adj[:] = old_adj
        
        else:
            # Try to recolor a random edge
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, old_c = col_edges[idx]
            new_c = random.randint(0, C-1)
            if new_c == old_c:
                continue
            
            # Check if recoloring is valid
            # Remove old edge, try to add with new colour
            old_edges = [list(e) for e in col_edges]
            old_adj = [row[:] for row in adj]
            
            _remove_edge(col_edges, adj, u, v)
            if _try_add(adj, u, v, new_c):
                _add_edge(col_edges, adj, u, v, new_c)
                col_edges[idx][2] = new_c  # update the stored colour
                # Edge count unchanged, so objective unchanged
                if len(col_edges) > best_count:
                    best_count = len(col_edges)
                    best_edges = [list(e) for e in col_edges]
                    best_adj = [row[:] for row in adj]
            else:
                # Restore
                col_edges[:] = old_edges
                adj[:] = old_adj
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # After SA, try local improvement on best state found
    if best_count > 0:
        improved = local_improve([list(e) for e in best_edges])
        if count_edges(improved) > best_count:
            best_edges = improved
            best_count = count_edges(improved)
    
    return best_edges

def construct():
    best = []
    best_count = 0
    
    # Run multiple greedy attempts with different seeds
    max_attempts = 30
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 20:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = result
            best_count = count
            
            # Try to improve locally
            if elapsed() < DEADLINE - 25:
                improved = local_improve([list(e) for e in best])
                improved_count = count_edges(improved)
                if improved_count > best_count:
                    best = improved
                    best_count = improved_count
        
        # Early exit if we've found a very dense graph
        if best_count >= 170:
            break
    
    # Run simulated annealing on the best solution found
    if best and elapsed() < DEADLINE - 30:
        sa_result = simulated_annealing(best, make_adj(best), max_iters=8000, T_init=5.0, T_min=0.005, cooling=0.998)
        sa_count = count_edges(sa_result)
        if sa_count > best_count:
            best = sa_result
            best_count = sa_count
    
    # Final local improvement
    if best and elapsed() < DEADLINE - 15:
        improved = local_improve([list(e) for e in best])
        improved_count = count_edges(improved)
        if improved_count > best_count:
            best = improved
            best_count = improved_count
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback: try a simple star
    star = [[0, v, v-1] for v in range(1, N)]
    if is_rainbow(star):
        return star
    
    return star
