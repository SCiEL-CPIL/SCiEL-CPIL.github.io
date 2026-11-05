"""Rewritten program: hill-climbing + simulated annealing to minimize max triangle degree sum."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill

def _score(adj, deg):
    """Compute max degree sum over all triangles."""
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _swap(adj, deg, e_remove, e_add):
    """Perform an edge swap: remove e_remove, add e_add. Returns (new_adj, new_deg)."""
    u, v = e_remove
    x, y = e_add
    # Remove (u,v)
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    # Add (x,y)
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1
    return adj, deg

def _undo_swap(adj, deg, e_remove, e_add):
    """Undo an edge swap."""
    u, v = e_remove
    x, y = e_add
    # Undo add (x,y)
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    # Undo remove (u,v)
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1
    return adj, deg

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    # Initialize: K_{8,8} + one internal edge
    adj = [[False]*N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    adj[0][1] = adj[1][0] = True  # internal edge in part A
    
    deg = [0]*N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    
    cur_score = _score(adj, deg)
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_score = cur_score
    
    # Precompute all possible edges and non-edges
    all_edges = [(i,j) for i in range(N) for j in range(i+1, N)]
    
    def get_edges_and_nonedges(adj):
        edges = []
        nonedges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    edges.append((i,j))
                else:
                    nonedges.append((i,j))
        return edges, nonedges
    
    edges, nonedges = get_edges_and_nonedges(adj)
    
    # Simulated annealing parameters
    max_iters = 200000
    start_temp = 5.0
    end_temp = 0.1
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    
    temp = start_temp
    iters_done = 0
    
    while iters_done < max_iters:
        elapsed = time.time() - t0
        if elapsed > DEADLINE:
            break
        
        if iters_done % 5000 == 0 and iters_done > 0:
            # Re-verify edge count
            ec = sum(sum(row) for row in adj) // 2
            if ec != TARGET_EDGES:
                break
            temp *= cooling_rate ** 5000
        
        # Pick random edge to remove
        e_remove = edges[rng.randrange(len(edges))]
        # Pick random non-edge to add
        e_add = nonedges[rng.randrange(len(nonedges))]
        
        u, v = e_remove
        x, y = e_add
        
        # Ensure we're not removing and adding the same edge
        if (u == x and v == y) or (u == y and v == x):
            iters_done += 1
            continue
        
        # Perform swap
        adj, deg = _swap(adj, deg, e_remove, e_add)
        new_score = _score(adj, deg)
        
        # Decide whether to accept
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.5
        else:
            delta = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()
        
        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            
            # Update edge/non-edge lists
            # e_remove becomes a non-edge, e_add becomes an edge
            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)
        else:
            # Undo
            adj, deg = _undo_swap(adj, deg, e_remove, e_add)
        
        iters_done += 1
        temp *= cooling_rate
    
    # Final check: use best found
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_adj[i][j]]}
