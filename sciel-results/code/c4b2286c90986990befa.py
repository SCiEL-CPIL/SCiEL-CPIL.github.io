import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before the 150s kill

def _score_from_adj(adj):
    """O(n^3) score: max deg(u)+deg(v)+deg(w) over all triangles."""
    deg = [0]*N
    for i in range(N):
        deg[i] = sum(adj[i])
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _count_edges(adj):
    return sum(sum(row) for row in adj) // 2

def _get_edges_and_nonedges(adj):
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append((i,j))
            else:
                nonedges.append((i,j))
    return edges, nonedges

def _apply_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    # Multiple restarts with different initial structures
    num_restarts = 25
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Initialize: K_{8,8} + one internal edge
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        
        # Add 1 random internal edge for diversity
        if r % 2 == 0:
            # Internal edge in part A
            u, v = rng.sample(range(8), 2)
        else:
            # Internal edge in part B
            u, v = rng.sample(range(8, 16), 2)
        adj[u][v] = adj[v][u] = True
        
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
        
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Simulated Annealing
        start_temp = 4.0
        end_temp = 0.1
        max_iters = 8000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 1:
                break
            
            # Pick random edge to remove
            e_remove = edges[rng.randrange(len(edges))]
            # Pick random non-edge to add
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            # Ensure we're not removing and adding the same edge
            if (u == x and v == y) or (u == y and v == x):
                temp *= cooling_rate
                continue
            
            # Perform swap
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score_from_adj(adj)
            
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
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            else:
                # Undo
                _undo_swap(adj, deg, e_remove, e_add)
            
            temp *= cooling_rate
        
        # Check if we should continue
        if time.time() - t0 > DEADLINE - 5:
            break
    
    # Fallback if something went wrong
    if best_adj is None:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    # Verify edge count
    ec = _count_edges(best_adj)
    if ec != TARGET_EDGES:
        # Fallback to known-good
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_adj[i][j]]}
