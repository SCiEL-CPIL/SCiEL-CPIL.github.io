"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 115.0  # safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE - 2:
            return
        
        # Upper bound
        available = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                available |= (1 << i)
        
        ub = popcount(current_mask) + popcount(available)
        if ub <= best[0]:
            return
        
        if idx == n:
            sz = popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        v = idx
        bit = 1 << v
        
        # Option 1: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v
        if not (forbidden_mask & bit):
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
    
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    
    # Start with complete tripartite graph K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    best_adj = adj[:]
    
    # Initial evaluation
    current_score = _max_triangle_free_induced(current_adj, n)
    best_score = current_score
    
    random.seed(42)
    
    # Local Search with Simulated Annealing
    T = 1.0
    no_improve_count = 0
    max_no_improve = 500
    
    while _time_left() > 5:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        new_adj = current_adj[:]
        if (new_adj[u] >> v) & 1:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(new_adj, n):
            continue
        
        # Evaluate score
        new_score = _max_triangle_free_induced(new_adj, n)
        
        # Accept if better or with probability based on simulated annealing
        if new_score < current_score:
            current_adj = new_adj
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_adj = new_adj[:]
                best_score = new_score
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_score - current_score
            if delta == 0:
                current_adj = new_adj
                current_score = new_score
            elif random.random() < (2.718281828 ** (-delta / T)):
                current_adj = new_adj
                current_score = new_score
            no_improve_count += 1
        
        # Cool down
        if no_improve_count >= max_no_improve:
            T *= 0.9
            no_improve_count = 0
            if T < 0.01:
                # Restart with fresh random perturbation
                current_adj = best_adj[:]
                current_score = best_score
                T = 0.5
                no_improve_count = 0
    
    # Return edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
