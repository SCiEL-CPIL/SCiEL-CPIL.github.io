import time
import random
import numpy as np
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.92

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    def _edge_count(adj):
        return sum(a.bit_count() for a in adj) // 2
    
    def _can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if adj[u] & (1 << v):
            return False
        
        # Case 1: v becomes a new common neighbor of some triple containing u
        # For any pair (a,b) in N(v), check if |N(u) ∩ N(a) ∩ N(b)| >= 2
        # (because after adding u-v, v would be the 3rd common neighbor)
        nv = adj[v]
        for a in range(N):
            if a == u or a == v or not (nv >> a) & 1:
                continue
            for b in range(a+1, N):
                if b == u or b == v or not (nv >> b) & 1:
                    continue
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        
        # Case 2: u becomes a new common neighbor of some triple containing v
        nu = adj[u]
        for a in range(N):
            if a == u or a == v or not (nu >> a) & 1:
                continue
            for b in range(a+1, N):
                if b == u or b == v or not (nu >> b) & 1:
                    continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 2:
                    return False
        
        return True
    
    def _greedy_fill(adj, time_limit):
        """Greedily add missing edges in random order."""
        missing = [(i, j) for i in range(N) for j in range(i+1, N) 
                   if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > time_limit:
                break
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def _find_all_violations(adj):
        """Find all triples with >= 3 common neighbors."""
        violations = []
        for u in range(N):
            for v in range(u+1, N):
                for w in range(v+1, N):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        violations.append((u, v, w, common))
        return violations
    
    def _repair(adj, time_limit):
        """Remove edges to eliminate K3,3 violations."""
        while time.time() < time_limit:
            viols = _find_all_violations(adj)
            if not viols:
                break
            # Pick a random violation and break it
            u, v, w, common = random.choice(viols)
            common_list = [i for i in range(N) if (common >> i) & 1]
            # Remove one edge from the common neighborhood structure
            # Strategy: remove edge between one of {u,v,w} and one common neighbor
            target = random.choice([u, v, w])
            neighbor = random.choice(common_list)
            adj[target] &= ~(1 << neighbor)
            adj[neighbor] &= ~(1 << target)
        return adj
    
    def _kick_and_fill(adj, kick_size, time_limit):
        """Remove kick_size random edges, then greedy fill."""
        existing = [(i, j) for i in range(N) for j in range(i+1, N) 
                    if (adj[i] & (1 << j))]
        if not existing:
            return adj
        random.shuffle(existing)
        for i in range(min(kick_size, len(existing))):
            u, v = existing[i]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        return _greedy_fill(adj, time_limit)
    
    def _local_search(adj, max_time):
        """Local search with kicks to escape local optima."""
        ls_start = time.time()
        best_adj = adj[:]
        best_count = _edge_count(adj)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() - ls_start < max_time and no_improve < max_no_improve:
            if time.time() > deadline - 3:
                break
            
            # Try greedy fill first
            adj_filled = _greedy_fill(adj[:], deadline)
            count = _edge_count(adj_filled)
            if count > best_count:
                best_count = count
                best_adj = adj_filled[:]
                adj = adj_filled[:]
                no_improve = 0
            else:
                # Kick: remove 2-4 random edges
                kick_size = random.choice([2, 3, 4])
                adj_kicked = _kick_and_fill(adj, kick_size, deadline)
                count = _edge_count(adj_kicked)
                if count > best_count:
                    best_count = count
                    best_adj = adj_kicked[:]
                    adj = adj_kicked[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    adj = best_adj[:]
        
        return best_adj, best_count
    
    best_adj = [0] * N
    best_count = 0
    
    # Phase 1: Multiple random restarts with repair + local search
    restarts = 0
    while time.time() < deadline - 5 and restarts < 50:
        restarts += 1
        
        # Random initial graph
        adj = [0] * N
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < 0.4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Repair to make K3,3-free
        adj = _repair(adj, deadline)
        
        # Greedy fill
        adj = _greedy_fill(adj, deadline)
        
        # Local search
        adj_ls, count_ls = _local_search(adj, 3.0)
        
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls[:]
    
    # Phase 2: Intensive local search on best solution
    if time.time() < deadline - 2:
        adj_final, count_final = _local_search(best_adj, deadline - time.time() - 2)
        if count_final > best_count:
            best_count = count_final
            best_adj = adj_final[:]
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] & (1 << v)):
                edges.append([u, v])
    return edges
