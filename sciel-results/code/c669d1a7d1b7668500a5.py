import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    full_mask = (1 << n) - 1

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no 3 vertices have 3 common neighbours (K3,3-free)."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a K3,3 violation. Returns (i,j,k) or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k)
        return None

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if adj[u] & (1 << v):
            return True
        old_u = adj[u]
        old_v = adj[v]
        
        # Check 1: For all w: |N(u) ∩ N(v) ∩ N(w)| < 3
        for w in range(n):
            if w == u or w == v:
                continue
            common = old_u & old_v & adj[w]
            if common.bit_count() >= 3:
                return False
        
        # Check 2: v is added to N(u), so for pairs (w1,w2) where v is common neighbor
        for w1 in range(n):
            if w1 == u or w1 == v:
                continue
            if not (adj[w1] & (1 << v)):
                continue
            for w2 in range(w1+1, n):
                if w2 == u or w2 == v:
                    continue
                if not (adj[w2] & (1 << v)):
                    continue
                common = old_u & adj[w1] & adj[w2]
                if common.bit_count() >= 2:
                    return False
        
        # Check 3: u is added to N(v), so for pairs (w1,w2) where u is common neighbor
        for w1 in range(n):
            if w1 == u or w1 == v:
                continue
            if not (adj[w1] & (1 << u)):
                continue
            for w2 in range(w1+1, n):
                if w2 == u or w2 == v:
                    continue
                if not (adj[w2] & (1 << u)):
                    continue
                common = old_v & adj[w1] & adj[w2]
                if common.bit_count() >= 2:
                    return False
        
        return True

    def make_valid(adj, max_iter=300):
        """Remove edges to make graph K3,3-free."""
        for _ in range(max_iter):
            if time.time() > deadline - 15:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j] & adj[k]
            if cij == 0:
                break
            # Remove edge (i, x) where x is a common neighbor
            cn = cij
            v = cn & (-cn)
            x = v.bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
        return adj

    def fast_local_search(adj, max_iter=5000):
        """Fast local search using efficient edge-addition check."""
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 5:
            # Try to add an edge
            added = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if added:
                iters += 1
                continue
            
            # Try swaps
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
            
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            for e_rem in edges_in[:50]:
                if swapped:
                    break
                for e_add in edges_out[:50]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid(adj):
                        swapped = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if not swapped:
                    continue
            
            if not swapped:
                break
            iters += 1
        
        return adj

    def simulated_annealing(adj, max_iter=3000):
        """Simulated annealing to improve the graph."""
        adj = adj[:]
        best_adj = adj[:]
        best_count = count_edges(adj)
        current_count = best_count
        temp = 2.0
        cooling = 0.9995
        
        iters = 0
        while iters < max_iter and time.time() < deadline - 5:
            # Random move: add or remove an edge
            u, v = random.choice(all_pairs)
            old_u, old_v = adj[u], adj[v]
            
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                delta = -1
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                delta = 1
            
            new_count = current_count + delta
            
            if not is_valid(adj):
                # Reject invalid moves
                adj[u], adj[v] = old_u, old_v
                iters += 1
                continue
            
            if new_count > current_count:
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
            else:
                # Accept with probability exp(delta/temp)
                if random.random() < np.exp(delta / temp):
                    current_count = new_count
                else:
                    adj[u], adj[v] = old_u, old_v
            
            temp *= cooling
            iters += 1
        
        return best_adj

    # Best solution tracking
    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start from complete graph, remove edges to make valid
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = make_valid(adj, max_iter=500)
    adj = fast_local_search(adj, max_iter=4000)
    update_best(adj)

    # Strategy 2: Multiple random restarts with greedy construction
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        random.seed(2024 + trial * 17)
        
        # Start with random sparse graph
        adj = [0] * n
        p = 0.3 + trial * 0.03
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Make valid
        adj = make_valid(adj, max_iter=200)
        
        # Greedy add
        adj = fast_local_search(adj, max_iter=3000)
        update_best(adj)
        
        # Simulated annealing polish
        if time.time() < deadline - 15:
            adj_sa = simulated_annealing(adj, max_iter=2000)
            update_best(adj_sa)

    # Strategy 3: Structured initializations - bipartite-like structures
    for split in range(6, 15):
        if time.time() > deadline - 20:
            break
        adj = [0] * n
        # Vertices 0..split-1 on one side, split..n-1 on other
        for i in range(split):
            for j in range(split, n):
                if random.random() < 0.85:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Make valid
        adj = make_valid(adj, max_iter=150)
        
        adj = fast_local_search(adj, max_iter=2500)
        update_best(adj)

    # Strategy 4: Further improvement on best
    if best_edges:
        adj = best_adj[:]
        adj = fast_local_search(adj, max_iter=4000)
        update_best(adj)
        
        # Final SA polish
        if time.time() < deadline - 10:
            adj_sa = simulated_annealing(adj, max_iter=3000)
            update_best(adj_sa)

    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
