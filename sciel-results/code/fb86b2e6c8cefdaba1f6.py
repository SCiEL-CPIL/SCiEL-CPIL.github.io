import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin for 150s limit

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
        """Check if no 3 vertices have 3 common neighbors (K3,3-free)."""
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
        
        # Check 1: For all w, |N(u) ∩ N(v) ∩ N(w)| < 3
        for w in range(n):
            if w == u or w == v:
                continue
            common = old_u & old_v & adj[w]
            if common.bit_count() >= 3:
                return False
        
        # Check 2: v is added to N(u). For pairs (w1,w2) where v is common neighbor of w1,w2,
        # old |N(u) ∩ N(w1) ∩ N(w2)| must be < 2
        # Build list of neighbors of v (excluding u)
        v_neighbors = []
        mask = old_v
        while mask:
            low = mask & (-mask)
            x = low.bit_length() - 1
            if x != u:
                v_neighbors.append(x)
            mask ^= low
        
        if len(v_neighbors) >= 2:
            for idx1 in range(len(v_neighbors)):
                w1 = v_neighbors[idx1]
                for idx2 in range(idx1+1, len(v_neighbors)):
                    w2 = v_neighbors[idx2]
                    common = old_u & adj[w1] & adj[w2]
                    if common.bit_count() >= 2:
                        return False
        
        # Check 3: u is added to N(v). For pairs (w1,w2) where u is common neighbor of w1,w2,
        # old |N(v) ∩ N(w1) ∩ N(w2)| must be < 2
        u_neighbors = []
        mask = old_u
        while mask:
            low = mask & (-mask)
            x = low.bit_length() - 1
            if x != v:
                u_neighbors.append(x)
            mask ^= low
        
        if len(u_neighbors) >= 2:
            for idx1 in range(len(u_neighbors)):
                w1 = u_neighbors[idx1]
                for idx2 in range(idx1+1, len(u_neighbors)):
                    w2 = u_neighbors[idx2]
                    common = old_v & adj[w1] & adj[w2]
                    if common.bit_count() >= 2:
                        return False
        
        return True

    def greedy_add(adj):
        """Try to add edges greedily until no more can be added."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    improved = True
                    break
                else:
                    adj[u], adj[v] = old_u, old_v
        return adj

    def fast_local_search(adj, max_iter=3000):
        """Fast local search using efficient edge-addition check."""
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 5:
            # Try to add an edge - scan in random order for diversity
            added = False
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
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
            
            # Try swaps - more systematic approach
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
            
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            # Try more swap combinations
            for e_rem in edges_in[:60]:
                if swapped:
                    break
                (u1, v1) = e_rem
                for e_add in edges_out[:60]:
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    # Skip if adding edge would be redundant
                    if adj[u2] & (1 << v2):
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
    
    # Iteratively remove edges that cause violations - use smarter removal
    for _ in range(500):
        if time.time() > deadline - 30:
            break
        viol = find_violation(adj)
        if viol is None:
            break
        i, j, k = viol
        cij = adj[i] & adj[j] & adj[k]
        if cij == 0:
            break
        # Try removing edges from each of the three vertices to common neighbors
        # and pick the one that allows most future additions
        best_removal = None
        best_score = -1
        for x in [i, j, k]:
            for y in [i, j, k]:
                if x == y:
                    continue
                if adj[x] & (1 << y):
                    continue
                # Try removing edge (x, y) if it helps
                # Actually, we should remove an edge incident to one of i,j,k
                # that goes to a common neighbor
                pass
        
        # Simpler: just remove edge from i to a common neighbor
        cn = cij
        v = cn & (-cn)
        x = v.bit_length() - 1
        adj[i] &= ~(1 << x)
        adj[x] &= ~(1 << i)
    
    adj = fast_local_search(adj, max_iter=4000)
    update_best(adj)

    # Strategy 2: Multiple random restarts with greedy construction
    for trial in range(20):
        if time.time() > deadline - 25:
            break
        random.seed(2024 + trial * 17)
        
        # Start with random sparse graph - vary density
        adj = [0] * n
        p = 0.25 + (trial % 10) * 0.05
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Make valid
        for _ in range(200):
            if time.time() > deadline - 20:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j] & adj[k]
            if cij == 0:
                break
            v = cij & (-cij)
            x = v.bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
        
        # Greedy add with more iterations
        adj = fast_local_search(adj, max_iter=3000)
        update_best(adj)

    # Strategy 3: Structured initializations
    # Try complete bipartite-like structures with varying splits
    for split in range(5, 16):
        if time.time() > deadline - 15:
            break
        adj = [0] * n
        # Vertices 0..split-1 on one side, split..n-1 on other
        for i in range(split):
            for j in range(split, n):
                if random.random() < 0.85:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Make valid
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j] & adj[k]
            if cij == 0:
                break
            v = cij & (-cij)
            x = v.bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
        
        adj = fast_local_search(adj, max_iter=2500)
        update_best(adj)

    # Strategy 4: Further improvement on best - multiple passes
    if best_edges:
        adj = best_adj[:]
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            adj = fast_local_search(adj, max_iter=4000)
            update_best(adj)

    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
