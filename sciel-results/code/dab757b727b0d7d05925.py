import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s limit
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += adj[i].bit_count()
        return c // 2
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def has_violation(adj):
        """Check if any triple has >= 3 common neighbors."""
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if common.bit_count() >= 3:
                        return True
        return False
    
    def clean(adj):
        """Remove edges until no violations."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        if common.bit_count() >= 3:
                            # Remove one edge: pick a common neighbor v and remove (i,v)
                            v = (common & -common).bit_length() - 1
                            adj[i] &= ~(1 << v)
                            adj[v] &= ~(1 << i)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) creates a violation."""
        # Only triples containing u or v can be affected
        # For triple {u, x, y}: j=v becomes common neighbor if v in N(x) and v in N(y)
        #   Need |N(u) & N(x) & N(y)| <= 1 (since v would be added to N(u))
        #   But only if v is in N(x) and N(y)
        # For triple {v, x, y}: i=u becomes common neighbor if u in N(x) and u in N(y)
        #   Need |N(v) & N(x) & N(y)| <= 1
        # For triple {u, v, x}: common neighbors gain u (if u in N(x)) and v (if v in N(x))
        #   Need |N(u) & N(v) & N(x) ∪ {u if u in N(x)} ∪ {v if v in N(x)}| <= 2
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                
                # Triple {u, x, y}
                # v becomes common neighbor if v in N(x) and v in N(y)
                if (adj[x] & (1 << v)) and (adj[y] & (1 << v)):
                    common = adj[u] & adj[x] & adj[y]
                    if common.bit_count() >= 2:
                        return False
                
                # Triple {v, x, y}
                # u becomes common neighbor if u in N(x) and u in N(y)
                if (adj[x] & (1 << u)) and (adj[y] & (1 << u)):
                    common = adj[v] & adj[x] & adj[y]
                    if common.bit_count() >= 2:
                        return False
        
        # Triple {u, v, x} for each x
        for x in range(n):
            if x == u or x == v:
                continue
            # Common neighbors of {u,v,x} after adding (u,v)
            # = (N(u) ∩ N(v) ∩ N(x)) ∪ {u if u∈N(x)} ∪ {v if v∈N(x)}
            common_base = adj[u] & adj[v] & adj[x]
            extra = 0
            if adj[x] & (1 << u):
                extra += 1
            if adj[x] & (1 << v):
                extra += 1
            if common_base.bit_count() + extra >= 3:
                return False
        
        return True
    
    def grow(adj):
        """Add all valid edges."""
        added = True
        while added:
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        if can_add(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            added = True
    
    def local_search(adj):
        """Try edge swaps to improve."""
        improved = True
        while improved:
            improved = False
            # Try removing an edge and adding another
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        # Try to find an edge to remove that allows adding (i,j)
                        for k in range(n):
                            for l in range(k+1, n):
                                if (adj[k] & (1 << l)) and k != i and l != i and k != j and l != j:
                                    # Remove (k,l), try add (i,j)
                                    adj[k] &= ~(1 << l)
                                    adj[l] &= ~(1 << k)
                                    if can_add(adj, i, j):
                                        adj[i] |= (1 << j)
                                        adj[j] |= (1 << i)
                                        # Try to add back (k,l)
                                        if not can_add(adj, k, l):
                                            # Keep the swap
                                            improved = True
                                        else:
                                            adj[k] |= (1 << l)
                                            adj[l] |= (1 << k)
                                    else:
                                        # Restore (k,l)
                                        adj[k] |= (1 << l)
                                        adj[l] |= (1 << k)
                                    if improved:
                                        break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
        return adj
    
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        adj = [0] * n
        
        # Random initialization with varying density
        density = 0.3 + 0.4 * (seed % 10) / 10.0
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Clean
        clean(adj)
        
        # Grow
        grow(adj)
        
        # Local search
        if time.time() < deadline - 10:
            local_search(adj)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges(adj)
    
    return best_edges
