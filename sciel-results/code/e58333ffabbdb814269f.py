import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    num_triples = len(triples)

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def is_valid_full(adj):
        for i, j, k in triples:
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            if bin(cm).count('1') >= 3:
                return False
        return True

    def try_add(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # Only triples containing u or v are affected
        # Actually, adding (u,v) increases common neighbors for triples containing u or v?
        # No, it changes N(u) and N(v).
        # A triple (a,b,c) is violated if |N(a) cap N(b) cap N(c)| >= 3.
        # Adding (u,v) adds v to N(u) and u to N(v).
        # So for a triple (a,b,c):
        # If u is in {a,b,c}, say a=u, then N(a) changes.
        # The new common neighbors are (OldCommon \ {v}) U ({v} if v in N(b) and v in N(c))
        # This logic is complex. 
        # Simpler: Just check all triples containing u or v?
        # No, adding (u,v) only affects the common neighbor count of triples that include u or v.
        # Because N(x) for x not in {u,v} is unchanged.
        # So we only need to check triples T such that u in T or v in T.
        
        # Triples containing u: (u, a, b)
        for a in range(n):
            if a == u: continue
            for b in range(a + 1, n):
                if b == u or b == v: continue
                # Triple is (u, a, b)
                # Common neighbors of u, a, b in NEW graph
                # N_new(u) = N_old(u) U {v}
                # N_new(a) = N_old(a)
                # N_new(b) = N_old(b)
                # Common = (N_old(u) U {v}) & N_old(a) & N_old(b)
                # = (N_old(u) & N_old(a) & N_old(b)) U ({v} & N_old(a) & N_old(b))
                
                old_cm = adj[u] & adj[a] & adj[b]
                # Mask out u, a, b
                mask = (1 << u) | (1 << a) | (1 << b)
                old_cm &= ~mask
                
                # Check if v is in N(a) and N(b)
                if (adj[a] >> v) & 1 and (adj[b] >> v) & 1:
                    # v is a new common neighbor
                    # But v might be u, a, or b? No, v is distinct.
                    # So new count = old_count + 1
                    if bin(old_cm).count('1') >= 2:
                        return False
                else:
                    # v is not a new common neighbor
                    # Count remains same
                    if bin(old_cm).count('1') >= 3:
                        # It was already invalid? Should not happen if adj was valid.
                        return False
        
        # Triples containing v (but not u, already covered)
        for a in range(n):
            if a == v: continue
            for b in range(a + 1, n):
                if b == v or b == u: continue
                # Triple is (v, a, b)
                old_cm = adj[v] & adj[a] & adj[b]
                mask = (1 << v) | (1 << a) | (1 << b)
                old_cm &= ~mask
                
                if (adj[a] >> u) & 1 and (adj[b] >> u) & 1:
                    if bin(old_cm).count('1') >= 2:
                        return False
                else:
                    if bin(old_cm).count('1') >= 3:
                        return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def repair(adj, max_iters=1000):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Find a violation
            viol = None
            for i, j, k in triples:
                cm = adj[i] & adj[j] & adj[k]
                mask = (1 << i) | (1 << j) | (1 << k)
                cm &= ~mask
                if bin(cm).count('1') >= 3:
                    viol = (i, j, k, cm)
                    break
            if viol is None:
                break
            i, j, k, cm = viol
            # Remove an edge (v, c) where v in {i,j,k} and c in common neighbors
            c_list = [x for x in range(n) if (cm >> x) & 1]
            if not c_list:
                break
            c = random.choice(c_list)
            v = random.choice([i, j, k])
            remove_edge(adj, v, c)
        return count_edges(adj)

    def greedy_add(adj):
        edges = count_edges(adj)
        remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        while remaining:
            if time.time() > deadline - 5:
                break
            random.shuffle(remaining)
            progress = False
            new_remaining = []
            for (u, v) in remaining:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    progress = True
                else:
                    new_remaining.append((u, v))
            remaining = new_remaining
            if not progress:
                break
        return edges

    def local_search(adj, max_rounds=200):
        edges = count_edges(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break
            found = False
            # Try swaps: remove one, add one or more
            random.shuffle(present)
            for (u1, v1) in present:
                if time.time() > deadline - 10:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                
                remove_edge(adj, u1, v1)
                edges -= 1
                
                # Try to add back
                added = 0
                # Try all absent edges
                # Optimization: Only try edges that might be valid
                new_absent = []
                for (u2, v2) in absent:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if try_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        edges += 1
                        added += 1
                    else:
                        new_absent.append((u2, v2))
                
                # Also try re-adding u1, v1
                if not ((adj[u1] >> v1) & 1):
                    if try_add(adj, u1, v1):
                        add_edge(adj, u1, v1)
                        edges += 1
                        added += 1
                
                if added == 0:
                    # Revert
                    add_edge(adj, u1, v1)
                    edges += 1
                    # Rebuild lists is expensive, just break and let greedy handle it?
                    # Or simply continue
                    break
                else:
                    found = True
                    # Rebuild present/absent
                    present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
                    absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
                    break
            
            if not found:
                # Try a kick: remove a random edge
                if present:
                    u1, v1 = random.choice(present)
                    remove_edge(adj, u1, v1)
                    edges -= 1
                    present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
                    absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        # Polish
        edges = greedy_add(adj)
        return edges

    rng = random.Random(42)
    best_edges = 0
    best_adj = None

    num_starts = 100

    for s in range(num_starts):
        if time.time() > deadline - 20:
            break

        # Random start
        adj = [0] * n
        density = 0.5 + 0.1 * (s % 5)
        for (i, j) in all_edges:
            if rng.random() < density:
                add_edge(adj, i, j)
        
        ec = repair(adj)
        ec = greedy_add(adj)
        ec = local_search(adj)
        
        if is_valid_full(adj):
            if ec > best_edges:
                best_edges = ec
                best_adj = [a for a in adj]

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
