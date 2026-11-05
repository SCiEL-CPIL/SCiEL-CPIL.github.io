import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_edges)

    # Precompute: for each vertex, list of (edge_index, other_vertex)
    vertex_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(all_edges):
        vertex_edges[u].append((idx, v))
        vertex_edges[v].append((idx, u))

    # Precompute triples (i<j<k) and for each triple the list of edges among them
    triples = []
    triple_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                # edges among {i,j,k}: (i,j),(i,k),(j,k)
                triple_edges.append(None)  # filled below

    # Map from (u,v) to edge index
    edge_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = idx

    # For each triple, the 3 internal edge indices
    for t, (i, j, k) in enumerate(triples):
        triple_edges[t] = (edge_idx[(i, j)], edge_idx[(i, k)], edge_idx[(j, k)])

    num_triples = len(triples)

    def is_valid_full(adj):
        """Check full validity: no triple has >=3 common neighbors."""
        for t in range(num_triples):
            i, j, k = triples[t]
            common = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            common &= ~mask
            if common & (common - 1) & (common - 2):  # at least 3 bits
                # more careful: count bits
                if bin(common).count('1') >= 3:
                    return False
        return True

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_add(adj, u, v):
        """Try to add edge (u,v). Returns True if valid after adding."""
        # After adding (u,v), a violation occurs if some triple has >=3 common neighbors.
        # The new edge (u,v) can only create violations in triples that contain u or v
        # (since common neighbors of a triple not containing u or v are unchanged).
        # For a triple containing u (say {u,a,b}): common neighbors = N(u)∩N(a)∩N(b).
        #   Adding v to N(u) means v is now a common neighbor iff v∈N(a) and v∈N(b).
        #   So the count increases by 1 iff (adj[v]&adj[a]&(1<<b)) and (adj[v]&adj[b]&(1<<a)).
        # Similarly for triples containing v.
        # Also triples containing both u and v: {u,v,a}: common = N(u)∩N(v)∩N(a).
        #   Adding edge doesn't change N(u)∩N(v) membership for a (a is in both N(u) and N(v)
        #   iff a adjacent to both; the new edge makes u∈N(v) and v∈N(u) but a is checked
        #   against N(u) and N(v): a∈N(u) means adj[u] has bit a, which is unchanged by adding (u,v)
        #   unless a is u or v. So for triple {u,v,a}, common neighbors unchanged? 
        #   Actually N(u) gains v, N(v) gains u. For a (third vertex), a∈N(u)∩N(v)∩N(a) is
        #   a∈N(u) and a∈N(v) and a∈N(a)=false. So a is never a common neighbor of {u,v,a}.
        #   The common neighbors of {u,v,a} are vertices x≠u,v,a with x∈N(u)∩N(v)∩N(a).
        #   Adding (u,v): N(u) gains v, N(v) gains u. x∈N(u) now includes x=v? No, x≠v.
        #   So for x≠u,v,a, x∈N(u)∩N(v)∩N(a) is unchanged. So triple {u,v,a} unaffected.
        
        # So only need to check triples containing exactly one of {u,v}.
        # For each a,b such that {u,a,b} is a triple (a<b, and u not in {a,b}):
        #   v becomes common neighbor iff v∈N(a) and v∈N(b), i.e. adj[v] has bits a and b.
        #   If so, the triple's common-neighbor count increases by 1.
        #   Violation if it was already 2 and now becomes 3.
        
        # Efficient: for each a adjacent to v, for each b adjacent to v with b>a,
        #   check if {u,a,b} is a triple (u not in {a,b}, and the triple is valid ordering).
        #   Actually we need u,a,b distinct. And we need to know current common-neighbor count of {u,a,b}.
        
        # Simpler: just check all triples containing u or v directly (there are at most 
        # C(19,2) + C(19,2) - C(18,1)*... roughly 2*171 = 342 triples, minus overlaps).
        # For n=20, triples containing u: C(19,2)=171. Same for v. Overlap (containing both): C(18,1)=18.
        # Total distinct: 171+171-18 = 324. That's a lot to check each time.
        
        # Better: only check triples where the count could reach 3.
        # A triple {u,a,b} (containing u, not v) gains v as common neighbor iff v∈N(a)∩N(b).
        # So: for each a with adj[v] bit a set, for each b with adj[v] bit b set, b>a,
        #   and u not in {a,b}: check triple {min(u,a,b), ..., ...}.
        #   Need current common-neighbor count of that triple to be exactly 2.
        
        # Let me just do the direct check but only for affected triples, computing count on the fly.
        # For speed, precompute for each triple its current common-neighbor bitmask? That's 1140 triples * 20 bits.
        
        # I'll use a hybrid: maintain triple_common[t] = bitmask of common neighbors of triple t.
        # This requires updating when edges are added/removed.
        
        # For now, do direct check on affected triples only.
        affected = set()
        for a in range(n):
            if a == u:
                continue
            if not (adj[v] >> a) & 1:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[v] >> b) & 1:
                    continue
                # Triple {u, a, b}: check if v is a common neighbor (yes, by construction)
                # and if current count (before adding) was >= 2, then after adding it's >= 3.
                # Current common neighbors of {u,a,b} (before adding (u,v)):
                #   N(u)∩N(a)∩N(b), excluding {u,a,b}.
                #   v is in this iff v∈N(u) (not yet, since we haven't added) and v∈N(a) and v∈N(b).
                #   v∈N(u) is false (edge not added), so v is NOT a current common neighbor.
                #   So current count doesn't include v. After adding, v is added iff v∈N(a)∩N(b) (true).
                #   So new count = old count + 1. Violation iff old count >= 2.
                cm = adj[u] & adj[a] & adj[b]
                mask = (1 << u) | (1 << a) | (1 << b)
                cm &= ~mask
                if bin(cm).count('1') >= 2:
                    return False
        # Same for triples containing v, not u:
        for a in range(n):
            if a == v:
                continue
            if not (adj[u] >> a) & 1:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[u] >> b) & 1:
                    continue
                cm = adj[v] & adj[a] & adj[b]
                mask = (1 << v) | (1 << a) | (1 << b)
                cm &= ~mask
                if bin(cm).count('1') >= 2:
                    return False
        return True

    # This try_add is O(deg(v)^2 + deg(u)^2) which for dense graphs is ~400*2 = 800 ops.
    # For 190 edges, that's ~152k ops per full pass. Very fast.

    # Strategy: 
    # 1. Build a good initial graph (e.g., complement of a sparse graph, or random dense + repair).
    # 2. Greedy add edges in random order, using try_add.
    # 3. Local search: try edge swaps (remove one, add another) to improve.
    # 4. Repeat with different random seeds.

    best_edges = 0
    best_adj = None

    # Precompute a good structured start: use a known construction.
    # The complement of a 20-vertex graph with min degree considerations.
    # Actually, let's use random dense starts + repair, plus a few structured ones.

    rng = random.Random(42)

    def repair(adj):
        """Remove edges until valid. Returns edge count."""
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Find a violating triple
            found = False
            for t in range(num_triples):
                i, j, k = triples[t]
                cm = adj[i] & adj[j] & adj[k]
                mask = (1 << i) | (1 << j) | (1 << k)
                cm &= ~mask
                if bin(cm).count('1') >= 3:
                    # Remove an edge: pick a random common neighbor c and a random vertex from {i,j,k}
                    c_list = [x for x in range(n) if (cm >> x) & 1]
                    c = rng.choice(c_list)
                    v = rng.choice([i, j, k])
                    remove_edge(adj, v, c)
                    found = True
                    break
            if not found:
                break
        return count_edges(adj)

    def greedy_add(adj):
        """Greedy add edges in random order."""
        edges = count_edges(adj)
        remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        while remaining:
            if time.time() > deadline - 3:
                break
            rng.shuffle(remaining)
            progress = False
            for (u, v) in remaining[:]:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    progress = True
                    remaining.remove((u, v))
                else:
                    remaining.remove((u, v))
            if not progress and not remaining:
                break
            if not progress:
                break
        return edges

    def local_search(adj):
        """Edge-swap local search to improve edge count."""
        edges = count_edges(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        max_rounds = 50
        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break
            rng.shuffle(present)
            rng.shuffle(absent)
            improved = False
            for (u1, v1) in present[:]:
                if time.time() > deadline - 3:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                # Try removing (u1,v1) and adding some (u2,v2)
                remove_edge(adj, u1, v1)
                edges -= 1
                best_add = None
                for (u2, v2) in absent:
                    if (u2, v2) == (u1, v1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    if try_add(adj, u2, v2):
                        best_add = (u2, v2)
                        break
                if best_add:
                    add_edge(adj, best_add[0], best_add[1])
                    edges += 1
                    # Update present/absent
                    present.remove((u1, v1))
                    absent.remove(best_add)
                    present.append(best_add)
                    absent.append((u1, v1))
                    improved = True
                    break
                else:
                    # Revert
                    add_edge(adj, u1, v1)
                    edges += 1
            if not improved:
                break
        return edges

    # Try multiple strategies
    num_starts = 30

    for s in range(num_starts):
        if time.time() > deadline - 10:
            break

        adj = [0] * n

        # Different start densities
        if s % 5 == 0:
            # Structured: complete graph minus a matching-like pattern
            for i in range(n):
                for j in range(i + 1, n):
                    # Skip edges where (i+j) % 4 == 0 to create structure
                    if (i + j) % 4 != 0:
                        add_edge(adj, i, j)
        elif s % 5 == 1:
            # Random dense
            for (i, j) in all_edges:
                if rng.random() < 0.7:
                    add_edge(adj, i, j)
        elif s % 5 == 2:
            # Random medium
            for (i, j) in all_edges:
                if rng.random() < 0.6:
                    add_edge(adj, i, j)
        elif s % 5 == 3:
            # Turan-like: 4 parts of 5, complete multipartite
            parts = [[i for i in range(k * 5, (k + 1) * 5)] for k in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    # Find parts
                    pi = i // 5
                    pj = j // 5
                    if pi != pj:
                        add_edge(adj, i, j)
        else:
            # Random with slight structure
            for (i, j) in all_edges:
                if rng.random() < 0.65 + 0.05 * (s % 10):
                    add_edge(adj, i, j)

        # Repair to validity
        ec = repair(adj)

        # Greedy add
        ec = greedy_add(adj)

        # Local search
        ec = local_search(adj)

        if ec > best_edges:
            best_edges = ec
            best_adj = [a for a in adj]

    # If no valid graph found, return empty
    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
