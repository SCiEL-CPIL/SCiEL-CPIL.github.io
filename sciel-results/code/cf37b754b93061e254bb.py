import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_edges)

    # Precompute triples (i<j<k)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))

    num_triples = len(triples)

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
        # Only triples containing exactly one of {u,v} can be affected.
        # Triple {u,a,b} (u not in {a,b}): v becomes common neighbor iff v∈N(a)∩N(b).
        #   New count = old count + 1. Violation iff old count >= 2.
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
    # 1. Build a good initial graph (structured + random starts).
    # 2. Repair to validity.
    # 3. Greedy add edges.
    # 4. Best-improvement local search (edge swaps).
    # 5. Polish: try all possible edge additions.
    # 6. Repeat with different starts.

    best_edges = 0
    best_adj = None

    rng = random.Random(42)

    def repair(adj):
        """Remove edges until valid. Returns edge count."""
        max_iters = 2000
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

    def best_improvement_local_search(adj):
        """Best-improvement edge-swap local search to improve edge count."""
        edges = count_edges(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        max_rounds = 30
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            best_swap = None  # (u1,v1,u2,v2)
            best_gain = 0  # We're trying to maintain edge count (swap) or gain (if possible)
            
            # Try all present edges for removal
            for (u1, v1) in present:
                if time.time() > deadline - 5:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                
                remove_edge(adj, u1, v1)
                edges -= 1
                
                # Try all absent edges for addition
                for (u2, v2) in absent:
                    if (u2, v2) == (u1, v1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    if try_add(adj, u2, v2):
                        # This swap is valid: remove (u1,v1), add (u2,v2)
                        # Net edge count change: 0 (swap)
                        # But we prefer swaps that allow further additions later
                        # For now, any valid swap is acceptable
                        if best_swap is None:
                            best_swap = (u1, v1, u2, v2)
                            break
                        # Keep first valid swap for this removal
                
                add_edge(adj, u1, v1)
                edges += 1
                
                if best_swap is not None:
                    break
            
            if best_swap is None:
                break
            
            # Apply the best swap
            u1, v1, u2, v2 = best_swap
            remove_edge(adj, u1, v1)
            add_edge(adj, u2, v2)
            present.remove((u1, v1))
            absent.remove((u2, v2))
            present.append((u2, v2))
            absent.append((u1, v1))
        
        return edges

    def polish(adj):
        """Try to add any remaining edges that can be added."""
        edges = count_edges(adj)
        remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        added = True
        while added:
            if time.time() > deadline - 3:
                break
            added = False
            new_remaining = []
            for (u, v) in remaining:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    added = True
                else:
                    new_remaining.append((u, v))
            remaining = new_remaining
        return edges

    def make_paley_start():
        """Paley-type graph: connect i,j if (i-j)^2 is a quadratic residue mod 19 (or similar)."""
        adj = [0] * n
        # Use a prime p close to n. For n=20, use p=19 and add vertex 19 specially.
        # Actually, let's use a simpler structured approach: 
        # Complete graph minus edges where (i-j) mod 5 is in {1, 4} (creates 5-cycles of non-edges)
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 5
                if d not in (1, 4):  # skip edges at distance 1 or 4 mod 5
                    add_edge(adj, i, j)
        return adj

    def make_turan_start(num_parts):
        """Complete multipartite graph with num_parts equal parts."""
        adj = [0] * n
        part_size = n // num_parts
        for i in range(n):
            for j in range(i + 1, n):
                pi = i // part_size
                pj = j // part_size
                if pi != pj:
                    add_edge(adj, i, j)
        return adj

    def make_random_start(density):
        """Random graph with given edge probability."""
        adj = [0] * n
        for (i, j) in all_edges:
            if rng.random() < density:
                add_edge(adj, i, j)
        return adj

    def make_circulant_start(skip_set):
        """Circulant graph: connect i to i+d for d not in skip_set."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d not in skip_set and (n - d) not in skip_set:
                    add_edge(adj, i, j)
        return adj

    # Define various structured starts
    start_generators = [
        make_paley_start,
        lambda: make_turan_start(4),
        lambda: make_turan_start(5),
        lambda: make_circulant_start({1, 4, 9, 16}),
        lambda: make_circulant_start({2, 3, 8, 12, 17}),
        lambda: make_random_start(0.7),
        lambda: make_random_start(0.65),
        lambda: make_random_start(0.6),
        lambda: make_random_start(0.75),
        lambda: make_circulant_start({1, 2, 8, 9, 11, 12}),
    ]

    num_starts = 40

    for s in range(num_starts):
        if time.time() > deadline - 15:
            break

        # Cycle through start generators
        gen = start_generators[s % len(start_generators)]
        adj = gen()

        # For random starts, vary the density slightly based on s
        if s % len(start_generators) >= 5:
            adj = [0] * n
            density = 0.55 + 0.02 * (s % 10)
            for (i, j) in all_edges:
                if rng.random() < density:
                    add_edge(adj, i, j)

        # Repair to validity
        ec = repair(adj)

        # Greedy add
        ec = greedy_add(adj)

        # Best-improvement local search
        ec = best_improvement_local_search(adj)

        # Polish
        ec = polish(adj)

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
