"""Strong edge colouring via simulated annealing on colour-class assignments."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, deadline):
        if not edges:
            return []
        n = max(max(u, v) for u, v in edges) + 1
        m = len(edges)

        # Build adjacency as sets for fast "joined by edge" checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute: for each edge i, the list of edges that conflict with it
        # Two edges conflict if they share a vertex or are joined by an edge
        conflict_list = [[] for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
                    continue
                # Joined by an edge?
                if v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1] or u2 in adj[u1]:
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)

        # Conflict count for each edge (degree in conflict graph)
        conflict_counts = [len(conflict_list[i]) for i in range(m)]

        # --- DSATUR-style greedy initialisation ---
        # Order edges by conflict count descending; break ties randomly
        rng_init = random.Random(42)
        order = sorted(range(m), key=lambda i: (-conflict_counts[i], rng_init.random()))

        colour = [-1] * m
        # colour_class[c] = set of edge indices with colour c
        colour_class = []

        for i in order:
            # Find colours forbidden for edge i
            forbidden = set()
            for j in conflict_list[i]:
                if colour[j] >= 0:
                    forbidden.add(colour[j])
            # Assign smallest available colour
            c = 0
            while c in forbidden:
                c += 1
            colour[i] = c
            if c >= len(colour_class):
                colour_class.append(set())
            colour_class[c].add(i)

        used_colours = len(colour_class)

        # Helper: check if moving edge i from colour c_old to colour c_new is valid
        # (no conflict in c_new)
        def can_move(i, c_new):
            for j in conflict_list[i]:
                if colour[j] == c_new:
                    return False
            return True

        def do_move(i, c_old, c_new):
            colour[i] = c_new
            colour_class[c_old].discard(i)
            if c_new >= len(colour_class):
                for _ in range(c_new - len(colour_class) + 1):
                    colour_class.append(set())
            colour_class[c_new].add(i)

        # --- Simulated Annealing ---
        # Objective: minimise number of distinct colour classes used
        # We fix a maximum number of colours K and try to fit all edges into [0, K-1]
        # Start with K = used_colours, try to reduce

        best_colour = colour[:]
        best_K = used_colours

        # Try to solve with progressively smaller K
        for K in range(used_colours, 0, -1):
            if time.time() > deadline:
                break

            # Reset to greedy colouring using at most K colours
            # If greedy uses more than K, we need SA to fix it
            colour_sa = [-1] * m
            colour_class_sa = []
            for i in order:
                forbidden = set()
                for j in conflict_list[i]:
                    if colour_sa[j] >= 0:
                        forbidden.add(colour_sa[j])
                c = 0
                while c in forbidden:
                    c += 1
                colour_sa[i] = c
                if c >= len(colour_class_sa):
                    colour_class_sa.append(set())
                colour_class_sa[c].add(i)

            current_K = len(colour_class_sa)
            if current_K <= K:
                # Already solved with K or fewer colours
                if current_K < best_K:
                    best_K = current_K
                    best_colour = colour_sa[:]
                break

            # Need SA to reduce from current_K to K
            # Temperature schedule
            T_start = 1.0
            T_end = 0.001
            n_iterations = 50000
            cooling = (T_end / T_start) ** (1.0 / max(1, n_iterations))

            T = T_start
            sa_colour = colour_sa[:]
            sa_cc = [set() for _ in range(current_K)]
            for i in range(m):
                sa_cc[sa_colour[i]].add(i)

            # Energy: number of conflicts (pairs of edges in same colour class that conflict)
            def energy(col, cc):
                e = 0
                for c in range(len(cc)):
                    members = list(cc[c])
                    for idx, i in enumerate(members):
                        for j in members[idx+1:]:
                            # Check if i and j conflict
                            # Use a set for fast lookup
                            if j in conflict_set_lookup[i]:
                                e += 1
                return e

            # Precompute conflict sets as Python sets for O(1) lookup
            conflict_set_lookup = [set(cl) for cl in conflict_list]

            current_energy = energy(sa_colour, sa_cc)
            best_sa_energy = current_energy
            best_sa_colour = sa_colour[:]
            best_sa_cc = [set(s) for s in sa_cc]

            for it in range(n_iterations):
                if time.time() > deadline:
                    break

                # Pick a random edge
                i = random.randrange(m)
                c_old = sa_colour[i]

                # Pick a random new colour in [0, K-1]
                c_new = random.randrange(K)
                if c_new == c_old:
                    continue

                # Remove i from old class
                sa_cc[c_old].discard(i)

                # Add i to new class
                if c_new >= len(sa_cc):
                    for _ in range(c_new - len(sa_cc) + 1):
                        sa_cc.append(set())
                sa_cc[c_new].add(i)
                sa_colour[i] = c_new

                # Compute energy change
                # Only need to check conflicts involving i in both old and new classes
                delta = 0
                # In new class: count conflicts with i
                for j in conflict_set_lookup[i]:
                    if sa_colour[j] == c_new and j != i:
                        delta += 1
                # In old class: we removed i, so conflicts between i and others in old class are gone
                for j in conflict_set_lookup[i]:
                    if sa_colour[j] == c_old and j != i:
                        delta -= 1

                new_energy = current_energy + delta

                if new_energy <= current_energy:
                    current_energy = new_energy
                    if current_energy < best_sa_energy:
                        best_sa_energy = current_energy
                        best_sa_colour = sa_colour[:]
                        best_sa_cc = [set(s) for s in sa_cc]
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < math.exp(-delta / T):
                        current_energy = new_energy

                T *= cooling

            # Check if we achieved zero energy with K colours
            if best_sa_energy == 0:
                # Recompress colours
                colour_map = {}
                new_colour = [0] * m
                next_c = 0
                for i in range(m):
                    if best_sa_colour[i] not in colour_map:
                        colour_map[best_sa_colour[i]] = next_c
                        next_c += 1
                    new_colour[i] = colour_map[best_sa_colour[i]]
                if next_c < best_K:
                    best_K = next_c
                    best_colour = new_colour
                break

        # Final recompress
        colour_map = {}
        new_colour = [0] * m
        next_c = 0
        for i in range(m):
            if best_colour[i] not in colour_map:
                colour_map[best_colour[i]] = next_c
                next_c += 1
            new_colour[i] = colour_map[best_colour[i]]
        return new_colour

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        cols = solve_instance(edges, DEADLINE)
        results.append(cols)

    return results
# EVOLVE-BLOCK-END
