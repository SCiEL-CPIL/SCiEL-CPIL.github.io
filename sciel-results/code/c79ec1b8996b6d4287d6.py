"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 20
    # Pre-compute all possible edges as bit positions
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    m_edges = len(all_edges)  # 190

    # Pre-compute for each edge (u,v), which other edges (w,x) would create a K2,3 if both present
    # Actually, we check validity by checking if adding edge (u,v) creates any K2,3.
    # A K2,3 is formed by two vertices u,v sharing >=3 common neighbors.
    # When we add edge (u,v), we need to check:
    # 1. u and v don't already have >=3 common neighbors
    # 2. For each neighbor w of u, v and w don't already have >=2 common neighbors (since u becomes a new common neighbor)
    # 3. For each neighbor x of v, u and x don't already have >=2 common neighbors

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free. adj is list of bitmasks."""
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() >= 3:
            return False

        # 2. For any w in N(u), u becomes a common neighbor of v and w.
        #    So |N(v) ∩ N(w)| must currently be < 2.
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() >= 2:
                return False
            neighbors_u ^= lsb

        # 3. For any x in N(v), v becomes a common neighbor of u and x.
        #    So |N(u) ∩ N(x)| must currently be < 2.
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            x = lsb.bit_length() - 1
            common_ux = adj[u] & adj[x]
            if common_ux.bit_count() >= 2:
                return False
            neighbors_v ^= lsb

        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def greedy_fill(order, adj=None):
        """Greedy fill: add edges in given order if valid."""
        if adj is None:
            adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def repair(adj):
        """Remove edges to fix K2,3 violations. Returns new adj and number removed."""
        # Find all K2,3 violations and remove one edge from each
        changed = True
        removed = 0
        while changed and time.time() < deadline - 5:
            changed = False
            # Check all pairs of vertices
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() >= 3:
                        # Remove a random edge from the common neighborhood
                        # Pick a random common neighbor w, remove edge (w,u) or (w,v)
                        common_list = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            w = lsb.bit_length() - 1
                            common_list.append(w)
                            mask ^= lsb
                        w = random.choice(common_list)
                        if random.random() < 0.5:
                            remove_edge(adj, u, w)
                        else:
                            remove_edge(adj, v, w)
                        removed += 1
                        changed = True
                        break
                if changed:
                    break
        return adj, removed

    def local_search(adj, count):
        """Try to add missing edges greedily."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        missing.append((i, j))
            if not missing:
                break
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    if time.time() > deadline - 5:
                        return count
        return count

    def genetic_algorithm():
        """
        Genetic algorithm with population of adjacency bitmasks.
        Crossover: union of two valid graphs, then repair by greedy edge removal.
        """
        pop_size = 15
        max_generations = 50
        mutation_rate = 0.1

        # Initialize population with diverse random greedy fills
        population = []
        for _ in range(pop_size):
            order = all_edges.copy()
            random.shuffle(order)
            adj, count = greedy_fill(order)
            count = local_search(adj, count)
            population.append((adj, count))

        # Sort by count descending
        population.sort(key=lambda x: x[1], reverse=True)
        best_adj, best_count = population[0]

        for gen in range(max_generations):
            if time.time() > deadline - 10:
                break

            new_population = []

            # Elitism: keep top 3
            for i in range(min(3, len(population))):
                new_population.append(population[i])

            # Generate offspring
            while len(new_population) < pop_size and time.time() < deadline - 10:
                # Tournament selection
                def tournament():
                    a = random.choice(population)
                    b = random.choice(population)
                    c = random.choice(population)
                    return max([a, b, c], key=lambda x: x[1])

                parent1 = tournament()
                parent2 = tournament()

                # Crossover: union of edges
                child_adj = [0] * n
                for i in range(n):
                    child_adj[i] = parent1[0][i] | parent2[0][i]

                # Repair: remove edges to fix violations
                child_adj, _ = repair(child_adj)
                child_count = count_edges(child_adj)

                # Mutation: randomly add/remove a few edges
                if random.random() < mutation_rate:
                    # Try to add a random missing edge
                    missing = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not (child_adj[i] >> j) & 1:
                                missing.append((i, j))
                    if missing:
                        u, v = random.choice(missing)
                        if is_valid_add(child_adj, u, v):
                            add_edge(child_adj, u, v)
                            child_count += 1

                # Local search on child
                child_count = local_search(child_adj, child_count)
                new_population.append((child_adj, child_count))

            # Update best
            for adj, count in new_population:
                if count > best_count:
                    best_adj = adj[:]
                    best_count = count

            # Sort and truncate
            new_population.sort(key=lambda x: x[1], reverse=True)
            population = new_population[:pop_size]

        return best_adj, best_count

    # Run genetic algorithm
    best_adj, best_count = genetic_algorithm()

    # Final polish with simulated annealing
    if time.time() < deadline - 5:
        T = 2.0
        cooling_rate = 0.999
        min_T = 0.01
        while T > min_T and time.time() < deadline - 3:
            # Try to add a random missing edge
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (best_adj[i] >> j) & 1:
                        missing.append((i, j))

            if missing:
                u, v = random.choice(missing)
                if is_valid_add(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    best_count += 1

            # Occasionally remove an edge to escape local optima
            if random.random() < 0.3:
                present = []
                for i in range(n):
                    mask = best_adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            present.append((i, j))
                        mask ^= lsb
                if present:
                    u, v = random.choice(present)
                    remove_edge(best_adj, u, v)
                    best_count -= 1
                    # Greedy re-add
                    for _ in range(3):
                        if time.time() > deadline - 3:
                            break
                        added = False
                        for i in range(n):
                            for j in range(i + 1, n):
                                if not (best_adj[i] >> j) & 1:
                                    if is_valid_add(best_adj, i, j):
                                        add_edge(best_adj, i, j)
                                        best_count += 1
                                        added = True
                                        break
                            if added:
                                break
                        if not added:
                            break

            T *= cooling_rate

    # Convert to edge list
    best_edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                best_edges.append([i, j])
            mask ^= lsb

    return best_edges
# EVOLVE-BLOCK-END
