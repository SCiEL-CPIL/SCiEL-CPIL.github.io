import time
import random
import math

def _is_triangle_free(adj_masks):
    for u in range(16):
        v_mask = adj_masks[u]
        while v_mask:
            v_bit = v_mask & -v_mask
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            if u < v:
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _is_3_colorable(adj_masks):
    n = 16
    degrees = [bin(m).count('1') for m in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    color = [-1] * n
    order_idx = [0] * n
    for i, v in enumerate(order):
        order_idx[v] = i

    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        neighbors = adj_masks[v]
        forbidden = 0
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        if forbidden == 0b111:
            return False
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False

    return backtrack(0)

def construct():
    deadline = time.time() + 110
    n = 16

    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2

    def build_seed():
        """Grötzsch graph (11 vertices, 20 edges) + 5 vertices with greedy connections."""
        adj = [0] * 16
        # C5: 0-1-2-3-4-0
        for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Mycielski copies 5-9
        for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Apex 10 connected to 5-9
        for v in range(5, 10):
            adj[10] |= (1 << v)
            adj[v] |= (1 << 10)
        # Add vertices 11-15: connect to vertices that don't create triangles
        for new_v in range(11, 16):
            # Find all vertices that can be connected without creating a triangle
            candidates = []
            for u in range(new_v):
                # Check: u and new_v not adjacent yet (always true for new vertex)
                # Check: no common neighbor between u and new_v
                # new_v has no neighbors yet, so any u works initially
                # But we need to check that adding edge (u, new_v) doesn't create triangle
                # Triangle would be u - w - new_v where w is common neighbor of u and new_v
                # Since new_v has no neighbors, no triangle is possible
                candidates.append(u)
            # Greedily add edges to maximize degree while keeping triangle-free
            # Sort by degree descending to connect to high-degree vertices first
            candidates.sort(key=lambda u: bin(adj[u]).count('1'), reverse=True)
            for u in candidates:
                if adj[u] & (1 << new_v):
                    continue
                # Check triangle-free: no common neighbor of u and new_v
                if adj[u] & adj[new_v]:
                    continue
                adj[u] |= (1 << new_v)
                adj[new_v] |= (1 << u)
        return adj

    def get_addable_edges(adj):
        """Return list of (u,v) that can be added without creating a triangle."""
        addable = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                addable.append((u, v))
        return addable

    def get_all_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                if u < v:
                    edges.append((u, v))
        return edges

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_polish(adj, time_limit):
        """Repeatedly try to add any addable edge that preserves 4-chromaticity."""
        adj = [m for m in adj]
        start = time.time()
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            addable = get_addable_edges(adj)
            random.shuffle(addable)
            for u, v in addable:
                if time.time() > time_limit:
                    break
                new_adj = [m for m in adj]
                add_edge(new_adj, u, v)
                if not _is_3_colorable(new_adj):
                    adj = new_adj
                    improved = True
        return adj

    def local_search_sa(adj, edges, time_limit):
        """Simulated annealing local search."""
        adj = [m for m in adj]
        best_adj = [m for m in adj]
        best_edges = edges
        current_edges = edges

        start_time = time.time()
        base_temp = 0.3
        min_temp = 0.01
        total_duration = max(0.1, time_limit - start_time)
        no_improve_start = start_time

        step = 0
        while time.time() < time_limit:
            step += 1
            elapsed_frac = min(1.0, (time.time() - start_time) / total_duration)
            temp = base_temp * (1.0 - elapsed_frac) + min_temp * elapsed_frac

            if time.time() - no_improve_start > 3.0:
                break

            move_type = random.random()

            if move_type < 0.55:
                # Add edge
                addable = get_addable_edges(adj)
                if not addable:
                    continue
                u, v = random.choice(addable)
                new_adj = [m for m in adj]
                add_edge(new_adj, u, v)

                if _is_3_colorable(new_adj):
                    continue

                adj = new_adj
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in adj]
                    no_improve_start = time.time()

            elif move_type < 0.80:
                # Remove edge (with acceptance)
                all_edges = get_all_edges(adj)
                if not all_edges:
                    continue
                u, v = random.choice(all_edges)
                new_adj = [m for m in adj]
                remove_edge(new_adj, u, v)

                if _is_3_colorable(new_adj):
                    continue

                # Accept removal with probability based on temperature
                delta = -1  # edges decreased
                if random.random() < math.exp(-delta / max(temp, 1e-9)):
                    adj = new_adj
                    current_edges -= 1

            else:
                # Swap: remove one edge, add another
                all_edges = get_all_edges(adj)
                if not all_edges:
                    continue
                u1, v1 = random.choice(all_edges)
                new_adj = [m for m in adj]
                remove_edge(new_adj, u1, v1)

                addable = get_addable_edges(new_adj)
                if not addable:
                    continue
                u2, v2 = random.choice(addable)
                add_edge(new_adj, u2, v2)

                if _is_3_colorable(new_adj):
                    continue

                adj = new_adj
                # Edge count unchanged

        # Final greedy polish
        adj = greedy_polish(adj, time_limit)
        edges_final = get_edge_count(adj)
        if edges_final > best_edges:
            best_adj = adj
            best_edges = edges_final

        return best_adj, best_edges

    best_adj = [0] * 16
    best_edges = 0

    # Build initial seed
    seed_adj = build_seed()
    seed_edges = get_edge_count(seed_adj)

    # Quick check: is seed 4-chromatic?
    if _is_3_colorable(seed_adj):
        # Seed is 3-colorable, need to fix
        # Try adding more edges greedily
        seed_adj = greedy_polish(seed_adj, deadline - 100)
        seed_edges = get_edge_count(seed_adj)
        if _is_3_colorable(seed_adj):
            # Fallback: use a known 4-chromatic triangle-free graph
            # Use the Grötzsch graph without the extra vertices
            adj = [0] * 16
            for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            for v in range(5, 10):
                adj[10] |= (1 << v)
                adj[v] |= (1 << 10)
            seed_adj = adj
            seed_edges = get_edge_count(seed_adj)

    # Run multiple restarts with simulated annealing
    num_restarts = 0
    while time.time() < deadline - 5:
        if num_restarts == 0:
            # First run: use the seed
            current_adj = [m for m in seed_adj]
            current_edges = seed_edges
        else:
            # Restart: perturb the best solution
            current_adj = [m for m in best_adj]
            current_edges = best_edges
            # Randomly remove a few edges to create diversity
            all_edges = get_all_edges(current_adj)
            random.shuffle(all_edges)
            num_remove = random.randint(2, 5)
            for i in range(min(num_remove, len(all_edges))):
                u, v = all_edges[i]
                remove_edge(current_adj, u, v)
            current_edges = get_edge_count(current_adj)
            if _is_3_colorable(current_adj):
                # Restore from best
                current_adj = [m for m in best_adj]
                current_edges = best_edges

        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break

        # Allocate time for this restart
        restart_time = min(remaining, 15.0)
        result_adj, result_edges = local_search_sa(current_adj, current_edges, time.time() + restart_time)

        if result_edges > best_edges:
            best_adj = result_adj
            best_edges = result_edges

        num_restarts += 1

    # Final polish on the best solution
    if time.time() < deadline - 2:
        best_adj = greedy_polish(best_adj, deadline - 2)
        best_edges = get_edge_count(best_adj)

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])

    return edges
