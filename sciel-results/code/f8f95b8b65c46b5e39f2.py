import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Combines fast bitset C6 detection, multiple greedy strategies,
    and local search with 1-for-2 swap moves and tabu.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def can_add(u, v, adj):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-v-x-y-z-w-u where x in N(v), w in N(u),
        y in N(x), z in N(w), y~z, all distinct.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        # Iterate over the smaller set first for efficiency
        if Nv.bit_count() <= Nu.bit_count():
            x_mask = Nv
            while x_mask:
                x_bit = x_mask & (-x_mask)
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit

                w_mask = Nu
                while w_mask:
                    w_bit = w_mask & (-w_mask)
                    w = w_bit.bit_length() - 1
                    w_mask ^= w_bit

                    if w == x:
                        continue

                    # y in N(x) \ {u, v, w}, z in N(w) \ {u, v, x}
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if not S_x:
                        continue
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    if not S_w:
                        continue

                    # Check if any edge (y, z) with y in S_x, z in S_w, y != z
                    y_mask = S_x
                    while y_mask:
                        y_bit = y_mask & (-y_mask)
                        y = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        else:
            x_mask = Nu
            while x_mask:
                x_bit = x_mask & (-x_mask)
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit

                w_mask = Nv
                while w_mask:
                    w_bit = w_mask & (-w_mask)
                    w = w_bit.bit_length() - 1
                    w_mask ^= w_bit

                    if w == x:
                        continue

                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if not S_x:
                        continue
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    if not S_w:
                        continue

                    y_mask = S_x
                    while y_mask:
                        y_bit = y_mask & (-y_mask)
                        y = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        return True

    def get_edges(adj):
        edges = []
        for i in range(N):
            mask = adj[i]
            while mask:
                j_bit = mask & (-mask)
                j = j_bit.bit_length() - 1
                mask ^= j_bit
                if i < j:
                    edges.append((i, j))
        return edges

    def greedy_random(seed):
        """Greedy with random edge ordering."""
        rng = random.Random(seed)
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)

        adj = [0] * N
        edge_count = 0

        for u, v in all_edges:
            if time_left() < 5:
                break
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1

        return adj, edge_count

    def greedy_pref_high_degree(seed):
        """Greedy that prefers edges between high-degree vertices."""
        rng = random.Random(seed)
        adj = [0] * N
        edge_count = 0

        # Start with some random edges
        for _ in range(20):
            u, v = rng.randint(0, N - 1), rng.randint(0, N - 1)
            if u != v:
                u, v = min(u, v), max(u, v)
                if not (adj[u] & (1 << v)) and can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1

        # Then greedily add edges, preferring those connecting high-degree vertices
        remaining = [(i, j) for i in range(N) for j in range(i + 1, N)
                     if not (adj[i] & (1 << j))]
        rng.shuffle(remaining)

        for u, v in remaining:
            if time_left() < 5:
                break
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1

        return adj, edge_count

    def structured_init(seed=42):
        """Initialize with blocks and inter-block edges."""
        adj = [0] * N
        edge_count = 0

        # 8 groups of 5, connect within groups as K5
        for g in range(8):
            group = list(range(g * 5, g * 5 + 5))
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = group[i], group[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1

        # Add inter-group edges
        rng = random.Random(seed)
        for _ in range(500):
            if time_left() < 10:
                break
            g1, g2 = rng.sample(range(8), 2)
            u = rng.randint(g1 * 5, g1 * 5 + 4)
            v = rng.randint(g2 * 5, g2 * 5 + 4)
            u, v = min(u, v), max(u, v)
            if not (adj[u] & (1 << v)) and can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1

        return adj, edge_count

    def local_search(adj, edge_count, max_iters=5000, seed=0):
        """Local search: try to add edges, and do 1-for-2 swap moves with tabu."""
        rng = random.Random(seed)
        current_adj = list(adj)
        current_count = edge_count
        best_local_adj = list(adj)
        best_local_count = edge_count
        no_improve = 0
        max_no_improve = 100

        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        tabu = {}
        tabu_iteration = 0

        for it in range(max_iters):
            if time_left() < 5:
                break

            improved = False
            tabu_iteration += 1

            # 1. Try to add a random non-edge (0-for-1 move)
            for _ in range(20):
                if time_left() < 5:
                    break
                u, v = rng.choice(all_pairs)
                if current_adj[u] & (1 << v):
                    continue
                if can_add(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_count += 1
                    improved = True
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_adj = list(current_adj)
                    no_improve = 0
                    break

            if improved:
                continue

            no_improve += 1
            if no_improve > max_no_improve:
                break

            # 2. Try 1-for-2 moves: Remove one edge, add two
            edge_list = get_edges(current_adj)
            if not edge_list:
                break

            # Try a few candidate removals
            for _ in range(5):
                if time_left() < 5:
                    break
                u_rem, v_rem = rng.choice(edge_list)

                # Check if this removal is tabu
                if tabu.get((u_rem, v_rem), 0) > tabu_iteration - 50:
                    continue

                # Temporarily remove edge
                current_adj[u_rem] &= ~(1 << v_rem)
                current_adj[v_rem] &= ~(1 << u_rem)

                added_count = 0
                added_edges = []

                # Try to add two edges
                for _ in range(100):
                    if time_left() < 5:
                        break
                    u_add, v_add = rng.choice(all_pairs)
                    if current_adj[u_add] & (1 << v_add):
                        continue
                    if (u_add, v_add) in added_edges:
                        continue
                    if can_add(u_add, v_add, current_adj):
                        current_adj[u_add] |= (1 << v_add)
                        current_adj[v_add] |= (1 << u_add)
                        added_edges.append((u_add, v_add))
                        added_count += 1
                        if added_count == 2:
                            break

                if added_count >= 2:
                    # Net gain: added_count - 1
                    current_count += (added_count - 1)
                    tabu[(u_rem, v_rem)] = tabu_iteration
                    improved = True
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_adj = list(current_adj)
                    no_improve = 0
                    break
                else:
                    # Revert: remove added edges, re-add removed edge
                    for u_add, v_add in added_edges:
                        current_adj[u_add] &= ~(1 << v_add)
                        current_adj[v_add] &= ~(1 << u_add)
                    current_adj[u_rem] |= (1 << v_rem)
                    current_adj[v_rem] |= (1 << u_rem)

        return best_local_adj, best_local_count

    # Phase 1: Multiple greedy restarts
    best_adj = [0] * N
    best_count = 0

    num_greedy = 80
    for seed in range(num_greedy):
        if time_left() < 10:
            break
        adj_try, ec = greedy_random(seed)
        if ec > best_count:
            best_count = ec
            best_adj = adj_try

    # Phase 2: Greedy with high-degree preference
    for seed in range(20):
        if time_left() < 10:
            break
        adj_try, ec = greedy_pref_high_degree(seed)
        if ec > best_count:
            best_count = ec
            best_adj = adj_try

    # Phase 3: Structured init + local search
    if time_left() > 10:
        adj_try, ec = structured_init()
        if ec > best_count:
            best_count = ec
            best_adj = adj_try

        if time_left() > 10:
            new_adj, new_count = local_search(adj_try, ec, max_iters=3000, seed=12345)
            if new_count > best_count:
                best_count = new_count
                best_adj = new_adj

    # Phase 4: Local search on best
    if best_count > 0 and time_left() > 10:
        new_adj, new_count = local_search(best_adj, best_count, max_iters=5000, seed=99999)
        if new_count > best_count:
            best_count = new_count
            best_adj = new_adj

    # Phase 5: More local search with different seeds
    if best_count > 0 and time_left() > 10:
        for ls_seed in range(10):
            if time_left() < 10:
                break
            new_adj, new_count = local_search(best_adj, best_count, max_iters=3000, seed=1000 + ls_seed)
            if new_count > best_count:
                best_count = new_count
                best_adj = new_adj

    # Final: try to add any remaining edges
    if best_count > 0 and time_left() > 5:
        for u in range(N):
            for v in range(u + 1, N):
                if time_left() < 2:
                    break
                if not (best_adj[u] & (1 << v)) and can_add(u, v, best_adj):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    best_count += 1

    # Convert to list of edges
    return [[u, v] for u, v in get_edges(best_adj)]
