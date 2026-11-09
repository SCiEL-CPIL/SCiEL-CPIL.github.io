<<<<<<< SEARCH
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 256

    # Build conflict adjacency as Python int bitmasks (bits 0..n, bit i means node i+1)
    # adj[i] is a bitmask of nodes that conflict with node i (i in 1..n)
    adj = [0] * (n + 1)

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Precompute degrees for better vertex ordering
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = adj[i].bit_count()

    # Greedy independent set (minimum degree heuristic) for initial lower bound
    def greedy_independent_set(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            # Find node with minimum degree in remaining
            min_deg = float('inf')
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        return selected

    all_mask = (1 << (n + 1)) - 2  # bits 1..n set

    # Initial lower bound with multiple random-order greedy tries
    best_set = greedy_independent_set(all_mask)
    best_size = best_set.bit_count()
    
    # Try a few different greedy strategies to improve initial bound
    for _ in range(3):
        # Greedy with max degree removal
        selected = 0
        remaining = all_mask
        while remaining:
            max_deg = -1
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        if selected.bit_count() > best_size:
            best_size = selected.bit_count()
            best_set = selected

    # Greedy coloring upper bound on the candidate subgraph
    # Returns an upper bound on the independence number of the subgraph induced by candidates
    def color_bound(candidates):
        """Greedy coloring of the conflict graph restricted to candidates.
        Returns the number of colors used, which upper-bounds the independence number."""
        if candidates == 0:
            return 0
        uncolored = candidates
        color_count = 0
        while uncolored:
            color_count += 1
            # Greedily pick an independent set for this color class
            # Use degree-based ordering for tighter bounds
            color_class = 0
            t = uncolored
            # Sort nodes by degree for better coloring
            nodes = []
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                nodes.append((degrees[node], node))
                t &= t - 1
            nodes.sort(reverse=True)
            for _, node in nodes:
                if (adj[node] & color_class) == 0:
                    color_class |= (1 << node)
            uncolored &= ~color_class
        return color_count

    # Branch and bound with coloring upper bound
    def branch_and_bound(candidates, current_size, selected_mask):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return

        # Quick bound: can't beat best
        if current_size + candidates.bit_count() <= best_size:
            return

        # Upper bound via greedy coloring
        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return

        # Pick branching vertex: highest degree in candidate subgraph
        # Use tie-breaking by number of neighbors with high degree
        max_deg = -1
        chosen_node = -1
        t = candidates
        while t:
            lsb = t & (-t)
            node = lsb.bit_length() - 1
            deg = (adj[node] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_node = node
            t &= t - 1

        if chosen_node == -1:
            return

        chosen_bit = 1 << chosen_node

        # Branch 1: Include chosen_node (more promising first)
        new_candidates = candidates & ~(adj[chosen_node] | chosen_bit)
        new_selected = selected_mask | chosen_bit
        branch_and_bound(new_candidates, current_size + 1, new_selected)

        # Branch 2: Exclude chosen_node
        new_candidates = candidates & ~chosen_bit
        branch_and_bound(new_candidates, current_size, selected_mask)

    branch_and_bound(all_mask, 0, 0)

    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)

    return result if result else [1]
=======
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 256

    # Build conflict adjacency as Python int bitmasks (bits 0..n, bit i means node i+1)
    # adj[i] is a bitmask of nodes that conflict with node i (i in 1..n)
    adj = [0] * (n + 1)

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Precompute degrees for better vertex ordering
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = adj[i].bit_count()

    all_mask = (1 << (n + 1)) - 2  # bits 1..n set

    # Greedy independent set: pick min-degree node
    def greedy_min_deg(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            min_deg = float('inf')
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        return selected

    # Greedy independent set: pick max-degree node first
    def greedy_max_deg(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            max_deg = -1
            chosen = -1
            t = remaining
            while t:
                lsb = t & (-t)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = node
                t &= t - 1
            if chosen == -1:
                break
            selected |= (1 << chosen)
            remaining &= ~(adj[chosen] | (1 << chosen))
        return selected

    # Multiple greedy passes for initial lower bound
    best_set = greedy_min_deg(all_mask)
    best_size = best_set.bit_count()

    for _ in range(5):
        s = greedy_max_deg(all_mask)
        if s.bit_count() > best_size:
            best_size = s.bit_count()
            best_set = s

    # 1-swap local search
    def local_search_1swap(selected_mask, max_iters=3000):
        nonlocal best_size, best_set
        selected = selected_mask
        size = selected.bit_count()
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            t_sel = selected
            while t_sel:
                lsb = t_sel & (-t_sel)
                to_remove = lsb.bit_length() - 1
                t_sel &= t_sel - 1

                sel_without = selected & ~lsb
                forbidden = 0
                t2 = sel_without
                while t2:
                    lsb2 = t2 & (-t2)
                    node2 = lsb2.bit_length() - 1
                    forbidden |= adj[node2]
                    t2 &= t2 - 1
                forbidden |= selected

                candidates = all_mask & ~forbidden
                if candidates == 0:
                    continue

                # Find candidate with 0 conflicts with sel_without
                t_cand = candidates
                best_cand = -1
                while t_cand:
                    lsb_c = t_cand & (-t_cand)
                    cand_node = lsb_c.bit_length() - 1
                    if (adj[cand_node] & sel_without) == 0:
                        best_cand = cand_node
                        break
                    t_cand &= t_cand - 1

                if best_cand != -1:
                    new_selected = selected | (1 << best_cand)
                    if new_selected.bit_count() > size:
                        selected = new_selected
                        size = selected.bit_count()
                        improved = True
                        if size > best_size:
                            best_size = size
                            best_set = selected
                        break

            if not improved:
                break
        return selected

    best_set = local_search_1swap(best_set)

    # Precompute node ordering by degree (descending) for fast coloring
    node_order = list(range(1, n + 1))
    node_order.sort(key=lambda x: -degrees[x])

    # Fast coloring bound using pre-sorted node order
    def color_bound_v2(candidates):
        if candidates == 0:
            return 0
        uncolored = candidates
        color_count = 0
        while uncolored:
            color_count += 1
            color_class = 0
            for node in node_order:
                if not (uncolored & (1 << node)):
                    continue
                if (adj[node] & color_class) == 0:
                    color_class |= (1 << node)
            uncolored &= ~color_class
        return color_count

    # Branch and bound
    def branch_and_bound(candidates, current_size, selected_mask):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return

        # Quick bound
        if current_size + candidates.bit_count() <= best_size:
            return

        # Upper bound via greedy coloring
        ub = current_size + color_bound_v2(candidates)
        if ub <= best_size:
            return

        # Pick branching vertex: highest degree in candidate subgraph
        max_deg = -1
        chosen_node = -1
        t = candidates
        while t:
            lsb = t & (-t)
            node = lsb.bit_length() - 1
            deg = (adj[node] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_node = node
            t &= t - 1

        if chosen_node == -1:
            return

        chosen_bit = 1 << chosen_node

        # Branch 1: Include chosen_node
        new_candidates = candidates & ~(adj[chosen_node] | chosen_bit)
        new_selected = selected_mask | chosen_bit
        branch_and_bound(new_candidates, current_size + 1, new_selected)

        # Branch 2: Exclude chosen_node
        new_candidates = candidates & ~chosen_bit
        branch_and_bound(new_candidates, current_size, selected_mask)

    branch_and_bound(all_mask, 0, 0)

    # Final local search polish
    if time.time() < deadline - 2.0:
        best_set = local_search_1swap(best_set, max_iters=5000)

    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)

    return result if result else [1]
>>>>>>> REPLACE
