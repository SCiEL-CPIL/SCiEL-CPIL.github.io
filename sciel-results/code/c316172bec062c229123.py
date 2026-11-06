"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            # Pivot: choose v in P|X maximizing |P & adj[v]|
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            # Iterate over set bits in PX
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            # Candidates: P minus neighbors of pivot
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def bitmask_to_list(mask):
        return [i for i in range(mask.bit_length()) if mask & (1 << i)]

    def greedy_transversal(cliques_masks, n, weight_fn=None):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques_masks:
            return 0

        # Count how many cliques each vertex appears in
        if weight_fn is None:
            vertex_weight = [0] * n
            for cm in cliques_masks:
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    vertex_weight[v] += 1
                    temp ^= lsb
        else:
            vertex_weight = weight_fn(cliques_masks, n)

        uncovered = set(cliques_masks)  # set of bitmask ints
        selected = 0
        selected_list = []

        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                count = sum(1 for cm in uncovered if cm & (1 << v))
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v < 0 or best_count <= 0:
                break
            selected |= (1 << best_v)
            selected_list.append(best_v)
            uncovered = {cm for cm in uncovered if not (cm & (1 << best_v))}

        return selected, selected_list

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=200):
        """Try to remove vertices; if it breaks coverage, stop."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                trial = selected_mask ^ (1 << v)
                # Check all cliques are still covered
                ok = True
                for cm in cliques_masks:
                    if not (cm & trial):
                        ok = False
                        break
                if ok:
                    selected_mask = trial
                    improved = True
                    break
        return selected_mask

    def local_search_swap(selected_mask, cliques_masks, n, max_iters=100):
        """Try removing one and adding another to reduce or maintain size while improving."""
        # This is more complex; for now, just do reduction
        return selected_mask

    def simulated_annealing(selected_mask, cliques_masks, n, start_temp=10.0, end_temp=0.1, max_iters=500):
        """SA to try to reduce the transversal size."""
        if not cliques_masks:
            return selected_mask

        def is_valid(mask):
            for cm in cliques_masks:
                if not (cm & mask):
                    return False
            return True

        def size(mask):
            return mask.bit_count()

        current = selected_mask
        current_size = size(current)
        best = current
        best_size = current_size

        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))

        for it in range(max_iters):
            # Generate neighbor: remove a random vertex from current, or add a random vertex
            if random.random() < 0.5 and current.bit_count() > 0:
                # Remove a random vertex
                bits = []
                temp_mask = current
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    bits.append(lsb)
                    temp_mask ^= lsb
                if not bits:
                    continue
                lsb = random.choice(bits)
                neighbor = current ^ lsb
            else:
                # Add a random vertex not in current
                available = ((1 << n) - 1) & ~current
                if available == 0:
                    continue
                temp_mask = available
                bits = []
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    bits.append(lsb)
                    temp_mask ^= lsb
                lsb = random.choice(bits)
                neighbor = current | lsb

            if not is_valid(neighbor):
                continue

            neighbor_size = size(neighbor)
            delta = neighbor_size - current_size

            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_size = neighbor_size
                if current_size < best_size:
                    best = current
                    best_size = current_size

            temp *= cooling

        return best

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        # Try multiple greedy strategies
        best_mask = None
        best_size = n + 1

        # Strategy 1: standard greedy (most cliques covered)
        mask1, _ = greedy_transversal(cliques, n)
        mask1 = local_search_reduce(mask1, cliques, n)
        s1 = mask1.bit_count()
        if s1 < best_size:
            best_size = s1
            best_mask = mask1

        # Strategy 2: weighted by clique size (larger cliques first)
        def weight_fn_size(cliques_masks, n):
            w = [0] * n
            for cm in cliques_masks:
                sz = cm.bit_count()
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    w[v] += sz
                    temp ^= lsb
            return w

        mask2, _ = greedy_transversal(cliques, n, weight_fn_size)
        mask2 = local_search_reduce(mask2, cliques, n)
        s2 = mask2.bit_count()
        if s2 < best_size:
            best_size = s2
            best_mask = mask2

        # Strategy 3: weighted by 1/clique_size (smaller cliques first)
        def weight_fn_inv_size(cliques_masks, n):
            w = [0.0] * n
            for cm in cliques_masks:
                sz = max(1, cm.bit_count())
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    w[v] += 1.0 / sz
                    temp ^= lsb
            return w

        mask3, _ = greedy_transversal(cliques, n, weight_fn_inv_size)
        mask3 = local_search_reduce(mask3, cliques, n)
        s3 = mask3.bit_count()
        if s3 < best_size:
            best_size = s3
            best_mask = mask3

        # Apply simulated annealing to the best found
        if time.time() < deadline - 5:
            best_mask = simulated_annealing(best_mask, cliques, n, start_temp=5.0, end_temp=0.01, max_iters=300)
            best_size = best_mask.bit_count()

        return bitmask_to_list(best_mask)

    start_time = time.time()
    deadline = start_time + 140

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: all vertices
            out.append(list(range(n)))
            continue
        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
