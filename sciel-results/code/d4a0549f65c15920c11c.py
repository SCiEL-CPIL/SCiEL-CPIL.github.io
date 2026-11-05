import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 57.0  # Safety margin before 60s

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency masks
    adj_mask = [0] * N
    forbidden = set()

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden.add(i)
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                if j != i:
                    adj_mask[i] |= (1 << j)

    candidates = [i for i in range(N) if i not in forbidden]
    n_cand = len(candidates)
    cand_set = set(candidates)
    degrees = [bin(adj_mask[i]).count('1') for i in range(N)]

    # Precompute candidate index mapping
    cand_index = {c: i for i, c in enumerate(candidates)}

    # Build adjacency in candidate-index space
    n = n_cand
    cand_adj = [0] * n
    for i, c in enumerate(candidates):
        m = adj_mask[c]
        for j, d in enumerate(candidates):
            if d != c and (m >> d) & 1:
                cand_adj[i] |= (1 << j)

    def popcount(x):
        return bin(x).count('1')

    def greedy_max_independent(order):
        """Greedy: add vertices in given order if no conflict."""
        current = 0
        for idx in order:
            if not (cand_adj[idx] & current):
                current |= (1 << idx)
        return current

    def local_improve(mask):
        """Add any vertex that has no conflict with current set."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if (mask >> i) & 1:
                    continue
                if not (cand_adj[i] & mask):
                    mask |= (1 << i)
                    improved = True
        return mask

    def local_search_12(mask, max_iters=200, deadline=None):
        """Local search with 1-removal, 2-removal, and swap moves."""
        mask = local_improve(mask)
        best_sz = popcount(mask)
        best_m = mask

        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break

            improved = False
            verts_in = [i for i in range(n) if (mask >> i) & 1]
            verts_out = [i for i in range(n) if not ((mask >> i) & 1)]

            # Try 1-removal + re-add
            random.shuffle(verts_in)
            for v in verts_in:
                new_m = mask & ~(1 << v)
                new_m = local_improve(new_m)
                sz = popcount(new_m)
                if sz > best_sz:
                    best_sz = sz
                    best_m = new_m
                    mask = new_m
                    improved = True
                    break

            if improved:
                continue

            # Try 2-removal + re-add
            if len(verts_in) >= 2:
                random.shuffle(verts_in)
                pairs_tried = 0
                max_pairs = 30
                for i in range(min(len(verts_in), 15)):
                    for j in range(i + 1, min(len(verts_in), 15)):
                        if pairs_tried >= max_pairs:
                            break
                        pairs_tried += 1
                        v1, v2 = verts_in[i], verts_in[j]
                        new_m = mask & ~(1 << v1) & ~(1 << v2)
                        new_m = local_improve(new_m)
                        sz = popcount(new_m)
                        if sz > best_sz:
                            best_sz = sz
                            best_m = new_m
                            mask = new_m
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                # Try 1-swap: remove one, add another
                if verts_in and verts_out:
                    random.shuffle(verts_out)
                    for v_out in verts_out[:20]:
                        # Find conflicts
                        conflicts = cand_adj[v_out] & mask
                        if popcount(conflicts) > 1:
                            continue
                        if conflicts == 0:
                            # Can just add
                            new_m = mask | (1 << v_out)
                            sz = popcount(new_m)
                            if sz > best_sz:
                                best_sz = sz
                                best_m = new_m
                                mask = new_m
                                improved = True
                                break
                        else:
                            # Remove the one conflicting vertex
                            c_bit = conflicts & (-conflicts)
                            c_idx = c_bit.bit_length() - 1
                            new_m = (mask & ~c_bit) | (1 << v_out)
                            new_m = local_improve(new_m)
                            sz = popcount(new_m)
                            if sz > best_sz:
                                best_sz = sz
                                best_m = new_m
                                mask = new_m
                                improved = True
                                break

            if not improved:
                break

        return best_m

    def simulated_annealing(s_mask, initial_temp=10.0, cooling=0.999, steps=3000, deadline=None):
        """
        Simulated Annealing with add/remove/swap moves.
        Uses mask representation for speed.
        """
        current = s_mask
        current_size = popcount(current)
        best = s_mask
        best_size = current_size
        temp = initial_temp

        for step in range(steps):
            if deadline and time.time() > deadline:
                break

            move_type = random.random()

            if move_type < 0.4:
                # Try to add a random vertex
                v = random.randint(0, n - 1)
                if (current >> v) & 1:
                    continue
                conflicts = cand_adj[v] & current
                if conflicts == 0:
                    new_m = current | (1 << v)
                    new_sz = popcount(new_m)
                    if new_sz > best_size:
                        best_size = new_sz
                        best = new_m
                    current = new_m
                    current_size = new_sz
                else:
                    # Accept with probability based on conflicts
                    if random.random() < math.exp(-len(bin(conflicts).count('1')) / max(temp, 0.1)):
                        # Remove conflicting vertices and add v
                        new_m = current & ~conflicts
                        new_m = new_m | (1 << v)
                        new_m = local_improve(new_m)
                        new_sz = popcount(new_m)
                        if new_sz > best_size:
                            best_size = new_sz
                            best = new_m
                        current = new_m
                        current_size = new_sz
            elif move_type < 0.7:
                # Try to remove a random vertex
                verts_in = [i for i in range(n) if (current >> i) & 1]
                if not verts_in:
                    continue
                v = random.choice(verts_in)
                new_m = current & ~(1 << v)
                new_m = local_improve(new_m)
                new_sz = popcount(new_m)
                if new_sz > best_size:
                    best_size = new_sz
                    best = new_m
                current = new_m
                current_size = new_sz
            else:
                # Swap: remove one, add another
                verts_in = [i for i in range(n) if (current >> i) & 1]
                verts_out = [i for i in range(n) if not ((current >> i) & 1)]
                if not verts_in or not verts_out:
                    continue
                v_in = random.choice(verts_in)
                v_out = random.choice(verts_out)
                conflicts = cand_adj[v_out] & current
                # Remove v_in and any other conflicts
                to_remove = (1 << v_in) | (conflicts & ~((1 << v_in)))
                new_m = current & ~to_remove
                new_m = new_m | (1 << v_out)
                new_m = local_improve(new_m)
                new_sz = popcount(new_m)
                if new_sz > best_size:
                    best_size = new_sz
                    best = new_m
                # Accept with probability
                delta = new_sz - current_size
                if delta >= 0 or random.random() < math.exp(-delta / max(temp, 0.1)):
                    current = new_m
                    current_size = new_sz

            temp *= cooling

        return best

    def branch_and_bound(remaining_mask, current_mask, depth, best_ref, deadline):
        """
        Backtracking with forward checking.
        """
        if time.time() > deadline:
            return current_mask

        if remaining_mask == 0:
            if popcount(current_mask) > best_ref[0]:
                best_ref[0] = popcount(current_mask)
                best_ref[1] = current_mask
            return current_mask

        # Upper bound
        upper_bound = popcount(current_mask) + popcount(remaining_mask)
        if upper_bound <= best_ref[0]:
            return current_mask

        # Pick vertex with highest degree in remaining
        rm = remaining_mask
        best_v = -1
        best_deg = -1
        temp = rm
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = popcount(cand_adj[v] & rm)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb

        if best_v < 0:
            return current_mask

        v_bit = 1 << best_v

        # Option 1: Include best_v
        new_current = current_mask | v_bit
        new_remaining = (remaining_mask & ~v_bit) & ~cand_adj[best_v]
        result1 = branch_and_bound(new_remaining, new_current, depth + 1, best_ref, deadline)
        if popcount(result1) > popcount(best_ref[1]):
            best_ref[0] = popcount(result1)
            best_ref[1] = result1

        # Option 2: Exclude best_v
        new_remaining2 = remaining_mask & ~v_bit
        result2 = branch_and_bound(new_remaining2, current_mask, depth + 1, best_ref, deadline)
        if popcount(result2) > popcount(best_ref[1]):
            best_ref[0] = popcount(result2)
            best_ref[1] = result2

        return best_ref[1]

    # Strategy 1: Greedy with multiple orderings + local improvement
    best_size = 0
    best_mask = 0

    # Multiple greedy orderings
    orderings = []
    orderings.append(sorted(range(n), key=lambda i: degrees[candidates[i]]))
    orderings.append(sorted(range(n), key=lambda i: -degrees[candidates[i]]))

    # Residue-based orderings
    for mod in [3, 4, 5, 7, 8]:
        for r in range(mod):
            verts = [i for i in range(n) if (candidates[i] + 1) % mod == r]
            orderings.append(sorted(verts, key=lambda i: degrees[candidates[i]]))
            orderings.append(sorted(verts, key=lambda i: -degrees[candidates[i]]))

    for order in orderings:
        m = greedy_max_independent(order)
        m = local_improve(m)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Random greedy orderings
    for _ in range(80):
        if time.time() > DEADLINE - 15:
            break
        order = list(range(n))
        random.shuffle(order)
        m = greedy_max_independent(order)
        m = local_improve(m)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Strategy 2: Local search from best and random starts
    ls_deadline = DEADLINE - 8
    best_mask = local_search_12(best_mask, max_iters=150, deadline=ls_deadline)
    best_size = popcount(best_mask)

    for _ in range(30):
        if time.time() > ls_deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        m = greedy_max_independent(order)
        m = local_improve(m)
        m = local_search_12(m, max_iters=50, deadline=ls_deadline)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Strategy 3: Simulated Annealing from best and random starts
    sa_deadline = DEADLINE - 4
    best_mask = simulated_annealing(best_mask, initial_temp=8.0, cooling=0.9995, steps=4000, deadline=sa_deadline)
    best_size = popcount(best_mask)

    for _ in range(15):
        if time.time() > sa_deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        m = greedy_max_independent(order)
        m = local_improve(m)
        m = simulated_annealing(m, initial_temp=5.0, cooling=0.9995, steps=2000, deadline=sa_deadline)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Strategy 4: Branch and bound on a subset if time permits
    if time.time() < DEADLINE - 5:
        # Get the current best set
        in_set = set()
        for i in range(n):
            if (best_mask >> i) & 1:
                in_set.add(i)

        # Vertices not in set
        not_in_set = [i for i in range(n) if i not in in_set]

        # Try to add vertices one by one
        for v in not_in_set[:50]:
            if time.time() > DEADLINE - 3:
                break
            conflicts = cand_adj[v] & best_mask
            if conflicts == 0:
                best_mask |= (1 << v)
                best_size += 1
            elif popcount(conflicts) == 1:
                # Remove one, add one
                c_bit = conflicts & (-conflicts)
                new_m = (best_mask & ~c_bit) | (1 << v)
                new_m = local_improve(new_m)
                if popcount(new_m) > best_size:
                    best_size = popcount(new_m)
                    best_mask = new_m

    # Final local improvement
    best_mask = local_improve(best_mask)
    best_size = popcount(best_mask)

    # Convert back to original indices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(candidates[i] + 1)

    return sorted(result)
