import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s

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
    degrees = [bin(adj_mask[i] & (sum(1 << c for c in candidates) if False else 0)).count('1') if False else bin(adj_mask[i]).count('1') for i in range(N)]

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

    def branch_and_bound(remaining_mask, current_mask, depth, best_ref, deadline):
        """
        Backtracking with forward checking.
        remaining_mask: vertices still available to consider
        current_mask: vertices already in the independent set
        Returns best mask found.
        """
        if time.time() > deadline:
            return current_mask

        if remaining_mask == 0:
            if popcount(current_mask) > best_ref[0]:
                best_ref[0] = popcount(current_mask)
                best_ref[1] = current_mask
            return current_mask

        # Upper bound: current size + number of remaining vertices
        upper_bound = popcount(current_mask) + popcount(remaining_mask)
        if upper_bound <= best_ref[0]:
            return current_mask

        # Pick the vertex with highest degree in remaining subgraph
        # Try both including and excluding it
        rm = remaining_mask
        # Find a vertex to branch on - pick one with max conflicts in remaining
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
        # Remove v and all its neighbors from remaining
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

    def bounded_bnb(full_mask, initial_bound, deadline):
        """Run branch and bound with a time limit."""
        best_ref = [initial_bound, 0]
        # Order vertices by degree for initial greedy
        order = sorted(range(n), key=lambda i: degrees[candidates[i]])
        greedy_mask = greedy_max_independent(order)
        greedy_mask = local_improve(greedy_mask)
        greedy_size = popcount(greedy_mask)
        best_ref = [greedy_size, greedy_mask]

        # Use BnB on the full set with time limit
        bnb_deadline = deadline - 2
        result = branch_and_bound(full_mask, 0, 0, best_ref, bnb_deadline)
        return result

    # Strategy 1: Greedy with multiple orderings + local improvement
    best_size = 0
    best_mask = 0

    # Multiple greedy orderings
    orderings = []
    orderings.append(sorted(range(n), key=lambda i: degrees[candidates[i]]))
    orderings.append(sorted(range(n), key=lambda i: -degrees[candidates[i]]))

    # Residue-based orderings
    for mod in [3, 4, 5, 7]:
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
    for _ in range(50):
        if time.time() > DEADLINE - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        m = greedy_max_independent(order)
        m = local_improve(m)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Strategy 2: Local search (1-move and 2-move)
    def local_search_12(mask, max_iters=200, deadline=None):
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

    # Run local search from best and from several random starts
    ls_deadline = DEADLINE - 5
    best_mask = local_search_12(best_mask, max_iters=150, deadline=ls_deadline)
    best_size = popcount(best_mask)

    for _ in range(20):
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

    # Strategy 3: Branch and bound on a subset
    # Use BnB on the remaining unassigned vertices to try to improve
    if time.time() < DEADLINE - 5:
        # Get the current best set
        in_set = set()
        for i in range(n):
            if (best_mask >> i) & 1:
                in_set.add(i)

        # Vertices not in set
        not_in_set = [i for i in range(n) if i not in in_set]

        # Try to add vertices one by one with BnB on small subproblems
        # For each vertex not in set, check if we can add it by removing minimal conflicts
        for v in not_in_set[:40]:
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
