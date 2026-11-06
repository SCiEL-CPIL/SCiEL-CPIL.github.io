import time
import sys
import random
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60s limit

    n = 256
    all_mask = (1 << n) - 1

    # Precompute conflict masks
    # conflict[i] is a bitmask of indices j such that (i+1) and (j+1) conflict
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Complementary adjacency (H): edge (u,v) in H iff NOT conflict
    adj = [0] * n
    for i in range(n):
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i)

    # Degree in H
    degrees = [adj[i].bit_count() for i in range(n)]

    # Order vertices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    rank = [0] * n
    for idx, v in enumerate(order):
        rank[v] = idx

    # Greedy coloring bound for MIS on candidate set P
    # Colors vertices such that no two adjacent in conflict graph share same color
    # Max color is upper bound on MIS size
    def color_bound(P):
        if P == 0:
            return 0
        verts = []
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            verts.append(i)
            p ^= lsb

        # Sort by rank (degree-based)
        verts.sort(key=lambda x: rank[x])

        colors = [-1] * n
        max_color = 0

        for v in verts:
            used = set()
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                if colors[u] >= 0:
                    used.add(colors[u])
                np_ ^= lsb

            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c

        return max_color

    best_size = 0
    best_mask = 0

    # Tomita-style Branch and Bound for Maximum Independent Set
    def bbmc(P, current):
        nonlocal best_size, best_mask

        if time.time() - start_time > timeout:
            return

        if P == 0:
            if len(current) > best_size:
                best_size = len(current)
                mask = 0
                for v in current:
                    mask |= (1 << v)
                best_mask = mask
            return

        bound = color_bound(P)
        if len(current) + bound <= best_size:
            return

        # Pick vertex with lowest rank (highest degree)
        v = -1
        min_rank = n + 1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            if rank[i] < min_rank:
                min_rank = rank[i]
                v = i
            p ^= lsb

        if v == -1:
            return

        v_bit = 1 << v

        # Include v
        new_P = P & adj[v]
        current.append(v)
        bbmc(new_P, current)
        current.pop()

        # Exclude v
        new_P_exc = P & ~v_bit
        bbmc(new_P_exc, current)

    # Initial greedy solution - multiple passes with different orderings
    def greedy_solve(seed_order):
        """Greedy independent set using a given vertex ordering."""
        mask = 0
        P = all_mask
        for v in seed_order:
            if not (P & (1 << v)):
                continue
            if (conflict[v] & mask) == 0:
                mask |= (1 << v)
                P &= adj[v]
        return mask

    # Try several greedy orderings
    greedy_orders = [
        order,  # degree descending
        list(reversed(order)),  # degree ascending
        sorted(range(n), key=lambda x: -degrees[x] + random.random()),
        sorted(range(n), key=lambda x: random.random()),
    ]

    for go in greedy_orders:
        gm = greedy_solve(go)
        if gm.bit_count() > best_size:
            best_size = gm.bit_count()
            best_mask = gm

    # Also try greedy with dynamic degree (recompute at each step)
    def greedy_dynamic():
        mask = 0
        P = all_mask
        while P:
            # Pick vertex with highest degree in remaining P
            v = -1
            max_deg = -1
            p = P
            while p:
                lsb = p & -p
                i = lsb.bit_length() - 1
                deg = (adj[i] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = i
                p ^= lsb
            if v == -1:
                break
            mask |= (1 << v)
            P &= adj[v]
        return mask

    gm = greedy_dynamic()
    if gm.bit_count() > best_size:
        best_size = gm.bit_count()
        best_mask = gm

    # Run BBMC
    bbmc(all_mask, [])

    # Local search improvement on best found
    if best_mask:
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break

        # Try to swap: remove one, add one if it leads to improvement
        for _ in range(3):  # Multiple passes of swap
            improved = False
            for i in range(n):
                if improved:
                    break
                if not ((best_mask >> i) & 1):
                    # Check if i can be added by removing some conflicting vertex
                    confs = conflict[i] & best_mask
                    if confs:
                        # Try removing each conflicting vertex
                        cp = confs
                        while cp:
                            lsb = cp & -cp
                            j = lsb.bit_length() - 1
                            new_mask = (best_mask & ~(1 << j)) | (1 << i)
                            # Check if new_mask is valid
                            valid = True
                            for k in range(n):
                                if (new_mask >> k) & 1 and k != i:
                                    if (conflict[i] >> k) & 1:
                                        valid = False
                                        break
                            if valid:
                                best_mask = new_mask
                                improved = True
                                break
                            cp ^= lsb
                    elif (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True

    # Simulated annealing refinement
    def mask_to_set(mask):
        s = set()
        p = mask
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            s.add(i)
            p ^= lsb
        return s

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    # Precompute conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n)]
    for i in range(n):
        c = conflict[i]
        p = c
        while p:
            lsb = p & -p
            j = lsb.bit_length() - 1
            conflict_sets[i].add(j)
            p ^= lsb

    def anneal(S_set, time_budget):
        """Simulated annealing over independent sets with multi-vertex moves."""
        S = set(S_set)
        T0 = 1.5
        T_end = 0.01
        cooling = 0.9998
        T = T0
        best_S = set(S)
        best_size_anneal = len(S)

        while time.time() - start_time < time_budget:
            if T < T_end:
                break

            if not S:
                break

            # Choose a random move type
            move_type = random.random()

            if move_type < 0.3 and len(S) > 0:
                # Remove a random vertex
                v = random.choice(tuple(S))
                S_new = S - {v}
                # Try to add back some vertices that were blocked by v
                cands = [u for u in conflict_sets[v] if u not in S_new]
                valid = [u for u in cands if not (conflict_sets[u] & S_new)]
                # Greedily add a random independent subset of valid
                rem = set(valid)
                add = set()
                while rem:
                    vc = random.choice(tuple(rem))
                    add.add(vc)
                    rem -= conflict_sets[vc]
                    rem.discard(vc)
                S_cand = S_new | add

            elif move_type < 0.5 and len(S) > 1:
                # Swap: remove one, add one
                v = random.choice(tuple(S))
                S_new = S - {v}
                cands = [u for u in range(n) if u not in S_new and not (conflict_sets[u] & S_new)]
                if cands:
                    u = random.choice(cands)
                    S_cand = S_new | {u}
                else:
                    S_cand = S_new
            else:
                # Try to add a random vertex
                not_in_S = [u for u in range(n) if u not in S]
                if not_in_S:
                    u = random.choice(not_in_S)
                    if not (conflict_sets[u] & S):
                        S_cand = S | {u}
                    else:
                        # Remove one conflicting vertex and add u
                        confs = conflict_sets[u] & S
                        if confs:
                            v = random.choice(tuple(confs))
                            S_new = S - {v}
                            if not (conflict_sets[u] & S_new):
                                S_cand = S_new | {u}
                            else:
                                S_cand = S
                        else:
                            S_cand = S
                else:
                    S_cand = S

            d = len(S_cand) - len(S)

            if d >= 0 or random.random() < math.exp(d / T):
                S = S_cand
                if len(S) > best_size_anneal:
                    best_size_anneal = len(S)
                    best_S = set(S)

            T *= cooling

        return best_S

    # Run annealing on the best solution
    remaining_time = timeout - (time.time() - start_time)
    if best_mask and remaining_time > 5:
        S_set = mask_to_set(best_mask)
        anneal_budget = min(remaining_time - 3, 40.0)
        if anneal_budget > 2:
            S_annealed = anneal(S_set, anneal_budget)
            if len(S_annealed) > best_size:
                best_size = len(S_annealed)
                best_mask = set_to_mask(S_annealed)

                # Try to add more vertices after annealing
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        if not ((best_mask >> i) & 1):
                            if (conflict[i] & best_mask) == 0:
                                best_mask |= (1 << i)
                                best_size += 1
                                improved = True
                                break

    # Second round of BBMC with tighter bound if time permits
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        # Re-run BBMC with the current best as starting point
        # This can help if the first BBMC didn't explore enough
        def bbmc_tight(P, current):
            nonlocal best_size, best_mask

            if time.time() - start_time > timeout:
                return

            if P == 0:
                if len(current) > best_size:
                    best_size = len(current)
                    mask = 0
                    for v in current:
                        mask |= (1 << v)
                    best_mask = mask
                return

            bound = color_bound(P)
            if len(current) + bound <= best_size:
                return

            # Pick vertex with lowest rank
            v = -1
            min_rank = n + 1
            p = P
            while p:
                lsb = p & -p
                i = lsb.bit_length() - 1
                if rank[i] < min_rank:
                    min_rank = rank[i]
                    v = i
                p ^= lsb

            if v == -1:
                return

            v_bit = 1 << v

            # Include v
            new_P = P & adj[v]
            current.append(v)
            bbmc_tight(new_P, current)
            current.pop()

            # Exclude v
            new_P_exc = P & ~v_bit
            bbmc_tight(new_P_exc, current)

        bbmc_tight(all_mask, [])

    # Final local search
    if best_mask:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break

    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    if not result:
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
