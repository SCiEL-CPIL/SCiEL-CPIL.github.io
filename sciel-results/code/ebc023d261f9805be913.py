import time
import math
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256

    # 1. Identify bad vertices (2*v is a perfect square)
    good_indices = []
    for v in range(1, N + 1):
        if int(math.isqrt(2 * v)) ** 2 == 2 * v:
            pass
        else:
            good_indices.append(v - 1)

    M = len(good_indices)
    if M == 0:
        return []

    # Build adjacency masks
    adj = [0] * N
    for i in good_indices:
        val_i = i + 1
        for j in good_indices:
            if i >= j:
                continue
            val_j = j + 1
            s = val_i + val_j
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def popcount(x):
        return x.bit_count()

    def greedy_is(mask):
        temp_mask = mask
        set_mask = 0
        count = 0
        while temp_mask:
            min_deg = float('inf')
            min_v = -1
            mm = temp_mask
            while mm:
                v = (mm & -mm).bit_length() - 1
                mm &= mm - 1
                deg = popcount(adj[v] & temp_mask)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v != -1:
                set_mask |= (1 << min_v)
                count += 1
                temp_mask &= ~adj[min_v]
                temp_mask &= ~(1 << min_v)
        return count, set_mask

    # Greedy coloring bound (Tomita-style)
    def greedy_color_bound(mask):
        """Upper bound on independent set size via greedy coloring of the subgraph."""
        if mask == 0:
            return 0
        colors = []
        remaining = mask
        while remaining:
            color_class = 0
            mm = remaining
            while mm:
                v = (mm & -mm).bit_length() - 1
                mm &= mm - 1
                if not (color_class & adj[v]):
                    color_class |= (1 << v)
            colors.append(color_class)
            remaining &= ~color_class
        return len(colors)

    # Modular residue seeding: group by residue mod small numbers
    def modular_seed(mask, mod):
        """Create a seed by picking vertices with favorable modular residues."""
        residues = {}
        mm = mask
        while mm:
            v = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            val = v + 1
            r = val % mod
            if r not in residues:
                residues[r] = 0
            residues[r] |= (1 << v)
        
        best_count = 0
        best_sub = 0
        for r, sub in residues.items():
            c, _ = greedy_is(sub)
            if c > best_count:
                best_count = c
                best_sub = sub
        return best_sub

    # Initial mask of all good vertices
    current_mask = 0
    for idx in good_indices:
        current_mask |= (1 << idx)

    # Domination Reduction
    def reduce_dominated(mask):
        changed = True
        while changed:
            changed = False
            verts = []
            tmp = mask
            while tmp:
                v = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                verts.append(v)
            
            for i in range(len(verts)):
                u = verts[i]
                if not (mask & (1 << u)):
                    continue
                nu = adj[u] | (1 << u)
                for j in range(len(verts)):
                    if i == j:
                        continue
                    v = verts[j]
                    if not (mask & (1 << v)):
                        continue
                    nv = adj[v] | (1 << v)
                    if (nu & ~nv) == 0:
                        mask &= ~(1 << u)
                        changed = True
                        break
                if changed:
                    break
        return mask

    current_mask = reduce_dominated(current_mask)

    # Initialize with multiple seeds
    init_size, init_mask = greedy_is(current_mask)
    best_size = init_size
    best_mask = init_mask

    # Try modular seeds
    for mod in [3, 4, 5, 6, 7, 8]:
        sub = modular_seed(current_mask, mod)
        c, sm = greedy_is(sub)
        if c > best_size:
            best_size = c
            best_mask = sm

    # Local Search: 1-2 Exchange
    def local_search_1_2(start_mask, max_iters=50):
        current_mask_ls = start_mask
        current_size_ls = popcount(start_mask)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            selected_list = []
            tmp = current_mask_ls
            while tmp:
                v = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                selected_list.append(v)

            random.shuffle(selected_list)

            for u in selected_list:
                mask_without_u = current_mask_ls & ~(1 << u)

                union_neighbors = 0
                tmp = mask_without_u
                while tmp:
                    v = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    union_neighbors |= adj[v]

                available = current_mask & ~mask_without_u & ~union_neighbors

                avail_list = []
                tmp = available
                while tmp:
                    v = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    avail_list.append(v)

                if len(avail_list) < 2:
                    continue

                random.shuffle(avail_list)

                found_pair = False
                for i in range(len(avail_list)):
                    v = avail_list[i]
                    for j in range(i + 1, len(avail_list)):
                        w = avail_list[j]
                        if not (adj[v] & (1 << w)):
                            new_mask = mask_without_u | (1 << v) | (1 << w)
                            new_size = current_size_ls + 1
                            if new_size > current_size_ls:
                                current_mask_ls = new_mask
                                current_size_ls = new_size
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break

            if not improved:
                break

        return current_mask_ls, current_size_ls

    # Run local search
    improved_mask, improved_size = local_search_1_2(best_mask, max_iters=80)
    if improved_size > best_size:
        best_size = improved_size
        best_mask = improved_mask

    # B&B Solver with coloring bound
    def solve(mask, current_size, current_set_mask):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return

        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_set_mask
            return

        # Prune with coloring bound
        bound = greedy_color_bound(mask)
        if current_size + bound <= best_size:
            return

        # Select vertex v from mask with highest degree in mask
        max_deg = -1
        v = -1
        mm = mask
        while mm:
            vv = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            deg = popcount(adj[vv] & mask)
            if deg > max_deg:
                max_deg = deg
                v = vv

        if v == -1:
            return

        # Branch 1: Include v
        new_mask_inc = mask & ~adj[v] & ~(1 << v)
        solve(new_mask_inc, current_size + 1, current_set_mask | (1 << v))

        # Branch 2: Exclude v
        new_mask_exc = mask & ~(1 << v)
        solve(new_mask_exc, current_size, current_set_mask)

    solve(current_mask, 0, 0)

    # Final local search polish
    final_mask, final_size = local_search_1_2(best_mask, max_iters=30)
    if final_size > best_size:
        best_size = final_size
        best_mask = final_mask

    result = []
    bm = best_mask
    while bm:
        v = (bm & -bm).bit_length() - 1
        bm &= bm - 1
        result.append(v + 1)

    return sorted(result)
