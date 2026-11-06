import time
import sys
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60s limit

    n = 256
    all_mask = (1 << n) - 1

    # ---- Build conflict graph (vectorized) ----
    # conflict[i] = bitmask of j != i such that (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    idx = np.arange(1, n + 1, dtype=np.int64)
    A = idx[:, None]          # (n,1)
    B = idx[None, :]          # (1,n)
    prod = A * B              # (n,n)
    s = A + B                 # (n,n)
    conflict_mat = (prod % s == 0)
    np.fill_diagonal(conflict_mat, False)

    # Pack into Python int masks (column-major so that row i = conflict of vertex i)
    conflict = [0] * n
    for i in range(n):
        mask_i = 0
        col = conflict_mat[:, i]
        for j in range(n):
            if col[j]:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Complementary adjacency (H): edge in H iff NOT conflict
    adj = [0] * n
    for i in range(n):
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i)

    degrees = [adj[i].bit_count() for i in range(n)]

    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    rank = [0] * n
    for idx_, v in enumerate(order):
        rank[v] = idx_

    # ---- Tomita greedy coloring bound on candidate set P ----
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
        verts.sort(key=lambda x: rank[x])

        colors = [-1] * n
        max_color = 0
        for v in verts:
            used = 0
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                if colors[u] >= 0:
                    used |= (1 << colors[u])
                np_ ^= lsb
            c = 0
            while used & (1 << c):
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        return max_color

    best_size = 0
    best_mask = 0

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

    # ---- Initial solution: all odd numbers (provably independent) ----
    odd_mask = 0
    for i in range(n):
        if (i + 1) % 2 == 1:
            odd_mask |= (1 << i)
    if odd_mask.bit_count() > best_size:
        best_size = odd_mask.bit_count()
        best_mask = odd_mask

    # Greedy with several orderings to seed a better lower bound
    def greedy_solve(seed_order):
        mask = 0
        P = all_mask
        for v in seed_order:
            if not (P & (1 << v)):
                continue
            if (conflict[v] & mask) == 0:
                mask |= (1 << v)
                P &= adj[v]
        return mask

    random.seed(12345)
    greedy_orders = [
        order,
        list(reversed(order)),
        sorted(range(n), key=lambda x: -degrees[x] + random.random()),
        sorted(range(n), key=lambda x: random.random()),
    ]
    for go in greedy_orders:
        gm = greedy_solve(go)
        if gm.bit_count() > best_size:
            best_size = gm.bit_count()
            best_mask = gm

    def greedy_dynamic():
        mask = 0
        P = all_mask
        while P:
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

    # ---- Run Tomita BBMC to find true MIS ----
    bbmc(all_mask, [])

    # ---- Local search: add any vertex that fits ----
    def polish_add(mask):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((mask >> i) & 1):
                    if (conflict[i] & mask) == 0:
                        mask |= (1 << i)
                        improved = True
                        break
        return mask

    if best_mask:
        best_mask = polish_add(best_mask)
        best_size = best_mask.bit_count()

    # ---- Simulated annealing refinement (with stagnation tracking) ----
    conflict_sets = [set() for _ in range(n)]
    for i in range(n):
        c = conflict[i]
        p = c
        while p:
            lsb = p & -p
            j = lsb.bit_length() - 1
            conflict_sets[i].add(j)
            p ^= lsb

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

    def anneal(S_set, time_budget):
        S = set(S_set)
        T0 = 1.5
        T_end = 0.01
        cooling = 0.9998
        T = T0
        best_S = set(S)
        best_size_anneal = len(S)
        stagnation = 0
        stagnation_limit = 800
        all_idx = list(range(n))

        while time.time() - start_time < time_budget:
            if T < T_end:
                T = T_end
            if not S:
                break

            move_type = random.random()
            S_cand = S
            if move_type < 0.3 and len(S) > 0:
                v = random.choice(tuple(S))
                S_new = S - {v}
                cands = [u for u in conflict_sets[v] if u not in S_new]
                valid = [u for u in cands if not (conflict_sets[u] & S_new)]
                rem = set(valid)
                add = set()
                while rem:
                    vc = random.choice(tuple(rem))
                    add.add(vc)
                    rem -= conflict_sets[vc]
                    rem.discard(vc)
                S_cand = S_new | add
            elif move_type < 0.5 and len(S) > 1:
                v = random.choice(tuple(S))
                S_new = S - {v}
                cands = [u for u in all_idx if u not in S_new and not (conflict_sets[u] & S_new)]
                if cands:
                    u = random.choice(cands)
                    S_cand = S_new | {u}
                else:
                    S_cand = S_new
            else:
                not_in_S = [u for u in all_idx if u not in S]
                if not_in_S:
                    u = random.choice(not_in_S)
                    if not (conflict_sets[u] & S):
                        S_cand = S | {u}
                    else:
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
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            T *= cooling
            if stagnation > stagnation_limit:
                T = max(T, 0.3)  # reheat
                stagnation = 0

        return best_S

    remaining_time = timeout - (time.time() - start_time)
    if best_mask and remaining_time > 5:
        S_set = mask_to_set(best_mask)
        anneal_budget = min(remaining_time - 3, 40.0)
        if anneal_budget > 2:
            S_annealed = anneal(S_set, anneal_budget)
            if len(S_annealed) > best_size:
                best_size = len(S_annealed)
                best_mask = set_to_mask(S_annealed)
                best_mask = polish_add(best_mask)
                best_size = best_mask.bit_count()

    # ---- Second BBMC pass if time permits (tighter pruning from better lower bound) ----
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        bbmc(all_mask, [])

    # ---- Final polish ----
    if best_mask:
        best_mask = polish_add(best_mask)
        best_size = best_mask.bit_count()

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
