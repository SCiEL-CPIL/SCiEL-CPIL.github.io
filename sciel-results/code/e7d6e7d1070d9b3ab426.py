import time

def construct():
    start_time = time.time()
    # Evaluator kills at 90s wall-clock. Return with a generous safety margin.
    deadline = start_time + 80.0

    N = 256
    w = [0.0] * (N + 1)
    for i in range(1, N + 1):
        w[i] = 1.0 / i

    # Suffix sums of weights for tight upper-bound pruning.
    # suffix[i] = sum_{k=i}^{N} w[k]
    suffix = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + w[i]

    present = [False] * (N + 1)
    chosen = [False] * (N + 1)
    best_val = 0.0
    best_mask = 0  # bitmask of best set (elements 1..256 -> bits 0..255)
    nodes = 0

    def conflict(x, p):
        # True if adding x to set described by p creates a nonconstant 3-AP.
        # x is middle: a, x, c  => c = 2x - a
        c = 2 * x - 1
        while c >= 1:
            if p[c]:
                return True
            c -= 2
        # x is end: a, m, x  => m = (a+x)/2
        m = (x - 1) // 2 + 1
        while m >= 1:
            if p[m]:
                return True
            m -= 2
        return False

    def greedy_init():
        # Add in increasing order (highest weight first) if valid.
        for x in range(1, N + 1):
            if not conflict(x, present):
                present[x] = True
                chosen[x] = True
                best_val = best_val + w[x]

    def set_bit(mask, x):
        return mask | (1 << (x - 1))

    def clear_bit(mask, x):
        return mask & ~(1 << (x - 1))

    def mask_to_list(mask):
        out = []
        x = 1
        while x <= N:
            if mask & (1 << (x - 1)):
                out.append(x)
            x += 1
        return out

    # ---- Warm start ----
    greedy_init()
    # build best_mask from chosen
    for x in range(1, N + 1):
        if chosen[x]:
            best_mask = set_bit(best_mask, x)

    # ---- Branch and bound ----
    def bb(x, cur):
        nonlocal best_val, best_mask, nodes
        # bound: cur + sum of weights of all remaining elements x..N
        if cur + suffix[x] <= best_val + 1e-15:
            return
        if x > N:
            if cur > best_val:
                best_val = cur
                # rebuild mask from chosen
                m = 0
                for k in range(1, N + 1):
                    if chosen[k]:
                        m = set_bit(m, k)
                best_mask = m
            return
        # Option 1: include x
        if not conflict(x, present):
            present[x] = True
            chosen[x] = True
            nodes += 1
            if time.time() > deadline:
                present[x] = False
                chosen[x] = False
                return
            bb(x + 1, cur + w[x])
            present[x] = False
            chosen[x] = False
            if time.time() > deadline:
                return
        # Option 2: exclude x
        nodes += 1
        bb(x + 1, cur)

    # Restart search from empty set (greedy gave incumbent; now refine).
    # Clear present/chosen.
    for x in range(1, N + 1):
        present[x] = False
        chosen[x] = False

    # We need the current best_val to be the greedy value (it is).
    # Run B&B.
    if time.time() < deadline:
        bb(1, 0.0)

    # ---- Hill-climb polish on best_mask ----
    # Rebuild present from best_mask.
    for x in range(1, N + 1):
        present[x] = bool(best_mask & (1 << (x - 1)))

    def hill_climb():
        nonlocal best_val, best_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try single adds
            for x in range(1, N + 1):
                if time.time() > deadline:
                    return
                if not present[x]:
                    if not conflict(x, present):
                        present[x] = True
                        best_val += w[x]
                        best_mask = set_bit(best_mask, x)
                        improved = True
            # Try single swaps (remove u, add v)
            for v in range(1, N + 1):
                if time.time() > deadline:
                    return
                if present[v]:
                    continue
                # find u in S whose removal makes v valid
                for u in range(1, N + 1):
                    if not present[u]:
                        continue
                    if u == v:
                        continue
                    # temporarily remove u
                    present[u] = False
                    if not conflict(v, present):
                        gain = w[v] - w[u]
                        if gain > 1e-15:
                            # accept
                            present[v] = True
                            best_val += gain
                            best_mask = set_bit(best_mask, v)
                            best_mask = clear_bit(best_mask, u)
                            improved = True
                            break  # v now in set
                    present[u] = True
                    if improved:
                        break
            # Try double swaps (remove u1,u2 add v1,v2) - limited
            # Only if time remains.
            if time.time() > deadline - 10.0:
                return
            # Enumerate candidate adds (v) not in S, and pairs of removals.
            # To keep bounded: for each pair of removals (u1<u2) in S, try to add
            # up to 2 best remaining v's.
            inS = [x for x in range(1, N + 1) if present[x]]
            notS = [x for x in range(1, N + 1) if not present[x]]
            # Limit work
            for i in range(len(inS)):
                if time.time() > deadline - 5.0:
                    return
                u1 = inS[i]
                for j in range(i + 1, len(inS)):
                    u2 = inS[j]
                    present[u1] = False
                    present[u2] = False
                    # find best v (and optional second v) to add
                    best_add = None
                    for v in notS:
                        if not conflict(v, present):
                            # try single
                            if best_add is None or w[v] > w[best_add]:
                                best_add = v
                    if best_add is not None:
                        # consider adding one
                        gain = w[best_add] - w[u1] - w[u2]
                        if gain > 1e-15:
                            present[best_add] = True
                            best_val += gain
                            best_mask = set_bit(best_mask, best_add)
                            best_mask = clear_bit(best_mask, u1)
                            best_mask = clear_bit(best_mask, u2)
                            improved = True
                        # consider adding two
                        for v2 in notS:
                            if v2 == best_add:
                                continue
                            if not conflict(v2, present):
                                gain2 = w[best_add] + w[v2] - w[u1] - w[u2]
                                if gain2 > 1e-15:
                                    present[v2] = True
                                    best_val += gain2
                                    best_mask = set_bit(best_mask, v2)
                                    best_mask = clear_bit(best_mask, u1)
                                    best_mask = clear_bit(best_mask, u2)
                                    improved = True
                                    break
                    present[u1] = True
                    present[u2] = True
                    if improved:
                        break
                if improved:
                    break

    if time.time() < deadline:
        hill_climb()

    return mask_to_list(best_mask)


if __name__ == "__main__":
    print(construct())
