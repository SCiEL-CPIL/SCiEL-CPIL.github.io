import time
import random


def construct():
    n = 128
    t0 = time.time()
    deadline = t0 + 84.0

    def bits_to_list(s):
        return [i + 1 for i in range(n) if (s >> i) & 1]

    def popcount(s):
        return bin(s).count('1')

    def is_valid_full(s):
        elems = bits_to_list(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    break
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        break
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return False
        return True

    def build_pc(s):
        pc = [0] * (2 * n + 1)
        elems = bits_to_list(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                pc[a + elems[j]] += 1
        return pc

    def try_add(s, pc, x):
        if (s >> (x - 1)) & 1:
            return s, pc, True
        if pc[x] >= 2:
            return s, pc, False
        for y in range(1, n - x + 1):
            if (s >> (y - 1)) & 1:
                pc[x + y] += 1
        for u in range(1, x):
            if (s >> (u - 1)) & 1:
                pc[x + u] += 1
        return (s | (1 << (x - 1))), pc, True

    def remove_elem(s, pc, x):
        for y in range(1, n - x + 1):
            if (s >> (y - 1)) & 1:
                pc[x + y] -= 1
        for u in range(1, x):
            if (s >> (u - 1)) & 1:
                pc[x + u] -= 1
        return s & ~(1 << (x - 1))

    def greedy_from_core(core, order, dl):
        s = core
        pc = build_pc(core)
        for x in order:
            if time.time() > dl:
                break
            s, pc, _ = try_add(s, pc, x)
        return s, pc

    def local_search(s, pc, dl):
        improved = True
        while improved:
            if time.time() > dl:
                break
            improved = False
            in_set = bits_to_list(s)
            out_set = [x for x in range(1, n + 1) if not ((s >> (x - 1)) & 1)]
            for a in in_set:
                if time.time() > dl:
                    break
                s2, pc2 = s, pc[:]
                s2 = remove_elem(s2, pc2, a)
                added = 0
                for b in out_set:
                    if b == a:
                        continue
                    s2, pc2, ok = try_add(s2, pc2, b)
                    if ok:
                        added += 1
                        if added >= 2:
                            break
                if added >= 2 and popcount(s2) > popcount(s):
                    s, pc = s2, pc2
                    improved = True
                    break
        return s, pc

    core = 0
    for x in range(64, 129):
        core |= (1 << (x - 1))

    s, pc = greedy_from_core(core, list(range(1, 64)), deadline - 6)
    best_s = s
    best_size = popcount(s)

    rng = random.Random(987654321)
    for t in range(300):
        if time.time() > deadline - 8:
            break
        order = list(range(1, 64))
        rng.shuffle(order)
        s2, pc2 = greedy_from_core(core, order, deadline - 8)
        sz = popcount(s2)
        if sz > best_size:
            best_s, best_size = s2, sz

    pc = build_pc(best_s)
    best_s, _ = local_search(best_s, pc, deadline - 3)

    if not is_valid_full(best_s):
        s = 0
        for x in range(64, 129):
            s |= (1 << (x - 1))
        best_s = s

    result = bits_to_list(best_s)
    return [int(v) for v in result]
