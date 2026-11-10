import time
import random

def construct():
    start = time.time()
    deadline = start + 80.0
    N = 128
    rng = random.Random(987654321)

    def time_left():
        return deadline - time.time()

    def can_add(mem, inS, x):
        # x is NOT in S. Return True if adding x creates no forbidden triple.
        # A forbidden triple must contain x: {x, a, b} with a<b in S,
        # and x+a, x+b, a+b all in S.
        for i in range(len(mem)):
            a = mem[i]
            xa = x + a
            if xa > N or not inS[xa]:
                continue
            for j in range(i + 1, len(mem)):
                b = mem[j]
                xb = x + b
                ab = a + b
                if xb <= N and inS[xb] and ab <= N and inS[ab]:
                    return False
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    def solve_restart(order):
        mem = [i for i in range(N // 2 + 1, N + 1)]
        inS = bytearray(N + 1)
        for i in mem:
            inS[i] = 1
        # Greedy add of small numbers in the given order.
        for x in order:
            if time_left() < 4.0:
                break
            if inS[x]:
                continue
            if can_add(mem, inS, x):
                mem.append(x)
                inS[x] = 1
        # Local search: add if possible, else swap to enable growth.
        for _ in range(60):
            if time_left() < 4.0:
                break
            added = False
            for x in range(1, N + 1):
                if time_left() < 4.0:
                    break
                if inS[x]:
                    continue
                if can_add(mem, inS, x):
                    mem.append(x)
                    inS[x] = 1
                    added = True
            if added:
                continue
            # No direct add possible; try a few swaps to escape the local optimum.
            sw = False
            for _ in range(50):
                if time_left() < 4.0:
                    break
                rm = rng.choice(mem)
                add = rng.randint(1, N)
                if inS[add]:
                    continue
                inS[rm] = 0
                mem.remove(rm)
                if can_add(mem, inS, add):
                    mem.append(add)
                    inS[add] = 1
                    sw = True
                    break
                else:
                    mem.append(rm)
                    inS[rm] = 1
            if not sw:
                break
        return mem

    base = [i for i in range(N // 2 + 1, N + 1)]
    best = base
    best_size = len(base)

    small = list(range(1, N // 2 + 1))
    orders = [list(small), list(reversed(small))]
    for _ in range(14):
        o = list(small)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time_left() < 6.0:
            break
        mem = solve_restart(o)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    if not full_valid(best):
        best = base

    return [int(x) for x in best]
