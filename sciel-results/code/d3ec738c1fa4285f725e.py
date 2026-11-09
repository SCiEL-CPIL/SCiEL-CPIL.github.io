import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 16
    k = 4

    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)  # 1820

    best_family = None
    best_size = 0

    rng = random.Random(12345)

    while time.time() < deadline:
        order = list(range(total))
        rng.shuffle(order)

        family = []
        inter = []  # inter[i] is a list of length i, inter[i][j] = (family[i]&family[j]).bit_count()

        for idx in order:
            S = all_sets[idx]
            L = len(family)

            if L < 2:
                if L == 1:
                    inter.append([(family[0] & S).bit_count()])
                family.append(S)
                continue

            d = [0] * L
            new_inter = [0] * L
            for i in range(L):
                v = (family[i] & S).bit_count()
                d[i] = v
                new_inter[i] = v

            valid = True
            for i in range(L):
                di = d[i]
                for j in range(i + 1, L):
                    if d[j] == di and inter[i][j] == di:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                family.append(S)
                inter.append(new_inter)

        sz = len(family)
        if sz > best_size:
            best_size = sz
            best_family = family[:]
            if best_size >= 250:
                break

    if best_family is None:
        return []

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(n) if (mask >> i) & 1]
        result.append(elements)

    return result
