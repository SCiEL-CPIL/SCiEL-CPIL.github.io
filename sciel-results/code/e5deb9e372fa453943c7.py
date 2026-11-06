import random
import time

def construct():
    n = 512
    best_size = 0
    best_set = [False] * (n + 1)
    
    start_time = time.time()
    max_time = 55.0

    def can_add(x, in_set):
        # x is 4th: a, a+d, a+2d, x
        for d in range(1, x // 3 + 1):
            a = x - 3 * d
            if in_set[a] and in_set[a + d] and in_set[a + 2 * d]:
                return False
        # x is 3rd: a, a+d, x, a+3d
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b3 = a + 3 * d
            if b3 <= n and in_set[a] and in_set[a + d] and in_set[b3]:
                return False
        # x is 2nd: a, x, a+2d, a+3d
        for d in range(1, x):
            a = x - d
            b2 = a + 2 * d
            b3 = a + 3 * d
            if b3 <= n and in_set[a] and in_set[b2] and in_set[b3]:
                return False
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            b1 = x + d
            b2 = x + 2 * d
            b3 = x + 3 * d
            if b3 <= n and in_set[b1] and in_set[b2] and in_set[b3]:
                return False
        return True

    # 1. Deterministic Greedy: Natural Order
    in_set = [False] * (n + 1)
    count = 0
    for x in range(1, n + 1):
        if can_add(x, in_set):
            in_set[x] = True
            count += 1
    if count > best_size:
        best_size = count
        best_set = in_set[:]

    # 2. Deterministic Greedy: Reverse Order
    in_set = [False] * (n + 1)
    count = 0
    for x in range(n, 0, -1):
        if can_add(x, in_set):
            in_set[x] = True
            count += 1
    if count > best_size:
        best_size = count
        best_set = in_set[:]

    # 3. Random Greedy Restarts
    while time.time() - start_time < max_time:
        order = list(range(1, n + 1))
        random.shuffle(order)
        in_set = [False] * (n + 1)
        count = 0
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                count += 1
        if count > best_size:
            best_size = count
            best_set = in_set[:]

    return [i for i in range(1, n + 1) if best_set[i]]
