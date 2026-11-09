import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute divisibility: divs[a] contains all sums s in [a, 512] where a | s
    max_sum = 2 * n
    divs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(a, max_sum + 1, a):
            divs[a].add(s)

    def can_add(S_set, x, S_list):
        if x in S_set:
            return True
        nS = len(S_list)

        # Case 1: a = x, need b,c in S with x < b <= c and x | (b+c)
        for i in range(nS):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, nS):
                c = S_list[j]
                if c < b:
                    continue
                if (b + c) in divs[x]:
                    return False

        # Case 2: a in S with a < x, c in S with c >= x, need a | (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) in divs[a]:
                    return False

        # Case 3: a,b in S with a < b <= x, need a | (b+x)
        for i in range(nS):
            a = S_list[i]
            if a >= x:
                break
            for j in range(i + 1, nS):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) in divs[a]:
                    return False

        return True

    best_set = set(range(192, 257))
    best_size = len(best_set)

    # Try multiple greedy orders
    orders = [
        list(range(1, n + 1)),          # ascending
        list(range(n, 0, -1)),          # descending
        list(range(1, n + 1, 2)),       # odds ascending
        list(range(n, 0, -2)),          # odds descending
        list(range(128, 0, -1)) + list(range(1, 128)),  # large then small
    ]

    for order in orders:
        if time.time() > deadline - 20.0:
            break
        S_set = set()
        S_list = []
        for x in order:
            if time.time() > deadline - 20.0:
                break
            if can_add(S_set, x, S_list):
                S_set.add(x)
                S_list.append(x)
                S_list.sort()
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()

    # Local search: repeatedly try to add any element
    S_set = best_set
    S_list = sorted(S_set)
    for _ in range(30):
        if time.time() > deadline - 10.0:
            break
        added = False
        for x in range(1, n + 1):
            if time.time() > deadline - 10.0:
                break
            if x in S_set:
                continue
            if can_add(S_set, x, S_list):
                S_set.add(x)
                S_list.append(x)
                S_list.sort()
                added = True
        if not added:
            break

    return sorted(S_set)
