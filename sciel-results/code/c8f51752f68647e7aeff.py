import time

def construct():
    start_time = time.time()
    deadline = start_time + 55

    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    LCM = 30030

    # Order indices by divisor size descending for early pruning
    order_desc = sorted(range(n), key=lambda i: -divisors[i])

    def solve_for_W(W):
        for S in range(1, LCM + 1):
            if time.time() > deadline:
                return None
            candidates = [None] * n
            feasible = True
            # Check most constrained first
            for i in order_desc:
                d = divisors[i]
                k = (S + d - 1) // d
                vals = []
                v = k * d
                upper = S + W
                while v <= upper:
                    if v <= max_val and v > 0:
                        vals.append(v)
                    v += d
                if not vals:
                    feasible = False
                    break
                candidates[i] = vals
            if not feasible:
                continue
            # Fill remaining
            for i in range(n):
                if candidates[i] is None:
                    d = divisors[i]
                    k = (S + d - 1) // d
                    vals = []
                    v = k * d
                    upper = S + W
                    while v <= upper:
                        if v <= max_val and v > 0:
                            vals.append(v)
                        v += d
                    candidates[i] = vals

            # DFS with MRV
            indices = sorted(range(n), key=lambda i: len(candidates[i]))
            assigned = [0] * n
            used = set()

            def dfs(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        assigned[i] = val
                        used.add(val)
                        if dfs(idx + 1):
                            return True
                        used.remove(val)
                        assigned[i] = 0
                return False

            if dfs(0):
                if len(set(assigned)) == n and all(
                    1 <= x <= max_val and x % divisors[i] == 0
                    for i, x in enumerate(assigned)
                ):
                    return assigned
        return None

    best_solution = None
    best_W = float('inf')

    low, high = 9, 10000
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        res = solve_for_W(mid)
        if res is not None:
            best_solution = res
            best_W = mid
            high = mid - 1
        else:
            low = mid + 1

    if best_solution is None:
        center = LCM // 2
        used = set()
        best_solution = []
        for d in divisors:
            k = center // d
            found = False
            for offset in range(-30, 31):
                val = (k + offset) * d
                if val > 0 and val <= max_val and val not in used:
                    best_solution.append(val)
                    used.add(val)
                    found = True
                    break
            if not found:
                best_solution.append(d)
        if len(set(best_solution)) != n or not all(
            1 <= x <= max_val and x % divisors[i] == 0
            for i, x in enumerate(best_solution)
        ):
            best_solution = divisors[:]

    return [int(x) for x in best_solution]
