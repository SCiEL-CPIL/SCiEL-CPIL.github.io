import time

def construct():
    deadline = time.time() + 85
    n = 256

    # Greedy initialization: add in order of decreasing reciprocal
    selected = []
    for x in range(1, n + 1):
        ok = True
        for a in selected:
            for b in selected:
                if a < b and x == 2 * b - a:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)

    best = list(selected)
    best_val = sum(1.0 / x for x in best)

    def can_add(s, x):
        for a in s:
            for b in s:
                if a < b:
                    if x == 2 * b - a:
                        return False
                elif b < a:
                    if x == 2 * a - b:
                        return False
        return True

    # Local search: try removing one element and greedily refilling
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        for elem in best:
            if time.time() > deadline - 5:
                break
            rem = [x for x in best if x != elem]
            rem_set = set(rem)
            new_list = list(rem)

            for x in range(1, n + 1):
                if x in rem_set:
                    continue
                if can_add(new_list, x):
                    new_list.append(x)
                    rem_set.add(x)

            val = sum(1.0 / x for x in new_list)
            if val > best_val + 1e-15:
                best = new_list
                best_val = val
                improved = True
                break

        if not improved:
            break

    return best
