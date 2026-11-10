def construct():
    n = 12
    subset_sums = {0}
    result = []
    used = set()
    for _ in range(n):
        v = 1
        while True:
            if v not in used:
                # Check disjointness
                new_sums = set()
                valid = True
                for s in subset_sums:
                    ns = s + v
                    if ns in subset_sums:
                        valid = False
                        break
                    if ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if valid:
                    subset_sums.update(new_sums)
                    result.append(v)
                    used.add(v)
                    break
            v += 1
    return result
