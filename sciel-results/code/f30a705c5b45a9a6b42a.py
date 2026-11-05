def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 80
    n = 255  # indices 0..254 correspond to values 2..256
    
    # Precompute conflicts
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = i + 2
        for j in range(i + 1, n):
            b = j + 2
            g = gcd(a, b)
            if a * b // g <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def calc_sum(s):
        return sum(1.0 / (i + 2) for i in s)
    
    def is_compat(idx, s):
        for i in s:
            if idx in conflict[i]:
                return False
        return True
    
    # Greedy init with multiple orderings
    def greedy(order):
        sel = set()
        for idx in order:
            if is_compat(idx, sel):
                sel.add(idx)
        return sel
    
    orders = [
        range(n),
        range(n - 1, -1, -1),
        sorted(range(n), key=lambda i: 1.0/(i+2)),  # same as range(n)
    ]
    
    best_set = set()
    best_score = 0.0
    for o in orders:
        s = greedy(o)
        sc = calc_sum(s)
        if sc > best_score:
            best_score = sc
            best_set = s
    
    # Local search
    def local_search(cur, cur_score):
        nonlocal best_set, best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Add moves
            for idx in range(n):
                if time.time() > deadline:
                    break
                if idx not in cur and is_compat(idx, cur):
                    ns = cur_score + 1.0 / (idx + 2)
                    if ns > cur_score + 1e-15:
                        cur.add(idx)
                        cur_score = ns
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_set = set(cur)
            # Swap moves
            if time.time() > deadline:
                break
            for rem in list(cur):
                if time.time() > deadline:
                    break
                wo = cur - {rem}
                base = cur_score - 1.0 / (rem + 2)
                for add in range(n):
                    if time.time() > deadline:
                        break
                    if add not in wo and is_compat(add, wo):
                        ns = base + 1.0 / (add + 2)
                        if ns > cur_score + 1e-15:
                            cur = set(wo)
                            cur.add(add)
                            cur_score = ns
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_set = set(cur)
                            break
        return cur, cur_score
    
    best_set, best_score = local_search(best_set, best_score)
    
    # Randomized restarts
    import random
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        order = list(range(n))
        rng.shuffle(order)
        s = greedy(order)
        sc = calc_sum(s)
        if sc > best_score:
            best_score = sc
            best_set = set(s)
        if time.time() < deadline:
            s, sc = local_search(s, sc)
    
    return [i + 2 for i in sorted(best_set)]
