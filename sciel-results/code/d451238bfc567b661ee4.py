def construct():
    from fractions import Fraction
    import random
    import time
    
    denoms = list(range(2, 49))  # 47 denominators
    n = len(denoms)
    vals = [Fraction(1, d) for d in denoms]
    one = Fraction(1, 1)
    
    best_sum = Fraction(0)
    best_set = set()
    
    def evaluate(s):
        # s is a set of indices; return sum
        tot = Fraction(0)
        for i in s:
            tot += vals[i]
        return tot
    
    def try_update(s):
        nonlocal best_sum, best_set
        if not s:
            return
        tot = evaluate(s)
        if tot < one and tot > best_sum:
            best_sum = tot
            best_set = set(s)
    
    # Greedy best-fit
    def greedy_best_fit():
        s = set()
        cur = Fraction(0)
        available = list(range(n))
        while available:
            # pick i in available maximizing cur + vals[i] < 1
            best_i = None
            best_val = None
            for i in available:
                nv = cur + vals[i]
                if nv < one:
                    if best_val is None or nv > best_val:
                        best_val = nv
                        best_i = i
            if best_i is None:
                break
            s.add(best_i)
            cur = best_val
            available.remove(best_i)
        return s, cur
    
    # Local search
    def local_search(s):
        cur = evaluate(s)
        improved = True
        while improved:
            improved = False
            # try add
            for i in range(n):
                if i not in s:
                    if cur + vals[i] < one:
                        ns = s | {i}
                        ns_sum = cur + vals[i]
                        if ns_sum > cur:
                            s = ns
                            cur = ns_sum
                            improved = True
            # try remove (to allow adding bigger? removing reduces sum, not helpful for max, but might allow swap)
            # try swap: remove one, add one
            for i in list(s):
                for j in range(n):
                    if j not in s:
                        ns_sum = cur - vals[i] + vals[j]
                        if ns_sum < one and ns_sum > cur:
                            s = (s - {i}) | {j}
                            cur = ns_sum
                            improved = True
        return s, cur
    
    # Simulated annealing
    def simulated_annealing(s, cur, steps=200000):
        import math
        T = Fraction(1, 100)  # temperature as Fraction? use float for SA
        # use float deficit for SA
        def deficit_float(s, cur):
            return float(one - cur)
        best_local = s
        best_local_sum = cur
        T_start = 0.01
        T_end = 1e-6
        for t in range(steps):
            # random move
            r = random.random()
            ns = set(s)
            if r < 0.4:
                # add
                avail = [i for i in range(n) if i not in s]
                if avail:
                    i = random.choice(avail)
                    ns.add(i)
            elif r < 0.7:
                # remove
                if s:
                    i = random.choice(list(s))
                    ns.discard(i)
            else:
                # swap
                if s:
                    i = random.choice(list(s))
                    avail = [j for j in range(n) if j not in s]
                    if avail:
                        j = random.choice(avail)
                        ns = ns - {i} | {j}
            if not ns:
                continue
            ns_sum = evaluate(ns)
            if ns_sum >= one:
                continue
            d_old = float(one - cur)
            d_new = float(one - ns_sum)
            if d_new < d_old:
                s = ns
                cur = ns_sum
                if cur > best_local_sum:
                    best_local = set(s)
                    best_local_sum = cur
            else:
                # accept with prob
                delta = d_new - d_old
                T = T_start * (T_end/T_start) ** (t/steps)
                if random.random() < math.exp(-delta / max(T, 1e-12)):
                    s = ns
                    cur = ns_sum
        return best_local, best_local_sum
    
    # Initial: greedy
    s, cur = greedy_best_fit()
    s, cur = local_search(s)
    try_update(s)
    
    # SA from this
    s2, cur2 = simulated_annealing(set(s), cur, steps=150000)
    try_update(s2)
    
    # A few random restarts
    for _ in range(3):
        s_r = set()
        cur_r = Fraction(0)
        # random greedy
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if cur_r + vals[i] < one:
                s_r.add(i)
                cur_r += vals[i]
        s_r, cur_r = local_search(s_r)
        try_update(s_r)
    
    return list(best_set)
