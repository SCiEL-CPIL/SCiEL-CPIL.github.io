def construct():
    from fractions import Fraction
    import time
    
    start = time.time()
    deadline = start + 55
    
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # DP: achievable sums (strictly below 1) -> one witness (list of indices)
    achievable = {Fraction(0): []}
    
    for i, frac in enumerate(fracs):
        new_sums = []
        for s, idxs in achievable.items():
            new_s = s + frac
            if new_s < 1:
                new_sums.append((new_s, idxs + [i]))
        
        for new_s, new_idxs in new_sums:
            if new_s not in achievable:
                achievable[new_s] = new_idxs
        
        if time.time() > deadline - 5:
            break
    
    # Find the best (maximum) sum strictly below 1
    best_sum = Fraction(0)
    best_idxs = []
    for s, idxs in achievable.items():
        if s > best_sum:
            best_sum = s
            best_idxs = idxs[:]
    
    if not best_idxs:
        return [0]
    
    # Local search: 1-swap improvements
    if time.time() < deadline - 3:
        in_set = set(best_idxs)
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            improved = False
            for out_i in sorted(in_set):
                if time.time() > deadline - 3:
                    break
                for in_i in range(n):
                    if in_i in in_set:
                        continue
                    new_sum = best_sum - fracs[out_i] + fracs[in_i]
                    if new_sum < 1 and new_sum > best_sum:
                        best_sum = new_sum
                        best_idxs.remove(out_i)
                        best_idxs.append(in_i)
                        in_set.remove(out_i)
                        in_set.add(in_i)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_idxs
