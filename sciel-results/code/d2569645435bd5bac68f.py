import time

def construct():
    deadline = time.time() + 80
    
    def check_exact(elements):
        n = len(elements)
        if n == 0:
            return True
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for e in elements:
            for k in range(n - 1, -1, -1):
                if sums_by_card[k]:
                    for s in list(sums_by_card[k]):
                        sums_by_card[k + 1].add(s + e)
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_card[i] & sums_by_card[j]:
                    return False
        return True
    
    # Find best AP satisfying sufficient condition
    best_size = 0
    best_set = []
    for n in range(60, 0, -1):
        b = 256 - n + 1
        if b < 1:
            continue
        valid = True
        for k in range(1, n + 1):
            min_sum_k = k * b + k * (k - 1) // 2
            if k - 1 == 0:
                max_sum_prev = 0
            else:
                max_sum_prev = (k - 1) * (b + n - 1) - (k - 1) * (k - 2) // 2
            if min_sum_k <= max_sum_prev:
                valid = False
                break
        if valid:
            best_size = n
            best_set = list(range(b, b + n))
            break
    
    if not best_set:
        return [256]
    
    current = sorted(best_set)
    used = set(current)
    candidates = [x for x in range(1, 257) if x not in used]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in candidates[:]:
            if time.time() > deadline:
                break
            if cand in used:
                continue
            new_set = sorted(current + [cand])
            if check_exact(new_set):
                current.append(cand)
                current.sort()
                used.add(cand)
                candidates.remove(cand)
                improved = True
                break
    
    return list(current) if check_exact(current) else list(best_set)
