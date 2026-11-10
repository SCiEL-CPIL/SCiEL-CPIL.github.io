import time

def construct():
    deadline = time.time() + 85.0
    
    def has_sum_200(bits):
        return (bits >> 200) & 1
    
    def add_element(bits, x):
        return bits | (bits << x)
    
    def is_valid(bits):
        return not has_sum_200(bits)
    
    # Start with known valid set {66, ..., 100}
    # Verify: max 2-sum = 99+100 = 199 < 200, min 3-sum = 66+67+68 = 201 > 200
    base = list(range(66, 101))
    bits = 1  # bit 0 set
    for x in base:
        bits = add_element(bits, x)
    
    assert is_valid(bits), "Base set should be valid"
    
    best_set = set(base)
    best_size = len(base)
    
    # Try greedy addition in different orders
    candidates = [x for x in range(1, 66)]
    
    # Order 1: descending
    for order in [sorted(candidates, reverse=True), sorted(candidates)]:
        cur_bits = bits
        cur_set = set(base)
        for x in order:
            if time.time() > deadline:
                break
            new_bits = add_element(cur_bits, x)
            if is_valid(new_bits):
                cur_bits = new_bits
                cur_set.add(x)
        if len(cur_set) > best_size:
            best_size = len(cur_set)
            best_set = cur_set
    
    # Local search: try to improve by removing one and adding one or two
    # First, build a list of all elements not in best_set
    not_in_best = [x for x in range(1, 101) if x not in best_set]
    
    # Try swap: remove one from best, add one from not_in_best
    best_list = list(best_set)
    for i in range(len(best_list)):
        if time.time() > deadline:
            break
        for j in range(len(not_in_best)):
            if time.time() > deadline:
                break
            # Remove best_list[i], add not_in_best[j]
            new_set = set(best_list[:i] + best_list[i+1:] + [not_in_best[j]])
            # Quick check: rebuild bits
            nb = 1
            for x in sorted(new_set):
                nb = add_element(nb, x)
            if is_valid(nb) and len(new_set) == best_size:
                # Same size, but maybe enables further additions
                # Try to add more from not_in_best
                cur_bits = nb
                cur_set = new_set
                for y in not_in_best:
                    if y in cur_set:
                        continue
                    if time.time() > deadline:
                        break
                    new_b = add_element(cur_bits, y)
                    if is_valid(new_b):
                        cur_bits = new_b
                        cur_set.add(y)
                if len(cur_set) > best_size:
                    best_size = len(cur_set)
                    best_set = cur_set
                    not_in_best = [x for x in range(1, 101) if x not in best_set]
    
    # Try 2-for-1 swap: remove two, add three
    if len(best_set) < 95 and time.time() < deadline - 5:
        best_list = list(best_set)
        not_in_best = [x for x in range(1, 101) if x not in best_set]
        n_best = len(best_list)
        n_not = len(not_in_best)
        # Limit search for speed
        max_i = min(n_best, 20)
        max_j = min(n_best, 20)
        max_k = min(n_not, 10)
        max_l = min(n_not, 10)
        max_m = min(n_not, 10)
        for i in range(max_i):
            if time.time() > deadline:
                break
            for j in range(i+1, max_j):
                if time.time() > deadline:
                    break
                for k in range(max_k):
                    if time.time() > deadline:
                        break
                    for l in range(k+1, max_l):
                        if time.time() > deadline:
                            break
                        for m in range(l+1, max_m):
                            if time.time() > deadline:
                                break
                            new_set = set(best_list)
                            new_set.discard(best_list[i])
                            new_set.discard(best_list[j])
                            new_set.add(not_in_best[k])
                            new_set.add(not_in_best[l])
                            new_set.add(not_in_best[m])
                            if len(new_set) <= best_size:
                                continue
                            nb = 1
                            for x in sorted(new_set):
                                nb = add_element(nb, x)
                            if is_valid(nb):
                                # Try to add more
                                cur_bits = nb
                                cur_set = new_set
                                for y in not_in_best:
                                    if y in cur_set:
                                        continue
                                    if time.time() > deadline:
                                        break
                                    new_b = add_element(cur_bits, y)
                                    if is_valid(new_b):
                                        cur_bits = new_b
                                        cur_set.add(y)
                                if len(cur_set) > best_size:
                                    best_size = len(cur_set)
                                    best_set = cur_set
                                    not_in_best = [x for x in range(1, 101) if x not in best_set]
    
    return list(best_set)
