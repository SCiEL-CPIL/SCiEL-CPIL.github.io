import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s limit
    
    def greedy_asc():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals
    
    def greedy_desc():
        vals = []
        sums_set = set()
        for x in range(256, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals
    
    def greedy_desc_start(start):
        vals = []
        sums_set = set()
        for x in range(start, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals
    
    def greedy_asc_start(start):
        vals = []
        sums_set = set()
        for x in range(start, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals
    
    # Try multiple greedy strategies
    candidates = []
    candidates.append(greedy_asc())
    candidates.append(greedy_desc())
    
    for start in range(256, 200, -1):
        candidates.append(greedy_desc_start(start))
    
    for start in range(1, 20):
        candidates.append(greedy_asc_start(start))
    
    # Also try some random-order greedy
    for seed in range(10):
        random.seed(seed)
        vals = []
        sums_set = set()
        pool = list(range(1, 257))
        random.shuffle(pool)
        for x in pool:
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        candidates.append(vals)
    
    best_seq = max(candidates, key=len)
    best_len = len(best_seq)
    
    # Efficient extension: try inserting elements one at a time
    def try_extend_fast(seq, time_budget=5.0):
        current = list(seq)
        used = set(current)
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            for pos in range(len(current) + 1):
                if pos == 0:
                    lo, hi = 1, (current[0] - 1 if current else 256)
                elif pos == len(current):
                    lo, hi = (current[-1] + 1, 256)
                else:
                    lo, hi = (current[pos-1] + 1, current[pos] - 1)
                
                # Try from high to low (larger values tend to work better)
                for x in range(hi, lo - 1, -1):
                    if x in used:
                        continue
                    # Quick check: compute new sums for the extended sequence
                    new_seq = current[:pos] + [x] + current[pos:]
                    n = len(new_seq)
                    expected = n * (n + 1) // 2
                    sums = set()
                    ok = True
                    for i in range(n):
                        s = 0
                        for j in range(i, n):
                            s += new_seq[j]
                            if s in sums:
                                ok = False
                                break
                            sums.add(s)
                        if not ok:
                            break
                    if ok and len(sums) == expected:
                        current = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return current
    
    extended = try_extend_fast(best_seq)
    if len(extended) > best_len:
        best_seq = extended
        best_len = len(best_seq)
    
    # Local search: try removing one element and reinserting elsewhere, then extend
    def local_search(seq, time_budget=15.0):
        current = list(seq)
        current_len = len(current)
        
        while (time.time() - start_time) < time_budget:
            # Try removing a random element
            if len(current) < 2:
                break
            i = random.randint(0, len(current) - 1)
            val = current.pop(i)
            
            # Try to reinsert at a different valid position
            inserted = False
            for pos in range(len(current) + 1):
                if pos == i:
                    continue
                if pos == 0:
                    if current and current[0] <= val:
                        continue
                elif pos == len(current):
                    if current and current[-1] >= val:
                        continue
                else:
                    if current[pos-1] >= val or current[pos] <= val:
                        continue
                
                new_seq = current[:pos] + [val] + current[pos:]
                n = len(new_seq)
                expected = n * (n + 1) // 2
                sums = set()
                ok = True
                for a in range(n):
                    s = 0
                    for b in range(a, n):
                        s += new_seq[b]
                        if s in sums:
                            ok = False
                            break
                        sums.add(s)
                    if not ok:
                        break
                if ok and len(sums) == expected:
                    current = new_seq
                    inserted = True
                    break
            
            if not inserted:
                current.insert(i, val)
            
            # Try to extend
            extended = try_extend_fast(current, time_budget=2.0)
            if len(extended) > current_len:
                current = extended
                current_len = len(current)
        
        return current
    
    searched = local_search(best_seq)
    if len(searched) > best_len:
        best_seq = searched
        best_len = len(best_seq)
    
    # Try multiple local search restarts from different greedy candidates
    for cand in candidates[:10]:
        if len(cand) < best_len:
            continue
        if (time.time() - start_time) > time_limit - 10:
            break
        extended_cand = try_extend_fast(cand, time_budget=3.0)
        if len(extended_cand) > best_len:
            best_seq = extended_cand
            best_len = len(best_seq)
        if (time.time() - start_time) > time_limit - 10:
            break
        searched_cand = local_search(extended_cand, time_budget=5.0)
        if len(searched_cand) > best_len:
            best_seq = searched_cand
            best_len = len(best_seq)
    
    return best_seq
