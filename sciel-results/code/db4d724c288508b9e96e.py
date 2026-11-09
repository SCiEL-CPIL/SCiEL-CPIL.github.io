import time
import random

def count_mono_4ap(colors):
    n = 256
    c = colors
    count = 0
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i = a - 1
            col = c[i]
            if col == c[i + d] == c[i + 2 * d] == c[i + 3 * d]:
                count += 1
    return count

def build_affected(n=256):
    """For each position p (0-indexed), store list of (d, a) such that p is in the 4-AP a, a+d, a+2d, a+3d"""
    affected = [[] for _ in range(n)]
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            for j in range(4):
                p = i0 + j * d
                if p < n:
                    affected[p].append((d, a))
    return affected

def thue_morse(n):
    seq = [0] * n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def local_search(colors, deadline, affected):
    c = list(colors)
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            old = c[p]
            c[p] = 1 - old
            # Compute delta
            delta = 0
            for (d, a) in affected[p]:
                i0 = a - 1
                idxs = [i0, i0 + d, i0 + 2 * d, i0 + 3 * d]
                col_before = c[idxs[0]] if p != idxs[0] else old
                col_before = c[idxs[1]] if p != idxs[1] else old
                col_before = c[idxs[2]] if p != idxs[2] else old
                col_before = c[idxs[3]] if p != idxs[3] else old
                # Check if all same (mono)
                mono_before = (col_before == c[idxs[1]] == c[idxs[2]] == c[idxs[3]])
                # Now c[p] is already flipped, so check current state
                mono_after = (c[idxs[0]] == c[idxs[1]] == c[idxs[2]] == c[idxs[3]])
                if mono_after and not mono_before:
                    delta += 1
                elif mono_before and not mono_after:
                    delta -= 1
            new = cur + delta
            if new < cur:
                cur = new
                improved = True
            else:
                c[p] = old
    return c, cur

def simulated_annealing(colors, deadline, affected, start_temp=10.0, cool_rate=0.9995, min_temp=0.01):
    c = list(colors)
    cur = count_mono_4ap(c)
    best = list(c)
    best_count = cur
    temp = start_temp
    
    while temp > min_temp and time.time() < deadline:
        p = random.randint(0, 255)
        old = c[p]
        c[p] = 1 - old
        
        # Compute delta
        delta = 0
        for (d, a) in affected[p]:
            i0 = a - 1
            idxs = [i0, i0 + d, i0 + 2 * d, i0 + 3 * d]
            # mono_before: use old value at p
            col0 = old if idxs[0] == p else c[idxs[0]]
            col1 = old if idxs[1] == p else c[idxs[1]]
            col2 = old if idxs[2] == p else c[idxs[2]]
            col3 = old if idxs[3] == p else c[idxs[3]]
            mono_before = (col0 == col1 == col2 == col3)
            # mono_after: c[p] is already flipped
            mono_after = (c[idxs[0]] == c[idxs[1]] == c[idxs[2]] == c[idxs[3]])
            if mono_after and not mono_before:
                delta += 1
            elif mono_before and not mono_after:
                delta -= 1
        
        new = cur + delta
        if new < cur:
            cur = new
            if new < best_count:
                best_count = new
                best = list(c)
        else:
            # Accept with probability exp(-delta/temp)
            if random.random() < pow(2.718281828, -delta / temp):
                cur = new
            else:
                c[p] = old
        
        temp *= cool_rate
    
    return best, best_count

def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin before 150s kill
    
    affected = build_affected(256)
    
    candidates = []
    candidates.append(thue_morse(256))
    candidates.append([(i // 2) % 2 for i in range(256)])
    candidates.append([(i // 3) % 2 for i in range(256)])
    candidates.append([(i // 4) % 2 for i in range(256)])
    tm_full = thue_morse(257)
    candidates.append(tm_full[1:257])
    
    def v2_color(n):
        out = []
        for i in range(1, n + 1):
            x = i
            p = 0
            while x % 2 == 0:
                x //= 2
                p += 1
            out.append(p % 2)
        return out
    candidates.append(v2_color(256))
    candidates.append([(i % 2) ^ ((i // 2) % 2) for i in range(256)])
    
    # Add some random candidates
    for _ in range(3):
        candidates.append([random.randint(0, 1) for _ in range(256)])
    
    best = None
    best_count = float('inf')
    
    # First pass: evaluate candidates and run greedy local search on each
    for cand in candidates:
        if time.time() > deadline - 30:  # Leave 30s for SA
            break
        cnt = count_mono_4ap(cand)
        if cnt < best_count:
            best_count = cnt
            best = list(cand)
        # Run local search on this candidate
        ls_result, ls_count = local_search(cand, deadline - 30, affected)
        if ls_count < best_count:
            best_count = ls_count
            best = ls_result
    
    if best is None:
        best = list(candidates[0])
        best_count = count_mono_4ap(best)
    
    # Run simulated annealing from the best result
    if time.time() < deadline - 5:
        sa_result, sa_count = simulated_annealing(best, deadline - 5, affected, start_temp=8.0, cool_rate=0.9997, min_temp=0.01)
        if sa_count < best_count:
            best = sa_result
            best_count = sa_count
    
    # Final greedy local search pass
    if time.time() < deadline - 2:
        final_result, final_count = local_search(best, deadline - 2, affected)
        if final_count < best_count:
            best = final_result
            best_count = final_count
    
    return best
