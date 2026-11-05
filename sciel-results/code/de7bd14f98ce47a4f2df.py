import itertools
import random
import time

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use TSP on permutations of 1..6
    symbols = list(range(1, 7))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Precompute overlap matrix
    overlap = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(6, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    best_length = float('inf')
    best_path = None
    
    start_time = time.time()
    time_limit = 110 # Leave some buffer
    
    # Greedy initialization
    def greedy_init():
        start = random.randint(0, num_perms - 1)
        visited = [False] * num_perms
        visited[start] = True
        path = [start]
        current = start
        for _ in range(num_perms - 1):
            best_next = -1
            best_ov = -1
            candidates = []
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        candidates = [j]
                    elif ov == best_ov:
                        candidates.append(j)
            if not candidates:
                break
            best_next = random.choice(candidates)
            visited[best_next] = True
            path.append(best_next)
            current = best_next
        return path

    def calc_overlap_sum(path):
        total = 0
        for k in range(1, len(path)):
            total += overlap[path[k-1]][path[k]]
        return total

    # Simulated Annealing
    def sa_refine(path):
        current_path = path[:]
        current_score = calc_overlap_sum(current_path)
        
        T = 50.0 # Initial temperature
        T_min = 0.1
        cooling = 0.999
        steps = 10000
        
        for s in range(steps):
            if T < T_min:
                break
            # Pick random i, j to swap
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Calculate delta overlap
            # Edges involved:
            # Left of i: (i-1, i)
            # Right of i: (i, i+1)
            # Left of j: (j-1, j)
            # Right of j: (j, j+1)
            
            old_ov = 0
            new_ov = 0
            
            # Helper to get overlap safely
            def get_ov(a, b):
                if a < 0 or a >= num_perms or b < 0 or b >= num_perms:
                    return 0
                return overlap[a][b]

            # Old
            if i > 0: old_ov += get_ov(current_path[i-1], current_path[i])
            if i < num_perms - 1: old_ov += get_ov(current_path[i], current_path[i+1])
            if j > 0: old_ov += get_ov(current_path[j-1], current_path[j])
            if j < num_perms - 1: old_ov += get_ov(current_path[j], current_path[j+1])
            
            # New (after swap)
            # p_i and p_j are swapped.
            # p_i is now at j, p_j is now at i.
            if i > 0: new_ov += get_ov(current_path[i-1], current_path[j])
            if i < num_perms - 1: new_ov += get_ov(current_path[j], current_path[i+1])
            if j > 0: new_ov += get_ov(current_path[j-1], current_path[i])
            if j < num_perms - 1: new_ov += get_ov(current_path[i], current_path[j+1])
            
            delta = new_ov - old_ov
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                # Accept
                current_path[i], current_path[j] = current_path[j], current_path[i]
                current_score += delta
                
            T *= cooling
        
        return current_path, current_score

    num_restarts = 10
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        path = greedy_init()
        path, score = sa_refine(path)
        
        length = 13 * num_perms - score
        
        if length < best_length:
            best_length = length
            best_path = path[:]
            
    if best_path is None:
        return standard_construct(n)
        
    # Construct string
    word = []
    p = perms[best_path[0]]
    block = list(p) + [7] + list(p)
    word.extend(block)
    
    for idx in range(1, len(best_path)):
        p = perms[best_path[idx]]
        block = list(p) + [7] + list(p)
        ov = overlap[best_path[idx-1]][best_path[idx]]
        word.extend(block[ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
