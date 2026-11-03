import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)
    return neigh, neigh_set, total

def _build_distance2_graph(q, n, total):
    """Build adjacency list for words at Hamming distance exactly 2."""
    dist2_adj = [[] for _ in range(total)]
    for idx in range(total):
        for jdx in range(idx + 1, total):
            # Count differing positions
            diff = 0
            a, b = idx, jdx
            for i in range(n):
                da = a % q
                db = b % q
                if da != db:
                    diff += 1
                a //= q
                b //= q
            if diff == 2:
                dist2_adj[idx].append(jdx)
                dist2_adj[jdx].append(idx)
    return dist2_adj

def _greedy_independent_set(dist2_adj, total, rng):
    """Greedy maximal independent set on distance-2 graph.
    Words in an independent set are pairwise at distance >= 3,
    so their radius-1 balls are disjoint. This gives a valid covering
    (every word is within distance 1 of itself, which is in the set... 
    wait, no. An independent set doesn't cover everything.
    
    Actually, we need a covering code. Let me reconsider.
    
    A better approach: greedy set cover where we pick words that cover the most
    uncovered words, but with a bias toward words that are far from already-chosen words.
    """
    pass

def _greedy_cover_init(q, n, radius, total, neigh, neigh_set, dist2_adj, deadline, rng):
    """Greedy set cover with distance-2 awareness.
    
    Strategy: repeatedly pick the word that covers the most uncovered words.
    Tie-breaking: prefer words that are at distance >= 3 from all already-chosen words
    (i.e., in the independent set of the distance-2 graph). This leads to a sparser
    initial code that is easier to optimize.
    """
    counts = [0] * total
    code = []
    remaining = [True] * total
    rem_count = total
    in_code = set()
    
    def add_to_code(c):
        nonlocal rem_count
        code.append(c)
        in_code.add(c)
        for u in neigh[c]:
            counts[u] += 1
            if counts[u] == 1:
                remaining[u] = False
                rem_count -= 1
    
    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        best_dist_penalty = float('inf')
        
        for c in range(total):
            if c in in_code:
                continue
            gain = 0
            for u in neigh[c]:
                if remaining[u]:
                    gain += 1
            if gain <= best_gain:
                continue
            # Tie-breaking: prefer words far from existing code
            min_dist_to_code = n + 1
            for oc in code:
                # Quick distance check
                diff = 0
                a, b = c, oc
                for i in range(n):
                    if a % q != b % q:
                        diff += 1
                    a //= q
                    b //= q
                if diff < min_dist_to_code:
                    min_dist_to_code = diff
                if min_dist_to_code <= 1:
                    break
            # We prefer higher min_dist_to_code (farther from existing)
            if gain > best_gain or (gain == best_gain and min_dist_to_code > best_dist_penalty):
                best_gain = gain
                best_c = c
                best_dist_penalty = min_dist_to_code
        
        if best_c == -1:
            break
        add_to_code(best_c)
    
    if rem_count > 0:
        # Fallback: cover remaining with arbitrary words
        for u in range(total):
            if remaining[u]:
                add_to_code(u)
                break
    
    return code, counts

def _local_search(code, counts, neigh, neigh_set, total, n, q, deadline, rng, best_size, best_code):
    """Simulated annealing local search to reduce code size."""
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c):
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    in_code = set(code)
    
    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.45:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    code.pop(idx)
                    in_code.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size[0]:
                        best_size[0] = len(code)
                        best_code[0] = code[:]
                        
        elif move_type < 0.85:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in in_code:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    in_code.discard(old_c)
                    in_code.add(new_c)
                    code[idx] = new_c
                    if len(code) < best_size[0]:
                        best_size[0] = len(code)
                        best_code[0] = code[:]

        else:
            # Try Add (Perturbation to escape local minima)
            new_c = rng.randint(0, total - 1)
            if new_c not in in_code:
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    in_code.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate

    return best_size[0], best_code[0]

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Build distance-2 graph for tie-breaking
    dist2_adj = _build_distance2_graph(q, n, total)
    
    # 3. Multiple restarts with different random seeds
    overall_best_size = total
    overall_best_code = None
    
    num_restarts = 5
    time_per_restart = (deadline - time.monotonic()) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 2.0:
            break
        
        rng = random.Random(42 + restart * 1000)
        
        # Greedy initialization with distance-2 awareness
        init_deadline = time.monotonic() + min(15.0, time_per_restart * 0.3)
        code, counts = _greedy_cover_init(q, n, radius, total, neigh, neigh_set, dist2_adj, init_deadline, rng)
        
        if overall_best_code is None or len(code) < overall_best_size:
            overall_best_size = len(code)
            overall_best_code = code[:]
        
        # Local search
        ls_deadline = time.monotonic() + min(time_per_restart * 0.7, deadline - time.monotonic() - 2.0)
        best_size = [len(code)]
        best_code = [code[:]]
        _local_search(code, counts, neigh, neigh_set, total, n, q, ls_deadline, rng, best_size, best_code)
        
        if best_size[0] < overall_best_size:
            overall_best_size = best_size[0]
            overall_best_code = best_code[0]
    
    # 4. Final cleanup pass if time remains
    if overall_best_code is not None and time.monotonic() < deadline - 5.0:
        # Try to remove any redundant codewords
        rng_final = random.Random(999)
        in_code = set(overall_best_code)
        counts_final = [0] * total
        for c in overall_best_code:
            for u in neigh[c]:
                counts_final[u] += 1
        
        # Try removing each codeword if possible
        for i in range(len(overall_best_code) - 1, -1, -1):
            c = overall_best_code[i]
            removable = True
            for u in neigh[c]:
                if counts_final[u] == 1:
                    removable = False
                    break
            if removable:
                overall_best_code.pop(i)
                in_code.discard(c)
                for u in neigh[c]:
                    counts_final[u] -= 1
    
    # Convert best_code indices to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
