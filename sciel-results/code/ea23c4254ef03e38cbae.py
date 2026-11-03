"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            # Find maximum overlap between suffix of new and prefix of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]

def _overlap(a, b):
    """Maximum overlap: suffix of a matches prefix of b."""
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses greedy + 2-opt + or-opt local search with incremental cost updates.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Initial order: the classic order
    order = list(range(n_perms))
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    
    # Precompute block lengths
    block_lens = [len(b) for b in blocks]
    
    # Build overlap matrix
    ov_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    
    def compute_cost(order):
        """Compute total cost for a given order."""
        cost = block_lens[order[0]]
        for k in range(1, len(order)):
            i, j = order[k-1], order[k]
            cost += block_lens[j] - ov_matrix[i][j]
        return cost
    
    def local_search_2opt(cur_order, cur_cost, t_limit):
        """2-opt local search with O(1) cost delta computation."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for i in range(n - 1):
                if (time.time() - start_time) > t_limit:
                    break
                for j in range(i + 1, n):
                    # Reverse segment [i+1..j]
                    old_edges = 0
                    new_edges = 0
                    # Edge from order[i] to order[i+1]
                    old_edges += block_lens[cur_order[i+1]] - ov_matrix[cur_order[i]][cur_order[i+1]]
                    # Edge from order[j] to order[j+1] (if exists)
                    if j + 1 < n:
                        old_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[j]][cur_order[j+1]]
                    # New edge from order[i] to order[j]
                    new_edges += block_lens[cur_order[j]] - ov_matrix[cur_order[i]][cur_order[j]]
                    # New edge from order[i+1] to order[j+1] (if exists)
                    if j + 1 < n:
                        new_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[i+1]][cur_order[j+1]]
                    
                    delta = new_edges - old_edges
                    if delta < 0:
                        cur_cost += delta
                        cur_order[i+1:j+1] = cur_order[i+1:j+1][::-1]
                        improved = True
        return cur_order, cur_cost
    
    def local_search_oropt(cur_order, cur_cost, t_limit):
        """Or-opt: move single element or small segment to a different position."""
        improved = True
        seg_sizes = [1, 2]
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for size in seg_sizes:
                if (time.time() - start_time) > t_limit:
                    break
                for i in range(n - size):
                    if (time.time() - start_time) > t_limit:
                        break
                    seg = cur_order[i:i+size]
                    for j in range(n - size + 1):
                        if (time.time() - start_time) > t_limit:
                            break
                        if j == i or j == i + size - 1:
                            continue
                        new_order = cur_order[:i] + cur_order[i+size:]
                        if j > i:
                            j -= size
                        new_order = new_order[:j] + seg + new_order[j:]
                        new_cost = compute_cost(new_order)
                        if new_cost < cur_cost:
                            cur_cost = new_cost
                            cur_order = new_order
                            improved = True
        return cur_order, cur_cost
    
    best_order = list(order)
    best_cost = compute_cost(best_order)
    
    # Phase 1: 2-opt from classic order
    t1 = min(time_limit * 0.4, time_limit - 2.0)
    if t1 > 0:
        best_order, best_cost = local_search_2opt(best_order, best_cost, t1)
    
    # Phase 2: or-opt refinement
    t2 = min(time_limit * 0.3, time_limit - 1.0)
    if t2 > 0:
        best_order, best_cost = local_search_oropt(best_order, best_cost, t2)
    
    # Phase 3: random restarts if time remains
    if (time.time() - start_time) < time_limit - 1.0:
        import random
        random.seed(42)
        n_restarts = 2
        for r in range(n_restarts):
            if (time.time() - start_time) > time_limit - 0.5:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            cur_order = list(rand_order)
            cur_cost = compute_cost(cur_order)
            cur_order, cur_cost = local_search_2opt(cur_order, cur_cost, time_limit - 0.5)
            cur_order, cur_cost = local_search_oropt(cur_order, cur_cost, time_limit - 0.3)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = cur_order
    
    return best_order, best_cost

def _greedy_superperm(n, time_limit=30.0):
    """
    Greedy maximal-overlap path construction with dead-end lookahead.
    Inspired by the alternative approach: build the superpermutation symbol by symbol,
    choosing the next symbol that extends to a new permutation while avoiding dead ends.
    """
    start_time = time.time()
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    used = set()
    
    # For each (n-1)-tuple prefix, the set of symbols that extend it to a permutation
    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, set()).add(p[-1])
    
    def is_dead_end(prefix):
        last = prefix[-1]
        for s in ext.get(prefix, ()):
            if s == last:
                continue
            p = prefix + (s,)
            if p not in used:
                return False
        return True
    
    # Start with the smallest permutation
    word = list(range(1, n + 1))
    used.add(tuple(word))
    
    while len(used) < len(all_perms):
        if time.time() - start_time > time_limit:
            return None
        
        suffix = tuple(word[-(n - 1):])
        last = word[-1]
        
        # Candidate symbols in ascending order, != last, that complete a NEW permutation
        cands = []
        for s in ext.get(suffix, ()):
            if s == last:
                continue
            p = suffix + (s,)
            if p not in used:
                cands.append((s, p))
        cands.sort()
        
        if not cands:
            return None
        
        chosen = None
        # Lookahead: prefer a candidate that does NOT create a dead-end suffix
        for s, p in cands:
            new_suffix = (suffix + (s,))[-(n - 1):]
            if not is_dead_end(new_suffix):
                chosen = (s, p)
                break
        
        if chosen is None:
            chosen = cands[0]
        
        s, p = chosen
        word.append(s)
        used.add(p)
    
    # Verify completeness
    if time.time() - start_time < time_limit:
        seen = set()
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i + n]))
        if len(seen) == len(all_perms):
            return word
    
    return None

def construct(n):
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
    
    # For n >= 4, use optimized construction
    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Get classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    best_word = classic_word
    best_len = classic_len
    
    # Strategy 1: Try greedy construction (inspired by alternative approach)
    # This can sometimes find shorter solutions by making different choices
    if n <= 6 and (time.time() - start_time) < deadline - 60.0:
        greedy_time = min(20.0, deadline - (time.time() - start_time) - 60.0)
        if greedy_time > 1.0:
            greedy_result = _greedy_superperm(n, time_limit=greedy_time)
            if greedy_result is not None:
                greedy_len = len(greedy_result)
                if greedy_len < best_len:
                    best_word = greedy_result
                    best_len = greedy_len
    
    # Strategy 2: Try to optimize the ordering at the top level (classic approach)
    if n_perms <= 720 and (time.time() - start_time) < deadline - 30.0:
        time_limit_for_tsp = min(40.0, deadline - (time.time() - start_time) - 30.0)
        if time_limit_for_tsp > 1.0:
            best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
            
            # Build the word using the optimized order
            blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap(word, b)
                word += b[ov:]
            
            if len(word) < best_len:
                best_word = word
                best_len = len(word)
    
    # Final validation
    if (time.time() - start_time) < deadline - 5.0:
        # Verify the best word contains all permutations
        all_perms = set(itertools.permutations(range(1, n + 1)))
        seen = set()
        w = best_word
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        
        if len(seen) < len(all_perms):
            # Fallback to classic if validation fails
            best_word = classic_word
            best_len = classic_len
    
    if (time.time() - start_time) > deadline:
        return "".join(map(str, classic_word))
    
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
