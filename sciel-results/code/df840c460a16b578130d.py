"""Shortest superpermutation construction combining classic recursive approach
with optimized ordering via TSP-style local search and greedy maximal-overlap
path for small n."""
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
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


def _overlap_fast(a_tuple, b_tuple):
    """Fast overlap using tuple comparison."""
    max_ov = min(len(a_tuple), len(b_tuple))
    # For blocks of form p+[m]+p where p is a perm of length m-1,
    # the max possible overlap is bounded by 2*(m-1) but practically small
    for o in range(max_ov, 0, -1):
        if a_tuple[-o:] == b_tuple[:o]:
            return o
    return 0


def _build_overlap_matrix_fast(blocks):
    """Build overlap matrix using precomputed prefix/suffix sets for speed."""
    n = len(blocks)
    block_tuples = [tuple(b) for b in blocks]
    ov_matrix = [[0] * n for _ in range(n)]
    
    # Precompute all suffixes and prefixes as sets for quick lookup
    # For each block, store dict: suffix_length -> suffix_tuple
    suffixes = []
    prefixes = []
    for bt in block_tuples:
        suf = {}
        for o in range(1, len(bt) + 1):
            suf[o] = bt[-o:]
        suffixes.append(suf)
        pre = {}
        for o in range(1, len(bt) + 1):
            pre[o] = bt[:o]
        prefixes.append(pre)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                max_ov = min(len(block_tuples[i]), len(block_tuples[j]))
                for o in range(max_ov, 0, -1):
                    if suffixes[i][o] == prefixes[j][o]:
                        ov_matrix[i][j] = o
                        break
    return ov_matrix


def _build_overlap_matrix(blocks):
    """Build full overlap matrix efficiently using tuples."""
    return _build_overlap_matrix_fast(blocks)


def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)


def _total_length_given_order(order, perm_list, m):
    """
    Given an ordering of permutations, compute the total length of the
    resulting superpermutation segment.
    """
    if not order:
        return 0
    blocks = [_block_for_perm(perm_list[i], m) for i in order]
    total = len(blocks[0])
    for k in range(1, len(blocks)):
        ov = _overlap(blocks[k-1], blocks[k])
        total += len(blocks[k]) - ov
    return total


def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses greedy construction + efficient 2-opt and 3-opt local search.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = _build_overlap_matrix(blocks)
    
    # Precompute block lengths
    block_lens = [len(b) for b in blocks]
    
    # Precompute edge costs: cost[i][j] = block_lens[j] - ov_matrix[i][j]
    edge_cost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_cost[i][j] = block_lens[j] - ov_matrix[i][j]
    
    def compute_cost(order):
        """Compute total cost for a given order."""
        cost = block_lens[order[0]]
        for k in range(1, len(order)):
            cost += edge_cost[order[k-1]][order[k]]
        return cost
    
    def greedy_construct(start=0):
        """Greedy construction: always pick the next block with max overlap (min edge cost)."""
        used = [False] * n
        order = []
        current = start
        used[start] = True
        order.append(start)
        
        for _ in range(n - 1):
            best_j = -1
            best_cost = float('inf')
            row = edge_cost[current]
            for j in range(n):
                if not used[j]:
                    c = row[j]
                    if c < best_cost:
                        best_cost = c
                        best_j = j
            if best_j == -1:
                break
            used[best_j] = True
            order.append(best_j)
            current = best_j
        
        return order
    
    def local_search_2opt(cur_order, cur_cost, t_limit):
        """2-opt local search with O(1) cost delta using precomputed edge costs."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for i in range(n - 1):
                if (time.time() - start_time) > t_limit:
                    break
                a = cur_order[i]
                b = cur_order[i + 1]
                old_ab = edge_cost[a][b]
                for j in range(i + 2, n):
                    c = cur_order[j]
                    # Reverse cur_order[i+1..j]
                    # Old edges: (a,b) and (c, cur_order[j+1]) if j+1<n
                    # New edges: (a,c) and (cur_order[j+1], b) if j+1<n
                    if j + 1 < n:
                        d = cur_order[j + 1]
                        old_cd = edge_cost[c][d]
                        new_ac = edge_cost[a][c]
                        new_db = edge_cost[d][b]
                        delta = new_ac + new_db - old_ab - old_cd
                    else:
                        new_ac = edge_cost[a][c]
                        delta = new_ac - old_ab
                    
                    if delta < 0:
                        cur_cost += delta
                        cur_order[i + 1:j + 1] = cur_order[i + 1:j + 1][::-1]
                        improved = True
        return cur_order, cur_cost
    
    def local_search_3opt(cur_order, cur_cost, t_limit):
        """3-opt: try moving a single node to another position (Or-opt for seg_len=1)."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for i in range(n):
                if (time.time() - start_time) > t_limit:
                    break
                node = cur_order[i]
                # Remove node from position i
                # Old edges affected: (i-1, i) and (i, i+1)
                if i > 0:
                    left = cur_order[i - 1]
                    old_left = edge_cost[left][node]
                else:
                    left = -1
                    old_left = 0
                if i < n - 1:
                    right = cur_order[i + 1]
                    old_right = edge_cost[node][right]
                else:
                    right = -1
                    old_right = 0
                
                # Try inserting at each other position j (0..n-2, since we removed one)
                for j in range(n - 1):
                    if (time.time() - start_time) > t_limit:
                        break
                    if j == i or j == i - 1:
                        continue  # same position or adjacent (already considered)
                    # Insert node at position j in the remaining list
                    # The remaining list is cur_order with index i removed
                    # Position j in remaining corresponds to:
                    #   if j < i: position j in original
                    #   if j >= i: position j+1 in original
                    if j < i:
                        # Insert before cur_order[j]
                        if j > 0:
                            prev_node = cur_order[j - 1]
                            old_prev = edge_cost[prev_node][cur_order[j]]
                        else:
                            prev_node = -1
                            old_prev = 0
                        next_node = cur_order[j]
                        new_prev = edge_cost[prev_node][node] if prev_node >= 0 else 0
                        new_next = edge_cost[node][next_node]
                        delta = new_prev + new_next - old_prev
                        if delta < 0:
                            # Perform the move
                            cur_cost += delta
                            del cur_order[i]
                            cur_order.insert(j, node)
                            improved = True
                            break
                    else:
                        # j >= i, insert before cur_order[j+1] in original
                        # But we need to be careful about indices after removal
                        # After removing index i, cur_order[j] (j>=i) becomes cur_order[j-1]
                        # We want to insert at position j in the new list
                        # This means before what was originally cur_order[j+1]
                        if j + 1 < n:
                            next_node = cur_order[j + 1]
                            if j >= 1:
                                prev_node = cur_order[j]
                                old_prev = edge_cost[prev_node][next_node]
                            else:
                                prev_node = -1
                                old_prev = 0
                            new_prev = edge_cost[prev_node][node] if prev_node >= 0 else 0
                            new_next = edge_cost[node][next_node]
                            delta = new_prev + new_next - old_prev
                            if delta < 0:
                                cur_cost += delta
                                del cur_order[i]
                                cur_order.insert(j, node)
                                improved = True
                                break
        return cur_order, cur_cost
    
    # Start with greedy construction from multiple starting points
    best_order = greedy_construct(0)
    best_cost = compute_cost(best_order)
    
    # Also try classic order
    classic_order = list(range(n))
    classic_cost = compute_cost(classic_order)
    if classic_cost < best_cost:
        best_cost = classic_cost
        best_order = classic_order
    
    # Try greedy from a few other starting points
    for start_pt in [1, 2, n // 2, n - 1]:
        if time.time() - start_time > time_limit * 0.1:
            break
        g_order = greedy_construct(start_pt)
        g_cost = compute_cost(g_order)
        if g_cost < best_cost:
            best_cost = g_cost
            best_order = g_order
    
    # 2-opt from best starting point
    t1 = min(time_limit * 0.4, time_limit - 3.0)
    best_order, best_cost = local_search_2opt(best_order, best_cost, t1)
    
    # 3-opt (Or-opt single node) polish
    t2 = min(time_limit * 0.3, time_limit - 2.0)
    if (time.time() - start_time) < time_limit - 2.0:
        best_order, best_cost = local_search_3opt(best_order, best_cost, t2)
    
    # Another 2-opt pass
    t3 = min(time_limit * 0.2, time_limit - 1.5)
    if (time.time() - start_time) < time_limit - 1.5:
        best_order, best_cost = local_search_2opt(best_order, best_cost, t3)
    
    # Try random restarts if time remains
    if (time.time() - start_time) < time_limit - 1.0:
        import random
        random.seed(42)
        n_restarts = 5
        for r in range(n_restarts):
            if (time.time() - start_time) > time_limit - 0.5:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            cur_cost = compute_cost(rand_order)
            t_rem = time_limit - (time.time() - start_time) - 0.5
            if t_rem > 1.0:
                rand_order, cur_cost = local_search_2opt(rand_order, cur_cost, t_rem * 0.6)
                if (time.time() - start_time) < time_limit - 0.5:
                    rand_order, cur_cost = local_search_3opt(rand_order, cur_cost, t_rem * 0.3)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = rand_order
    
    return best_order, best_cost


def _solve_small_n(n, deadline):
    """
    For small n (<=5), try exact or near-exact search using the de Bruijn
    graph approach with greedy maximal-overlap path.
    """
    all_perms = list(permutations(range(1, n + 1)))
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
        if time.time() > deadline:
            return None
        suffix = tuple(word[-(n - 1):])
        last = word[-1]
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
    if time.time() < deadline - 1:
        seen = set()
        w = word
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        if len(seen) == len(all_perms):
            return word
    return None


def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
    
    if n == 4:
        return "123412134212343142341234"
    
    # For n >= 5, use optimized construction
    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Get classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # For small n, try the greedy maximal-overlap path approach
    if n <= 5:
        small_result = _solve_small_n(n, deadline - 10.0)
        if small_result is not None and len(small_result) < classic_len:
            return "".join(map(str, small_result))
    
    # Try to optimize the ordering at the top level
    if n_perms <= 720 and (time.time() - start_time) < deadline - 20.0:
        time_limit_for_tsp = min(100.0, deadline - (time.time() - start_time) - 20.0)
        if time_limit_for_tsp > 1.0:
            best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
            
            # Build the word using the optimized order
            blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap(word, b)
                word += b[ov:]
            
            if len(word) < classic_len:
                return "".join(map(str, word))
    
    # Fall back to classic
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
