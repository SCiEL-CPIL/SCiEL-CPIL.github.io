import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # Build the standard (Graham-Lagarias) superpermutation word for m symbols,
    # but at each level we extract the list of (m-1)-permutations in order of
    # appearance and then optimize their ordering via overlap + local search.
    
    def build_optimized(m):
        """Build shortest known superpermutation for m symbols via level-by-level optimization."""
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        # Recursively build optimized word for m-1
        word_prev = build_optimized(m - 1)
        
        # Extract (m-1)-permutations from word_prev in order of first appearance
        perms = []
        seen = set()
        for i in range(len(word_prev) - (m - 1) + 1):
            window = tuple(word_prev[i:i + (m - 1)])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)  # should be (m-1)!
        
        # Compute overlap matrix: overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        max_k = m - 2
        if max_k <= 0:
            overlap = np.zeros((num_perms, num_perms), dtype=np.int32)
        else:
            overlap = np.zeros((num_perms, num_perms), dtype=np.int32)
            for i in range(num_perms):
                pi = perms[i]
                for j in range(num_perms):
                    pj = perms[j]
                    for k in range(min(max_k, m - 1), -1, -1):
                        if k == 0:
                            break
                        if pi[-k:] == pj[:k]:
                            overlap[i][j] = k
                            break
        
        block_len = 2 * (m - 1) + 1  # length of each block: p + [m] + p
        
        def calc_total_length(order):
            total = block_len
            for idx in range(len(order) - 1):
                total += block_len - overlap[order[idx], order[idx + 1]]
            return total
        
        # Greedy initial ordering
        def greedy_order(start_idx=0):
            unvisited = set(range(num_perms))
            order = [start_idx]
            unvisited.remove(start_idx)
            current = start_idx
            while unvisited:
                best_next = -1
                best_ov = -1
                for v in unvisited:
                    ov = overlap[current, v]
                    if ov > best_ov or (ov == best_ov and v < best_next):
                        best_ov = ov
                        best_next = v
                order.append(best_next)
                unvisited.remove(best_next)
                current = best_next
            return order
        
        best_order = greedy_order(0)
        best_len = calc_total_length(best_order)
        
        # Try a few random restarts
        n_restarts = min(10, max(1, num_perms // 50))
        for _ in range(n_restarts):
            if time.time() > deadline - 20.0:
                break
            candidate = greedy_order(random.randint(0, num_perms - 1))
            cand_len = calc_total_length(candidate)
            if cand_len < best_len:
                best_len = cand_len
                best_order = candidate
        
        # Local search: adjacent swaps
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline - 10.0:
                    break
                u = best_order[i]
                v = best_order[i + 1]
                delta = 0
                if i > 0:
                    prev = best_order[i - 1]
                    delta += (block_len - overlap[prev, v]) - (block_len - overlap[prev, u])
                if i + 1 < num_perms - 1:
                    nxt = best_order[i + 2]
                    delta += (block_len - overlap[u, nxt]) - (block_len - overlap[v, nxt])
                delta += (block_len - overlap[v, u]) - (block_len - overlap[u, v])
                
                if delta < 0:
                    best_order[i], best_order[i + 1] = v, u
                    best_len += delta
                    improved = True
        
        # 2-opt: reverse segments (only for smaller permutation counts to save time)
        if num_perms <= 1000:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(num_perms - 1):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 2, num_perms):
                        if time.time() > deadline - 5.0:
                            break
                        delta = 0
                        if i > 0:
                            prev = best_order[i - 1]
                            delta += (block_len - overlap[prev, best_order[j]]) - (block_len - overlap[prev, best_order[i]])
                        if j < num_perms - 1:
                            nxt = best_order[j + 1]
                            delta += (block_len - overlap[best_order[i], nxt]) - (block_len - overlap[best_order[j], nxt])
                        for k in range(i, j):
                            delta += (block_len - overlap[best_order[k + 1], best_order[k]]) - (block_len - overlap[best_order[k], best_order[k + 1]])
                        
                        if delta < 0:
                            best_order[i:j + 1] = best_order[i:j + 1][::-1]
                            best_len += delta
                            improved = True
                            break
        
        # Build the word for m symbols using the optimized ordering
        final_word = []
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of final_word
            ov = 0
            max_ov = min(len(final_word), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if final_word[len(final_word) - o:] == block[:o]:
                    ov = o
                    break
            final_word += block[ov:]
        
        return final_word
    
    word = build_optimized(n)
    return "".join(map(str, word))
