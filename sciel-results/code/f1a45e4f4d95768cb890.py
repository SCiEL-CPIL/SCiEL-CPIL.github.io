import itertools
import time
import random
import sys
from collections import deque

def construct(n):
    # For n <= 4, the standard recursive construction is optimal.
    if n <= 4:
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
                # Find max overlap
                max_k = min(len(new), len(block))
                overlap = 0
                for o in range(max_k, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n = 5, 6, 7, we use a specialized optimization.
    # The problem is equivalent to finding a Hamiltonian path in the de Bruijn graph of order n-1
    # with edge weights based on overlap.
    # Specifically, we want to order the permutations of 1..n-1 to maximize the total overlap
    # between consecutive blocks (p 7 p).
    
    if n in (5, 6, 7):
        start_time = time.time()
        deadline = start_time + 115.0 # Leave 5s buffer for construction and safety
        
        # Get the "standard" order of permutations of 1..n-1
        # This is the order they appear in the standard superpermutation for n-1
        def get_standard_perms(m):
            if m == 1:
                return [(1,)]
            if m == 2:
                return [(1, 2), (2, 1)]
            
            prev_perms = get_standard_perms(m-1)
            new_perms = []
            seen = set()
            
            # Simulate the word construction to find the order
            # But we only need the order of the perms of length m-1.
            # The standard construction inserts m into each perm of m-1.
            # The order of perms of m-1 in the word for m is the same as their order in the word for m-1?
            # No. The word for m is built from blocks p m p.
            # The windows of length m-1 in the word for m are:
            # 1. Suffix of p m p (length m-1) -> (p[1:], m)
            # 2. Overlap region?
            
            # Let's just extract from the word.
            word = [1]
            for k in range(2, m):
                p_list, s_list = [], set()
                for i in range(len(word) - k + 2):
                    w = tuple(word[i:i + k - 1])
                    if len(set(w)) == k - 1 and w not in s_list:
                        s_list.add(w)
                        p_list.append(w)
                
                new_word = []
                for p in p_list:
                    block = list(p) + [k] + list(p)
                    max_ov = min(len(new_word), len(block))
                    ov = 0
                    for o in range(max_ov, -1, -1):
                        if new_word[len(new_word) - o:] == block[:o]:
                            ov = o
                            break
                    new_word += block[ov:]
                word = new_word
            
            # Now extract perms of length m-1 from word
            p_list, s_list = [], set()
            for i in range(len(word) - (m-1) + 1):
                w = tuple(word[i:i + m-1])
                if len(set(w)) == m-1 and w not in s_list:
                    s_list.add(w)
                    p_list.append(w)
            return p_list

        perms = get_standard_perms(n-1)
        N = len(perms)
        
        # Precompute overlap matrix
        # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # Max k is n-2
        max_k = n - 2
        overlap_matrix = [[0] * N for _ in range(N)]
        
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_matrix[i][j] = 0 # Distinct perms, but technically if same, overlap is n-1? No, we visit each once.
                    continue
                pj = perms[j]
                # Check overlaps
                for k in range(max_k, -1, -1):
                    if pi[-k:] == pj[:k]:
                        overlap_matrix[i][j] = k
                        break
        
        # Cost matrix: cost[i][j] = (2*(n-1)+1) - overlap[i][j]
        # We want to minimize total cost.
        # Total length = first_block_len + sum(cost[i][j])
        # first_block_len = 2*(n-1) + 1
        
        base_len = 2 * (n - 1) + 1
        
        def calc_total_len(order):
            total = base_len
            for i in range(len(order) - 1):
                u = order[i]
                v = order[i + 1]
                total += base_len - overlap_matrix[u][v]
            return total
        
        # Initial order: standard order
        best_order = list(range(N))
        best_len = calc_total_len(best_order)
        
        # Local Search: 2-opt
        # We can try to improve the order by reversing segments.
        # This is a TSP-like problem.
        
        improved = True
        while improved:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    # Reverse best_order[i+1..j]
                    # Check if it improves total length
                    # Old edges: (best[i], best[i+1]) and (best[j], best[j+1]) (if j < N-1)
                    # New edges: (best[i], best[j]) and (best[i+1], best[j+1]) (if j < N-1)
                    
                    A = best_order[i]
                    B = best_order[i+1]
                    D = best_order[j]
                    
                    old_cost = base_len - overlap_matrix[A][B]
                    if j < N - 1:
                        E = best_order[j+1]
                        old_cost += base_len - overlap_matrix[D][E]
                        
                        new_cost = base_len - overlap_matrix[A][D]
                        new_cost += base_len - overlap_matrix[B][E]
                        
                        if new_cost < old_cost:
                            best_order[i+1:j+1] = reversed(best_order[i+1:j+1])
                            improved = True
                    else:
                        # D is last
                        # Old: ... A, B ... D
                        # New: ... A, D ... B
                        # This changes the whole suffix.
                        # Cost change:
                        # Old: C(A,B) + ... + C(prev_D, D)
                        # New: C(A,D) + ... + C(prev_D, B)
                        # This is harder to calculate locally without summing.
                        # For simplicity, skip j=N-1 in 2-opt or handle carefully.
                        # Given N is small for n=5,6, we can afford to recalc? No, N=720 for n=6, N=5040 for n=7.
                        # Recalc is O(N). 2-opt is O(N^2). Total O(N^3) if we recalc every time.
                        # For N=5040, N^3 is ~1e11. Too slow.
                        # We need local delta.
                        # For j=N-1, the "path" ends at D.
                        # Old path: ... A -> B -> ... -> D
                        # New path: ... A -> D -> ... -> B
                        # The cost of the segment B...D is reversed.
                        # Cost(B...D) = sum_{k=i+1}^{j-1} C(best[k], best[k+1])
                        # Since the graph is directed, reversing changes edges (u,v) to (v,u).
                        # C(u,v) != C(v,u) generally.
                        # So we must sum the costs.
                        # This makes 2-opt slow if we have to sum.
                        # However, for n=7, N=5040.
                        # We can precompute prefix sums of costs?
                        # But the costs change as we swap.
                        
                        # Alternative: Simulated Annealing or Greedy with backtracking.
                        # Or just use the standard order if 2-opt is too slow.
                        # The standard order for n=7 is 5913.
                        # The best known is 5905.
                        # 2-opt might get us closer.
                        
                        # Let's try a simpler local move: Swap adjacent elements.
                        # This is O(1) cost update.
                        pass
        
        # Let's implement a faster local search: Or-opt (move a segment to another position) or just adjacent swaps.
        # Adjacent swaps:
        # Swap best_order[k] and best_order[k+1].
        # Old: ... A, B, C ... -> Cost(A,B) + Cost(B,C)
        # New: ... A, C, B ... -> Cost(A,C) + Cost(C,B)
        
        improved = True
        while improved:
            improved = False
            for k in range(N - 1):
                if time.time() > deadline:
                    break
                A = best_order[k-1] if k > 0 else None
                B = best_order[k]
                C = best_order[k+1]
                D = best_order[k+2] if k+2 < N else None
                
                old_cost = 0
                if A is not None:
                    old_cost += base_len - overlap_matrix[A][B]
                old_cost += base_len - overlap_matrix[B][C]
                if D is not None:
                    old_cost += base_len - overlap_matrix[C][D]
                
                new_cost = 0
                if A is not None:
                    new_cost += base_len - overlap_matrix[A][C]
                new_cost += base_len - overlap_matrix[C][B]
                if D is not None:
                    new_cost += base_len - overlap_matrix[B][D]
                
                if new_cost < old_cost:
                    best_order[k], best_order[k+1] = best_order[k+1], best_order[k]
                    improved = True
        
        # Construct the word
        current_str = ""
        for idx in best_order:
            p = perms[idx]
            block = "".join(map(str, p)) + str(n) + "".join(map(str, p))
            if not current_str:
                current_str = block
            else:
                # Find max overlap
                max_k = 0
                # Max overlap is n-1 (suffix of prev block, prefix of new block)
                # Prev block ends with p_prev. New block starts with p_new.
                # Overlap is max k such that p_prev[-k:] == p_new[:k]
                # We already have this in overlap_matrix, but we need to align with the actual string.
                # Actually, the overlap is exactly overlap_matrix[prev_idx][idx].
                # But we need to know the previous index.
                pass
        
        # Re-implement construction to be safe
        # We need to keep track of the last block's suffix to find overlap.
        # Or just use the overlap_matrix values.
        
        word_parts = []
        prev_idx = None
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            
            if prev_idx is None:
                word_parts.append(block)
            else:
                ov = overlap_matrix[prev_idx][idx]
                # Append block[ov:]
                word_parts.append(block[ov:])
            
            prev_idx = idx
        
        # Flatten
        final_word = []
        for part in word_parts:
            final_word.extend(part)
        
        return "".join(map(str, final_word))

    # Fallback for other n
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
            max_k = min(len(new), len(block))
            overlap = 0
            for o in range(max_k, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
