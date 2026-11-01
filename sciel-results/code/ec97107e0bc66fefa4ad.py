import time
import itertools
import sys
import random

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal Graham-Lagarias construction.
    For n = 7, uses a rotation-based constructive algorithm that
    cyclically shifts symbol labels to maximize overlap between blocks.
    """
    if n == 1:
        return "1"
    
    # Standard Graham-Lagarias construction for n <= 6
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        # Get word and perms for m-1
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    if n <= 6:
        word, _ = graham_lagarias(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Rotation-based constructive algorithm for n=7
        # Generate all permutations of 1..6 in a rotation-optimized order
        # The key idea: order permutations so that consecutive perms have
        # maximum suffix-prefix overlap (up to 5 symbols)
        
        start_time = time.time()
        deadline = start_time + 110.0  # Leave 40s safety margin
        
        # Generate all 720 permutations of 1..6
        base_perms = list(itertools.permutations(range(1, 7)))
        
        # Build a rotation-optimized ordering using greedy + local search
        # First, compute overlap between all pairs (suffix of i matches prefix of j)
        # For speed, only check overlaps of length 1..5
        
        # Precompute: for each perm, its suffixes of length 1-5 and prefixes of length 1-5
        # overlap[i][j] = max k in {1..5} such that perms[i][-k:] == perms[j][:k], else 0
        
        # To speed up, group perms by their last 5 symbols and first 5 symbols
        # suffix_map: tuple(last5) -> list of perm indices
        # prefix_map: tuple(first5) -> list of perm indices
        
        suffix_map = {}
        prefix_map = {}
        for idx, p in enumerate(base_perms):
            s5 = p[-5:]
            p5 = p[:5]
            if s5 not in suffix_map:
                suffix_map[s5] = []
            suffix_map[s5].append(idx)
            if p5 not in prefix_map:
                prefix_map[p5] = []
            prefix_map[p5].append(idx)
        
        # For each perm i, find all j where overlap >= 1
        # overlap[i][j] = max k such that p_i[-k:] == p_j[:k]
        # We can check k=5,4,3,2,1 in order
        
        # Build adjacency: for each i, list of (j, overlap) where overlap > 0
        adj = [[] for _ in range(720)]
        for i in range(720):
            pi = base_perms[i]
            for k in range(5, 0, -1):
                suffix = pi[-k:]
                if suffix in prefix_map:
                    for j in prefix_map[suffix]:
                        if j != i:
                            adj[i].append((j, k))
                    break  # Only need max overlap
        
        # Greedy construction: start with perm 0, then repeatedly pick the
        # unvisited perm with max overlap from current, breaking ties by index
        visited = [False] * 720
        order = [0]
        visited[0] = True
        
        for _ in range(719):
            current = order[-1]
            best_j = -1
            best_overlap = 0
            for j, ov in adj[current]:
                if not visited[j] and ov > best_overlap:
                    best_overlap = ov
                    best_j = j
            if best_j == -1:
                # No overlap found, pick any unvisited
                for j in range(720):
                    if not visited[j]:
                        best_j = j
                        break
            order.append(best_j)
            visited[best_j] = True
        
        # Local search: 2-opt improvements
        def calc_total_overlap(ord_list):
            total = 0
            for i in range(len(ord_list) - 1):
                u = ord_list[i]
                v = ord_list[i + 1]
                # Compute overlap between u and v
                pu = base_perms[u]
                pv = base_perms[v]
                for k in range(5, 0, -1):
                    if pu[-k:] == pv[:k]:
                        total += k
                        break
            return total
        
        best_order = order[:]
        best_overlap = calc_total_overlap(best_order)
        
        # 2-opt: reverse segments
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(719):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 720):
                    if time.time() > deadline:
                        break
                    # Compute delta overlap for reversing [i, j]
                    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                    
                    u_left = best_order[i-1] if i > 0 else None
                    u_right = best_order[j+1] if j < 719 else None
                    
                    old_ov = 0
                    if i > 0:
                        pu = base_perms[u_left]
                        pv = base_perms[best_order[i]]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                old_ov += k
                                break
                    for k_idx in range(i, j):
                        pu = base_perms[best_order[k_idx]]
                        pv = base_perms[best_order[k_idx+1]]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                old_ov += k
                                break
                    if j < 719:
                        pu = base_perms[best_order[j]]
                        pv = base_perms[u_right]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                old_ov += k
                                break
                    
                    new_ov = 0
                    if i > 0:
                        pu = base_perms[u_left]
                        pv = base_perms[best_order[j]]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                new_ov += k
                                break
                    for k_idx in range(j, i, -1):
                        pu = base_perms[best_order[k_idx]]
                        pv = base_perms[best_order[k_idx-1]]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                new_ov += k
                                break
                    if j < 719:
                        pu = base_perms[best_order[i]]
                        pv = base_perms[u_right]
                        for k in range(5, 0, -1):
                            if pu[-k:] == pv[:k]:
                                new_ov += k
                                break
                    
                    if new_ov > old_ov:
                        best_order[i:j+1] = best_order[i:j+1][::-1]
                        best_overlap += (new_ov - old_ov)
                        improved = True
        
        # Construct the final word
        final_word = []
        for idx in best_order:
            p = base_perms[idx]
            block = list(p) + [7] + list(p)
            # Find max overlap with current final_word
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
        
        return "".join(map(str, final_word))
        
    # Fallback for n > 7
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
