"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    all_syms = set(range(1, n+1))
    
    def get_missing(node):
        node_set = set(node)
        for s in all_syms:
            if s not in node_set:
                return s
        return None
    
    def get_next(node):
        s = get_missing(node)
        return node[1:] + (s,)
    
    def node_to_perm(node):
        s = get_missing(node)
        return node + (s,)
    
    all_nodes = list(permutations(range(1, n+1), n-1))
    visited = set()
    cycles = []
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
    
    cycle_perms = []
    for cycle in cycles:
        perms = [node_to_perm(node) for node in cycle]
        cycle_perms.append(perms)
    
    m = len(cycles)
    if m == 1:
        word = []
        perms = cycle_perms[0]
        for perm in perms:
            if not word:
                word.extend(perm)
            else:
                word.extend(perm[n-1:])
        return "".join(map(str, word))

    overlaps = [[0]*m for _ in range(m)]
    for i in range(m):
        end_perm = cycle_perms[i][-1]
        for j in range(m):
            if i == j:
                continue
            start_perm = cycle_perms[j][0]
            max_ovl = 0
            for k in range(1, n):
                if end_perm[n-k:] == start_perm[:k]:
                    max_ovl = k
            overlaps[i][j] = max_ovl
    
    used = [False] * m
    order = []
    
    best_start = 0
    best_score = -1
    for i in range(m):
        score = sum(overlaps[i][j] for j in range(m) if j != i)
        if score > best_score:
            best_score = score
            best_start = i
    
    order.append(best_start)
    used[best_start] = True
    current = best_start
    
    for _ in range(m - 1):
        best_next = -1
        best_ovl = -1
        for j in range(m):
            if not used[j]:
                if overlaps[current][j] > best_ovl:
                    best_ovl = overlaps[current][j]
                    best_next = j
        if best_next == -1:
            for j in range(m):
                if not used[j]:
                    best_next = j
                    break
        order.append(best_next)
        used[best_next] = True
        current = best_next
        
    def calc_score(ord_list):
        score = 0
        for k in range(len(ord_list) - 1):
            score += overlaps[ord_list[k]][ord_list[k+1]]
        return score

    current_score = calc_score(order)
    
    if m <= 30:
        improved = True
        max_iters = 500
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            best_gain = 0
            best_i, best_j = -1, -1
            for i in range(m - 1):
                for j in range(i + 1, m):
                    old_link_1 = overlaps[order[i-1]][order[i]] if i > 0 else 0
                    old_link_2 = overlaps[order[j]][order[j+1]] if j < m-1 else 0
                    
                    new_link_1 = overlaps[order[i-1]][order[j]] if i > 0 else 0
                    new_link_2 = overlaps[order[i]][order[j+1]] if j < m-1 else 0
                    
                    gain = (new_link_1 + new_link_2) - (old_link_1 + old_link_2)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
                        
            if best_i != -1:
                order[best_i:best_j+1] = order[best_i:best_j+1][::-1]
                current_score += best_gain
                improved = True

    word = []
    for idx, ci in enumerate(order):
        perms = cycle_perms[ci]
        for p_idx, perm in enumerate(perms):
            if not word:
                word.extend(perm)
            else:
                if p_idx == 0:
                    max_ovl = 0
                    for k in range(1, n):
                        if k > len(word):
                            break
                        if tuple(word[-k:]) == perm[:k]:
                            max_ovl = k
                    word.extend(perm[max_ovl:])
                else:
                    word.extend(perm[n-1:])

    return "".join(map(str, word))

# EVOLVE-BLOCK-END
