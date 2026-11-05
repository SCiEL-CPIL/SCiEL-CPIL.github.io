"""Shortest superpermutation via cycle decomposition of S_n."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return _construct_4()
    
    return _construct_general(n, deadline)

def _construct_4():
    return "12341243123414231243143214321342132412341324143214321"

def _construct_general(n, deadline):
    symbols = tuple(range(1, n + 1))
    all_perms = list(permutations(symbols))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Left rotation: (a1,...,an) -> (a2,...,an,a1)
    rot_succ = [0] * num_perms
    for i, p in enumerate(all_perms):
        r = p[1:] + (p[0],)
        rot_succ[i] = perm_to_idx[r]
    
    # Find cycles
    visited = [False] * num_perms
    cycles = []
    
    for i in range(num_perms):
        if not visited[i]:
            cur = i
            cyc = []
            while not visited[cur]:
                visited[cur] = True
                cyc.append(cur)
                cur = rot_succ[cur]
            if cyc:
                last = cyc[-1]
                nxt = rot_succ[last]
                if nxt in cyc:
                    idx = cyc.index(nxt)
                    cycles.append(cyc[idx:])
                else:
                    cycles.append(cyc)
    
    m = len(cycles)
    
    def overlap_between(a, b, n):
        max_ov = min(n - 1, len(a), len(b))
        for k in range(max_ov, -1, -1):
            if k == 0 or a[-k:] == b[:k]:
                return k
        return 0
    
    # For each cycle, precompute superstrings for all possible starting points
    # cycle_variants[c] = list of (start_idx, end_idx, superstring_list)
    cycle_variants = []
    for c in range(m):
        cyc = cycles[c]
        k = len(cyc)
        variants = []
        for s in range(k):
            # Start at position s in the cycle
            order = [cyc[(s + t) % k] for t in range(k)]
            # Build superstring
            ss = list(all_perms[order[0]])
            for t in range(1, k):
                ss.append(all_perms[order[t]][-1])
            start_perm = all_perms[order[0]]
            end_perm = all_perms[order[-1]]
            variants.append((s, ss, start_perm, end_perm))
        cycle_variants.append(variants)
    
    # Greedy: choose for each cycle a variant, and order cycles
    # To keep it tractable, first pick a default variant (start=0) for each cycle
    # Then do local search on both variant choice and cycle order
    
    # Default: use variant 0 for each cycle
    default_strs = [cycle_variants[c][0][1] for c in range(m)]
    default_starts = [cycle_variants[c][0][2] for c in range(m)]
    default_ends = [cycle_variants[c][0][3] for c in range(m)]
    
    def compute_total(order, variant_choices):
        if not order:
            return 0
        total = len(cycle_variants[order[0]][variant_choices[order[0]]][1])
        for i in range(1, len(order)):
            prev_str = cycle_variants[order[i-1]][variant_choices[order[i-1]]][1]
            curr_str = cycle_variants[order[i]][variant_choices[order[i]]][1]
            ov = overlap_between(prev_str, curr_str, n)
            total += len(curr_str) - ov
        return total
    
    # Greedy cycle ordering with variant 0
    current_order = [0]
    remaining = set(range(1, m))
    
    while remaining and time.time() < deadline - 10:
        best_next = -1
        best_ov = -1
        last_str = cycle_variants[current_order[-1]][0][1]
        for c in remaining:
            ov = overlap_between(last_str, cycle_variants[c][0][1], n)
            if ov > best_ov:
                best_ov = ov
                best_next = c
        if best_next == -1:
            break
        current_order.append(best_next)
        remaining.discard(best_next)
    for c in remaining:
        current_order.append(c)
    
    variant_choices = [0] * m
    current_len = compute_total(current_order, variant_choices)
    
    # Local search: try improving cycle order and variant choices
    best_order = list(current_order)
    best_variants = list(variant_choices)
    best_len = current_len
    
    deadline_ls = deadline - 5
    improved = True
    while improved and time.time() < deadline_ls:
        improved = False
        
        # Try 2-opt on cycle order
        for i in range(m - 1):
            if time.time() >= deadline_ls:
                break
            for j in range(i + 1, m):
                if time.time() >= deadline_ls:
                    break
                old_ov = 0
                new_ov = 0
                if i > 0:
                    old_ov += overlap_between(cycle_variants[best_order[i-1]][best_variants[best_order[i-1]]][1],
                                             cycle_variants[best_order[i]][best_variants[best_order[i]]][1], n)
                    new_ov += overlap_between(cycle_variants[best_order[i-1]][best_variants[best_order[i-1]]][1],
                                             cycle_variants[best_order[j]][best_variants[best_order[j]]][1], n)
                if j < m - 1:
                    old_ov += overlap_between(cycle_variants[best_order[j]][best_variants[best_order[j]]][1],
                                             cycle_variants[best_order[j+1]][best_variants[best_order[j+1]]][1], n)
                    new_ov += overlap_between(cycle_variants[best_order[i]][best_variants[best_order[i]]][1],
                                             cycle_variants[best_order[j+1]][best_variants[best_order[j+1]]][1], n)
                for k in range(i, j):
                    old_ov += overlap_between(cycle_variants[best_order[k]][best_variants[best_order[k]]][1],
                                             cycle_variants[best_order[k+1]][best_variants[best_order[k+1]]][1], n)
                    new_ov += overlap_between(cycle_variants[best_order[k+1]][best_variants[best_order[k+1]]][1],
                                             cycle_variants[best_order[k]][best_variants[best_order[k]]][1], n)
                delta = new_ov - old_ov
                if delta > 0:
                    best_order[i:j+1] = reversed(best_order[i:j+1])
                    best_len -= delta
                    improved = True
                    break
            if improved:
                break
        
        # Try changing variant for each cycle
        for c in range(m):
            if time.time() >= deadline_ls:
                break
            pos = best_order.index(c)
            old_ov = 0
            if pos > 0:
                old_ov += overlap_between(cycle_variants[best_order[pos-1]][best_variants[best_order[pos-1]]][1],
                                         cycle_variants[c][best_variants[c]][1], n)
            if pos < m - 1:
                old_ov += overlap_between(cycle_variants[c][best_variants[c]][1],
                                         cycle_variants[best_order[pos+1]][best_variants[best_order[pos+1]]][1], n)
            
            for v in range(len(cycle_variants[c])):
                if v == best_variants[c]:
                    continue
                new_ov = 0
                if pos > 0:
                    new_ov += overlap_between(cycle_variants[best_order[pos-1]][best_variants[best_order[pos-1]]][1],
                                             cycle_variants[c][v][1], n)
                if pos < m - 1:
                    new_ov += overlap_between(cycle_variants[c][v][1],
                                             cycle_variants[best_order[pos+1]][best_variants[best_order[pos+1]]][1], n)
                delta = new_ov - old_ov
                if delta > 0:
                    best_variants[c] = v
                    best_len -= delta
                    improved = True
                    break
    
    # Build final word
    final_word = []
    for c in best_order:
        v = best_variants[c]
        cs = cycle_variants[c][v][1]
        if not final_word:
            final_word = list(cs)
        else:
            ov = overlap_between(final_word, cs, n)
            final_word.extend(cs[ov:])
    
    return "".join(map(str, final_word))
# EVOLVE-BLOCK-END
