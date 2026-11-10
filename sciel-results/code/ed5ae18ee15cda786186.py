"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
import math
from collections import Counter

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin within 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute all K4s and their edge indices
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            idx_map[(i,j)] = idx
            idx += 1
    
    # Precompute K4 structure
    k4_list = list(itertools.combinations(range(n), 4))
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    
    for k4_idx, v in enumerate(k4_list):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i,j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(k4_idx)
        k4_edges.append(e_indices)
    
    num_k4 = len(k4_list)
    
    def count_violations(colors):
        """Count number of K4s with < 5 distinct colours."""
        violations = 0
        for e_indices in k4_edges:
            col_set = set()
            for ei in e_indices:
                col_set.add(colors[ei])
            if len(col_set) < 5:
                violations += 1
        return violations
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    def num_colours_used(colors):
        return len(set(colors))
    
    # Base solution: (i+j) mod 17
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors.copy()
    best_num = num_colours_used(base_colors)  # 17
    
    if not is_valid(base_colors):
        raise ValueError("Base solution invalid")
    
    def local_search_fast(colors, max_time):
        """
        Fast local search: minimize violations*1000 + num_colours.
        Uses direct recomputation of affected K4s for delta evaluation.
        """
        cur = colors.copy()
        cur_viol = count_violations(cur)
        cur_num = len(set(cur))
        best = cur.copy()
        best_num = cur_num
        cur_palette = list(set(cur))
        
        temp = 2.0
        cooling = 0.998
        min_temp = 0.05
        
        while time.time() < max_time and temp > min_temp:
            ei = random.randint(0, num_edges - 1)
            old_col = cur[ei]
            
            if random.random() < 0.7:
                new_col = random.choice(cur_palette)
            else:
                new_col = random.randint(0, max(cur_num - 1, 10))
            
            if new_col == old_col:
                continue
            
            # Evaluate delta by checking affected K4s in old and new state
            # New state violations for affected K4s
            cur[ei] = new_col
            new_affected_viol = 0
            for k4i in edge_to_k4[ei]:
                e_indices = k4_edges[k4i]
                cs = set()
                for ej in e_indices:
                    cs.add(cur[ej])
                if len(cs) < 5:
                    new_affected_viol += 1
            
            # Old state violations for affected K4s
            cur[ei] = old_col
            old_affected_viol = 0
            for k4i in edge_to_k4[ei]:
                e_indices = k4_edges[k4i]
                cs = set()
                for ej in e_indices:
                    cs.add(cur[ej])
                if len(cs) < 5:
                    old_affected_viol += 1
            
            delta_viol = new_affected_viol - old_affected_viol
            new_num = len(set(cur))
            delta_num = new_num - cur_num
            
            old_cost = cur_viol * 1000 + cur_num
            new_cost = (cur_viol + delta_viol) * 1000 + new_num
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0:
                cur[ei] = new_col
                cur_viol += delta_viol
                cur_num = new_num
                cur_palette = list(set(cur))
                if cur_viol == 0 and cur_num < best_num:
                    best = cur.copy()
                    best_num = cur_num
            else:
                if random.random() < math.exp(-delta_cost / temp):
                    cur[ei] = new_col
                    cur_viol += delta_viol
                    cur_num = new_num
                    cur_palette = list(set(cur))
                # else: cur[ei] already reverted to old_col
            
            temp *= cooling
        
        return best, best_num
    
    def try_reduce_palette(colors, target_size, max_time):
        """Try to reduce distinct colour count to target_size via greedy recoloring."""
        cur = colors.copy()
        cur_num = len(set(cur))
        best = cur.copy()
        best_num = cur_num
        
        if cur_num <= target_size:
            return best, best_num
        
        col_counts = Counter(cur)
        sorted_cols = sorted(col_counts.items(), key=lambda x: x[1])
        keep_cols = set(c for c, _ in sorted_cols[-target_size:])
        elim_cols = set(c for c, _ in sorted_cols[:cur_num - target_size])
        
        for iteration in range(8000):
            if time.time() > max_time:
                break
            
            elim_edges = [i for i in range(num_edges) if cur[i] in elim_cols]
            if not elim_edges:
                break
            
            ei = random.choice(elim_edges)
            old_col = cur[ei]
            
            candidates = list(keep_cols)
            random.shuffle(candidates)
            
            found = False
            for new_col in candidates:
                if new_col == old_col:
                    continue
                cur[ei] = new_col
                ok = True
                for k4i in edge_to_k4[ei]:
                    e_indices = k4_edges[k4i]
                    cs = set()
                    for ej in e_indices:
                        cs.add(cur[ej])
                    if len(cs) < 5:
                        ok = False
                        break
                if ok:
                    found = True
                    break
                cur[ei] = old_col
            
            if found:
                cur_num = len(set(cur))
                if cur_num < best_num:
                    best = cur.copy()
                    best_num = cur_num
                if cur_num <= target_size:
                    break
        
        return best, best_num
    
    random.seed(42)
    
    # Phase 1: Local search from base
    t1 = deadline - 45
    best1, num1 = local_search_fast(base_colors, t1)
    if num1 < best_num:
        best_colors = best1
        best_num = num1
    
    # Phase 2: Targeted palette reduction
    if time.time() < deadline - 15:
        for target in range(best_num - 1, 12, -1):
            if time.time() > deadline - 10:
                break
            t2 = min(deadline - 10, time.time() + 8)
            reduced, rnum = try_reduce_palette(best_colors, target, t2)
            if rnum < best_num:
                best_colors = reduced
                best_num = rnum
                if time.time() < deadline - 8:
                    t3 = min(deadline - 5, time.time() + 5)
                    best3, num3 = local_search_fast(best_colors, t3)
                    if num3 < best_num:
                        best_colors = best3
                        best_num = num3
    
    # Phase 3: Random restarts
    if time.time() < deadline - 10:
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            rand_colors = [random.randint(0, 13) for _ in range(num_edges)]
            viol = count_violations(rand_colors)
            if viol == 0:
                rnum = num_colours_used(rand_colors)
                if rnum < best_num:
                    best_colors = rand_colors
                    best_num = rnum
            else:
                t4 = min(deadline - 5, time.time() + 3)
                fixed, fnum = local_search_fast(rand_colors, t4)
                if fnum < best_num:
                    best_colors = fixed
                    best_num = fnum
    
    # Final verification
    final_viol = count_violations(best_colors)
    if final_viol > 0:
        best_colors = base_colors
        best_num = num_colours_used(base_colors)
    
    # Renormalize
    col_map = {}
    new_colors = []
    next_col = 0
    for c in best_colors:
        if c not in col_map:
            col_map[c] = next_col
            next_col += 1
        new_colors.append(col_map[c])
    
    return new_colors

# EVOLVE-BLOCK-END
