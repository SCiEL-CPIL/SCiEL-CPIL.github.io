import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Edge list and index
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list)
    
    # For each K4, store the list of edge indices
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in itertools.combinations(k4, 2):
            edges.append(edge_index[(a, b)])
        k4_edges.append(edges)
    
    # For each edge, the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Precompute: for each K4, the 6 edge indices as a numpy array
    k4_edges_np = [np.array(e, dtype=np.int32) for e in k4_edges]
    
    # Helper: compute violation count and per-K4 color frequency
    def compute_violations(colours):
        """Returns (total_violations, per_k4_distinct_count, per_k4_color_freq)"""
        colours = np.array(colours, dtype=np.int32)
        total_viol = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        for k in range(num_k4s):
            edges = k4_edges_np[k]
            cvals = colours[edges]
            distinct = len(np.unique(cvals))
            distinct_counts[k] = distinct
            if distinct < 5:
                total_viol += 1
        return total_viol, distinct_counts
    
    def distinct_count_from_colours(colours, k):
        """Compute distinct color count for K4 k given current colours array."""
        edges = k4_edges_np[k]
        cvals = colours[edges]
        return len(np.unique(cvals))
    
    # Try structured colorings first
    def try_structured():
        """Try various structured colorings and return best valid one."""
        candidates = []
        
        # Sum mod p
        for p in range(13, 20):
            c = [((i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sum_mod_%d' % p))
        
        # Difference mod p
        for p in range(13, 20):
            c = [((i - j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'diff_mod_%d' % p))
        
        # Product mod p
        for p in range(13, 20):
            c = [((i * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'prod_mod_%d' % p))
        
        # Sum of squares mod p
        for p in range(13, 20):
            c = [((i * i + j * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sq_sum_mod_%d' % p))
        
        # i*j + i + j mod p
        for p in range(13, 20):
            c = [((i * j + i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'ijij_mod_%d' % p))
        
        # F4-based: assign each vertex a distinct element of F4^2 (16 elements)
        # F4 = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
        # Represent elements as pairs (a, b) in F2^2 with multiplication:
        # (a,b)*(c,d) = (ac+bd, ad+bc+bd)
        # Vertex i -> element i in F4^2
        # Edge color = hash of coordinate-wise difference projected to 13 colors
        
        # F4 addition is just XOR on the 2-bit representation
        # F4 elements as 2-bit pairs: 0=(0,0), 1=(1,0), alpha=(0,1), alpha^2=(1,1)
        f4_elements = [(0,0), (1,0), (0,1), (1,1)]
        
        # Assign 16 vertices to distinct elements of F4^2
        vertex_f4 = []
        for a in range(4):
            for b in range(4):
                vertex_f4.append((a, b))  # (coord1, coord2) each in F4
        
        # For each edge, compute the difference in F4^2 and hash to color
        for C in [13, 14, 15]:
            c = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Compute difference in F4^2
                    # F4 subtraction is same as addition (char 2)
                    d1 = vertex_f4[i][0] ^ vertex_f4[j][0]  # XOR for F4 addition
                    d2 = vertex_f4[i][1] ^ vertex_f4[j][1]
                    # Hash the pair (d1, d2) to a color
                    # d1, d2 are in {0,1,2,3} representing F4 elements
                    val = (d1 * 5 + d2 * 3 + i * 2 + j) % C
                    c.append(val)
            candidates.append((c, 'f4_%d' % C))
        
        # Also try a different F4 hash
        for C in [13, 14, 15]:
            c = []
            for i in range(n):
                for j in range(i + 1, n):
                    d1 = vertex_f4[i][0] ^ vertex_f4[j][0]
                    d2 = vertex_f4[i][1] ^ vertex_f4[j][1]
                    # Use F4 multiplication to get a single F4 element
                    # (d1 * d2) in F4
                    # F4 mult: (a,b)*(c,d) = (ac+bd, ad+bc+bd)
                    a, b = f4_elements[d1]
                    c_, d_ = f4_elements[d2]
                    r1 = (a * c_ + b * d_) % 2
                    r2 = (a * d_ + b * c_ + b * d_) % 2
                    # Find which F4 element (r1, r2) is
                    prod_idx = 0
                    for k, (x, y) in enumerate(f4_elements):
                        if x == r1 and y == r2:
                            prod_idx = k
                            break
                    val = (prod_idx * 7 + i * 3 + j * 5) % C
                    c.append(val)
            candidates.append((c, 'f4_mult_%d' % C))
        
        # Check each candidate
        best_valid = None
        best_num_colors = float('inf')
        for c, name in candidates:
            num_used = len(set(c))
            if num_used >= best_num_colors:
                continue
            valid = True
            for k in range(num_k4s):
                edges = k4_edges[k]
                cset = set(c[e] for e in edges)
                if len(cset) < 5:
                    valid = False
                    break
            if valid:
                best_valid = c
                best_num_colors = num_used
                if best_num_colors <= 13:
                    return best_valid
        
        return best_valid
    
    # Fast local search with simulated annealing
    def local_search(colours, C, max_time=60.0):
        """
        colours: list of ints (edge colors)
        C: number of colors available
        Returns: (best_colours, best_cost, found_valid)
        """
        colours = list(colours)
        
        # Initial violation count
        total_viol, distinct_counts = compute_violations(colours)
        current_cost = total_viol
        
        best_colours = colours[:]
        best_cost = current_cost
        found_valid = (current_cost == 0)
        
        if found_valid:
            return best_colours, best_cost, True
        
        # Simulated annealing parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        max_iter = 500000
        iter_count = 0
        stall_count = 0
        max_stall = 2000
        
        # Per-K4 color frequency dicts
        freq = [dict() for _ in range(num_k4s)]
        for k in range(num_k4s):
            edges = k4_edges[k]
            for e in edges:
                c = colours[e]
                freq[k][c] = freq[k].get(c, 0) + 1
        
        # Maintain set of violating K4s incrementally
        viol_set = set(k for k in range(num_k4s) if distinct_counts[k] < 5)
        
        def compute_delta(e_idx, old_c, new_c):
            if old_c == new_c:
                return 0
            delta = 0
            for k in edge_to_k4s[e_idx]:
                old_distinct = distinct_counts[k]
                old_count = freq[k].get(old_c, 0)
                new_count = freq[k].get(new_c, 0)
                
                new_distinct = old_distinct
                if old_count == 1:
                    new_distinct -= 1
                if new_count == 0:
                    new_distinct += 1
                
                old_viol = 1 if old_distinct < 5 else 0
                new_viol = 1 if new_distinct < 5 else 0
                delta += new_viol - old_viol
            return delta
        
        def apply_move(e_idx, old_c, new_c):
            nonlocal current_cost
            if old_c == new_c:
                return
            colours[e_idx] = new_c
            for k in edge_to_k4s[e_idx]:
                old_count = freq[k].get(old_c, 0)
                if old_count == 1:
                    del freq[k][old_c]
                    distinct_counts[k] -= 1
                else:
                    freq[k][old_c] = old_count - 1
                
                new_count = freq[k].get(new_c, 0)
                if new_count == 0:
                    freq[k][new_c] = 1
                    distinct_counts[k] += 1
                else:
                    freq[k][new_c] = new_count + 1
                
                if distinct_counts[k] < 5:
                    viol_set.add(k)
                elif k in viol_set:
                    viol_set.discard(k)
        
        while iter_count < max_iter:
            if time.time() > deadline:
                break
            
            if not viol_set:
                found_valid = True
                best_cost = 0
                best_colours = colours[:]
                break
            
            k4_idx = random.choice(list(viol_set))
            
            best_delta = float('inf')
            best_move = None
            
            for e_idx in k4_edges[k4_idx]:
                old_c = colours[e_idx]
                candidates = set()
                for _ in range(min(C, 8)):
                    candidates.add(random.randint(0, C - 1))
                for nc in range(min(C, 10)):
                    candidates.add(nc)
                
                for new_c in candidates:
                    if new_c == old_c:
                        continue
                    d = compute_delta(e_idx, old_c, new_c)
                    if d < best_delta:
                        best_delta = d
                        best_move = (e_idx, old_c, new_c)
            
            if best_move:
                e_idx, old_c, new_c = best_move
                if best_delta <= 0 or (T > T_min and random.random() < np.exp(-best_delta / T)):
                    apply_move(e_idx, old_c, new_c)
                    current_cost += best_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                e_idx = random.choice(k4_edges[k4_idx])
                old_c = colours[e_idx]
                new_c = random.randint(0, C - 1)
                if new_c != old_c:
                    d = compute_delta(e_idx, old_c, new_c)
                    apply_move(e_idx, old_c, new_c)
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
            
            iter_count += 1
            T = max(T_min, T * cooling)
            
            if stall_count > max_stall:
                for _ in range(15):
                    e_idx = random.randint(0, num_edges - 1)
                    old_c = colours[e_idx]
                    new_c = random.randint(0, C - 1)
                    if new_c != old_c:
                        d = compute_delta(e_idx, old_c, new_c)
                        apply_move(e_idx, old_c, new_c)
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colours = colours[:]
                            if best_cost == 0:
                                found_valid = True
                                break
                stall_count = 0
                T = 5.0
        
        if not found_valid and best_cost == 0:
            found_valid = True
        
        return best_colours, best_cost, found_valid
    
    def reduce_colors(colours, max_time=15.0):
        """Try to reduce the number of distinct colors in a valid coloring."""
        colours = list(colours)
        used_colors = set(colours)
        num_used = len(used_colors)
        
        if num_used <= 13:
            return colours, num_used
        
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        freq = [dict() for _ in range(num_k4s)]
        for k in range(num_k4s):
            edges = k4_edges[k]
            for e in edges:
                c = colours[e]
                freq[k][c] = freq[k].get(c, 0) + 1
            distinct_counts[k] = len(freq[k])
        
        remaining = sorted(used_colors, key=lambda c: sum(1 for e in range(num_edges) if colours[e] == c))
        
        for target_color in remaining:
            if time.time() > deadline:
                break
            if target_color not in set(colours):
                continue
            
            target_edges = [e for e in range(num_edges) if colours[e] == target_color]
            if not target_edges:
                continue
            
            success = True
            for e_idx in target_edges:
                if time.time() > deadline:
                    break
                old_c = colours[e_idx]
                if old_c != target_color:
                    continue
                
                found = False
                for new_c in used_colors:
                    if new_c == target_color:
                        continue
                    valid = True
                    for k in edge_to_k4s[e_idx]:
                        old_distinct = distinct_counts[k]
                        old_count = freq[k].get(target_color, 0)
                        new_count = freq[k].get(new_c, 0)
                        
                        new_distinct = old_distinct
                        if old_count == 1:
                            new_distinct -= 1
                        if new_count == 0:
                            new_distinct += 1
                        
                        if new_distinct < 5:
                            valid = False
                            break
                    
                    if valid:
                        colours[e_idx] = new_c
                        for k in edge_to_k4s[e_idx]:
                            oc = freq[k].get(target_color, 0)
                            if oc == 1:
                                del freq[k][target_color]
                                distinct_counts[k] -= 1
                            else:
                                freq[k][target_color] = oc - 1
                            nc = freq[k].get(new_c, 0)
                            if nc == 0:
                                freq[k][new_c] = 1
                                distinct_counts[k] += 1
                            else:
                                freq[k][new_c] = nc + 1
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                used_colors.discard(target_color)
        
        final_used = len(set(colours))
        return colours, final_used
    
    # Main search
    best_result = None
    best_cost = float('inf')
    
    # Try structured first
    structured = try_structured()
    if structured is not None:
        num_used = len(set(structured))
        if num_used < best_cost:
            best_cost = num_used
            best_result = structured
        if best_cost <= 13:
            return best_result
    
    # Try local search with various C and initializations
    for C in [13, 14, 15]:
        if time.time() > deadline:
            break
        
        num_restarts = 8 if C == 13 else 5
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Use F4-based initialization for first restart, random for others
            if r == 0:
                # F4-based initialization
                f4_elements = [(0,0), (1,0), (0,1), (1,1)]
                vertex_f4 = [(a, b) for a in range(4) for b in range(4)]
                colours = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d1 = vertex_f4[i][0] ^ vertex_f4[j][0]
                        d2 = vertex_f4[i][1] ^ vertex_f4[j][1]
                        val = (d1 * 5 + d2 * 3 + i * 2 + j) % C
                        colours.append(val)
            else:
                colours = [random.randint(0, C - 1) for _ in range(num_edges)]
            
            ls_time = 25.0 if C == 13 else 20.0
            best_colours, best_c, found = local_search(colours, C, max_time=ls_time)
            
            if found:
                red_time = 12.0 if C == 13 else 8.0
                reduced, reduced_num = reduce_colors(best_colours, max_time=red_time)
                if reduced_num < best_cost:
                    best_cost = reduced_num
                    best_result = reduced
                if best_cost <= 13:
                    return best_result
    
    # If no valid found, try with C=16, 17 as fallback
    for C in [16, 17]:
        if time.time() > deadline:
            break
        colours = [random.randint(0, C - 1) for _ in range(num_edges)]
        best_colours, best_c, found = local_search(colours, C, max_time=15.0)
        if found:
            reduced, reduced_num = reduce_colors(best_colours, max_time=8.0)
            if reduced_num < best_cost:
                best_cost = reduced_num
                best_result = reduced
    
    # Final fallback: seed
    if best_result is None:
        best_result = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify and return
    return best_result
