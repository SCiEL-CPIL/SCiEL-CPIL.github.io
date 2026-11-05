import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute K4s and their edges
    k4s = []
    k4_edges = []

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)

    num_k4s = len(k4s)

    # Precompute edge_to_k4s mapping for fast delta computation
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)

    def verify_valid(colors_list):
        for e_list in k4_edges:
            cols = set([colors_list[e] for e in e_list])
            if len(cols) < 5:
                return False
        return True

    def count_bad_k4s(colors_list):
        bad = 0
        for e_list in k4_edges:
            cols = set([colors_list[e] for e in e_list])
            if len(cols) < 5:
                bad += 1
        return bad

    # Baseline: (i+j) % 17
    def get_baseline():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    baseline = get_baseline()
    if not verify_valid(baseline):
        return [random.randint(0, 119) for _ in range(num_edges)]

    best_colors_used = 17
    best_result = baseline.copy()

    # Delta-based bad K4 count for edge color changes
    def delta_bad_edge(e_idx, old_color, new_color, colors):
        delta = 0
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            others = [x for x in e_list if x != e_idx]
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            c4 = colors[others[3]]
            c5 = colors[others[4]]
            s_before = {old_color, c1, c2, c3, c4, c5}
            bad_before = 1 if len(s_before) < 5 else 0
            s_after = {new_color, c1, c2, c3, c4, c5}
            bad_after = 1 if len(s_after) < 5 else 0
            delta += bad_after - bad_before
        return delta

    # SA on edge colors
    def sa_edge_colors(target_C, num_restarts, max_iters, init_colors=None):
        for r in range(num_restarts):
            if time.time() > deadline - 3:
                return None, None
            
            if init_colors is not None and r == 0:
                colors = init_colors.copy()
            else:
                colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            
            current_bad = count_bad_k4s(colors)
            
            if current_bad == 0:
                return colors, target_C
            
            T = 2.0
            cooling = 0.9998
            T_min = 0.01
            no_improve = 0
            best_bad = current_bad
            best_colors = colors.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue
                
                delta = delta_bad_edge(e_idx, old_color, new_color, colors)
                
                if delta < 0:
                    colors[e_idx] = new_color
                    current_bad += delta
                    no_improve = 0
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = colors.copy()
                elif delta == 0:
                    if random.random() < 0.2:
                        colors[e_idx] = new_color
                        no_improve = 0
                else:
                    if random.random() < min(1.0, np.exp(-delta / max(T, 0.01))):
                        colors[e_idx] = new_color
                        current_bad += delta
                    no_improve += 1
                
                if current_bad == 0:
                    return colors, target_C
                
                if no_improve > 300:
                    T = max(T_min, T * 0.9)
                    no_improve = 0
                else:
                    T = max(T_min, T * cooling)
            
            if best_bad < current_bad:
                return best_colors, target_C
        
        return None, None

    # Try to find valid colorings with fewer colors
    # Start with 13, then 14, then 15
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 15:
            break
        
        # Multiple restarts with SA
        num_restarts = 20 if target_C == 13 else 30
        max_iters = 5000
        
        colors, c_used = sa_edge_colors(target_C, num_restarts, max_iters)
        
        if colors is not None and c_used is not None:
            if verify_valid(colors):
                if c_used < best_colors_used:
                    best_colors_used = c_used
                    best_result = colors.copy()
                break
    
    # If we still have 17, try harder with 15
    if best_colors_used == 17:
        if time.time() < deadline - 10:
            colors, c_used = sa_edge_colors(15, 50, 8000)
            if colors is not None and c_used is not None:
                if verify_valid(colors):
                    if c_used < best_colors_used:
                        best_colors_used = c_used
                        best_result = colors.copy()

    # Polish pass: try to reduce colors used by remapping
    if best_colors_used > 13:
        if time.time() < deadline - 5:
            # Try to remap colors to use fewer distinct values
            used_colors = set(best_result)
            if len(used_colors) > 13:
                # Try to merge some colors
                color_list = sorted(used_colors)
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    # Try removing one color by mapping it to another
                    if len(color_list) <= 13:
                        break
                    remove_idx = random.randint(0, len(color_list) - 1)
                    target_idx = random.randint(0, len(color_list) - 1)
                    if target_idx == remove_idx:
                        continue
                    remove_color = color_list[remove_idx]
                    target_color = color_list[target_idx]
                    
                    # Try the remap
                    new_colors = [target_color if c == remove_color else c for c in best_result]
                    if verify_valid(new_colors):
                        best_result = new_colors
                        color_list.remove(remove_color)
                    else:
                        # Try a different target
                        continue
    
    # Final verification
    if verify_valid(best_result):
        return best_result
    else:
        return baseline
