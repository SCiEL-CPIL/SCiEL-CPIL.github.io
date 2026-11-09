import time
import random
import sys

def construct():
    N = 128
    start_time = time.time()
    time_limit = 85.0
    
    # Precompute APs
    # all_aps: list of (a, d)
    # ap_index[i]: list of ap_ids containing i
    # ap_details[i][ap_id]: tuple of the other 3 indices in the AP
    ap_index = [[] for _ in range(N)]
    all_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            ap_id = len(all_aps)
            all_aps.append((a, d))
            for j in range(4):
                idx = a + j * d
                ap_index[idx].append(ap_id)
                
    num_aps = len(all_aps)
    
    # Precompute details for fast lookup
    # details[i] is a list of tuples (ap_id, (idx1, idx2, idx3)) where idx1..3 are the other indices
    details = [None] * N
    for i in range(N):
        details[i] = []
        for ap_id in ap_index[i]:
            a, d = all_aps[ap_id]
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            others = tuple(x for x in idxs if x != i)
            details[i].append((ap_id, others))

    def count_violations(colors):
        total_v = 0
        for ap_id in range(num_aps):
            a, d = all_aps[ap_id]
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2 * d]
            c3 = colors[a + 3 * d]
            # Check if distinct colors < 3
            if len({c0, c1, c2, c3}) < 3:
                total_v += 1
        return total_v

    def solve(K):
        # Initialize
        colors = [random.randint(0, K - 1) for _ in range(N)]
        # Calculate initial violations
        violated = [False] * num_aps
        total_v = 0
        for ap_id in range(num_aps):
            a, d = all_aps[ap_id]
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2 * d]
            c3 = colors[a + 3 * d]
            if len({c0, c1, c2, c3}) < 3:
                violated[ap_id] = True
                total_v += 1
        
        if total_v == 0:
            return colors
            
        best_colors = colors[:]
        best_total_v = total_v
        
        # Local Search with Simulated Annealing-like behavior
        # We try to reduce total_v. If stuck, we allow random moves.
        current_temp = 1.0
        cooling_rate = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick a random index
            i = random.randint(0, N - 1)
            old_c = colors[i]
            
            # Try all K colors
            best_c = old_c
            best_v = total_v
            
            for c_new in range(K):
                if c_new == old_c:
                    continue
                
                delta = 0
                # Iterate over APs containing i
                for ap_id, others in details[i]:
                    was_v = violated[ap_id]
                    # Check new violation status
                    # others is a tuple of 3 ints
                    o0, o1, o2 = others
                    # Fast check for len({c_new, o0, o1, o2}) < 3
                    # This is True if the set has size 1 or 2.
                    # Size 1: c_new==o0==o1==o2
                    # Size 2: exactly two distinct values among the four.
                    
                    # Optimization: 
                    # If o0, o1, o2 are all distinct (3 colors), then adding c_new makes >=3 colors. So is_v_new is False.
                    # If o0, o1, o2 have 2 distinct colors, say {A, B}. 
                    #   Then {c_new, A, B} has 3 colors if c_new not in {A, B}, else 2 colors.
                    # If o0, o1, o2 have 1 distinct color, say {A}.
                    #   Then {c_new, A} has 2 colors if c_new != A, else 1 color.
                    
                    # Let's just use the set logic, it's very fast in CPython for small sets
                    is_v_new = len({c_new, o0, o1, o2}) < 3
                    
                    if was_v != is_v_new:
                        if is_v_new:
                            delta += 1
                        else:
                            delta -= 1
                
                new_total_v = total_v + delta
                if new_total_v < best_v:
                    best_v = new_total_v
                    best_c = c_new
            
            # Apply best_c if it improves or with some probability (SA)
            if best_c != old_c:
                # Update state
                colors[i] = best_c
                for ap_id, others in details[i]:
                    o0, o1, o2 = others
                    is_v_new = len({best_c, o0, o1, o2}) < 3
                    if violated[ap_id] != is_v_new:
                        if is_v_new:
                            total_v += 1
                        else:
                            total_v -= 1
                        violated[ap_id] = is_v_new
                
                if total_v < best_total_v:
                    best_total_v = total_v
                    best_colors = colors[:]
                
                if total_v == 0:
                    return colors
            else:
                # No improvement found. 
                # Accept a random move with probability based on temperature to escape local minima
                # Simple heuristic: if we are stuck, just pick a random different color
                # This is a weak form of SA, but effective for small N
                if random.random() < current_temp:
                    c_rand = random.randint(0, K - 1)
                    if c_rand != old_c:
                        colors[i] = c_rand
                        for ap_id, others in details[i]:
                            o0, o1, o2 = others
                            is_v_new = len({c_rand, o0, o1, o2}) < 3
                            if violated[ap_id] != is_v_new:
                                if is_v_new:
                                    total_v += 1
                                else:
                                    total_v -= 1
                                violated[ap_id] = is_v_new
                        
                        if total_v < best_total_v:
                            best_total_v = total_v
                            best_colors = colors[:]
                        
                        if total_v == 0:
                            return colors
            
            current_temp *= cooling_rate
            if current_temp < 0.01:
                current_temp = 0.01

        return best_colors if best_total_v < float('inf') else None

    # Try K=3 first, then K=4 if needed
    # The problem asks to minimize the number of labels used.
    # K=3 is the theoretical minimum for 3-coloring 4-APs (it's known that 3 colors suffice for some N, but we need to check if it works for N=128).
    # Actually, the problem says "Minimize the number of labels actually used".
    # So we should try K=3 first. If we find a valid solution with 3 colors, that's optimal (since we need at least 3 colors for any non-constant 4-AP).
    
    result = solve(3)
    if result is not None:
        # Verify result
        if all(len({result[a + j * d] for j in range(4)}) >= 3 for a, d in all_aps):
            return result
    
    # If K=3 failed, try K=4
    result = solve(4)
    if result is not None:
        if all(len({result[a + j * d] for j in range(4)}) >= 3 for a, d in all_aps):
            return result
            
    # Fallback: Greedy construction
    colors = []
    for x in range(1, N + 1):
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x
            # x is the 4th term: x, x-d, x-2d, x-3d
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - j * d for j in range(4)]
                if any(idx < 0 for idx in idxs):
                    continue
                c_vals = set()
                c_vals.add(color)
                for idx in idxs[:3]:
                    c_vals.add(colors[idx])
                if len(c_vals) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
    return colors
