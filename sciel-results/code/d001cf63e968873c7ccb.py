import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    line_points_list = [list(s) for s in line_points]
    
    def init_C_parabola():
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C

    def init_C_random(rng, density=0.05):
        in_C = np.zeros(q2, dtype=bool)
        for i in range(q2):
            if rng.random() < density:
                in_C[i] = True
        return in_C

    def compute_W_and_wcount(in_C):
        W = [set() for _ in range(q2)]
        wcount = np.zeros(q2, dtype=np.int32)
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
                    wcount[i] += 1
        return W, wcount

    def local_search(in_C, W, wcount, rng, max_time=15.0):
        """Simulated annealing style local search on complement C."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        start_time = time.time()
        temperature = 10.0
        cooling = 0.995
        
        # Build list of all points not in C (candidates for addition)
        # and points in C (candidates for removal)
        
        while time.time() - start_time < max_time and time.time() < _DEADLINE - _MARGIN:
            # Choose a random point to try adding or removing
            if rng.random() < 0.7:
                # Try adding a point
                candidates = [i for i in range(q2) if not in_C[i]]
                if not candidates:
                    break
                p_idx = rng.choice(candidates)
                
                # Check if safe to add
                safe = True
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != p_idx:
                            if lid in W[x_idx] and wcount[x_idx] == 1:
                                safe = False
                                break
                    if not safe:
                        break
                
                if safe:
                    # Add p_idx
                    in_C[p_idx] = True
                    for lid in point_lines[p_idx]:
                        for x_idx in line_points_list[lid]:
                            if x_idx != p_idx:
                                if lid in W[x_idx]:
                                    W[x_idx].discard(lid)
                                    wcount[x_idx] -= 1
                else:
                    # Unsafe: try with probability based on temperature
                    if rng.random() < temperature:
                        # Force add even if unsafe (will be fixed later)
                        in_C[p_idx] = True
                        for lid in point_lines[p_idx]:
                            for x_idx in line_points_list[lid]:
                                if x_idx != p_idx:
                                    if lid in W[x_idx]:
                                        W[x_idx].discard(lid)
                                        wcount[x_idx] -= 1
            else:
                # Try removing a point
                candidates = [i for i in range(q2) if in_C[i]]
                if not candidates:
                    break
                p_idx = rng.choice(candidates)
                
                # Remove p_idx
                in_C[p_idx] = False
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != p_idx:
                            if lid in W[x_idx]:
                                # This line was a witness for x_idx, now it's not valid anymore
                                # because p_idx is no longer in C, so the line might be valid again
                                # Actually, removing a point from C makes more lines valid (witnesses)
                                # So we need to recheck if lid becomes a witness for x_idx
                                # lid is a witness for x_idx if no other point in C is on lid (other than x_idx)
                                # Since we removed p_idx, lid might now be a witness
                                # Check if any other point in C is on lid
                                has_other = False
                                for q_idx in line_points_list[lid]:
                                    if q_idx != x_idx and q_idx != p_idx and in_C[q_idx]:
                                        has_other = True
                                        break
                                if not has_other:
                                    W[x_idx].add(lid)
                                    wcount[x_idx] += 1
                            else:
                                # lid was not a witness before, might become one now
                                has_other = False
                                for q_idx in line_points_list[lid]:
                                    if q_idx != x_idx and q_idx != p_idx and in_C[q_idx]:
                                        has_other = True
                                        break
                                if not has_other:
                                    W[x_idx].add(lid)
                                    wcount[x_idx] += 1
                
                # After removal, some points might now have wcount > 0 that were 0 before
                # We should check if any point has wcount == 0 and try to fix it
                # For simplicity, we'll rely on the next iterations to fix issues
        
            temperature *= cooling
        
        return in_C, W, wcount

    def greedy_add(in_C, W, wcount, max_time=10.0):
        """Greedy add points that are safe and have low damage."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        start_time = time.time()
        
        # Build heap of (cost, point) for points not in C
        heap = []
        for i in range(q2):
            if not in_C[i]:
                cost = 0
                for lid in point_lines[i]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != i and lid in W[x_idx]:
                            cost += 1
                heapq.heappush(heap, (cost, i))
        
        while heap and time.time() - start_time < max_time and time.time() < _DEADLINE - _MARGIN:
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
            
            # Check safety
            safe = True
            current_cost = 0
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx != p_idx:
                        if lid in W[x_idx]:
                            current_cost += 1
                            if wcount[x_idx] == 1:
                                safe = False
                                break
                if not safe:
                    break
            
            if not safe:
                continue
            
            # Add p_idx
            in_C[p_idx] = True
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx != p_idx:
                        if lid in W[x_idx]:
                            W[x_idx].discard(lid)
                            wcount[x_idx] -= 1
                            if not in_C[x_idx]:
                                new_cost = 0
                                for lid2 in point_lines[x_idx]:
                                    for y_idx in line_points_list[lid2]:
                                        if y_idx != x_idx and lid2 in W[y_idx]:
                                            new_cost += 1
                                heapq.heappush(heap, (new_cost, x_idx))
        
        return in_C, W, wcount

    best_in_C = None
    best_size = -1
    
    rng = random.Random(42)
    
    # Multiple strategies
    strategies = []
    
    # Strategy 1: Parabola + greedy
    strategies.append(("parabola_greedy", init_C_parabola, greedy_add))
    
    # Strategy 2: Random + local search
    for density in [0.02, 0.05, 0.10]:
        strategies.append((f"random_{density}_ls", 
                          lambda d=density: init_C_random(rng, d), 
                          local_search))
    
    # Strategy 3: Parabola + local search
    strategies.append(("parabola_ls", init_C_parabola, local_search))
    
    # Strategy 4: Random + greedy
    for density in [0.02, 0.05]:
        strategies.append((f"random_{density}_greedy",
                          lambda d=density: init_C_random(rng, d),
                          greedy_add))
    
    num_trials = len(strategies) * 2  # 2 runs per strategy
    time_per_trial = 18.0  # 10 * 18 = 180s, leaving plenty of margin
    
    for trial in range(num_trials):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        strat_name, init_func, search_func = strategies[trial % len(strategies)]
        
        in_C = init_func()
        W, wcount = compute_W_and_wcount(in_C)
        
        in_C, W, wcount = search_func(in_C, W, wcount, rng, max_time=time_per_trial)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
    
    # If we still have time, do more local search on the best solution
    if best_in_C is not None and time.time() < _DEADLINE - _MARGIN - 30:
        W, wcount = compute_W_and_wcount(best_in_C)
        best_in_C, W, wcount = local_search(best_in_C, W, wcount, rng, max_time=30.0)
        best_size = int(np.sum(best_in_C))
    
    if best_in_C is None:
        best_in_C = init_C_parabola()
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
