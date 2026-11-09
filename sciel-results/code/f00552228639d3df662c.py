import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute structures for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # For each triple, list of (other_triple_indices_in_quad)
        quad_neighbors = [[] for _ in range(num_triples)]
        # For each triple, list of (other_triple_indices_in_quint)
        quint_neighbors = [[] for _ in range(num_triples)]
        
        # Iterate over all quads
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            
            quad_neighbors[t1].append((t2, t3, t4))
            quad_neighbors[t2].append((t1, t3, t4))
            quad_neighbors[t3].append((t1, t2, t4))
            quad_neighbors[t4].append((t1, t2, t3))
            
        # Iterate over all quints
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            
            for i, t_idx in enumerate(triple_indices_in_quint):
                others = tuple(triple_indices_in_quint[j] for j in range(10) if j != i)
                quint_neighbors[t_idx].append(others)
                
        return triple_list, num_triples, triple_to_idx, quad_neighbors, quint_neighbors

    def run_local_search(target_n, initial_colors, deadline):
        """
        Runs simulated annealing to minimize violations.
        Returns (colors, num_violations)
        """
        triple_list, num_triples, triple_to_idx, quad_neighbors, quint_neighbors = build_structures(target_n)
        
        # Color array: 0 for Red, 1 for Blue
        # Use numpy int8 for speed
        colors = np.array(initial_colors, dtype=np.int8)
        
        # Initial violation count
        red_k4_count = 0
        blue_k5_count = 0
        
        # Precompute indices for all quads and quints for fast access
        # quad_idx_list[i] = (t1, t2, t3, t4)
        quad_idx_list = []
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            quad_idx_list.append((t1, t2, t3, t4))
            
        # quint_idx_list[i] = (t1, ..., t10)
        quint_idx_list = []
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            quint_idx_list.append(tuple(triple_indices_in_quint))

        # Initial full scan
        for q in quad_idx_list:
            if colors[q[0]] == 0 and colors[q[1]] == 0 and colors[q[2]] == 0 and colors[q[3]] == 0:
                red_k4_count += 1
                
        for q in quint_idx_list:
            # Check if all 10 are blue
            # Optimization: check first few, then rest
            if colors[q[0]] == 1 and colors[q[1]] == 1 and colors[q[2]] == 1 and colors[q[3]] == 1 and colors[q[4]] == 1 and \
               colors[q[5]] == 1 and colors[q[6]] == 1 and colors[q[7]] == 1 and colors[q[8]] == 1 and colors[q[9]] == 1:
                blue_k5_count += 1

        current_cost = red_k4_count + blue_k5_count
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        # Precompute random indices? No, just use randint.
        
        while time.time() < deadline - 1:
            # Choose a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            
            # Calculate delta red_k4
            delta_r4 = 0
            if old_color == 0: # Currently Red -> Flipping to Blue
                # Check quads containing t_idx
                for others in quad_neighbors[t_idx]:
                    # If the other 3 are red, this quad WAS red K4
                    if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
                        delta_r4 -= 1
            else: # Currently Blue -> Flipping to Red
                for others in quad_neighbors[t_idx]:
                    # If the other 3 are red, this quad BECOMES red K4
                    if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
                        delta_r4 += 1
                        
            # Calculate delta blue_k5
            delta_b5 = 0
            if old_color == 1: # Currently Blue -> Flipping to Red
                # Check quints containing t_idx
                for others in quint_neighbors[t_idx]:
                    # If the other 9 are blue, this quint WAS blue K5
                    # Check all 9
                    if colors[others[0]] == 1 and colors[others[1]] == 1 and colors[others[2]] == 1 and colors[others[3]] == 1 and colors[others[4]] == 1 and \
                       colors[others[5]] == 1 and colors[others[6]] == 1 and colors[others[7]] == 1 and colors[others[8]] == 1:
                        delta_b5 -= 1
            else: # Currently Red -> Flipping to Blue
                for others in quint_neighbors[t_idx]:
                    # If the other 9 are blue, this quint BECOMES blue K5
                    if colors[others[0]] == 1 and colors[others[1]] == 1 and colors[others[2]] == 1 and colors[others[3]] == 1 and colors[others[4]] == 1 and \
                       colors[others[5]] == 1 and colors[others[6]] == 1 and colors[others[7]] == 1 and colors[others[8]] == 1:
                        delta_b5 += 1

            delta = delta_r4 + delta_b5
            
            # Simulated Annealing decision
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept move
                colors[t_idx] = 1 - old_color
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        # Found valid coloring
                        return best_colors.tolist(), 0
            
            T *= cooling
            if T < T_min:
                T = T_min
                # Optional: restart temperature if stuck for a while?
                # Simple restart logic: if no improvement in X steps, reset T
                # We'll keep it simple for now, maybe add a stagnation counter later if needed.
                # For now, just keep cooling.
                
        return best_colors.tolist(), best_cost

    # Strategy: Try increasing n
    # We start with n=34 (the target) and try to find a valid coloring.
    # If we fail, we try n=33, etc.
    # Given time constraints, we focus most time on n=34.
    
    targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    # Time allocation: most time to 34
    # 115s total.
    # 34: 60s
    # 33: 20s
    # 32: 10s
    # 31: 5s
    # 30: 5s
    # rest: 5s total
    
    time_allocations = {
        34: 60,
        33: 20,
        32: 10,
        31: 5,
        30: 5,
        29: 2,
        28: 2,
        27: 1,
        26: 1,
        25: 1
    }
    
    best_found = (9, mod_color(9, 3))
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        alloc = time_allocations.get(n, 2)
        current_deadline = time.time() + alloc
        
        max_restarts = 20
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline:
                break
                
            # Generate initial coloring
            if r == 0:
                # Modular 3
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 1:
                # Modular 5
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 2:
                # Modular 7
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 3:
                # Modular 2
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 4:
                # Random 0.5
                init = [0 if random.random() < 0.5 else 1 for _ in range(len(list(combinations(range(n), 3))))]
            elif r == 5:
                # Random 0.4 (more red)
                init = [0 if random.random() < 0.4 else 1 for _ in range(len(list(combinations(range(n), 3))))]
            elif r == 6:
                # Random 0.6 (more blue)
                init = [0 if random.random() < 0.6 else 1 for _ in range(len(list(combinations(range(n), 3))))]
            else:
                # Random with varying bias
                bias = 0.3 + 0.4 * (r % 5) / 4.0
                init = [0 if random.random() < bias else 1 for _ in range(len(list(combinations(range(n), 3))))]

            colors, cost = run_local_search(n, init, current_deadline)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
            else:
                # If we found a low cost but not zero, we might still want to keep it if it's better than fallback?
                # But the task requires a VALID coloring. So we only update best_found if cost == 0.
                pass
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
