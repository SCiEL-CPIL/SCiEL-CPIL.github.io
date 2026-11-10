import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Precompute triple indices for given n
    def get_triples(n):
        return list(itertools.combinations(range(n), 3))
    
    def build_matrices(n):
        triples = get_triples(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        num_fours = len(four_sets)
        num_fives = len(five_sets)
        
        # M: shape (num_fours, 4)
        M = np.zeros((num_fours, 4), dtype=np.int32)
        for i, quad in enumerate(four_sets):
            M[i, 0] = triple_to_idx[(quad[0], quad[1], quad[2])]
            M[i, 1] = triple_to_idx[(quad[0], quad[1], quad[3])]
            M[i, 2] = triple_to_idx[(quad[0], quad[2], quad[3])]
            M[i, 3] = triple_to_idx[(quad[1], quad[2], quad[3])]
            
        # Q: shape (num_fives, 10)
        Q = np.zeros((num_fives, 10), dtype=np.int32)
        for i, quint in enumerate(five_sets):
            idx = 0
            for t in itertools.combinations(quint, 3):
                Q[i, idx] = triple_to_idx[t]
                idx += 1
                
        # Build reverse index: for each triple, which four-sets and five-sets contain it
        triple_to_fours = [[] for _ in range(num_triples)]
        triple_to_fives = [[] for _ in range(num_triples)]
        
        for i in range(num_fours):
            for j in range(4):
                triple_to_fours[M[i, j]].append(i)
        
        for i in range(num_fives):
            for j in range(10):
                triple_to_fives[Q[i, j]].append(i)
                
        return triples, M, Q, num_triples, triple_to_fours, triple_to_fives

    def calculate_cost(colors, M, Q):
        # colors is 1D numpy array of 0s and 1s
        # Red K4: all 4 triples in M are 0
        # Blue K5: all 10 triples in Q are 1
        red_viol = np.all(colors[M] == 0, axis=1)
        blue_viol = np.all(colors[Q] == 1, axis=1)
        return int(np.sum(red_viol) + np.sum(blue_viol))

    def solve_for_n(n, time_budget):
        triples, M, Q, num_triples, triple_to_fours, triple_to_fives = build_matrices(n)
        num_fours = M.shape[0]
        num_fives = Q.shape[0]
        
        # For incremental cost updates, track:
        # - red_count[four_idx]: number of red triples in that four-set
        # - blue_count[five_idx]: number of blue triples in that five-set
        
        def init_counts(colors):
            red_counts = np.zeros(num_fours, dtype=np.int32)
            blue_counts = np.zeros(num_fives, dtype=np.int32)
            
            # Count red triples in each four-set
            for i in range(num_fours):
                red_counts[i] = np.sum(colors[M[i]] == 0)
            
            # Count blue triples in each five-set
            for i in range(num_fives):
                blue_counts[i] = np.sum(colors[Q[i]] == 1)
            
            return red_counts, blue_counts
        
        def get_cost_from_counts(red_counts, blue_counts):
            return int(np.sum(red_counts == 4) + np.sum(blue_counts == 10))
        
        # Initial random coloring
        colors = np.random.randint(0, 2, num_triples)
        red_counts, blue_counts = init_counts(colors)
        current_cost = get_cost_from_counts(red_counts, blue_counts)
        
        best_cost = current_cost
        best_colors = colors.copy()
        best_red_counts = red_counts.copy()
        best_blue_counts = blue_counts.copy()
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.005
        restart_threshold = 3000  # If no improvement in this many steps, restart
        steps_since_improvement = 0
        
        deadline = start_time + time_budget
        
        while time.time() < deadline:
            if current_cost == 0:
                return colors.tolist()
            
            # Pick random triple
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Update counts incrementally
            # For four-sets containing this triple
            for fi in triple_to_fours[t_idx]:
                if old_color == 0:
                    red_counts[fi] -= 1
                # new_color is 1, so no change to red count
            
            # For five-sets containing this triple
            for fi in triple_to_fives[t_idx]:
                if old_color == 1:
                    blue_counts[fi] -= 1
                # new_color is 0, so no change to blue count
            
            # Apply flip
            colors[t_idx] = new_color
            
            # Calculate new cost
            new_cost = get_cost_from_counts(red_counts, blue_counts)
            
            # Acceptance
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    best_red_counts = red_counts.copy()
                    best_blue_counts = blue_counts.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # Probabilistic accept
                delta = new_cost - current_cost
                if T > min_T:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_cost = new_cost
                        steps_since_improvement += 1
                    else:
                        # Reject, revert
                        colors[t_idx] = old_color
                        # Revert counts
                        for fi in triple_to_fours[t_idx]:
                            if old_color == 0:
                                red_counts[fi] += 1
                        for fi in triple_to_fives[t_idx]:
                            if old_color == 1:
                                blue_counts[fi] += 1
                        steps_since_improvement += 1
                else:
                    # Reject, revert
                    colors[t_idx] = old_color
                    for fi in triple_to_fours[t_idx]:
                        if old_color == 0:
                            red_counts[fi] += 1
                    for fi in triple_to_fives[t_idx]:
                        if old_color == 1:
                            blue_counts[fi] += 1
                    steps_since_improvement += 1
            
            T *= cooling_rate
            
            # Restart logic
            if steps_since_improvement > restart_threshold:
                # Reset temperature and randomize
                T = 1.0
                colors = np.random.randint(0, 2, num_triples)
                red_counts, blue_counts = init_counts(colors)
                current_cost = get_cost_from_counts(red_counts, blue_counts)
                steps_since_improvement = 0
                # If current is better than best, update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    best_red_counts = red_counts.copy()
                    best_blue_counts = blue_counts.copy()

        if best_cost == 0:
            return best_colors.tolist()
        return None

    # Try n=35 first, then 34, then 33...
    # We allocate time based on likelihood. 35 is hard, 34 is known.
    
    target_ns = [35, 34, 33, 32]
    
    # Dynamic time allocation
    remaining_time = time_limit - (time.time() - start_time)
    
    for n in target_ns:
        if time.time() - start_time > time_limit - 5:
            break
            
        if n == 35:
            budget = min(45.0, remaining_time * 0.45)
        elif n == 34:
            budget = min(45.0, remaining_time * 0.40)
        else:
            budget = min(20.0, remaining_time * 0.15)
            
        if budget < 5.0:
            continue
            
        result = solve_for_n(n, budget)
        
        if result is not None:
            return {'n': n, 'colors': result}
            
        remaining_time = time_limit - (time.time() - start_time)
        
    # Fallback: return a valid small n
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}
