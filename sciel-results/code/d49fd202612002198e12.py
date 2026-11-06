"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def build_constraint_data(n):
        """Build all constraint data for a given n."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        triple_index = {}
        for i, t in enumerate(triples):
            triple_index[t] = i
        
        k4_triple_indices = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idxs = [
                triple_index[(a, b, c)],
                triple_index[(a, b, d)],
                triple_index[(a, c, d)],
                triple_index[(b, c, d)]
            ]
            k4_triple_indices.append(idxs)
        
        k5_triple_indices = []
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            idxs = [
                triple_index[(a, b, c)],
                triple_index[(a, b, d)],
                triple_index[(a, b, e)],
                triple_index[(a, c, d)],
                triple_index[(a, c, e)],
                triple_index[(a, d, e)],
                triple_index[(b, c, d)],
                triple_index[(b, c, e)],
                triple_index[(b, d, e)],
                triple_index[(c, d, e)]
            ]
            k5_triple_indices.append(idxs)
        
        num_k4 = len(k4_triple_indices)
        num_k5 = len(k5_triple_indices)
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k4_i, idxs in enumerate(k4_triple_indices):
            for tri_idx in idxs:
                triple_to_k4s[tri_idx].append(k4_i)
        
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k5_i, idxs in enumerate(k5_triple_indices):
            for tri_idx in idxs:
                triple_to_k5s[tri_idx].append(k5_i)
        
        return {
            'triples': triples,
            'num_triples': num_triples,
            'triple_index': triple_index,
            'k4_triple_indices': k4_triple_indices,
            'k5_triple_indices': k5_triple_indices,
            'num_k4': num_k4,
            'num_k5': num_k5,
            'triple_to_k4s': triple_to_k4s,
            'triple_to_k5s': triple_to_k5s
        }
    
    def compute_penalty(colors, data):
        """Compute the penalty (number of violations)."""
        num_k4 = data['num_k4']
        num_k5 = data['num_k5']
        k4_triple_indices = data['k4_triple_indices']
        k5_triple_indices = data['k5_triple_indices']
        
        penalty = 0
        for idxs in k4_triple_indices:
            if all(colors[i] == 0 for i in idxs):
                penalty += 1
        for idxs in k5_triple_indices:
            if all(colors[i] == 1 for i in idxs):
                penalty += 1
        return penalty
    
    def local_search(colors, data, max_iters=5000):
        """First-improvement local search."""
        num_triples = data['num_triples']
        triple_to_k4s = data['triple_to_k4s']
        triple_to_k5s = data['triple_to_k5s']
        k4_triple_indices = data['k4_triple_indices']
        k5_triple_indices = data['k5_triple_indices']
        num_k4 = data['num_k4']
        num_k5 = data['num_k5']
        
        # Initialize counts
        red_counts_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_counts_k5 = np.zeros(num_k5, dtype=np.int8)
        
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_counts_k4[k4_i] = sum(1 for idx in idxs if colors[idx] == 0)
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_counts_k5[k5_i] = sum(1 for idx in idxs if colors[idx] == 1)
        
        current_penalty = np.sum(red_counts_k4 == 4) + np.sum(blue_counts_k5 == 10)
        best_penalty = current_penalty
        best_colors = colors[:]
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            for t in range(num_triples):
                if time.time() > deadline:
                    return best_penalty, best_colors
                
                old_color = colors[t]
                delta = 0
                
                if old_color == 0:
                    for k4_i in triple_to_k4s[t]:
                        if red_counts_k4[k4_i] == 4:
                            delta -= 1
                        red_counts_k4[k4_i] -= 1
                    for k5_i in triple_to_k5s[t]:
                        if blue_counts_k5[k5_i] == 9:
                            delta += 1
                        blue_counts_k5[k5_i] += 1
                else:
                    for k4_i in triple_to_k4s[t]:
                        if red_counts_k4[k4_i] == 3:
                            delta += 1
                        red_counts_k4[k4_i] += 1
                    for k5_i in triple_to_k5s[t]:
                        if blue_counts_k5[k5_i] == 10:
                            delta -= 1
                        blue_counts_k5[k5_i] -= 1
                
                new_penalty = current_penalty + delta
                
                if delta < 0:
                    colors[t] = 1 - old_color
                    current_penalty = new_penalty
                    improved = True
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors[:]
                        if best_penalty == 0:
                            return 0, best_colors
                else:
                    # Revert
                    if old_color == 0:
                        for k4_i in triple_to_k4s[t]:
                            red_counts_k4[k4_i] += 1
                        for k5_i in triple_to_k5s[t]:
                            blue_counts_k5[k5_i] -= 1
                    else:
                        for k4_i in triple_to_k4s[t]:
                            red_counts_k4[k4_i] -= 1
                        for k5_i in triple_to_k5s[t]:
                            blue_counts_k5[k5_i] += 1
        
        return best_penalty, best_colors
    
    def simulated_annealing(colors, data, max_iters=20000):
        """Simulated annealing with efficient delta calculation."""
        num_triples = data['num_triples']
        triple_to_k4s = data['triple_to_k4s']
        triple_to_k5s = data['triple_to_k5s']
        k4_triple_indices = data['k4_triple_indices']
        k5_triple_indices = data['k5_triple_indices']
        num_k4 = data['num_k4']
        num_k5 = data['num_k5']
        
        red_counts_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_counts_k5 = np.zeros(num_k5, dtype=np.int8)
        
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_counts_k4[k4_i] = sum(1 for idx in idxs if colors[idx] == 0)
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_counts_k5[k5_i] = sum(1 for idx in idxs if colors[idx] == 1)
        
        current_penalty = np.sum(red_counts_k4 == 4) + np.sum(blue_counts_k5 == 10)
        best_penalty = current_penalty
        best_colors = colors[:]
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.001
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            t = random.randrange(num_triples)
            old_color = colors[t]
            delta = 0
            
            if old_color == 0:
                for k4_i in triple_to_k4s[t]:
                    if red_counts_k4[k4_i] == 4:
                        delta -= 1
                    red_counts_k4[k4_i] -= 1
                for k5_i in triple_to_k5s[t]:
                    if blue_counts_k5[k5_i] == 9:
                        delta += 1
                    blue_counts_k5[k5_i] += 1
            else:
                for k4_i in triple_to_k4s[t]:
                    if red_counts_k4[k4_i] == 3:
                        delta += 1
                    red_counts_k4[k4_i] += 1
                for k5_i in triple_to_k5s[t]:
                    if blue_counts_k5[k5_i] == 10:
                        delta -= 1
                    blue_counts_k5[k5_i] -= 1
            
            new_penalty = current_penalty + delta
            
            if delta < 0 or (delta == 0 and random.random() < 0.3) or (delta > 0 and random.random() < math.exp(-delta / T)):
                colors[t] = 1 - old_color
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_colors = colors[:]
                    if best_penalty == 0:
                        return 0, best_colors
            else:
                if old_color == 0:
                    for k4_i in triple_to_k4s[t]:
                        red_counts_k4[k4_i] += 1
                    for k5_i in triple_to_k5s[t]:
                        blue_counts_k5[k5_i] -= 1
                else:
                    for k4_i in triple_to_k4s[t]:
                        red_counts_k4[k4_i] -= 1
                    for k5_i in triple_to_k5s[t]:
                        blue_counts_k5[k5_i] += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        return best_penalty, best_colors
    
    def generate_algebraic_init(n, num_triples, seed_offset=0):
        """Generate initialization using modular arithmetic structure."""
        colors = [0] * num_triples
        rng = random.Random(42 + seed_offset)
        
        for i, (a, b, c) in enumerate(combinations(range(n), 3)):
            # Try different modular structures
            mod_val = (a * 7 + b * 11 + c * 13 + seed_offset) % 17
            if mod_val < 8:
                colors[i] = 0
            else:
                colors[i] = 1
        
        # Add some randomization
        for _ in range(num_triples // 10):
            idx = rng.randrange(num_triples)
            colors[idx] = 1 - colors[idx]
        
        return colors
    
    def verify_coloring(n, colors):
        """Verify that a coloring is valid."""
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if all(colors[triple_index[(a, b, c)]] == 0 for _ in [0]) and \
               colors[triple_index[(a, b, c)]] == 0 and \
               colors[triple_index[(a, b, d)]] == 0 and \
               colors[triple_index[(a, c, d)]] == 0 and \
               colors[triple_index[(b, c, d)]] == 0:
                return False
        
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            triples_in_quint = [
                (a, b, c), (a, b, d), (a, b, e),
                (a, c, d), (a, c, e), (a, d, e),
                (b, c, d), (b, c, e), (b, d, e),
                (c, d, e)
            ]
            if all(colors[triple_index[t]] == 1 for t in triples_in_quint):
                return False
        
        return True
    
    # Try n=34 first with multiple strategies
    target_n = 34
    data_34 = build_constraint_data(target_n)
    num_triples_34 = data_34['num_triples']
    
    best_penalty = float('inf')
    best_colors = None
    
    # Strategy 1: Algebraic initialization + local search + SA
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        colors = generate_algebraic_init(target_n, num_triples_34, seed_offset=attempt * 7)
        
        # First do local search
        penalty, colors = local_search(colors, data_34, max_iters=3000)
        
        # Then SA
        penalty, colors = simulated_annealing(colors, data_34, max_iters=15000)
        
        if penalty < best_penalty:
            best_penalty = penalty
            best_colors = colors[:]
        
        if best_penalty == 0:
            break
    
    if best_penalty == 0 and best_colors is not None:
        if verify_coloring(target_n, best_colors):
            return {'n': target_n, 'colors': best_colors}
    
    # Strategy 2: Random initialization with SA
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        colors = [random.randint(0, 1) for _ in range(num_triples_34)]
        penalty, colors = simulated_annealing(colors, data_34, max_iters=20000)
        
        if penalty < best_penalty:
            best_penalty = penalty
            best_colors = colors[:]
        
        if best_penalty == 0:
            break
    
    if best_penalty == 0 and best_colors is not None:
        if verify_coloring(target_n, best_colors):
            return {'n': target_n, 'colors': best_colors}
    
    # Try smaller n values
    for n in range(33, 25, -1):
        if time.time() > deadline - 10:
            break
        
        data_n = build_constraint_data(n)
        num_triples_n = data_n['num_triples']
        
        # Try a few initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            colors = generate_algebraic_init(n, num_triples_n, seed_offset=attempt * 13)
            penalty, colors = local_search(colors, data_n, max_iters=2000)
            penalty, colors = simulated_annealing(colors, data_n, max_iters=10000)
            
            if penalty == 0:
                if verify_coloring(n, colors):
                    return {'n': n, 'colors': colors}
        
        # Random init
        colors = [random.randint(0, 1) for _ in range(num_triples_n)]
        penalty, colors = simulated_annealing(colors, data_n, max_iters=15000)
        if penalty == 0:
            if verify_coloring(n, colors):
                return {'n': n, 'colors': colors}
    
    # Final fallback: n=9 with simple modular coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
