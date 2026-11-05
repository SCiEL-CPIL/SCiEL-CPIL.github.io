import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute some combinatorial values
    def C(n, k):
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def build_structure(n):
        """Build index structures for K4 and K5 violations."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build K4 list: each K4 is a tuple of 4 triple indices
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        
        # Build K5 list: each K5 is a tuple of 10 triple indices
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
        
        # Build reverse maps: triple_idx -> list of clique indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)
        
        return num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s (all 4 red) and blue K5s (all 10 blue)."""
        red_k4 = 0
        for idxs in k4s:
            if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                red_k4 += 1
        
        blue_k5 = 0
        for idxs in k5s:
            # Check if all 10 are blue (value 1)
            cnt = 0
            for i in idxs:
                cnt += colors[i]
            if cnt == 10:
                blue_k5 += 1
        
        return red_k4, blue_k5

    def heuristic_init(n, num_triples, k4s, k5s, seed=0):
        """
        Initialize with a structured heuristic that minimizes violations.
        Uses a greedy approach: start random, then do a fast greedy repair.
        """
        rng = np.random.RandomState(seed)
        colors = rng.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Build reverse maps for this n
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)
        
        # Greedy repair: repeatedly find a violated clique and flip a random triple in it
        # This is fast and gets us close to zero violations
        max_iters = 5000
        for _ in range(max_iters):
            red_k4, blue_k5 = count_violations(colors, k4s, k5s)
            if red_k4 == 0 and blue_k5 == 0:
                break
            
            # Pick a random violated clique
            if red_k4 > 0 and (blue_k5 == 0 or rng.rand() < red_k4 / (red_k4 + blue_k5)):
                # Pick a random red K4
                while True:
                    k4_idx = rng.randint(len(k4s))
                    idxs = k4s[k4_idx]
                    if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                        break
                # Flip a random triple in this K4 to blue
                t = idxs[rng.randint(4)]
                colors[t] = 1
            else:
                # Pick a random blue K5
                while True:
                    k5_idx = rng.randint(len(k5s))
                    idxs = k5s[k5_idx]
                    cnt = sum(colors[i] for i in idxs)
                    if cnt == 10:
                        break
                # Flip a random triple in this K5 to red
                t = idxs[rng.randint(10)]
                colors[t] = 0
        
        return colors

    def local_search(n, time_limit, seed=0):
        """
        Perform local search with simulated annealing to find a valid coloring.
        """
        num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        # Try multiple initializations
        best_colors = None
        best_cost = float('inf')
        
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            colors = heuristic_init(n, num_triples, k4s, k5s, seed=seed + restart)
            
            red_k4, blue_k5 = count_violations(colors, k4s, k5s)
            current_cost = red_k4 + blue_k5
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Simulated annealing
            temp = 5.0
            cool_rate = 0.9995
            min_temp = 0.1
            
            # Time slice for this restart
            remaining = time_limit - time.time()
            if remaining <= 0:
                break
            step_deadline = time.time() + max(0.05, remaining * 0.9)
            
            while time.time() < step_deadline:
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Update K4s containing t
                for k4_idx in triple_to_k4s[t]:
                    idxs = k4s[k4_idx]
                    if old_color == 0:
                        # Flipping 0 -> 1
                        # Check if it was a red K4
                        if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                            delta_cost -= 1
                    else:
                        # Flipping 1 -> 0
                        # Check if it becomes a red K4 (other 3 must be 0)
                        c0 = colors[idxs[0]] if idxs[0] != t else 0
                        c1 = colors[idxs[1]] if idxs[1] != t else 0
                        c2 = colors[idxs[2]] if idxs[2] != t else 0
                        c3 = colors[idxs[3]] if idxs[3] != t else 0
                        if c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0:
                            delta_cost += 1
                
                # Update K5s containing t
                for k5_idx in triple_to_k5s[t]:
                    idxs = k5s[k5_idx]
                    if old_color == 1:
                        # Flipping 1 -> 0
                        # Check if it was a blue K5
                        cnt = 0
                        for i in idxs:
                            cnt += colors[i]
                        if cnt == 10:
                            delta_cost -= 1
                    else:
                        # Flipping 0 -> 1
                        # Check if it becomes a blue K5 (other 9 must be 1)
                        cnt = 0
                        for i in idxs:
                            if i == t:
                                continue
                            cnt += colors[i]
                        if cnt == 9:
                            delta_cost += 1
                
                new_cost = current_cost + delta_cost
                
                # Acceptance
                if new_cost < current_cost:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif new_cost == current_cost and np.random.rand() < 0.3:
                    colors[t] = new_color
                elif new_cost > current_cost and np.random.rand() < np.exp(-new_cost / max(temp, 0.01)):
                    colors[t] = new_color
                    current_cost = new_cost
                
                temp *= cool_rate
                if temp < min_temp:
                    temp = min_temp
        
        return best_colors, (best_cost == 0)

    # Main: try to find the largest valid n
    best_n = 9
    # Fallback: simple modular coloring for n=9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try high n first
    targets = [34, 33, 32, 31, 30, 28, 25, 20, 15]
    
    # Allocate time: give more time to higher n
    time_alloc = {}
    total_time = deadline - start_time
    # Weights
    weights = {34: 0.35, 33: 0.20, 32: 0.15, 31: 0.10, 30: 0.08, 28: 0.05, 25: 0.04, 20: 0.02, 15: 0.01}
    for n in targets:
        time_alloc[n] = total_time * weights.get(n, 0.01)
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        t_limit = time.time() + time_alloc.get(n, 2.0)
        
        try:
            colors, is_valid = local_search(n, t_limit, seed=42)
            if is_valid:
                best_n = n
                best_colors = colors.tolist()
                if n >= 34:
                    break
        except Exception:
            pass
    
    return {'n': best_n, 'colors': best_colors}
