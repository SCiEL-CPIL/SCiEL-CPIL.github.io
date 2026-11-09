import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)

        # 1. Build Conflict Graph (Adjacency Matrix)
        # Edge (i, j) exists if v_i + v_j is in V (for i != j) OR 2*v_i == v_j
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vi == vj:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1

        degrees = np.sum(adj, axis=1)

        # 2. Greedy Initialization
        def greedy_init():
            rem = np.ones(n, dtype=bool)
            selected = np.zeros(n, dtype=bool)
            rem_indices = list(range(n))
            cur_degrees = degrees.copy()
            
            while rem_indices:
                if time.time() - start_time > DEADLINE - 5:
                    break
                min_idx = 0
                min_val = float('inf')
                for idx in rem_indices:
                    if cur_degrees[idx] < min_val:
                        min_val = cur_degrees[idx]
                        min_idx = idx
                
                selected[min_idx] = True
                new_rem = []
                for idx in rem_indices:
                    if idx == min_idx:
                        continue
                    if adj[min_idx, idx]:
                        pass
                    else:
                        new_rem.append(idx)
                rem_indices = new_rem
            
            return selected

        x = greedy_init()
        
        # Calculate initial energy
        LAMBDA = 10.0
        
        def get_energy(x_vec):
            size = np.sum(x_vec)
            conflicts = np.dot(x_vec, np.dot(adj, x_vec)) / 2.0
            return -size + LAMBDA * conflicts

        current_energy = get_energy(x)
        best_x = x.copy()
        best_energy = current_energy
        
        best_valid_x = np.zeros(n, dtype=bool)
        best_valid_size = 0
        
        if np.dot(x, np.dot(adj, x)) == 0:
            best_valid_x = x.copy()
            best_valid_size = np.sum(x)
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        adj_f = adj.astype(np.float32)
        x_f = x.astype(np.float32)
        
        def get_energy_fast(x_f_vec):
            size = np.sum(x_f_vec)
            conflicts = np.dot(x_f_vec, np.dot(adj_f, x_f_vec)) / 2.0
            return -size + LAMBDA * conflicts
        
        current_energy = get_energy_fast(x_f)
        
        # 3. Simulated Annealing with 2-flip moves
        while T > T_min:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            x_f[i] = 1.0 - x_f[i]
            x_f[j] = 1.0 - x_f[j]
            
            new_energy = get_energy_fast(x_f)
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
                if abs(current_energy + np.sum(x_f)) < 1e-5:
                    size = int(np.sum(x_f))
                    if size > best_valid_size:
                        best_valid_size = size
                        best_valid_x = x_f.copy()
            else:
                x_f[i] = 1.0 - x_f[i]
                x_f[j] = 1.0 - x_f[j]
            
            T *= cooling
            
        if best_valid_size == 0:
             res = [v for v in values if v % 2 != 0]
             return res
        
        res = [values[i] for i in range(n) if best_valid_x[i] == 1.0]
        
        if not is_sum_free(res):
             s = set(res)
             changed = True
             while changed:
                 changed = False
                 for a in list(s):
                     for b in list(s):
                         if a + b in s:
                             s.remove(a)
                             changed = True
                             break
                     if changed:
                         break
             res = list(s)
        
        return res

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    weights = [1, 2, 4]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
             vals = instance(size, seed)
             res = [x for x in vals if x % 2 != 0]
             results.append(res)
             continue
        
        budget = remaining * (weights[i] / total_weight)
        budget = max(budget, 2.0)
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        if not is_sum_free(res):
             res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results
