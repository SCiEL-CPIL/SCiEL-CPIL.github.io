"""
Rewritten program with Structured Initialization and Efficient Local Search.
Focus: Finding a valid 2-coloring for n=34 (R(4,5;3) > 34) using Simulated Annealing
with vectorized updates and restarts.
"""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def build_precomputation(n):
        """
        Precomputes indices for fast violation counting.
        Returns:
        - M: number of triples
        - k4_edges: np.array of shape (N4, 4) with triple indices for each K4
        - k5_edges: np.array of shape (N5, 10) with triple indices for each K5
        - k4_adj: list of lists, k4_adj[i] contains K4 indices containing triple i
        - k5_adj: list of lists, k5_adj[i] contains K5 indices containing triple i
        """
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        N4 = len(k4_list)
        N5 = len(k5_list)
        
        k4_edges = np.zeros((N4, 4), dtype=np.int32)
        k4_adj = [[] for _ in range(M)]
        
        for i, k4 in enumerate(k4_list):
            # 4 triples in a K4
            for j in range(4):
                # Remove k4[j] and sort
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                k4_edges[i, j] = idx
                k4_adj[idx].append(i)
                
        k5_edges = np.zeros((N5, 10), dtype=np.int32)
        k5_adj = [[] for _ in range(M)]
        
        for i, k5 in enumerate(k5_list):
            # 10 triples in a K5
            for j, combo in enumerate(combinations(range(5), 3)):
                t = tuple(sorted([k5[x] for x in combo]))
                idx = triple_to_idx[t]
                k5_edges[i, j] = idx
                k5_adj[idx].append(i)
                
        return M, k4_edges, k5_edges, k4_adj, k5_adj

    def run_sa(n, time_budget, M, k4_edges, k5_edges, k4_adj, k5_adj):
        """
        Runs Simulated Annealing.
        """
        # Initialize colors: 0 (Red), 1 (Blue)
        colors = np.random.randint(0, 2, size=M, dtype=np.int8)
        
        # Initialize counts
        # red_counts[k] = number of red edges in k4_edges[k]
        # blue_counts[k] = number of blue edges in k5_edges[k]
        red_counts = np.sum(1 - colors[k4_edges], axis=1).astype(np.int32)
        blue_counts = np.sum(colors[k5_edges], axis=1).astype(np.int32)
        
        cur_violations = int(np.sum(red_counts == 4)) + int(np.sum(blue_counts == 10))
        
        best_violations = cur_violations
        best_colors = colors.copy()
        
        T = 20.0
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 3000
        kicks = 0
        max_kicks = 20
        
        while time.time() < start_time + time_budget:
            if cur_violations == 0:
                return colors.tolist()
            
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = colors.copy()
                no_improve = 0
                # Reheat slightly
                T = max(T * 1.05, 5.0)
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                kicks += 1
                if kicks > max_kicks:
                    return None  # Restart needed
                
                # Kick: flip 10% of edges
                num_flips = M // 10
                flip_indices = np.random.choice(M, num_flips, replace=False)
                colors[flip_indices] = 1 - colors[flip_indices]
                
                # Recount
                red_counts = np.sum(1 - colors[k4_edges], axis=1).astype(np.int32)
                blue_counts = np.sum(colors[k5_edges], axis=1).astype(np.int32)
                cur_violations = int(np.sum(red_counts == 4)) + int(np.sum(blue_counts == 10))
                
                T = 20.0
                no_improve = 0
                continue
            
            # Single Flip
            idx = random.randint(0, M - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            
            delta_v = 0
            
            # Update K4s
            for k4_idx in k4_adj[idx]:
                old_c = red_counts[k4_idx]
                if old_val == 0: # was Red
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                
                was_v = (old_c == 4)
                is_v = (new_c == 4)
                
                if was_v and not is_v: delta_v -= 1
                elif not was_v and is_v: delta_v += 1
                
                red_counts[k4_idx] = new_c
            
            # Update K5s
            for k5_idx in k5_adj[idx]:
                old_c = blue_counts[k5_idx]
                if old_val == 1: # was Blue
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                
                was_v = (old_c == 10)
                is_v = (new_c == 10)
                
                if was_v and not is_v: delta_v -= 1
                elif not was_v and is_v: delta_v += 1
                
                blue_counts[k5_idx] = new_c
            
            new_v = cur_violations + delta_v
            
            if new_v < cur_violations:
                colors[idx] = new_val
                cur_violations = new_v
            else:
                if T > T_min:
                    # Avoid overflow/underflow issues
                    if delta_v > 0:
                        # If delta is large, exp is 0
                        if delta_v < 50 * T: 
                            prob = np.exp(-delta_v / T)
                            if random.random() < prob:
                                colors[idx] = new_val
                                cur_violations = new_v
                    # else reject
                # else reject
            
            T *= cooling
        
        if best_violations == 0:
            return best_colors.tolist()
        return None

    # Strategy:
    # 1. Precompute for n=34
    # 2. Run SA
    # 3. If fail, try n=33
    
    # n=34
    M, k4_edges, k5_edges, k4_adj, k5_adj = build_precomputation(34)
    
    # Run multiple attempts if time permits
    # Each attempt uses remaining time minus a buffer
    attempts = 0
    max_attempts = 3
    while time.time() < deadline - 10 and attempts < max_attempts:
        budget = deadline - time.time() - 5
        res = run_sa(34, budget, M, k4_edges, k5_edges, k4_adj, k5_adj)
        if res is not None:
            return {'n': 34, 'colors': res}
        attempts += 1
        
    # n=33
    if time.time() < deadline - 5:
        M, k4_edges, k5_edges, k4_adj, k5_adj = build_precomputation(33)
        budget = deadline - time.time() - 2
        res = run_sa(33, budget, M, k4_edges, k5_edges, k4_adj, k5_adj)
        if res is not None:
            return {'n': 33, 'colors': res}
            
    # Fallback: n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
