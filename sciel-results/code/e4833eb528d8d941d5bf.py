import time
import math
import sys
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    # conflicts[i] is a list of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflicts = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                edges.append((i, j))
    
    # If no edges, all nodes are independent
    if not edges:
        return valid_nodes
    
    # Build conflict masks for fast operations
    conflict_masks = [0] * n
    for i in range(n):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    # Try to solve using MILP for Minimum Vertex Cover
    # Minimize sum(x_i) subject to x_i + x_j >= 1 for all (i,j) in E
    # x_i in {0, 1}
    best_ilp_mask = None
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        m = n
        
        # Objective: minimize sum(x_i)
        c = np.ones(m)
        
        num_edges = len(edges)
        
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (u, v) in enumerate(edges):
            row_indices.extend([idx, idx])
            col_indices.extend([u, v])
            data.extend([-1, -1])
        
        A = csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, m))
        
        constraints = LinearConstraint(A, lb=-np.inf, ub=-np.ones(num_edges))
        
        integrality = np.ones(m)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 60, "presolve": True}
        )
        
        if res.status == 0:
            cover_mask = res.x
            independent_set_indices = []
            for i in range(m):
                if cover_mask[i] < 0.5:
                    independent_set_indices.append(valid_nodes[i])
            
            if independent_set_indices:
                valid = True
                for i in range(len(independent_set_indices)):
                    for j in range(i, len(independent_set_indices)):
                        s = independent_set_indices[i] + independent_set_indices[j]
                        r = math.isqrt(s)
                        if r * r == s:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Convert to mask
                    best_ilp_mask = 0
                    for node in independent_set_indices:
                        idx = node_index[node]
                        best_ilp_mask |= (1 << idx)
            
    except Exception as e:
        pass
    
    # Greedy MIS as initial solution
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflict_masks[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflict_masks[best_v])
        return chosen
    
    best_mask = greedy_mis()
    best_size = bin(best_mask).count('1')
    
    # If ILP gave a better result, use it
    if best_ilp_mask is not None:
        ilp_size = bin(best_ilp_mask).count('1')
        if ilp_size > best_size:
            best_mask = best_ilp_mask
            best_size = ilp_size
    
    # Local search: try adding vertices
    def local_search_add(mask):
        current = mask
        current_size = bin(current).count('1')
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in range(n):
                if not (current & (1 << v)):
                    if not (conflict_masks[v] & current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
        return current, current_size
    
    best_mask, best_size = local_search_add(best_mask)
    
    # Simulated Annealing with swap moves
    # Move: remove vertex v from set, add vertex u not in set
    # Accept if size increases, or with probability exp(-delta/T) if size decreases
    # We focus on size-preserving swaps to escape local optima, then try to add
    
    def simulated_annealing(initial_mask, initial_size):
        if time.time() > deadline - 15:
            return initial_mask, initial_size
        
        current = initial_mask
        current_size = initial_size
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        num_iterations = 50000
        
        # Precompute for speed
        rng = random.Random(42)
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            # Temperature
            progress = it / num_iterations
            T = T_start * (T_end / T_start) ** progress
            
            # Choose a random vertex in the set to potentially remove
            set_bits = []
            temp = current
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                set_bits.append(v)
                temp ^= lsb
            
            if not set_bits:
                break
            
            v = rng.choice(set_bits)
            v_bit = 1 << v
            
            # Find candidates to add: vertices not in set that don't conflict with (current without v)
            current_without_v = current & ~v_bit
            candidates = []
            for u in range(n):
                if not (current & (1 << u)):
                    if not (conflict_masks[u] & current_without_v):
                        candidates.append(u)
            
            if not candidates:
                # Can't do a swap, try just removing v (worse) or skip
                continue
            
            u = rng.choice(candidates)
            u_bit = 1 << u
            
            new_mask = current_without_v | u_bit
            new_size = current_size  # swap preserves size
            
            # After swap, try to add more vertices
            final_mask = new_mask
            final_size = new_size
            for w in range(n):
                if not (final_mask & (1 << w)):
                    if not (conflict_masks[w] & final_mask):
                        final_mask |= (1 << w)
                        final_size += 1
            
            # Decide whether to accept
            if final_size > current_size:
                current = final_mask
                current_size = final_size
            else:
                # Accept with probability exp(-(current_size - final_size)/T)
                delta = current_size - final_size
                if delta == 0:
                    # Same size, accept with some probability to explore
                    if rng.random() < 0.5:
                        current = final_mask
                else:
                    prob = math.exp(-delta / max(T, 0.001))
                    if rng.random() < prob:
                        current = final_mask
                        current_size = final_size
        
        # Final local search
        current, current_size = local_search_add(current)
        return current, current_size
    
    sa_mask, sa_size = simulated_annealing(best_mask, best_size)
    if sa_size > best_size:
        best_mask = sa_mask
        best_size = sa_size
    
    # Try multiple SA restarts with different seeds if time permits
    if time.time() < deadline - 30:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 15:
                break
            rng = random.Random(seed)
            # Random restart: start from a random feasible set
            random_mask = 0
            for i in range(n):
                if rng.random() < 0.5:
                    if not (conflict_masks[i] & random_mask):
                        random_mask |= (1 << i)
            random_mask, random_size = local_search_add(random_mask)
            
            sa_mask2, sa_size2 = simulated_annealing(random_mask, random_size)
            if sa_size2 > best_size:
                best_mask = sa_mask2
                best_size = sa_size2
    
    # Final B&B with time limit to try to prove optimality or find better
    if time.time() < deadline - 20:
        def bnb(available_mask, current_size, chosen_mask, best):
            if time.time() > deadline - 5:
                return
            if current_size + bin(available_mask).count('1') <= best[0]:
                return
            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                    best[1] = chosen_mask
                return
            
            # Pick vertex with highest degree in available
            best_v = -1
            best_deg = -1
            temp = available_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflict_masks[v] & available_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return
            
            v_bit = 1 << best_v
            
            # Include
            new_avail = available_mask & ~(v_bit | conflict_masks[best_v])
            bnb(new_avail, current_size + 1, chosen_mask | v_bit, best)
            
            # Exclude
            new_avail2 = available_mask & ~v_bit
            bnb(new_avail2, current_size, chosen_mask, best)
        
        best = [best_size, best_mask]
        all_mask = (1 << n) - 1
        bnb(all_mask, 0, 0, best)
        
        if best[0] > best_size:
            best_size = best[0]
            best_mask = best[1]
    
    best_mask = best[1] if 'best_mask' in dir() and best_mask != best[1] else best_mask
    # Actually let me fix this - the bnb updates best[1] which is the mask
    # Let me re-extract
    
    # Re-extract from best_mask which should be the final best
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
        
    return result_nodes
