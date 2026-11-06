import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s for safety
    
    # Target n values to try in descending order
    target_ns = [34, 33, 32, 31, 30, 25, 20, 15, 10]
    
    # We will try to find a valid coloring for each n
    # If we find one for a larger n, we stop and return it.
    # If we run out of time, we return the best valid one found so far.
    # If we find none, we must return a trivially valid one (e.g. n=3).
    
    best_result = None
    best_n = -1
    
    for n in target_ns:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Precompute structures for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Map from edge index to the triple
        # We need to know, for a given K4 or K5, which edge indices are involved.
        
        # Precompute K4s and K5s
        k4s = []
        k4_edges = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            k4s.append(quad)
            k4_edges.append(edges)
            
        k5s = []
        k5_edges = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            k5s.append(quint)
            k5_edges.append(edges)
            
        num_k4 = len(k4_edges)
        num_k5 = len(k5_edges)
        
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        for i, edges in enumerate(k4_edges):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        edge_to_k5s = [[] for _ in range(num_triples)]
        for i, edges in enumerate(k5_edges):
            for e in edges:
                edge_to_k5s[e].append(i)
                
        # Local Search
        # We will run multiple restarts if possible
        max_restarts = 50
        current_time = time.time()
        
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Calculate initial cost
            # This is O(num_k4 * 4 + num_k5 * 10) which is fine for n=34
            # num_k4 = C(34,4) ~ 46k, num_k5 = C(34,5) ~ 278k
            # 46k*4 + 278k*10 ~ 3M ops. Very fast.
            
            def calc_cost(colors):
                cost = 0
                # Check K4s
                for edges in k4_edges:
                    # edges is list of 4 indices
                    if colors[edges[0]] == 0 and colors[edges[1]] == 0 and colors[edges[2]] == 0 and colors[edges[3]] == 0:
                        cost += 1
                # Check K5s
                for edges in k5_edges:
                    # edges is list of 10 indices
                    # Using sum is fast
                    if np.sum(colors[edges]) == 10:
                        cost += 1
                return cost
                
            cost = calc_cost(colors)
            
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            # Simulated Annealing / Hill Climbing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            max_iter_per_restart = 50000
            
            for i in range(max_iter_per_restart):
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                # Pick random edge
                e = random.randint(0, num_triples - 1)
                
                # Calculate delta cost
                delta = 0
                old_color = colors[e]
                new_color = 1 - old_color
                
                # Check K4s containing e
                for k4_idx in edge_to_k4s[e]:
                    edges = k4_edges[k4_idx]
                    # Check if K4 was red
                    was_red = True
                    for ed in edges:
                        if ed == e:
                            if old_color != 0:
                                was_red = False
                                break
                        else:
                            if colors[ed] != 0:
                                was_red = False
                                break
                    if was_red:
                        delta -= 1
                        
                    # Check if K4 becomes red
                    becomes_red = True
                    for ed in edges:
                        if ed == e:
                            if new_color != 0:
                                becomes_red = False
                                break
                        else:
                            if colors[ed] != 0:
                                becomes_red = False
                                break
                    if becomes_red:
                        delta += 1
                        
                # Check K5s containing e
                for k5_idx in edge_to_k5s[e]:
                    edges = k5_edges[k5_idx]
                    # Check if K5 was blue
                    was_blue = True
                    for ed in edges:
                        if ed == e:
                            if old_color != 1:
                                was_blue = False
                                break
                        else:
                            if colors[ed] != 1:
                                was_blue = False
                                break
                    if was_blue:
                        delta -= 1
                        
                    # Check if K5 becomes blue
                    becomes_blue = True
                    for ed in edges:
                        if ed == e:
                            if new_color != 1:
                                becomes_blue = False
                                break
                        else:
                            if colors[ed] != 1:
                                becomes_blue = False
                                break
                    if becomes_blue:
                        delta += 1
                        
                # Accept/Reject
                if delta < 0:
                    colors[e] = new_color
                    cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[e] = new_color
                        cost += delta
                        
                T *= cooling
                if T < T_min:
                    break
                    
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            # If we finished a restart and didn't find 0, we can try another restart
            # But we should also keep track of the best cost found?
            # No, we just need ANY valid coloring.
            
    # If we didn't find a valid coloring for large n, we need to return a valid one.
    # We can easily construct a valid one for small n.
    # For n=1,2,3: any coloring is valid.
    # Let's just return n=3 with all 0s.
    # Or better, try to find one for n=10 quickly?
    # Actually, the loop above tries n=10 last. If it failed, it's because of time.
    # Let's ensure we always have a fallback.
    
    # Fallback: n=3
    triples_3 = list(combinations(range(3), 3))
    return {'n': 3, 'colors': [0] * len(triples_3)}

# EVOLVE-BLOCK-END
