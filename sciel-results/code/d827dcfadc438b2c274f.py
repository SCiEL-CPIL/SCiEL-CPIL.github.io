import numpy as np
import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s margin

    n = 16
    # Generate all triples
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)
    
    # Generate constraints
    constraint_to_triples = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            
                            for j in range(1, 6):
                                pair1 = (verts[0], verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                
                                pairings = [
                                    [(remaining[0], remaining[1]), (remaining[2], remaining[3])],
                                    [(remaining[0], remaining[2]), (remaining[1], remaining[3])],
                                    [(remaining[0], remaining[3]), (remaining[1], remaining[2])]
                                ]
                                for p in pairings:
                                    pair2 = p[0]
                                    pair3 = p[1]
                                    
                                    t_indices = []
                                    for a in pair1:
                                        for b in pair2:
                                            for c in pair3:
                                                t_idx = get_idx(a, b, c)
                                                t_indices.append(t_idx)
                                                triple_to_constraints[t_idx].append(len(constraint_to_triples))
                                    
                                    constraint_to_triples.append(t_indices)
                                    
    num_constraints = len(constraint_to_triples)
    
    # Precompute constraint overlap
    constraint_overlap = np.zeros(num_constraints, dtype=np.int32)
    for c in range(num_constraints):
        for t in constraint_to_triples[c]:
            constraint_overlap[c] += len(triple_to_constraints[t])
    
    # Sort constraints by overlap (descending)
    constraint_order = np.argsort(-constraint_overlap)
    
    best_size = -1
    best_present = None
    
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            break
            
        random.seed(restart * 12345 + 67890)
        
        present = np.ones(num_triples, dtype=bool)
        counts = np.full(num_constraints, 8, dtype=np.int32)
        
        is_violated = np.ones(num_constraints, dtype=bool)
        violated_list = list(range(num_constraints))
        
        viol_deg = np.zeros(num_triples, dtype=np.int32)
        for c in violated_list:
            for t in constraint_to_triples[c]:
                viol_deg[t] += 1
        
        removed_count = 0
        removed_triples = []
        
        # Phase 1: Improved Greedy removal with constraint-aware scoring
        while violated_list:
            if time.time() > deadline - 8:
                break
                
            idx = random.randint(0, len(violated_list) - 1)
            c = violated_list[idx]
            
            if not is_violated[c]:
                violated_list[idx] = violated_list[-1]
                violated_list.pop()
                continue
                
            # Find best triple to remove using enhanced scoring
            best_t = -1
            best_score = -1
            for t in constraint_to_triples[c]:
                if present[t]:
                    # Score based on violation degree and constraint overlap
                    score = float(viol_deg[t])
                    # Bonus for triples involved in many violated constraints
                    score += 0.1 * sum(1 for c2 in triple_to_constraints[t] if is_violated[c2])
                    # Small random perturbation
                    score += random.random() * 0.3
                    if score > best_score:
                        best_score = score
                        best_t = t
                        
            if best_t == -1:
                break
                
            present[best_t] = False
            removed_count += 1
            removed_triples.append(best_t)
            
            for c2 in triple_to_constraints[best_t]:
                counts[c2] -= 1
                if counts[c2] == 7:
                    if is_violated[c2]:
                        is_violated[c2] = False
                        for t2 in constraint_to_triples[c2]:
                            viol_deg[t2] -= 1
        
        # Phase 2: Simulated Annealing with adaptive parameters
        M = 500  # Reduced penalty to allow more exploration
        T_start = 30.0
        T_end = 0.5
        cooling = 0.998
        T = T_start
        
        steps = 0
        max_sa_steps = 30000
        no_improve_count = 0
        best_local_size = num_triples - removed_count
        
        while T > T_end and steps < max_sa_steps:
            if time.time() > deadline - 2:
                break
            
            steps += 1
            
            # Adaptive move ratio: more adds when many removed, more removes when few
            add_prob = 0.6 + 0.2 * min(1.0, len(removed_triples) / 500.0)
            r = random.random()
            
            if r < add_prob:
                # Try Add
                if not removed_triples:
                    continue
                t_idx = random.randint(0, len(removed_triples) - 1)
                t = removed_triples[t_idx]
                
                if present[t]: 
                    continue
                
                # Calculate delta
                delta_viol = 0
                for c2 in triple_to_constraints[t]:
                    if counts[c2] == 7:
                        delta_viol += 1
                
                delta_size = 1
                delta_E = -delta_size + M * delta_viol
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    present[t] = True
                    removed_triples[t_idx] = removed_triples[-1]
                    removed_triples.pop()
                    removed_count -= 1
                    for c2 in triple_to_constraints[t]:
                        counts[c2] += 1
                        if counts[c2] == 8:
                            if not is_violated[c2]:
                                is_violated[c2] = True
                                for t2 in constraint_to_triples[c2]:
                                    viol_deg[t2] += 1
                    
                    current_size = num_triples - removed_count
                    if current_size > best_local_size:
                        best_local_size = current_size
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                # Try Remove
                t = random.randint(0, num_triples - 1)
                if not present[t]:
                    continue
                
                # Calculate delta
                delta_viol = 0
                for c2 in triple_to_constraints[t]:
                    if counts[c2] == 8:
                        delta_viol -= 1
                
                delta_size = -1
                delta_E = 1 - M * delta_viol  # Corrected: delta_E = +1 - M*delta_viol
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    present[t] = False
                    removed_count += 1
                    removed_triples.append(t)
                    for c2 in triple_to_constraints[t]:
                        counts[c2] -= 1
                        if counts[c2] == 7:
                            if is_violated[c2]:
                                is_violated[c2] = False
                                for t2 in constraint_to_triples[c2]:
                                    viol_deg[t2] -= 1
                    
                    no_improve_count += 1
            
            T *= cooling
            
            # Adaptive cooling: speed up if stuck
            if no_improve_count > 500:
                T *= 0.99
                no_improve_count = 0
        
        # Phase 3: Final polish - try to add back any triple that doesn't violate
        if time.time() < deadline - 1:
            for t in removed_triples[:]:
                if present[t]:
                    continue
                delta_viol = 0
                for c2 in triple_to_constraints[t]:
                    if counts[c2] == 7:
                        delta_viol += 1
                if delta_viol == 0:
                    present[t] = True
                    removed_triples.remove(t)
                    removed_count -= 1
                    for c2 in triple_to_constraints[t]:
                        counts[c2] += 1
                        if counts[c2] == 8:
                            if not is_violated[c2]:
                                is_violated[c2] = True
                                for t2 in constraint_to_triples[c2]:
                                    viol_deg[t2] += 1
        
        # Calculate final size
        current_size = num_triples - removed_count
        if current_size > best_size:
            best_size = current_size
            best_present = present.copy()
    
    if best_size == -1:
        return []
    
    result = []
    for i in range(num_triples):
        if best_present[i]:
            result.append(list(triples[i]))
            
    return result
