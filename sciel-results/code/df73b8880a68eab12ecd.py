# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time
import numpy as np

def _jc_distance_matrix(sequences):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0]) if n > 0 else 0
    
    dist_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            s1 = np.frombuffer(seq_list[i], dtype=np.uint8)
            s2 = np.frombuffer(seq_list[j], dtype=np.uint8)
            mismatches = np.sum(s1 != s2)
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_dict[(i, j)] = d
            dist_dict[(j, i)] = d
    return labels, dist_dict

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 150 # 150s safety margin
    
    labels, dist_dict = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 3:
        if n == 0: return ""
        if n == 1: return f"({labels[0]});"
        if n == 2: return f"({labels[0]},{labels[1]});"
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    nodes = list(range(n))
    D = dist_dict
    trees = {i: labels[i] for i in range(n)}
    next_id = -1
    
    while len(nodes) > 3:
        m = len(nodes)
        r = {node: 0.0 for node in nodes}
        for i in nodes:
            for j in nodes:
                if i != j:
                    r[i] += D[(i, j)]
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for idx_i, i in enumerate(nodes):
            for j in nodes[idx_i + 1:]:
                criterion = (m - 2) * D[(i, j)] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        new_node = next_id
        next_id -= 1
        trees[new_node] = f"({trees[i]},{trees[j]})"
        
        for k in nodes:
            if k == i or k == j:
                continue
            D[(new_node, k)] = 0.5 * (D[(i, k)] + D[(j, k)] - D[(i, j)])
            D[(k, new_node)] = D[(new_node, k)]
            
        nodes.remove(i)
        nodes.remove(j)
        nodes.append(new_node)
        
    return "(" + ",".join(trees[x] for x in nodes) + ");"
# EVOLVE-BLOCK-END
