import numpy as np
import itertools
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

D = 11

def construct():
    # 1. Generate all vectors
    all_vectors = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(comb, signs):
                v[k] = s
            all_vectors.append(v)
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    support_to_indices = {}
    for i, v in enumerate(all_vectors):
        support = tuple(k for k in range(D) if v[k] != 0)
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
        
    edges = []
    supports = list(support_to_indices.keys())
    
    for i_s, s1 in enumerate(supports):
        vecs1 = support_to_indices[s1]
        for j_s, s2 in enumerate(supports):
            if j_s <= i_s: continue
            
            inter = set(s1) & set(s2)
            inter_size = len(inter)
            
            if inter_size == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                sig_map1 = {}
                for idx1 in vecs1:
                    v1 = all_vectors[idx1]
                    sig = tuple(v1[p] for p in pos_in_s1)
                    if sig not in sig_map1:
                        sig_map1[sig] = []
                    sig_map1[sig].append(idx1)
                    
                for idx2 in support_to_indices[s2]:
                    v2 = all_vectors[idx2]
                    sig2 = tuple(v2[p] for p in pos_in_s2)
                    if sig2 in sig_map1:
                        for idx1 in sig_map1[sig2]:
                            edges.append((idx1, idx2))
            # inter_size == 4 implies s1 == s2, no conflicts between distinct vectors with same support
            # inter_size < 3 implies dot product <= 2, no conflicts

    # 3. Solve ILP
    if not edges:
        return np.array(all_vectors, dtype=np.int64)
        
    m = len(edges)
    n = N
    
    row_indices = []
    col_indices = []
    data = []
    
    for k, (u, v) in enumerate(edges):
        row_indices.append(k)
        col_indices.append(u)
        data.append(1)
        
        row_indices.append(k)
        col_indices.append(v)
        data.append(1)
        
    A = csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
    
    c = np.ones(n)
    
    constraint = LinearConstraint(A, ub=np.ones(m))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status not in [0, 1]:
        # Fallback: Return empty or a simple greedy result
        # To be safe against timeouts or errors, return empty
        return np.array([], dtype=np.int64).reshape(0, D)

    x = res.x
    selected_indices = np.where(x > 0.5)[0]
    
    result_vectors = [all_vectors[i] for i in selected_indices]
    
    if not result_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(result_vectors, dtype=np.int64)
