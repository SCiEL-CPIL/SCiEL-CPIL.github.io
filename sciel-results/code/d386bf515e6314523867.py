import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Build Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def is_5_5_free(adj, n):
        for subset in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def get_k4s_i4s(adj, n):
        k4s = []
        i4s = []
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(frozenset(subset))
            if is_indep:
                i4s.append(frozenset(subset))
        return k4s, i4s

    k4s, i4s = get_k4s_i4s(adj, n)
    current_n = n

    def try_add_vertex(adj, k4s, i4s, current_n):
        # For each vertex v, compute:
        #   in_k4_count[v] = number of K4s containing v
        #   in_i4_count[v] = number of I4s containing v
        in_k4_count = [0] * current_n
        in_i4_count = [0] * current_n
        for k in k4s:
            for v in k:
                in_k4_count[v] += 1
        for i in i4s:
            for v in i:
                in_i4_count[v] += 1

        best_S = None
        best_penalty = float('inf')

        for attempt in range(200):
            # Initialize S: vertices with fewer I4s are more likely to be excluded (in V\S)
            # Actually, we want S to intersect every I4, so we want vertices that are in many I4s to be IN S
            # And we want S to avoid containing K4s, so vertices in many K4s should be excluded from S
            S = set()
            for v in range(current_n):
                # Heuristic: include v in S if in_i4_count is high and in_k4_count is low
                prob = in_i4_count[v] / (max(1, in_i4_count[v] + in_k4_count[v]))
                if random.random() < prob:
                    S.add(v)
            
            for _ in range(1000):
                # Check violations
                bad_k4 = None
                for k in k4s:
                    if k.issubset(S):
                        bad_k4 = k
                        break
                
                bad_i4 = None
                if not bad_k4:
                    V_minus_S = set(range(current_n)) - S
                    for i in i4s:
                        if i.issubset(V_minus_S):
                            bad_i4 = i
                            break
                
                if not bad_k4 and not bad_i4:
                    return S
                
                penalty = (1 if bad_k4 else 0) + (1 if bad_i4 else 0)
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_S = S.copy()
                
                if bad_k4:
                    candidates = list(bad_k4 & S)
                    if candidates:
                        # Remove the vertex in bad_k4 with highest in_k4_count (most "dangerous")
                        v_to_remove = max(candidates, key=lambda v: in_k4_count[v])
                        S.remove(v_to_remove)
                    else:
                        break
                elif bad_i4:
                    candidates = list(bad_i4 - S)
                    if candidates:
                        # Add the vertex in bad_i4 with highest in_i4_count
                        v_to_add = max(candidates, key=lambda v: in_i4_count[v])
                        S.add(v_to_add)
                    else:
                        break
        
        # If we found a near-miss, try to repair it
        if best_S is not None and best_penalty == 1:
            # Check if best_S is valid
            valid = True
            for k in k4s:
                if k.issubset(best_S):
                    valid = False
                    break
            if valid:
                V_minus_S = set(range(current_n)) - best_S
                for i in i4s:
                    if i.issubset(V_minus_S):
                        valid = False
                        break
            if valid:
                return best_S
        
        return None

    while current_n < 42 and time.time() < deadline - 15:
        S = try_add_vertex(adj, k4s, i4s, current_n)
        if S is None:
            found = False
            for _ in range(10):
                S = try_add_vertex(adj, k4s, i4s, current_n)
                if S is not None:
                    found = True
                    break
            if not found:
                break
        
        new_idx = current_n
        adj.append(0)
        for v in S:
            adj[v] |= (1 << new_idx)
            adj[new_idx] |= (1 << v)
        
        current_n += 1
        k4s, i4s = get_k4s_i4s(adj, current_n)

    if not is_5_5_free(adj, current_n):
        return 37, [(i, j) for i in range(37) for j in range(i + 1, 37) if (j - i) % 37 in residues]
    
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
                
    return current_n, edges
