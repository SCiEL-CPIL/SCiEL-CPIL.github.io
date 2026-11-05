import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n = 20
    
    def check_4cycle_violation(edge_dict, new_edge):
        """Check if adding new_edge creates a 4-cycle violation.
        edge_dict: {(u,v): color} for u<v, plus (v,u): color
        new_edge: (u, v, color)
        Returns True if violation exists.
        """
        u, v, c = new_edge
        
        # A 4-cycle involving new edge (u,v) has form u-a-b-v-u or u-b-a-v-u
        # i.e., two paths of length 3 between u and v
        # For each pair (a,b) where a-b is an edge, check if u-a, a-b, b-v form path
        # and u-v closes it, OR u-b, b-a, a-v
        
        # More precisely: 4-cycle u,a,b,v means edges (u,a),(a,b),(b,v),(v,u)
        # We need to check all such a,b where (a,b) is in edge_dict
        # and (u,a) and (b,v) are in edge_dict (or new edge)
        
        # Get neighbors of u and v from existing edges
        u_neighbors = {}
        v_neighbors = {}
        for (a, b), col in edge_dict.items():
            if a < b:
                if a == u:
                    u_neighbors[b] = col
                elif b == u:
                    u_neighbors[a] = col
                if a == v:
                    v_neighbors[b] = col
                elif b == v:
                    v_neighbors[a] = col
        
        # Add new edge's contribution
        u_neighbors[v] = c
        v_neighbors[u] = c
        
        # For 4-cycle u-a-b-v: need (u,a) in u_neighbors, (a,b) in edge_dict, (b,v) in v_neighbors
        # and the 4 colours distinct
        for a, col_ua in u_neighbors.items():
            if a == v:
                continue
            # Find all b such that (a,b) is an edge
            for (x, y), col_ab in edge_dict.items():
                if x < y:
                    if x == a:
                        b = y
                        if b in v_neighbors:
                            col_bv = v_neighbors[b]
                            if len({col_ua, col_ab, col_bv, c}) < 4:
                                return True
                    elif y == a:
                        b = x
                        if b in v_neighbors:
                            col_bv = v_neighbors[b]
                            if len({col_ua, col_ab, col_bv, c}) < 4:
                                return True
        return False
    
    # Find a working bipartite colouring formula
    def check_bipartite_formula(a, b, mod):
        for i1 in range(10):
            for i2 in range(i1+1, 10):
                for j1 in range(10):
                    for j2 in range(j1+1, 10):
                        c11 = (a*i1 + b*j1) % mod
                        c21 = (a*i2 + b*j1) % mod
                        c22 = (a*i2 + b*j2) % mod
                        c12 = (a*i1 + b*j2) % mod
                        if len({c11, c21, c22, c12}) < 4:
                            return False
        return True
    
    best_formula = None
    for mod in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        for a in range(1, mod):
            for b in range(1, mod):
                if check_bipartite_formula(a, b, mod):
                    best_formula = (a, b, mod)
                    break
            if best_formula:
                break
        if best_formula:
            break
    
    if best_formula:
        a, b, mod = best_formula
    else:
        a, b, mod = 1, 1, 10
    
    # Build K_{10,10}
    edges = []
    edge_dict = {}
    for i in range(10):
        for j in range(10):
            color = (a*i + b*j) % mod
            u, v = i, 10+j
            edges.append([u, v, color])
            edge_dict[(u, v)] = color
            edge_dict[(v, u)] = color
    
    # Verify base
    def full_verify(edge_list):
        edge_d = {}
        for u, v, c in edge_list:
            edge_d[(u, v)] = c
            edge_d[(v, u)] = c
        for u, v, c in edge_list:
            if check_4cycle_violation(edge_d, (u, v, c)):
                return False
        return True
    
    if not full_verify(edges):
        return [[0, v, 0] for v in range(1, 20)]
    
    # Try to add intra-partition edges
    for partition in [range(10), range(10, 20)]:
        for i_idx in range(len(partition)):
            for j_idx in range(i_idx+1, len(partition)):
                if time.time() > DEADLINE:
                    return edges
                i, j = partition[i_idx], partition[j_idx]
                for color in range(20):
                    if check_4cycle_violation(edge_dict, (i, j, color)):
                        continue
                    # Accept
                    edges.append([i, j, color])
                    edge_dict[(i, j)] = color
                    edge_dict[(j, i)] = color
                    break
    
    return edges
