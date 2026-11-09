import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145
    n = 20
    
    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def is_valid_addition(adj_bits, u, v):
        """
        Check if adding edge (u, v) creates a K3,3.
        Assumes (u, v) is NOT currently in the graph.
        """
        Nu = adj_bits[u]
        Nv = adj_bits[v]
        
        # Type 1: u, v in same partition.
        # Need w != u, v such that |Nu ∩ Nv ∩ Nw| >= 3
        # Note: Nu and Nv do not contain u or v respectively.
        # Intersection Nu ∩ Nv contains common neighbors.
        # We need to check if any w has >= 3 neighbors in Nu ∩ Nv.
        # Actually, the condition is |N(u) ∩ N(v) ∩ N(w)| >= 3.
        # Since u, v not in N(u) ∩ N(v), and w not in N(w), 
        # we just check popcount of (Nu & Nv & Nw) >= 3.
        
        common_uv = Nu & Nv
        # If common_uv has < 3 bits, no w can have 3 common neighbors with u,v.
        if common_uv.bit_count() < 3:
             # Wait, w must be connected to those 3.
             # If common_uv has < 3 bits, it's impossible.
             pass
        else:
             for w in range(n):
                 if w == u or w == v:
                     continue
                 if (common_uv & adj_bits[w]).bit_count() >= 3:
                     return False

        # Type 2: u in A, v in B.
        # Need w1, w2 in N(v)\{u} and x1, x2 in N(u)\{v}
        # such that w_i connected to x_j for all i,j.
        # x1, x2 must be in N(u) ∩ N(v).
        # Let C = N(u) ∩ N(v).
        # We need to pick x1, x2 from C.
        # Then w1, w2 must be in N(x1) ∩ N(x2) ∩ N(v) \ {u}.
        # And |{w1, w2}| >= 2.
        
        C = Nu & Nv
        # Get list of vertices in C
        verts_C = []
        mask = C
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts_C.append(idx)
            mask ^= lsb
            
        if len(verts_C) < 2:
            return True
            
        for i in range(len(verts_C)):
            for j in range(i+1, len(verts_C)):
                x1 = verts_C[i]
                x2 = verts_C[j]
                # Find w candidates: neighbors of x1, x2, and v.
                # w must be in N(x1) ∩ N(x2) ∩ N(v).
                # And w != u.
                w_mask = adj_bits[x1] & adj_bits[x2] & Nv
                w_mask &= ~(1 << u) # exclude u
                if w_mask.bit_count() >= 2:
                    return False
                    
        return True

    def greedy_maximize(adj_bits, time_limit):
        """
        Greedily add edges in random order if valid.
        """
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj_bits[i] >> j) & 1:
                    missing.append((i, j))
                    
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > time_limit:
                break
            if is_valid_addition(adj_bits, u, v):
                add_edge(adj_bits, u, v)

    # Initialize empty
    adj_bits = [0] * n
    best_adj = adj_bits[:]
    best_edges = 0
    
    # Initial greedy
    greedy_maximize(adj_bits, deadline)
    best_adj = adj_bits[:]
    best_edges = count_edges(adj_bits)
    
    # Iterated Local Search / Random Restart with Kicks
    while time.time() < deadline - 5:
        current_adj = best_adj[:]
        
        # Try to improve current_adj
        # 1. Try adding missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (current_adj[i] >> j) & 1:
                    missing.append((i, j))
        
        if not missing:
            break # Complete graph (unlikely)
            
        random.shuffle(missing)
        improved = False
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(current_adj, u, v):
                add_edge(current_adj, u, v)
                improved = True
                # Re-shuffle remaining? Or just continue?
                # Continuing is fine, as we're iterating a shuffled list.
                # But new edges might open up other edges? No, adding edges only restricts future additions.
                # So if we can't add now, we can't add later unless we remove something.
        
        if count_edges(current_adj) > best_edges:
            best_adj = current_adj[:]
            best_edges = count_edges(current_adj)
            
        # Kick: Remove some random edges to escape local optimum
        # Remove k random edges
        k = random.randint(2, 5)
        existing = []
        for i in range(n):
            for j in range(i+1, n):
                if (current_adj[i] >> j) & 1:
                    existing.append((i, j))
                    
        if existing:
            to_remove = random.sample(existing, min(k, len(existing)))
            for u, v in to_remove:
                remove_edge(current_adj, u, v)
                
            # Re-greedy
            greedy_maximize(current_adj, deadline)
            
            if count_edges(current_adj) > best_edges:
                best_adj = current_adj[:]
                best_edges = count_edges(current_adj)

    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
