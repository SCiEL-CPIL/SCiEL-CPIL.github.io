import time
import random
import math
import numpy as np
import sys
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

def construct():
    """
    Maximize edges in a K2,3-free graph on 20 vertices.
    
    Strategy:
    1. Start from multiple random greedy initializations
    2. Apply aggressive local search with swap operations
    3. Use simulated annealing with structured moves
    4. Final systematic polish pass
    """
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    def can_add_edge_fast(adj, u, v):
        """Check if adding edge (u,v) maintains K2,3-free property."""
        if adj[u] & (1 << v):
            return True
        
        # Check: common neighbors of u and v must be <= 2
        common_uv = adj[u] & adj[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # For each w in N(u) \ {v}, common(w, v) must be <= 1
        neighbors_u = adj[u] & ~(1 << v)
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_wv = adj[w] & adj[v]
            if bin(common_wv).count('1') > 1:
                return False
        
        # For each w in N(v) \ {u}, common(w, u) must be <= 1
        neighbors_v = adj[v] & ~(1 << u)
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_wu = adj[w] & adj[u]
            if bin(common_wu).count('1') > 1:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                temp -= lsb
        return edges
    
    def greedy_fill(adj, order=None, deadline=None):
        """Greedily add edges in the given order."""
        if order is None:
            order = all_pairs
        
        for u, v in order:
            if deadline is not None and time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge_fast(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search_aggressive(adj, max_iterations, deadline):
        """
        Aggressive local search with multiple swap attempts per iteration.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        stagnant_count = 0
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try multiple random moves per iteration
            found_improvement = False
            
            for _ in range(30):  # 30 attempts per iteration
                if time.time() > deadline:
                    break
                
                move_type = random.random()
                
                if move_type < 0.4:
                    # Try to add a random non-edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if adj[u] & (1 << v):
                        continue
                    
                    if can_add_edge_fast(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                            stagnant_count = 0
                        found_improvement = True
                        break
                elif move_type < 0.7:
                    # Try swap: remove random edge, add another
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if adj[u] & (1 << v):
                        continue
                    
                    # Find an edge to remove
                    incident = []
                    temp = adj[u]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        incident.append((u, w))
                        temp -= lsb
                    temp = adj[v]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        if w != u:
                            incident.append((v, w))
                        temp -= lsb
                    
                    if incident:
                        rem_u, rem_v = random.choice(incident)
                        remove_edge(adj, rem_u, rem_v)
                        current_edges -= 1
                        
                        if can_add_edge_fast(adj, u, v):
                            add_edge(adj, u, v)
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                                stagnant_count = 0
                            found_improvement = True
                        else:
                            # Revert
                            add_edge(adj, rem_u, rem_v)
                            current_edges += 1
                else:
                    # Try 2-swap: remove two edges, add two different edges
                    u1 = random.randint(0, n - 1)
                    v1 = random.randint(0, n - 1)
                    if u1 == v1 or not (adj[u1] & (1 << v1)):
                        continue
                    
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    
                    # Remove (u1,v1), try to add (u2,v2)
                    remove_edge(adj, u1, v1)
                    current_edges -= 1
                    
                    if not (adj[u2] & (1 << v2)) and can_add_edge_fast(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                            stagnant_count = 0
                        found_improvement = True
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        current_edges += 1
            
            if not found_improvement:
                stagnant_count += 1
                if stagnant_count > 50:
                    break
        
        return best_adj, best_edges
    
    def simulated_annealing(adj, deadline):
        """
        Simulated annealing with structured moves.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        T = 3.0
        T_end = 0.1
        cooling_rate = 0.9998
        
        while time.time() < deadline and T > T_end:
            move_type = random.random()
            
            if move_type < 0.5:
                # Add/remove edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                if adj[u] & (1 << v):
                    # Remove edge
                    remove_edge(adj, u, v)
                    current_edges -= 1
                else:
                    # Try to add edge
                    if can_add_edge_fast(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                    else:
                        continue
            else:
                # Swap operation
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or (adj[u] & (1 << v)):
                    continue
                
                incident = []
                temp = adj[u]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    incident.append((u, w))
                    temp -= lsb
                temp = adj[v]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    if w != u:
                        incident.append((v, w))
                    temp -= lsb
                
                if not incident:
                    continue
                
                rem_u, rem_v = random.choice(incident)
                remove_edge(adj, rem_u, rem_v)
                current_edges -= 1
                
                if can_add_edge_fast(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                else:
                    add_edge(adj, rem_u, rem_v)
                    current_edges += 1
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
            
            T *= cooling_rate
        
        return best_adj, best_edges
    
    def systematic_polish(adj, deadline):
        """
        Systematic pass that tries all possible edge additions and swaps.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all possible edge additions
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        return adj, count_edges(adj)
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_fast(adj, u, v):
                            add_edge(adj, u, v)
                            improved = True
            
            # Try all possible swaps
            if time.time() > deadline:
                return adj, count_edges(adj)
            
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        return adj, count_edges(adj)
                    if adj[u] & (1 << v):
                        continue
                    
                    # Try removing each edge incident to u or v
                    for w in range(n):
                        if time.time() > deadline:
                            return adj, count_edges(adj)
                        if w == u or w == v:
                            continue
                        
                        # Try removing (u,w)
                        if adj[u] & (1 << w):
                            remove_edge(adj, u, w)
                            if can_add_edge_fast(adj, u, v):
                                add_edge(adj, u, v)
                                improved = True
                            else:
                                add_edge(adj, u, w)
                        
                        # Try removing (v,w)
                        if adj[v] & (1 << w):
                            remove_edge(adj, v, w)
                            if can_add_edge_fast(adj, u, v):
                                add_edge(adj, u, v)
                                improved = True
                            else:
                                add_edge(adj, v, w)
        
        return adj, count_edges(adj)
    
    # --- Main optimization loop ---
    best_adj = None
    best_edges = 0
    deadline = start_time + time_limit
    
    # Strategy 1: Multiple random greedy initializations
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        pairs_copy = all_pairs[:]
        random.shuffle(pairs_copy)
        
        greedy_fill(adj, order=pairs_copy, deadline=deadline)
        curr_edges = count_edges(adj)
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj[:]
    
    # Strategy 2: Aggressive local search on best found
    if best_adj is not None and time.time() < deadline - 10:
        improved_adj, improved_edges = local_search_aggressive(
            best_adj[:], 
            max_iterations=200000, 
            deadline=deadline - 10
        )
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj
    
    # Strategy 3: Simulated annealing for further improvement
    if best_adj is not None and time.time() < deadline - 5:
        sa_adj, sa_edges = simulated_annealing(best_adj[:], deadline - 5)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # Strategy 4: Final systematic polish
    if best_adj is not None and time.time() < deadline - 2:
        polished_adj, polished_edges = systematic_polish(best_adj[:], deadline - 2)
        if polished_edges > best_edges:
            best_edges = polished_edges
            best_adj = polished_adj
    
    # Fallback: 5 disjoint K4s
    if best_adj is None:
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(best_adj, u, v)
    
    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
