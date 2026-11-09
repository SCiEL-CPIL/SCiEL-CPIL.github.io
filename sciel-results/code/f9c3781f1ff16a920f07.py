import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20

    def count_k33_violations(adj_bits):
        """
        Return list of K3,3 violations. Each violation is a tuple (a,b,c) 
        where {a,b,c} is a triple with >=3 common neighbors.
        Also return a dict mapping each edge to the number of violations it participates in.
        """
        violations = []
        edge_violation_count = {}
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    common_count = common.bit_count()
                    if common_count >= 3:
                        violations.append((i, j, k))
                        # The edges involved in this violation:
                        # All edges from {i,j,k} to the common neighbors
                        common_list = []
                        for v in range(n):
                            if (common >> v) & 1:
                                common_list.append(v)
                        # Edges: (i,v), (j,v), (k,v) for each v in common_list
                        for v in common_list:
                            for u in [i, j, k]:
                                if u < v:
                                    key = (u, v)
                                else:
                                    key = (v, u)
                                edge_violation_count[key] = edge_violation_count.get(key, 0) + 1
        
        return violations, edge_violation_count

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def is_k33_free(adj_bits):
        """
        Check if the graph is K3,3-free.
        For all distinct a, b, c: |N(a) ∩ N(b) ∩ N(c)| <= 2.
        """
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    if common.bit_count() >= 3:
                        return False
        return True

    def hitting_set_removal(initial_adj, max_iterations=200):
        """
        Start with initial_adj and greedily remove edges that participate 
        in the most K3,3 violations until the graph is K3,3-free.
        """
        adj = initial_adj[:]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
                
            violations, edge_violation_count = count_k33_violations(adj)
            
            if not violations:
                break  # K3,3-free
            
            # Find the edge with the most violations
            max_count = 0
            best_edge = None
            for edge, cnt in edge_violation_count.items():
                if cnt > max_count:
                    max_count = cnt
                    best_edge = edge
            
            if best_edge is None:
                break
            
            # Remove the edge
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return adj

    def local_search_polish(adj, max_steps=500):
        """
        Try to add missing edges or perform edge swaps to increase density
        while maintaining K3,3-freeness.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding each missing edge
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                    
                if (current_adj[u] >> v) & 1:
                    continue  # Edge already exists
                
                # Try adding edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                if is_k33_free(current_adj):
                    current_edges += 1
                    improved = True
                else:
                    # Revert
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
        
        return current_adj, current_edges

    def random_dense_graph(density=0.8):
        """
        Generate a random graph with given edge probability.
        """
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def structured_initial_graph():
        """
        Create a structured initial graph based on a known construction.
        Use a circulant-like structure with carefully chosen connection patterns.
        """
        adj = [0] * n
        
        # Try a pattern based on modular arithmetic
        # Connect i to i+d mod n for certain offsets d
        offsets = [1, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19]
        
        for i in range(n):
            for d in offsets:
                j = (i + d) % n
                if j != i and i < j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj

    best_adj = [0] * n
    best_edges = 0
    
    # Strategy 1: Hitting set removal from dense random graphs
    for r in range(50):
        if time.time() > deadline - 20:
            break
            
        # Try different densities
        density = 0.7 + 0.05 * (r % 5)
        initial = random_dense_graph(density)
        
        # Apply hitting set removal
        adj = hitting_set_removal(initial, max_iterations=100)
        
        # Polish with local search
        adj, e_count = local_search_polish(adj, max_steps=200)
        
        if e_count > best_edges:
            best_edges = e_count
            best_adj = adj[:]
    
    # Strategy 2: Hitting set removal from structured graphs
    for r in range(20):
        if time.time() > deadline - 15:
            break
            
        # Create a structured base and perturb it
        base = structured_initial_graph()
        
        # Randomly remove some edges
        adj = base[:]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        # Remove a random subset
        num_removals = random.randint(5, 20)
        for u, v in all_edges[:num_removals]:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Apply hitting set removal
        adj = hitting_set_removal(adj, max_iterations=100)
        
        # Polish with local search
        adj, e_count = local_search_polish(adj, max_steps=200)
        
        if e_count > best_edges:
            best_edges = e_count
            best_adj = adj[:]
    
    # Strategy 3: Final polish on the best graph found
    if time.time() < deadline - 5:
        best_adj, best_edges = local_search_polish(best_adj, max_steps=1000)
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
