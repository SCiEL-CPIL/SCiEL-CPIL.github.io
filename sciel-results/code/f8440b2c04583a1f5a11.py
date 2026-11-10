import random
import time
import sys

def _has_c6_through(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    This is true iff there is a simple path of length 5 from u to v in the current graph.
    We use a BFS-layering approach with pruning to find if such a path exists.
    """
    # S_k is the set of vertices reachable from u by a simple path of length k
    # that does not visit v.
    
    S0 = {u}
    
    # Step 1: Neighbors of u, excluding v
    S1 = set()
    for x in adj[u]:
        if x != v:
            S1.add(x)
    if not S1:
        return False
        
    # Step 2: Neighbors of S1, excluding S0, S1, and v
    visited = S0 | S1
    S2 = set()
    for y in S1:
        for x in adj[y]:
            if x not in visited and x != v:
                S2.add(x)
    if not S2:
        return False
        
    visited |= S2
    # Step 3: Neighbors of S2, excluding visited and v
    S3 = set()
    for y in S2:
        for x in adj[y]:
            if x not in visited and x != v:
                S3.add(x)
    if not S3:
        return False
        
    visited |= S3
    # Step 4: Neighbors of S3, excluding visited and v
    S4 = set()
    for y in S3:
        for x in adj[y]:
            if x not in visited and x != v:
                S4.add(x)
    if not S4:
        return False
        
    # Step 5: Check if v is a neighbor of any node in S4
    # If yes, we have u -> ... -> s4 -> v, a path of length 5.
    for s4_node in S4:
        if v in adj[s4_node]:
            return True
            
    return False

def _count_edges(adj):
    return sum(len(s) for s in adj) // 2

def construct():
    n = 40
    start_time = time.time()
    time_limit = 115 # Leave 5s for safety
    
    # Initialize with a random graph
    # Start with a moderate number of edges to allow growth
    # Or start empty and grow?
    # Local search usually works better starting from a valid solution.
    # Let's start with a random sparse graph and ensure no C6.
    
    def random_valid_graph(target_edges):
        adj = [set() for _ in range(n)]
        edges = 0
        # Try to add random edges
        # To avoid C6, we check before adding.
        # This is slow if we do it for many edges.
        # Instead, start with 0 edges and add greedily?
        # That's the current approach.
        # Let's try starting with a random permutation of edges and adding greedily.
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if edges >= target_edges:
                break
            if not _has_c6_through(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edges += 1
        return adj

    # Start with a greedy construction to get a decent baseline
    # Then use Simulated Annealing to improve
    
    # Greedy phase
    adj = random_valid_graph(120) # Try to get to 120 edges
    current_edges = _count_edges(adj)
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges
    
    # Simulated Annealing Phase
    T = 1.0
    cooling_rate = 0.999
    
    # Maintain a list of existing edges for fast removal
    # and a set for fast existence check
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
                
    existing_edges_list = list(edge_set)
    
    # We need a list of missing edges for random addition
    # Generating all missing edges is 780 - current_edges.
    # We can maintain a list of missing edges.
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing_edges_list = [e for e in all_edges if e not in edge_set]
    
    while time.time() - start_time < time_limit:
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_adj = [set(s) for s in adj]
        new_edges = current_edges
        valid_move = False
        removed_edge = None
        added_edge = None

        if move_type == 'add':
            if not missing_edges_list:
                continue
            # Pick a random missing edge
            idx = random.randint(0, len(missing_edges_list) - 1)
            u, v = missing_edges_list[idx]
            
            if not _has_c6_through(u, v, new_adj):
                new_adj[u].add(v)
                new_adj[v].add(u)
                new_edges += 1
                valid_move = True
                added_edge = (u, v)
                
        elif move_type == 'remove':
            if not existing_edges_list:
                continue
            # Pick a random existing edge
            idx = random.randint(0, len(existing_edges_list) - 1)
            u, v = existing_edges_list[idx]
            
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_edges -= 1
            valid_move = True
            removed_edge = (u, v)
            
        else: # swap
            # Remove e1, Add e2
            if not existing_edges_list or not missing_edges_list:
                continue
            idx1 = random.randint(0, len(existing_edges_list) - 1)
            u1, v1 = existing_edges_list[idx1]
            
            # Find a missing edge to add
            # To avoid picking the same edge, we pick from missing list
            idx2 = random.randint(0, len(missing_edges_list) - 1)
            u2, v2 = missing_edges_list[idx2]
            
            # Ensure we don't remove and add the same edge (though they are from disjoint sets, so this is impossible)
            # Remove e1
            new_adj[u1].remove(v1)
            new_adj[v1].remove(u1)
            
            # Add e2
            if not _has_c6_through(u2, v2, new_adj):
                new_adj[u2].add(v2)
                new_adj[v2].add(u2)
                # new_edges stays same
                valid_move = True
                removed_edge = (u1, v1)
                added_edge = (u2, v2)
            else:
                continue # Move invalid, skip
                
        if not valid_move:
            continue

        # Acceptance criterion
        delta = new_edges - current_edges
        if delta > 0:
            accept = True
        else:
            # delta is 0 or negative
            # If delta == 0, we might accept to escape local minima
            # If delta < 0, we accept with prob exp(delta/T)
            if delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < pow(2.718, delta / T)
                
        if accept:
            adj = new_adj
            current_edges = new_edges
            
            # Update lists
            if removed_edge:
                edge_set.remove(removed_edge)
                existing_edges_list.remove(removed_edge)
                missing_edges_list.append(removed_edge)
            if added_edge:
                edge_set.add(added_edge)
                existing_edges_list.append(added_edge)
                missing_edges_list.remove(added_edge)
                
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [set(s) for s in adj]
                
        T *= cooling_rate
        
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
                
    return result
