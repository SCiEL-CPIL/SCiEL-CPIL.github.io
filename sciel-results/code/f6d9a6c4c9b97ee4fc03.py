"""Construct a graph on 12 vertices with min degree 3 and minimum distinct cycle lengths."""
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 40  # 40s safety margin before 60s limit
    
    # Fallback: Two K_{3,3} (Raw value 2, Score 2.5)
    fallback_edges = []
    for u in range(3):
        for v in range(3, 6):
            fallback_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            fallback_edges.append([u, v])
    fallback_edges = [[int(a), int(b)] for a, b in fallback_edges]
    
    best_edges = fallback_edges
    best_raw_value = 2
    
    # Fast cycle-length checker with early termination
    def get_cycle_lengths(edges, max_lengths=None):
        """Returns set of distinct cycle lengths. If max_lengths is given,
        stops early if more than max_lengths distinct lengths are found."""
        adj = [set() for _ in range(12)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        lengths = set()
        
        # For each start node, find cycles where start is the minimum node
        for start in range(12):
            # DFS: path starts at start, only visit nodes > start
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 4:
                        lengths.add(len(path))
                        if max_lengths is not None and len(lengths) > max_lengths:
                            return lengths
                    elif neighbor > start and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        
        return lengths
    
    # Generate random 3-regular bipartite graph on 6+6 vertices
    # Using random permutation approach
    def random_3reg_bipartite():
        """Generate random 3-regular bipartite graph with partitions 0-5 and 6-11."""
        # Each left node connects to 3 right nodes, each right node connects to 3 left nodes
        # This is equivalent to a random 3-regular bipartite graph
        # Use configuration model or random assignment
        
        # Method: random assignment with degree constraints
        # Left nodes 0-5 each need degree 3, right nodes 6-11 each need degree 3
        # Total edges = 18
        
        # Use random matching approach:
        # Create multiset of left nodes (each repeated 3 times) and right nodes (each repeated 3 times)
        # Randomly pair them up
        left_multiset = []
        for i in range(6):
            left_multiset.extend([i] * 3)
        right_multiset = []
        for i in range(6, 12):
            right_multiset.extend([i] * 3)
        
        random.shuffle(right_multiset)
        edges = set()
        for i in range(18):
            u = left_multiset[i]
            v = right_multiset[i]
            edges.add((min(u, v), max(u, v)))
        
        # Check if this gives a simple graph (no multi-edges)
        if len(edges) != 18:
            return None
        
        return list(edges)
    
    # Alternative: direct random 3-regular bipartite via random permutations
    def random_3reg_bipartite_v2():
        """Generate random 3-regular bipartite graph using random permutation composition."""
        # A 3-regular bipartite graph can be represented as union of 3 perfect matchings
        # Generate 3 random perfect matchings and take their union
        left_nodes = list(range(6))
        right_nodes = list(range(6, 12))
        
        edges = set()
        for _ in range(3):
            # Random perfect matching
            perm = right_nodes[:]
            random.shuffle(perm)
            for i in range(6):
                u = left_nodes[i]
                v = perm[i]
                edges.add((min(u, v), max(u, v)))
        
        if len(edges) != 18:
            return None
        
        return list(edges)
    
    # Local search: try edge swaps to reduce distinct cycle lengths
    def local_search(edges, max_time=2.0):
        """Try to improve the graph by edge swaps."""
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        current_lengths = get_cycle_lengths(list(edges_set))
        current_count = len(current_lengths)
        
        if current_count == 1:
            return list(edges_set), 1
        
        search_start = time.time()
        improved = True
        while improved and time.time() - search_start < max_time:
            improved = False
            edges_list = list(edges_set)
            random.shuffle(edges_list)
            
            for i in range(len(edges_list)):
                if time.time() - search_start >= max_time:
                    break
                u1, v1 = edges_list[i]
                
                # Try swapping v1 with another right node
                # u1 is in left partition (0-5), v1 is in right partition (6-11)
                if u1 > 5:
                    u1, v1 = v1, u1  # ensure u1 is left, v1 is right
                
                # Find another edge (u2, v2) where u2 != u1
                for j in range(len(edges_list)):
                    if j == i:
                        continue
                    u2, v2 = edges_list[j]
                    if u2 > 5:
                        u2, v2 = v2, u2
                    
                    if u2 == u1:
                        continue
                    
                    # Try swap: remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
                    new_edges = edges_set - {(u1, v1), (u2, v2)}
                    new_edge1 = (min(u1, v2), max(u1, v2))
                    new_edge2 = (min(u2, v1), max(u2, v1))
                    
                    if new_edge1 in new_edges or new_edge2 in new_edges:
                        continue
                    if new_edge1 == new_edge2:
                        continue
                    
                    new_edges.add(new_edge1)
                    new_edges.add(new_edge2)
                    
                    # Check degrees are still valid (they should be, since it's a swap)
                    # Check cycle lengths
                    new_lengths = get_cycle_lengths(list(new_edges), max_lengths=current_count)
                    new_count = len(new_lengths)
                    
                    if new_count < current_count:
                        edges_set = new_edges
                        current_lengths = new_lengths
                        current_count = new_count
                        improved = True
                        if current_count == 1:
                            return list(edges_set), 1
                        break
        
        return list(edges_set), current_count
    
    # Search Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate random 3-regular bipartite graph
        edges = random_3reg_bipartite_v2()
        if edges is None:
            continue
        
        # Check cycle lengths with early termination
        lengths = get_cycle_lengths(edges, max_lengths=best_raw_value)
        count = len(lengths)
        
        if count == 1:
            best_edges = [[int(u), int(v)] for u, v in edges]
            best_raw_value = 1
            break
        
        if count < best_raw_value:
            # Try local search to improve further
            refined_edges, refined_count = local_search(edges, max_time=1.0)
            if refined_count < best_raw_value:
                best_edges = [[int(u), int(v)] for u, v in refined_edges]
                best_raw_value = refined_count
                if best_raw_value == 1:
                    break
            else:
                best_edges = [[int(u), int(v)] for u, v in edges]
                best_raw_value = count
    
    # Return best found
    # Ensure types are correct
    out = []
    seen = set()
    for u, v in best_edges:
        if u == v:
            continue
        a, b = (min(u, v), max(u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([int(a), int(b)])
        
    return out
