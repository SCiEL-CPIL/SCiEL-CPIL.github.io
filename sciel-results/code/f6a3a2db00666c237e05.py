import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 16
    
    # Precompute all 4-sets
    all_4sets = list(combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # For each 4-set, store as a frozenset for O(1) intersection checks
    # Precompute disjoint pairs of 4-sets
    disjoint_pairs = []
    for i, A in enumerate(all_4sets):
        A_set = frozenset(A)
        for j, B in enumerate(all_4sets):
            if i >= j:
                continue
            if not A_set & frozenset(B):
                disjoint_pairs.append((i, j))
    
    def count_q3s(adj):
        """Count Q3 subgraphs. adj[i] is bitmask of neighbors."""
        count = 0
        for idx_a, idx_b in disjoint_pairs:
            A = all_4sets[idx_a]
            B = all_4sets[idx_b]
            # Count non-edges between A and B
            non_edge_count = 0
            non_edges = []
            for u in A:
                for v in B:
                    if not (adj[u] & (1 << v)):
                        non_edge_count += 1
                        non_edges.append((u, v))
                        if non_edge_count > 4:
                            break
                if non_edge_count > 4:
                    break
            if non_edge_count > 4:
                continue
            # Check if non_edges form a matching
            used_u = set()
            used_v = set()
            is_match = True
            for u, v in non_edges:
                if u in used_u or v in used_v:
                    is_match = False
                    break
                used_u.add(u)
                used_v.add(v)
            if is_match:
                count += 1
        return count
    
    def is_q3_free(adj):
        """Quick check: returns True if no Q3 found."""
        for idx_a, idx_b in disjoint_pairs:
            A = all_4sets[idx_a]
            B = all_4sets[idx_b]
            non_edge_count = 0
            non_edges = []
            for u in A:
                for v in B:
                    if not (adj[u] & (1 << v)):
                        non_edge_count += 1
                        non_edges.append((u, v))
                        if non_edge_count > 4:
                            break
                if non_edge_count > 4:
                    break
            if non_edge_count > 4:
                continue
            used_u = set()
            used_v = set()
            is_match = True
            for u, v in non_edges:
                if u in used_u or v in used_v:
                    is_match = False
                    break
                used_u.add(u)
                used_v.add(v)
            if is_match:
                return False
        return True
    
    # Start with K16 and remove edges
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    # Greedy removal: find Q3s and remove edges
    # First, let's do a quick greedy pass
    removed_edges = set()
    
    # Try removing edges from K16 to get Q3-free
    # Use random order for diversity
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(123)
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 60:
            break
        # Try removing edge (u,v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        if is_q3_free(adj):
            removed_edges.add((u, v))
        else:
            # Restore
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    current_edges = sum(bin(a).count('1') for a in adj) // 2
    best_adj = adj[:]
    best_edges = current_edges
    
    # Now try to add back edges using simulated annealing
    missing_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                     if not (best_adj[i] & (1 << j))]
    
    # Simulated annealing on edge additions
    temperature = 1.0
    cooling = 0.9995
    
    adj = best_adj[:]
    current_count = count_q3s(adj)
    
    for iteration in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random missing edge to try adding
        if not missing_edges:
            break
        idx = random.randint(0, len(missing_edges) - 1)
        u, v = missing_edges[idx]
        
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        new_count = count_q3s(adj)
        edge_count = sum(bin(a).count('1') for a in adj) // 2
        
        if new_count == 0 and edge_count > best_edges:
            best_adj = adj[:]
            best_edges = edge_count
        elif new_count <= current_count:
            current_count = new_count
            # Keep the edge
            missing_edges.pop(idx)
        else:
            # Accept with probability exp(-delta/T)
            delta = new_count - current_count
            if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                current_count = new_count
                missing_edges.pop(idx)
            else:
                # Undo
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        temperature *= cooling
    
    # Final polish: try all single edge additions
    adj = best_adj[:]
    for u, v in missing_edges:
        if time.time() > deadline - 2:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if is_q3_free(adj):
            edge_count = sum(bin(a).count('1') for a in adj) // 2
            if edge_count > best_edges:
                best_adj = adj[:]
                best_edges = edge_count
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
