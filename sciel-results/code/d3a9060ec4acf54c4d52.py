import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2) # 7 * 64 = 448
    
    # 1. Precompute edges and their indices
    # edge_list[idx] = (u, v)
    # edge_index[(u, v)] = idx
    edge_list = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_list.append((u, v))
                edge_index[(u, v)] = idx
                idx += 1
                
    # 2. Precompute conflicts
    # conflicts[idx] = list of 6 tuples of 3 edge indices
    # For each edge idx (u, v) with dimension i:
    # For each j != i:
    #   u2 = u ^ (1<<j)
    #   v2 = v ^ (1<<j)
    #   e1 = edge_index[(min(u, u2), max(u, u2))]
    #   e2 = edge_index[(min(v, v2), max(v, v2))]
    #   e3 = edge_index[(min(u2, v2), max(u2, v2))]
    #   conflicts[idx].append((e1, e2, e3))
    
    conflicts = [[] for _ in range(total_edges)]
    for idx, (u, v) in enumerate(edge_list):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u2 = u ^ bit_j
            v2 = v ^ bit_j
            
            # Get indices
            # Note: u < u2 or u2 < u? 
            # The edge is between u and u2.
            # We stored keys as (min, max).
            e1_key = (min(u, u2), max(u, u2))
            e2_key = (min(v, v2), max(v, v2))
            e3_key = (min(u2, v2), max(u2, v2))
            
            e1 = edge_index[e1_key]
            e2 = edge_index[e2_key]
            e3 = edge_index[e3_key]
            
            conflicts[idx].append((e1, e2, e3))

    def is_blocked(idx, S):
        """Check if adding edge idx to set S creates a C4."""
        for e1, e2, e3 in conflicts[idx]:
            if e1 in S and e2 in S and e3 in S:
                return True
        return False

    def refill(S):
        """Try to add any unblocked edge to S. Returns new S (modified in place)."""
        added = True
        while added:
            added = False
            for idx in range(total_edges):
                if idx not in S:
                    if not is_blocked(idx, S):
                        S.add(idx)
                        added = True
        return S

    def random_greedy():
        """Generate a random valid set using greedy."""
        S = set()
        order = list(range(total_edges))
        random.shuffle(order)
        for idx in order:
            if not is_blocked(idx, S):
                S.add(idx)
        # Refill to catch any edges that became unblocked? 
        # Greedy from random order usually ends in a maximal independent set of the conflict graph.
        # But "maximal" doesn't mean "maximum". Refilling might add more if we missed some?
        # Actually, if we process in random order, an edge rejected early might be acceptable later?
        # No, adding edges only increases conflicts. If it was blocked, it remains blocked.
        # So greedy produces a maximal C4-free set.
        return S

    best_set = set()
    best_count = 0
    
    deadline = time.time() + 110
    
    # Initial runs
    for _ in range(10):
        S = random_greedy()
        if len(S) > best_count:
            best_count = len(S)
            best_set = set(S)
            
    # Local Search / Simulated Annealing
    current_set = set(best_set)
    current_count = best_count
    
    # Temperature for simulated annealing
    # We want to escape local optima.
    # Move: Remove k random edges, then refill.
    # If new count > old count, accept.
    # If new count < old count, accept with probability exp(-delta/T).
    # Given the discrete nature and small size, simple hill climbing with random restarts on perturbation might suffice.
    
    t = 1.0
    min_t = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Perturbation
        # Remove a small number of random edges
        k = random.randint(2, 8)
        if len(current_set) < k:
            k = len(current_set)
        
        # Pick k random edges to remove
        # Using list conversion is O(N), but N=448 is small.
        # Or use random.sample on list(S)
        edges_to_remove = random.sample(list(current_set), k)
        for e in edges_to_remove:
            current_set.remove(e)
            
        # Refill
        refill(current_set)
        
        new_count = len(current_set)
        
        if new_count >= current_count:
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_set = set(current_set)
                # Reset temperature or cool down
                t = 1.0 
        else:
            # Accept if better or with probability
            delta = current_count - new_count
            # Simple annealing
            if random.random() < pow(2.718, -delta / max(t, 0.1)):
                current_count = new_count
                # Do not update best if worse
            # else: reject, restore current_set? 
            # We modified current_set in place. We need to restore if rejected.
            # To avoid complex restoration, we can work on a copy?
            # Copying a set of 448 ints is fast.
            # Let's optimize: keep a backup of the state before perturbation.
            
        t = max(min_t, t * cooling)
        
        # If we are stuck for a while, maybe restart?
        # The annealing should handle it.

    # Return best_set as list of lists of vertices
    # best_set contains edge indices.
    # We need to return [ [u, v], ... ]
    result = []
    for idx in best_set:
        u, v = edge_list[idx]
        result.append([u, v])
        
    return result
