import time
import random
import itertools
import sys

# Q3 structure
# Vertices 0..7
# Edges:
# 0-1, 0-2, 0-4
# 1-3, 1-5
# 2-3, 2-6
# 3-7
# 4-5, 4-6
# 5-7
# 6-7

Q3_ADJ = [
    [1, 2, 4],
    [0, 3, 5],
    [0, 3, 6],
    [1, 2, 7],
    [0, 5, 6],
    [1, 4, 7],
    [2, 4, 7],
    [3, 5, 6]
]

Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 16
    all_vertices = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    def find_blocking_q3(u, v, adj_mask):
        """
        Finds a Q3 embedding that uses edge (u,v) and is otherwise present in adj_mask.
        Returns a list of edges (tuple of 2 ints) forming the Q3, or None.
        """
        def backtrack(assigned, available):
            unassigned = [x for x in range(8) if x not in assigned]
            if not unassigned:
                # All assigned, return the mapping
                return assigned
            
            # Heuristic: Pick unassigned Q3 vertex with most assigned neighbors
            def constraint_count(q3_v):
                count = 0
                for nb in Q3_ADJ[q3_v]:
                    if nb in assigned:
                        count += 1
                return count
            
            unassigned.sort(key=constraint_count, reverse=True)
            q3_v = unassigned[0]
            
            candidates = list(available)
            
            for nb in Q3_ADJ[q3_v]:
                if nb in assigned:
                    g_nb = assigned[nb]
                    mask = adj_mask[g_nb]
                    # Filter candidates
                    # Optimization: Use bit operations if possible, but list comprehension is fine for small N
                    candidates = [c for c in candidates if (mask >> c) & 1]
                    if not candidates:
                        return None
            
            for g_cand in candidates:
                new_assigned = dict(assigned)
                new_assigned[q3_v] = g_cand
                new_available = set(available)
                new_available.remove(g_cand)
                
                result = backtrack(new_assigned, new_available)
                if result is not None:
                    return result
            
            return None

        # Iterate over all Q3 edges to see if (u,v) can be part of a Q3
        for i, j in Q3_EDGES:
            # Case 1: u->i, v->j
            assigned = {i: u, j: v}
            available = set(all_vertices) - {u, v}
            
            result_map = backtrack(assigned, available)
            if result_map is not None:
                # Convert mapping to edge list
                q3_edges_in_g = []
                for q3_u, q3_v in Q3_EDGES:
                    g_u = result_map[q3_u]
                    g_v = result_map[q3_v]
                    # Store in canonical form for consistency if needed, but tuple is fine
                    q3_edges_in_g.append((min(g_u, g_v), max(g_u, g_v)))
                return q3_edges_in_g
            
            # Case 2: u->j, v->i
            assigned = {j: u, i: v}
            available = set(all_vertices) - {u, v}
            
            result_map = backtrack(assigned, available)
            if result_map is not None:
                q3_edges_in_g = []
                for q3_u, q3_v in Q3_EDGES:
                    g_u = result_map[q3_u]
                    g_v = result_map[q3_v]
                    q3_edges_in_g.append((min(g_u, g_v), max(g_u, g_v)))
                return q3_edges_in_g
        
        return None

    def random_greedy_init(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edges = list(all_edges)
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates Q3
            # We don't need the specific Q3 for initialization, just existence
            # But find_blocking_q3 returns None if no Q3, so we can use it.
            # It's slightly slower than a pure boolean check, but we need it for the kick later.
            # For initialization, a pure boolean check is faster.
            # Let's implement a fast boolean check inline or reuse logic.
            # Actually, find_blocking_q3 does the same work. Let's just use it.
            if find_blocking_q3(u, v, adj_mask) is None:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask

    def count_edges(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2

    def greedy_hitting_set(families):
        """
        Greedy set cover.
        families: list of sets.
        Returns a set of elements to remove.
        """
        if not families:
            return set()
        
        # Flatten to find all elements
        all_elements = set()
        for f in families:
            all_elements.update(f)
        
        # If no families, return empty
        if not all_elements:
            return set()
            
        # Count occurrences of each element
        # We want to pick the element that covers the most *uncovered* families.
        # Greedy:
        # 1. Start with all families uncovered.
        # 2. Pick element e that appears in the most uncovered families.
        # 3. Add e to solution, mark those families as covered.
        # 4. Repeat.
        
        uncovered = list(families)
        solution = set()
        
        while uncovered:
            # Find best element
            best_elem = None
            max_count = 0
            # To speed up, we can just iterate through all_elements
            # But all_elements might be large.
            # However, families are small (12 edges each).
            # So we can iterate through elements in the first few families?
            # No, just iterate through all_elements.
            
            # Optimization: Only consider elements that are in at least one uncovered family
            candidate_elements = set()
            for f in uncovered:
                candidate_elements.update(f)
                
            for e in candidate_elements:
                count = 0
                for f in uncovered:
                    if e in f:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_elem = e
            
            if best_elem is None or max_count == 0:
                break # Should not happen if uncovered is not empty and families are not empty
                
            solution.add(best_elem)
            
            # Remove covered families
            new_uncovered = []
            for f in uncovered:
                if best_elem not in f:
                    new_uncovered.append(f)
            uncovered = new_uncovered
            
        return solution

    best_adj = None
    best_count = 0
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj = random_greedy_init(r)
        current_count = count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        stuck_iterations = 0
        max_stuck = 50 # Reduced, as kicks are more expensive but effective
        
        while time.time() < deadline and stuck_iterations < max_stuck:
            missing = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
            
            if not missing:
                break
            
            random.shuffle(missing)
            
            added = False
            for u, v in missing[:20]: # Try a few
                if find_blocking_q3(u, v, adj) is None:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    added = True
                    stuck_iterations = 0
                    break
            
            if not added:
                stuck_iterations += 1
                
                # Smart Kick
                # Find blocking Q3s for a subset of missing edges
                # To save time, only check first 10 missing edges
                blockers = []
                for u, v in missing[:10]:
                    q3_edges = find_blocking_q3(u, v, adj)
                    if q3_edges:
                        blockers.append(set(q3_edges))
                
                if blockers:
                    # Find hitting set
                    edges_to_remove = greedy_hitting_set(blockers)
                    
                    if edges_to_remove:
                        # Remove them
                        for u, v in edges_to_remove:
                            if (adj[u] >> v) & 1:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_count -= 1
                        stuck_iterations = 0 # Reset after kick
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
