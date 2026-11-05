"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict sets: conflict[i] is set of j such that vertices[i] + vertices[j] is a square
    # Using 0-indexed internally
    conflict = [set() for _ in range(N)]
    valid_vertices = []
    
    for i in range(N):
        a = i + 1
        # Check self conflict: 2*a should not be a perfect square
        if 2 * a in squares:
            continue  # This vertex cannot be in the set
        valid_vertices.append(i)
    
    # Build conflicts for valid vertices
    for i in valid_vertices:
        a = i + 1
        for j in valid_vertices:
            if j == i:
                continue
            b = j + 1
            if a + b in squares:
                conflict[i].add(j)
    
    def is_compatible(candidate_set, vertex):
        """Check if vertex can be added to candidate_set"""
        for v in candidate_set:
            if v in conflict[vertex]:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an ordering of vertices"""
        result = []
        for v in order:
            if is_compatible(result, v):
                result.append(v)
        return result
    
    def local_improve(candidate_set):
        """Try to add any non-conflicting vertices not in the set"""
        changed = True
        while changed:
            changed = False
            current_set = set(candidate_set)
            for v in valid_vertices:
                if v not in current_set and is_compatible(candidate_set, v):
                    candidate_set.append(v)
                    current_set.add(v)
                    changed = True
        return candidate_set
    
    def local_search(candidate_set, max_iterations=1000):
        """Swap-based local search to improve the set"""
        current_set = set(candidate_set)
        best_set = set(candidate_set)
        
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try removing one element and adding another
            for remove_v in list(current_set):
                temp_set = current_set - {remove_v}
                # Try to add a vertex not in temp_set
                for add_v in valid_vertices:
                    if add_v in temp_set:
                        continue
                    if is_compatible(list(temp_set), add_v):
                        new_set = temp_set | {add_v}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            current_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return list(best_set)
    
    # Run multiple random restarts
    best_result = []
    
    # First, try a deterministic greedy approach
    # Order by degree (fewest conflicts first)
    degree_order = sorted(valid_vertices, key=lambda v: len(conflict[v]))
    result = greedy_construct(degree_order)
    result = local_improve(result)
    result = local_search(result)
    best_result = result
    
    # Random restarts
    while time.time() - start_time < DEADLINE:
        random.shuffle(valid_vertices)
        result = greedy_construct(valid_vertices)
        result = local_improve(result)
        result = local_search(result)
        
        if len(result) > len(best_result):
            best_result = result
    
    # Convert back to 1-indexed
    return [v + 1 for v in best_result]
# EVOLVE-BLOCK-END
