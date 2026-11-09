import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 256
    
    # 1. Identify valid nodes (no self-conflict)
    # 2*v must not be a perfect square
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
        
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    n = len(valid_nodes)
    if n == 0:
        return [1] # Fallback, though 1 is valid (2*1=2 not square)

    # 2. Build conflict graph using bitsets
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # neighbors[i] is a bitmask of nodes that conflict with valid_nodes[i]
    neighbors = [0] * n
    
    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i + 1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                # Set bit j in neighbors[i] and bit i in neighbors[j]
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    # 3. Greedy Lower Bound
    # We want a good initial solution to prune the B&B tree.
    # Heuristic: Always pick the node with the minimum degree in the remaining candidates.
    # This is a standard approximation for MIS.
    
    def greedy_mis(n, neighbors):
        candidates = (1 << n) - 1
        selected = []
        while candidates:
            # Find node with min degree in candidates
            best_node = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                # Extract lowest set bit
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                # Count bits in neighbors[idx] & candidates
                deg = (neighbors[idx] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                temp ^= lsb
                
            if best_node == -1:
                break
            
            selected.append(best_node)
            # Remove best_node and its neighbors from candidates
            candidates &= ~(neighbors[best_node] | (1 << best_node))
            
        return selected

    initial_greedy = greedy_mis(n, neighbors)
    best_size = len(initial_greedy)
    best_solution = initial_greedy[:]
    
    # 4. Branch and Bound
    # State: candidates (int), current_size (int)
    # We use a recursive function. To avoid recursion depth issues (n~240), 
    # Python's default limit is 1000, which should be fine for depth <= 240.
    # However, the branching factor can be high.
    
    # Optimization: Order nodes by degree? 
    # The greedy step picks dynamically. In B&B, we also pick dynamically.
    
    def popcount(x):
        return x.bit_count()
        
    # Global variables for B&B
    current_best_size = best_size
    current_best_set = best_solution[:]
    
    # To speed up B&B, we can pre-compute the "degree" of each node in the original graph.
    # But dynamic degree is better.
    
    # Let's use an iterative B&B or a recursive one with a strict time check.
    # Given the small N, recursive is cleaner.
    
    import sys
    sys.setrecursionlimit(3000)
    
    def bnb(candidates, current_set, current_size):
        nonlocal current_best_size, current_best_set
        
        # Time check
        if time.time() > deadline - 5.0:
            return
        
        # Prune: Even if we pick all remaining candidates, can we beat best?
        if current_size + popcount(candidates) <= current_best_size:
            return
            
        if candidates == 0:
            if current_size > current_best_size:
                current_best_size = current_size
                current_best_set = current_set[:]
            return
        
        # Heuristic: Pick a node from candidates
        # Min-degree heuristic in the subgraph induced by candidates
        best_node = -1
        min_deg = n + 1
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (neighbors[idx] & candidates).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_node = idx
            temp ^= lsb
            
        if best_node == -1:
            return
            
        bit = 1 << best_node
        
        # Branch 1: Include best_node
        # New candidates: candidates without best_node and its neighbors
        new_candidates = candidates & ~(neighbors[best_node] | bit)
        current_set.append(best_node)
        bnb(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude best_node
        # New candidates: candidates without best_node
        new_candidates = candidates & ~bit
        bnb(new_candidates, current_set, current_size)

    # Start B&B
    # We can start with all nodes as candidates
    # To make it faster, we can sort the nodes initially? 
    # The dynamic min-degree selection handles ordering.
    
    # One optimization: If the graph is disconnected, solve components separately.
    # Given the structure of square sums, it's likely connected or has few components.
    # We'll stick to single B&B for simplicity unless it's too slow.
    
    # Let's try running B&B.
    # If N is large, this might still be slow. 
    # But N ~ 240. 
    # The greedy bound is usually quite close to optimal for random-ish graphs.
    
    # To be safe, let's add a simple local search improvement on the greedy result first.
    # Swap-based local search:
    # Try to add a node if it doesn't conflict. If it does, try to remove a conflicting node.
    
    def improve_local_search(solution_indices):
        sol_set = set(solution_indices)
        improved = True
        while improved:
            improved = False
            # Try adding nodes not in solution
            for i in range(n):
                if i in sol_set:
                    continue
                # Check if i can be added
                if (neighbors[i] & sol_set) == 0:
                    sol_set.add(i)
                    improved = True
                    break # Restart scan
            # Try swapping: remove j from sol, add i
            # This is O(N^2) per iteration, might be slow if done too much.
            # Given B&B is the main solver, we rely on it.
        return list(sol_set)
    
    # Actually, the B&B will find the optimal. 
    # Let's just run B&B.
    
    # Initial call
    # We use the greedy solution as the initial best.
    # We don't need to pass the greedy set to B&B, just the size.
    # But we do need to track the best set.
    
    # To avoid deep recursion overhead, we can use a stack-based approach if needed, 
    # but for N=240, recursion depth is bounded by N.
    
    # Let's run it.
    bnb((1 << n) - 1, [], 0)
    
    # Construct final result
    result_nodes = [valid_nodes[i] for i in current_best_set]
    
    # Final validation
    result_set = set(result_nodes)
    for v in result_nodes:
        if 2 * v in squares:
            # Invalid, fallback to greedy
            result_nodes = [valid_nodes[i] for i in initial_greedy]
            break
    else:
        valid = True
        for i in range(len(result_nodes)):
            for j in range(i + 1, len(result_nodes)):
                if result_nodes[i] + result_nodes[j] in squares:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result_nodes = [valid_nodes[i] for i in initial_greedy]
            
    return result_nodes

# Run once to ensure no errors in logic
# construct()
