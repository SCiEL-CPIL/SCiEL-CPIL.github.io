"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill

    # Range [2, 256]
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict graph using bitsets for fast operations
    # conflict_mask[i] is a bitmask of nodes that conflict with node i
    conflict_mask = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            aj = nums[j]
            if _lcm(ai, aj) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    if time.time() - start_time > deadline:
        return [2]
    
    # Weights: 1/x
    weights = [1.0 / x for x in nums]
    
    # Greedy upper bound for a set of available nodes:
    # Sort by weight descending, greedily pick non-conflicting
    def greedy_upper_bound(available_mask):
        if available_mask == 0:
            return 0.0
        bound = 0.0
        # Get list of available nodes
        avail = []
        for i in range(n):
            if (available_mask >> i) & 1:
                avail.append(i)
        # Sort by weight descending
        avail.sort(key=lambda i: -weights[i])
        selected_mask = 0
        for i in avail:
            if (selected_mask & conflict_mask[i]) == 0:
                selected_mask |= (1 << i)
                bound += weights[i]
        return bound
    
    # Better upper bound: LP relaxation approximation
    # For each node, the max we can get is bounded by its weight plus neighbors
    # Simple bound: sum of weights of available nodes (loose)
    # Better: for each connected component, bound by max independent set weight
    # Use a simple greedy coloring bound
    
    def color_bound(available_mask):
        # Greedy coloring of the subgraph induced by available nodes
        # Each color class is an independent set, so max weight <= sum of max weights per color
        if available_mask == 0:
            return 0.0
        # Get available nodes
        avail = []
        for i in range(n):
            if (available_mask >> i) & 1:
                avail.append(i)
        # Greedy coloring
        color_of = {}
        colors = []  # list of lists of nodes per color
        for node in avail:
            # Find first color where node doesn't conflict with any member
            placed = False
            for c in range(len(colors)):
                conflict = False
                for other in colors[c]:
                    if (conflict_mask[node] >> other) & 1:
                        conflict = True
                        break
                if not conflict:
                    colors[c].append(node)
                    color_of[node] = c
                    placed = True
                    break
            if not placed:
                colors.append([node])
                color_of[node] = len(colors) - 1
        # Upper bound: sum over colors of max weight in that color
        bound = 0.0
        for c in colors:
            max_w = max(weights[node] for node in c)
            bound += max_w
        return bound
    
    # Branch and bound
    best_weight = 0.0
    best_set = set()
    
    def branch_and_bound(available_mask, current_set, current_weight, depth):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > deadline:
            return
        
        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound
        ub = color_bound(available_mask)
        if current_weight + ub <= best_weight:
            return
        
        # Also try greedy upper bound as a check
        ub2 = greedy_upper_bound(available_mask)
        ub = min(ub, ub2)
        if current_weight + ub <= best_weight:
            return
        
        # Choose next node to branch on
        # Heuristic: pick node with highest weight among available
        avail = []
        for i in range(n):
            if (available_mask >> i) & 1:
                avail.append(i)
        avail.sort(key=lambda i: -weights[i])
        
        # Try including the first node
        node = avail[0]
        # Include node
        new_set = list(current_set) + [node]
        new_available = available_mask & ~(1 << node)
        # Remove all conflicts of node
        new_available &= ~conflict_mask[node]
        branch_and_bound(new_available, new_set, current_weight + weights[node], depth + 1)
        
        if time.time() - start_time > deadline:
            return
        
        # Exclude node
        new_available = available_mask & ~(1 << node)
        branch_and_bound(new_available, current_set, current_weight, depth + 1)
    
    # Initial greedy solution to set a good lower bound
    order = sorted(range(n), key=lambda i: -weights[i])
    greedy_set = set()
    greedy_weight = 0.0
    for idx in order:
        compatible = True
        for s_idx in greedy_set:
            if (conflict_mask[idx] >> s_idx) & 1:
                compatible = False
                break
        if compatible:
            greedy_set.add(idx)
            greedy_weight += weights[idx]
    
    best_weight = greedy_weight
    best_set = set(greedy_set)
    
    # Run branch and bound
    branch_and_bound((1 << n) - 1, [], 0.0, 0)
    
    if time.time() - start_time > deadline:
        pass  # Return best found
    
    if not best_set:
        return [256]
    
    result = [nums[i] for i in best_set]
    
    # Verify the solution
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if _lcm(result[i], result[j]) <= 256:
                # Invalid, fall back to greedy
                return [nums[i] for i in greedy_set]
    
    return result

# EVOLVE-BLOCK-END
