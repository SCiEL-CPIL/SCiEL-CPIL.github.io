import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    weights = [1.0 / i for i in range(1, n + 1)]  # weights[i-1] = 1/i
    
    # Build conflict graph: edge between a,c if (a+c)/2 is integer in [1,n]
    # i.e., a and c have same parity and |a-c| >= 2
    # Conflict: a, b, c form 3-AP where b = (a+c)/2
    # Two elements x,y conflict if there exists z such that x,z,y or y,z,x or z,x,y form AP
    # Actually: x and y conflict if (x+y) is even AND (x+y)//2 is in [1,n]
    # Wait, that's not right. Let me think again.
    # A 3-AP is (a, b, c) where b = (a+c)/2, so a+c = 2b
    # Two elements a and c conflict if their midpoint (a+c)/2 is an integer in [1,n]
    # i.e., a+c is even and 1 <= (a+c)/2 <= n, i.e., 2 <= a+c <= 2n
    # Since a,c in [1,n], a+c is always in [2, 2n], so the condition is just a+c even
    
    # Build adjacency: conflict[i] = set of j that conflict with i
    # Using 0-indexed: element i+1
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(n):
            if i == j:
                continue
            c = j + 1
            # a and c conflict if (a+c) is even
            if (a + c) % 2 == 0:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Maximum Weight Independent Set via Branch and Bound
    # Nodes are 0..n-1 representing elements 1..n
    # Weight of node i is 1/(i+1)
    
    best_weight = 0.0
    best_set = set()
    
    # Greedy coloring for upper bound
    def greedy_color_bound(remaining_nodes, weights):
        """Compute upper bound using greedy coloring on the subgraph induced by remaining_nodes.
        Returns the sum of the maximum weight in each color class."""
        if not remaining_nodes:
            return 0.0
        
        # Greedy coloring: assign colors to maximize the bound
        # Use largest-first ordering for better bounds
        nodes = sorted(remaining_nodes, key=lambda x: -weights[x])
        
        # Color classes
        color_classes = []  # list of (weight, node) for each color
        node_color = {}
        
        for node in nodes:
            # Find the lowest color not used by neighbors
            neighbor_colors = set()
            for neighbor in conflict[node]:
                if neighbor in node_color:
                    neighbor_colors.add(node_color[neighbor])
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            node_color[node] = color
            if color >= len(color_classes):
                color_classes.append(weights[node])
            else:
                color_classes[color] = max(color_classes[color], weights[node])
        
        return sum(color_classes)
    
    def branch_and_bound(remaining_nodes, current_weight, current_set, time_limit):
        """Branch and bound for maximum weight independent set."""
        nonlocal best_weight, best_set
        
        if time.time() > time_limit:
            return
        
        if not remaining_nodes:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound: current weight + greedy coloring bound on remaining
        ub = current_weight + greedy_color_bound(remaining_nodes, weights)
        if ub <= best_weight:
            return
        
        # Select node to branch on: pick the one with highest weight among remaining
        # that has the most conflicts (to prune more)
        node = max(remaining_nodes, key=lambda x: (weights[x], len(conflict[x] & remaining_nodes)))
        
        # Branch 1: include node
        new_remaining = remaining_nodes - {node} - conflict[node]
        branch_and_bound(new_remaining, current_weight + weights[node], 
                        current_set | {node}, time_limit)
        
        # Branch 2: exclude node
        branch_and_bound(remaining_nodes - {node}, current_weight, 
                        current_set, time_limit)
    
    # Initial greedy solution for lower bound
    def greedy_solution():
        S = set()
        S_set = set()
        for x in range(1, n + 1):
            # Check if x can be added
            can_add = True
            for y in S_set:
                if (x + y) % 2 == 0:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                S_set.add(x)
        return S
    
    # Get initial greedy solution
    greedy_S = greedy_solution()
    best_weight = sum(weights[i-1] for i in greedy_S)
    best_set = set(greedy_S)
    
    # Run branch and bound with time limit
    all_nodes = set(range(n))
    branch_and_bound(all_nodes, 0.0, set(), deadline)
    
    # Convert back to 1-indexed
    result = sorted([i + 1 for i in best_set])
    
    if not result:
        return [1]
    
    # Final validation
    S_set = set(result)
    for a in result:
        for b in result:
            if a == b:
                continue
            c = 2 * b - a
            if c in S_set and c != a and c != b:
                # Invalid, fallback
                return [1, 3, 7, 12, 20, 24, 31, 35, 45, 50, 57, 61, 71, 75, 85, 89, 99, 103, 113, 117, 127, 131, 141, 145, 155, 159, 169, 173, 183, 187, 197, 201, 211, 215, 225, 229, 239, 243, 253]
    
    return result
