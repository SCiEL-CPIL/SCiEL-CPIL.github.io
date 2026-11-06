"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    num_valid = len(valid_numbers)
    
    # Partition into odd and even numbers (bipartition)
    # For a + b to be a perfect square, one must be odd and one even
    # (since squares are either 0 or 1 mod 4, and odd+even is odd, 
    #  even+odd is odd, odd+odd is even, even+even is even)
    # Wait: odd + even = odd. Squares can be odd (1,9,25,...) or even (4,16,36,...)
    # Actually, the bipartition is NOT simply odd/even.
    # Let me reconsider: a + b is a perfect square.
    # If a and b are both odd: a+b is even. Even squares: 4,16,36,64,...
    # If a and b are both even: a+b is even. Same.
    # If one odd, one even: a+b is odd. Odd squares: 1,9,25,49,...
    # So the graph is NOT bipartite by parity! Both odd-odd and even-even pairs can conflict.
    
    # Let me re-examine. The conflict graph has edges between:
    # - odd and even (when sum is odd square)
    # - odd and odd (when sum is even square)  
    # - even and even (when sum is even square)
    # So it's NOT bipartite. The directive's premise is flawed.
    
    # However, I can still use a different exact approach: 
    # Since the graph structure is specific, let me try a different strategy:
    # Use maximum independent set via complement of minimum vertex cover is only for bipartite.
    # For general graphs, MIS is NP-hard.
    
    # Alternative strategy: Since the problem size is manageable (~180 valid numbers),
    # use a branch-and-bound with strong bounds, OR use the fact that we can model
    # this as a maximum independent set and solve with a specialized approach.
    
    # Actually, let me reconsider the bipartite claim more carefully.
    # a + b = square. 
    # Consider the graph where vertices are valid numbers, edges connect a,b if a+b is square.
    # This graph is NOT bipartite in general.
    
    # NEW STRATEGY: Use a different exact method - formulate as a maximum independent set
    # and solve using a branch-and-bound with greedy coloring upper bound, but with
    # much better initial solutions and more aggressive pruning.
    # 
    # Even better: Use the complement graph and find minimum vertex cover via
    # maximum matching ONLY IF the graph is bipartite. Since it's not, let me try:
    # 
    # STRATEGY: Split the problem. The conflict graph can be decomposed.
    # Actually, let me just implement a very strong branch-and-bound with bitmasks.
    
    # Build conflict sets
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Make symmetric
    for i in range(num_valid):
        for j in conflict[i]:
            conflict[j].add(i)
    
    # Precompute conflict bitmasks
    adj_bits = []
    for i in range(num_valid):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        adj_bits.append(bits)
    
    # Greedy coloring for upper bound
    def greedy_color_bound(available_mask):
        """Upper bound on independent set size using greedy coloring of available vertices."""
        uncolored = available_mask
        colors = 0
        while uncolored:
            colors += 1
            # Greedily assign this color to as many non-adjacent vertices as possible
            color_class = 0
            remaining = uncolored
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                best_v = -1
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = bin(adj_bits[v] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                # Add best_v to color class
                color_class |= (1 << best_v)
                remaining &= ~(1 << best_v)
                # Remove neighbors from remaining
                remaining &= ~adj_bits[best_v]
            uncolored &= ~color_class
        return colors
    
    # Branch and bound
    best_size = [0]
    best_set = [set()]
    nodes_visited = [0]
    
    def bb_search(available_mask, current_set, current_size):
        """Branch and bound for maximum independent set."""
        if time.time() > deadline:
            return
        nodes_visited[0] += 1
        if nodes_visited[0] % 100000 == 0 and nodes_visited[0] > 1000000:
            # Time check more frequently
            if time.time() > deadline:
                return
        
        if available_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size[0]:
            return
        
        # Pick vertex with maximum degree in available subgraph
        min_deg = float('inf')
        best_v = -1
        temp = available_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = bin(adj_bits[v] & available_mask).count('1')
            if deg > min_deg:  # Wait, we want max degree for branching
                min_deg = deg
                best_v = v
        
        if best_v == -1:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Branch 1: Include best_v
        new_available = available_mask & ~adj_bits[best_v] & ~(1 << best_v)
        bb_search(new_available, current_set | {best_v}, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_available = available_mask & ~(1 << best_v)
        bb_search(new_available, current_set, current_size)
    
    # Get a good initial solution
    def greedy_min_degree():
        """Greedy: repeatedly pick the vertex with minimum degree in remaining graph."""
        remaining = available_mask_full = 0
        for i in range(num_valid):
            available_mask_full |= (1 << i)
        
        selected = set()
        remaining = available_mask_full
        while remaining:
            min_deg = float('inf')
            best_idx = -1
            temp = remaining
            while temp:
                idx = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = bin(adj_bits[idx] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
            selected.add(best_idx)
            remaining &= ~(1 << best_idx)
            remaining &= ~adj_bits[best_idx]
        return selected
    
    initial = greedy_min_degree()
    best_size[0] = len(initial)
    best_set[0] = set(initial)
    
    # Run branch and bound
    full_mask = 0
    for i in range(num_valid):
        full_mask |= (1 << i)
    
    bb_search(full_mask, set(), 0)
    
    # Convert back to values
    if best_set[0]:
        best_values = [valid_numbers[i] for i in best_set[0]]
        # Verify validity
        best_set_vals = set(best_values)
        for a in best_set_vals:
            if 2 * a in squares:
                # Invalid, fallback
                fallback = greedy_min_degree()
                best_values = [valid_numbers[i] for i in fallback]
                break
            for b in best_set_vals:
                if b >= a and a + b in squares:
                    fallback = greedy_min_degree()
                    best_values = [valid_numbers[i] for i in fallback]
                    break
        return sorted(best_values)
    else:
        return [valid_numbers[0]]
# EVOLVE-BLOCK-END
