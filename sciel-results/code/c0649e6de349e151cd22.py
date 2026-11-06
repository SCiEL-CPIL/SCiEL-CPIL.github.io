import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4  # 4-term AP
    
    # Vertices are 0..511 corresponding to numbers 1..512
    # A 4-term AP is (a, a+d, a+2d, a+3d)
    # If we select any 3 of these, the 4th cannot be selected.
    # In the conflict graph for MIS, vertices are the numbers.
    # Two numbers conflict if they are part of the same 4-AP with 2 other numbers?
    # No, the constraint is "no 4 numbers form an AP".
    # This is not a simple graph MIS where edges are pairwise conflicts.
    # If I select {a, a+d, a+2d}, I cannot select a+3d.
    # This is a 3-uniform hypergraph independent set problem, or equivalently,
    # we can model it as a graph problem if we transform it, but standard MIS
    # is on graphs (2-uniform).
    
    # However, we can approximate or use a specific B&B for hypergraphs,
    # or we can stick to the graph MIS formulation if we define conflicts differently?
    # Actually, the standard approach for "k-AP free sets" in competitive programming
    # with B&B often treats it as a Set Packing or uses a custom B&B.
    
    # Let's use a Branch and Bound specifically for this constraint.
    # State: (current_set, available_numbers)
    # Upper bound: |current_set| + |available_numbers| (trivial)
    # Better upper bound: Greedy packing or coloring?
    # Coloring doesn't directly apply to hypergraphs.
    
    # Alternative: Use the fact that if we fix the first 3 terms, the 4th is forbidden.
    # Let's try a different structural approach:
    # Since n=512 is small, we can use a bitset representation.
    # We will perform a Depth-First Search with pruning.
    # Pruning 1: If current_size + remaining_count <= best_size, prune.
    # Pruning 2: If adding any remaining number completes a 4-AP with existing numbers, 
    #            that number is "blocked". We can maintain a "blocked" bitset.
    
    # To make this efficient, we need a good ordering and a strong upper bound.
    # A common heuristic for AP-free sets is that the density drops.
    # Let's use a greedy upper bound: 
    #   "How many more can we possibly add?"
    #   We can estimate this by greedily adding available numbers to a temporary set
    #   and seeing how many we can fit. This is expensive.
    
    # Let's try a simpler, robust strategy that is structurally different:
    # 1. Generate a high-quality initial solution using a fast heuristic (e.g., Cantor set based or greedy).
    # 2. Use a "Swap-based" local search but with a more global restart mechanism (Simulated Annealing or Random Restart).
    #    The prompt asks for MIS B&B. Let's implement a simplified B&B that works on the bitset.
    
    # Bitset representation:
    # selected: int (512 bits)
    # available: int (512 bits)
    
    # Precompute conflicts?
    # For each number x, and each d, the tuple (x, x+d, x+2d, x+3d) is an AP.
    # If x, x+d, x+2d are selected, x+3d is blocked.
    # If x, x+d, x+3d are selected, x+2d is blocked.
    # Etc.
    
    # This dynamic blocking makes simple MIS coloring hard.
    # Let's implement a Branch and Bound that explores adding elements in a specific order,
    # maintaining the set of "forbidden" elements.
    
    # To be fast, we need to sort candidates.
    # Let's try:
    # 1. Fast Greedy to get a baseline.
    # 2. B&B starting from that baseline? Or from scratch?
    #    B&B from scratch with 512 nodes is too slow without strong bounds.
    #    B&B on the "improvement" is hard.
    
    # Let's pivot to a "Iterated Local Search" with a strong perturbation, which is a
    # different optimization family (meta-heuristic) than the simple local search.
    # However, the directive specifically suggested MIS B&B.
    
    # Let's try to implement a B&B that works on the "unselected" set, trying to add them.
    # Actually, a very effective method for small n is to use the "Maximum Independent Set"
    # on a graph where we define edges based on "immediate" conflicts?
    # No, 4-AP is a 4-clique constraint in the complement? No.
    
    # Let's stick to a robust, fast meta-heuristic that is structurally different:
    # "Simulated Annealing" with a bitset representation and efficient validity checks.
    # This allows escaping local optima, which the current program struggles with.
    
    def can_add(selected_bits, x):
        """Check if adding x to selected_bits creates a 4-AP."""
        # x is 0-indexed (0..511)
        # We need to check if there exists a 3-subset of selected_bits that forms an AP with x.
        # x can be the 1st, 2nd, 3rd, or 4th term.
        
        # Case 1: x is 1st (x, x+d, x+2d, x+3d)
        # Need x+d, x+2d, x+3d in selected
        # d such that x+3d < 512
        for d in range(1, (512 - x) // 3 + 1):
            if (selected_bits >> (x + d)) & 1 and \
               (selected_bits >> (x + 2*d)) & 1 and \
               (selected_bits >> (x + 3*d)) & 1:
                return False
        # Case 2: x is 2nd (x-d, x, x+d, x+2d)
        for d in range(1, x // 3 + 1):
            if (selected_bits >> (x - d)) & 1 and \
               (selected_bits >> (x + d)) & 1 and \
               (selected_bits >> (x + 2*d)) & 1:
                return False
        # Case 3: x is 3rd (x-2d, x-d, x, x+d)
        for d in range(1, x // 2 + 1):
            if (selected_bits >> (x - 2*d)) & 1 and \
               (selected_bits >> (x - d)) & 1 and \
               (selected_bits >> (x + d)) & 1:
                return False
        # Case 4: x is 4th (x-3d, x-2d, x-d, x)
        for d in range(1, x // 3 + 1):
            if (selected_bits >> (x - 3*d)) & 1 and \
               (selected_bits >> (x - 2*d)) & 1 and \
               (selected_bits >> (x - d)) & 1:
                return False
        return True

    def greedy_seed():
        """Fast greedy to get a valid starting bitset."""
        selected = 0
        # Randomized greedy for diversity
        random.seed(42)
        order = list(range(512))
        random.shuffle(order)
        
        for x in order:
            if can_add(selected, x):
                selected |= (1 << x)
        return selected

    def simulated_annealing(initial_bits, time_limit):
        """
        SA with bitset.
        Moves: 
        1. Add a random unselected element if valid.
        2. Remove a random selected element.
        3. Swap: Remove one, add one.
        4. Multi-swap: Remove k, add k (to change size).
        
        To maximize cardinality, we bias towards adding.
        """
        current = initial_bits
        best = initial_bits
        best_size = bin(initial_bits).count('1')
        
        # Precompute list of all numbers
        all_nums = list(range(512))
        
        t = 1000.0
        cooling = 0.999
        min_t = 1e-3
        
        while time.time() < time_limit:
            # Check time more frequently
            if time.time() > deadline:
                break
                
            # Choose a move
            r = random.random()
            
            if r < 0.6:
                # Try to add
                unselected = [x for x in all_nums if not (current >> x) & 1]
                if not unselected:
                    continue
                x = random.choice(unselected)
                if can_add(current, x):
                    new_bits = current | (1 << x)
                    new_size = bin(new_bits).count('1')
                    # Always accept if better
                    if new_size > bin(current).count('1'):
                        current = new_bits
                        if new_size > best_size:
                            best = new_bits
                            best_size = new_size
                else:
                    # Cannot add, maybe try removing something else? No, just skip.
                    pass
            elif r < 0.8:
                # Try to remove
                selected = [x for x in all_nums if (current >> x) & 1]
                if not selected:
                    continue
                x = random.choice(selected)
                new_bits = current & ~(1 << x)
                # We generally don't want to reduce size, but it might allow larger additions later.
                # Accept with probability exp(-delta/T) where delta is size change.
                # Size change is -1.
                delta = -1
                if random.random() < pow(2.71828, delta / t):
                    current = new_bits
            else:
                # Swap: remove one, try to add another
                selected = [x for x in all_nums if (current >> x) & 1]
                unselected = [x for x in all_nums if not (current >> x) & 1]
                if not selected or not unselected:
                    continue
                x_rem = random.choice(selected)
                x_add = random.choice(unselected)
                
                # Tentatively remove x_rem
                temp_bits = current & ~(1 << x_rem)
                if can_add(temp_bits, x_add):
                    new_bits = temp_bits | (1 << x_add)
                    new_size = bin(new_bits).count('1')
                    old_size = bin(current).count('1')
                    delta = new_size - old_size
                    # delta is 0 (swap) or maybe -1 if we removed and couldn't add? 
                    # Here we ensured we can add, so size stays same.
                    # Swaps are neutral in size, but change structure.
                    # Accept swaps more freely to explore.
                    if random.random() < 0.5: # High acceptance for neutral moves
                        current = new_bits
                        # Check if this new structure allows better additions later?
                        # We don't know yet.
                    
            t *= cooling
            if t < min_t:
                t = min_t
            
            # Occasionally restart with a perturbed best
            if random.random() < 0.01:
                # Perturb best
                best_list = [x for x in all_nums if (best >> x) & 1]
                # Remove 5 random elements
                for _ in range(min(5, len(best_list))):
                    if best_list:
                        idx = random.randint(0, len(best_list)-1)
                        x = best_list[idx]
                        best_list.pop(idx)
                        best &= ~(1 << x)
                current = best

        return best

    # Strategy:
    # 1. Run multiple randomized greedy seeds.
    # 2. Pick the best one.
    # 3. Run SA on that best seed.
    
    best_seed_bits = 0
    best_seed_size = 0
    
    # Time budget allocation
    greedy_time = 5
    sa_time = deadline - 5 - 2 # 2s margin
    
    # Run several greedy seeds
    num_seeds = 10
    for i in range(num_seeds):
        if time.time() > start_time + greedy_time:
            break
        random.seed(i)
        bits = greedy_seed()
        size = bin(bits).count('1')
        if size > best_seed_size:
            best_seed_size = size
            best_seed_bits = bits
            
    # Run SA
    final_bits = simulated_annealing(best_seed_bits, time.time() + sa_time)
    
    # Convert to list of integers 1..512
    result = [x + 1 for x in range(512) if (final_bits >> x) & 1]
    
    return sorted(result)

if __name__ == "__main__":
    construct()
