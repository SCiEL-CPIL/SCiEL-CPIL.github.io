import random
import time
import math
import itertools

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n > 4, use Simulated Annealing with efficient cost updates
    def get_standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    std_word = get_standard_superperm(n)
    
    # Extract standard order of permutations
    std_order = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        w = tuple(std_word[i:i+n])
        if w not in seen:
            seen.add(w)
            std_order.append(w)
            
    if len(std_order) != math.factorial(n):
        return "".join(map(str, std_word))
        
    # Precompute overlap function
    def overlap(p1, p2):
        # Max k such that p1[-k:] == p2[:k]
        # p1, p2 are tuples of length n
        # We can use slicing
        for k in range(n, -1, -1):
            if p1[n-k:] == p2[:k]:
                return k
        return 0

    # Initialize state
    order = std_order[:]
    N = len(order)
    
    # costs[i] is the cost of edge from order[i] to order[i+1]
    # cost = n - overlap
    costs = [0] * (N - 1)
    for i in range(N - 1):
        costs[i] = n - overlap(order[i], order[i+1])
        
    current_length = n + sum(costs)
    
    start_time = time.time()
    time_limit = 110.0
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.9999
    min_T = 1e-5
    
    best_order = order[:]
    best_length = current_length
    
    # Helper to update costs after a swap
    # We will implement moves inline for speed
    
    while time.time() - start_time < time_limit:
        # Choose a move: Adjacent Swap or Random Swap
        # 80% Adjacent, 20% Random
        if random.random() < 0.8:
            i = random.randint(0, N - 2)
            j = i + 1
            # Adjacent swap i, j
            # Affected edges: (i-1, i), (i, i+1), (i+1, i+2)
            # Indices in costs array:
            # Edge (i-1, i) is at index i-1
            # Edge (i, i+1) is at index i
            # Edge (i+1, i+2) is at index i+1
            
            # Calculate old delta
            old_cost = 0
            if i > 0:
                old_cost += costs[i-1]
            old_cost += costs[i]
            if i < N - 2:
                old_cost += costs[i+1]
                
            # Perform swap
            order[i], order[j] = order[j], order[i]
            
            # Calculate new delta
            new_cost = 0
            if i > 0:
                new_cost += n - overlap(order[i-1], order[i])
            new_cost += n - overlap(order[i], order[i+1])
            if i < N - 2:
                new_cost += n - overlap(order[i+1], order[i+2])
                
            delta = new_cost - old_cost
            
            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Update costs array
                if i > 0:
                    costs[i-1] = n - overlap(order[i-1], order[i])
                costs[i] = n - overlap(order[i], order[i+1])
                if i < N - 2:
                    costs[i+1] = n - overlap(order[i+1], order[i+2])
                    
                current_length += delta
                if current_length < best_length:
                    best_length = current_length
                    best_order = order[:]
            else:
                # Revert swap
                order[i], order[j] = order[j], order[i]
                
        else:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if j == i + 1:
                continue # Already handled as adjacent
            
            # Random swap i, j (non-adjacent)
            # Affected edges:
            # (i-1, i) -> (i-1, j)
            # (i, i+1) -> (j, i+1)
            # (j-1, j) -> (j-1, i)
            # (j, j+1) -> (i, j+1)
            
            old_cost = 0
            if i > 0:
                old_cost += costs[i-1]
            old_cost += costs[i]
            old_cost += costs[j-1]
            if j < N - 1:
                old_cost += costs[j]
                
            # Perform swap
            order[i], order[j] = order[j], order[i]
            
            new_cost = 0
            if i > 0:
                new_cost += n - overlap(order[i-1], order[i])
            new_cost += n - overlap(order[i], order[i+1])
            new_cost += n - overlap(order[j-1], order[j])
            if j < N - 1:
                new_cost += n - overlap(order[j], order[j+1])
                
            delta = new_cost - old_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if i > 0:
                    costs[i-1] = n - overlap(order[i-1], order[i])
                costs[i] = n - overlap(order[i], order[i+1])
                costs[j-1] = n - overlap(order[j-1], order[j])
                if j < N - 1:
                    costs[j] = n - overlap(order[j], order[j+1])
                    
                current_length += delta
                if current_length < best_length:
                    best_length = current_length
                    best_order = order[:]
            else:
                order[i], order[j] = order[j], order[i]
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Reconstruct word from best_order
    word = list(best_order[0])
    for i in range(1, len(best_order)):
        p_prev = best_order[i-1]
        p_curr = best_order[i]
        ov = overlap(p_prev, p_curr)
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
