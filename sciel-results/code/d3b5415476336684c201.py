import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    
    # Build conflict adjacency as bitmasks
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Precompute degrees
    degrees = [bin(conflicts[i]).count('1') for i in range(n)]
    
    # Initial greedy independent set
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflicts[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflicts[best_v])
        return chosen
    
    # Get initial solution
    best_mask = greedy_mis()
    best_size = bin(best_mask).count('1')
    
    # Local search: try to add vertices or swap
    def local_search(initial_mask, time_limit):
        current = initial_mask
        current_size = bin(current).count('1')
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each non-member vertex if it doesn't conflict
            for v in range(n):
                if not (current & (1 << v)):
                    if not (conflicts[v] & current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
            # Try 1-for-1 swaps
            for v in range(n):
                if current & (1 << v):
                    current_without_v = current & ~(1 << v)
                    for u in range(n):
                        if not (current & (1 << u)):
                            if not (conflicts[u] & current_without_v):
                                new_size = bin(current_without_v | (1 << u)).count('1')
                                if new_size > current_size:
                                    current = current_without_v | (1 << u)
                                    current_size = new_size
                                    improved = True
        return current, bin(current).count('1')
    
    best_mask, best_size = local_search(best_mask, deadline - 10)
    
    # Simulated annealing
    def simulated_annealing(initial_mask, time_limit):
        random.seed(42)
        current = initial_mask
        current_size = bin(current).count('1')
        best_sa = current
        best_sa_size = current_size
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        while T > min_T and time.time() < time_limit:
            # Generate a neighbor: random swap
            in_set = [i for i in range(n) if current & (1 << i)]
            not_in_set = [i for i in range(n) if not (current & (1 << i))]
            
            if not in_set or not not_in_set:
                break
            
            v = random.choice(in_set)
            u = random.choice(not_in_set)
            
            # Try swap: remove v, add u
            candidate = current & ~(1 << v)
            candidate |= (1 << u)
            
            # Check if valid
            valid = True
            for i in range(n):
                if candidate & (1 << i):
                    if i != u and (conflicts[u] & (1 << i)):
                        valid = False
                        break
            
            if valid:
                cand_size = bin(candidate).count('1')
                delta = cand_size - current_size
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current = candidate
                    current_size = cand_size
                    
                    if current_size > best_sa_size:
                        best_sa = current
                        best_sa_size = current_size
            
            T *= cooling
        
        return best_sa, best_sa_size
    
    # Run simulated annealing with multiple restarts
    sa_time_limit = deadline - 15
    num_sa_runs = 5
    
    for run in range(num_sa_runs):
        if time.time() > sa_time_limit:
            break
        
        # Add some random perturbation for diversity
        perturbed = best_mask
        if run > 0:
            # Remove 2-3 random elements and re-add
            in_set = [i for i in range(n) if perturbed & (1 << i)]
            if len(in_set) > 3:
                to_remove = random.sample(in_set, min(3, len(in_set) // 4))
                for v in to_remove:
                    perturbed &= ~(1 << v)
                perturbed, _ = local_search(perturbed, start_time + 0.5)
        
        sa_result, sa_size = simulated_annealing(perturbed, sa_time_limit)
        
        if sa_size > best_size:
            best_mask = sa_result
            best_size = sa_size
    
    # Final local search polish
    best_mask, best_size = local_search(best_mask, deadline - 5)
    
    # Extract the result
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    # Final verification: ensure no pair sum is a perfect square
    for idx_a in range(len(result_nodes)):
        for idx_b in range(idx_a, len(result_nodes)):
            s = result_nodes[idx_a] + result_nodes[idx_b]
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid - fallback to a single valid node
                return [valid_nodes[0]]
    
    return result_nodes
