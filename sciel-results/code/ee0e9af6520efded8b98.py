import time
import random
import math

MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def is_P_smooth(n, P):
    """Check if n is P-smooth (all prime factors in P)"""
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    while n % 7 == 0:
        n //= 7
    while n % 11 == 0:
        n //= 11
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Precompute prime sets to try
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    # Generate P-smooth numbers for each set
    def generate_smooth_numbers(P, limit=20000):
        """Generate all P-smooth numbers up to limit"""
        smooth = set()
        
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(P):
                smooth.add(current)
                return
            p = P[idx]
            val = current
            while val <= limit:
                smooth.add(val)
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        # Remove 1 since we need positive integers and sums >= 2
        smooth.discard(1)
        return smooth
    
    best_found = None
    best_cost = float('inf')
    
    for P in prime_sets:
        if time.time() > deadline - 10:
            break
            
        # Generate P-smooth numbers
        smooth_nums = generate_smooth_numbers(P)
        
        # Build adjacency list
        # adj[x] is set of y such that x+y is P-smooth
        adj = [set() for _ in range(10001)]
        
        for S in smooth_nums:
            # x + y = S => y = S - x
            low = max(1, S - 10000)
            high = min(10000, S - 1)
            for x in range(low, high + 1):
                y = S - x
                if 1 <= y <= 10000 and x != y:
                    adj[x].add(y)
        
        # Find clique of size 12 using randomized backtracking
        # Sort nodes by degree descending
        nodes = list(range(1, 10001))
        nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Try to find clique from high-degree nodes
        found_clique = None
        
        # Randomized search with multiple restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Start from a random high-degree node
            start_idx = random.randint(0, min(200, len(nodes) - 1))
            start_node = nodes[start_idx]
            
            # Backtracking search
            def find_clique(current_clique, candidates, target_size):
                if len(current_clique) == target_size:
                    return current_clique[:]
                
                if len(candidates) < target_size - len(current_clique):
                    return None
                
                # Filter candidates that are connected to all in current_clique
                valid_candidates = []
                for v in candidates:
                    is_connected = True
                    for u in current_clique:
                        if v not in adj[u]:
                            is_connected = False
                            break
                    if is_connected:
                        valid_candidates.append(v)
                
                if len(valid_candidates) < target_size - len(current_clique):
                    return None
                
                # Try each valid candidate
                for i, v in enumerate(valid_candidates):
                    new_clique = current_clique + [v]
                    new_candidates = [u for u in valid_candidates[i+1:] if u in adj[v]]
                    
                    res = find_clique(new_clique, new_candidates, target_size)
                    if res:
                        return res
                
                return None
            
            # Initial candidates: top 500 high-degree nodes
            initial_candidates = nodes[:500]
            res = find_clique([start_node], initial_candidates, 12)
            if res:
                found_clique = res
                break
        
        if found_clique:
            cost = count_distinct_primes(found_clique)
            if cost <= len(P):
                best_found = found_clique
                best_cost = cost
                if cost <= 2:
                    break
    
    # If we found a good solution, return it
    if best_found is not None and best_cost <= 4:
        return best_found
    
    # Fallback: Simulated Annealing with better initialization
    # Start with a set that has low prime count
    best_arr = [3 + 6 * k for k in range(12)]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() < deadline - 2:
        if T < 0.01:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr
