import time
import random
import math
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(max_val=10000, max_prime=7):
    """Generate smooth numbers up to max_val with prime factors <= max_prime."""
    smooths = set()
    primes = [2, 3, 5, 7]
    
    def gen(idx, current):
        if current > max_val:
            return
        smooths.add(current)
        if idx < len(primes):
            p = primes[idx]
            for i in range(1, 15):  # Limit exponent to keep numbers manageable
                gen(idx + 1, current * p)
                current *= p
                if current > max_val:
                    break
    
    gen(0, 1)
    return sorted(smooths)

def compute_pair_weight(a, b, A):
    """Compute the weight of including both a and b in the set A."""
    # Count how many sums/products collide with existing elements
    weight = 0
    
    # Sum collisions
    sum_collisions = 0
    for x in A:
        if x != a and x != b:
            if (a + b) == (a + x) or (a + b) == (b + x) or (a + b) == (x + x):
                sum_collisions += 1
            if (a + x) == (b + x) or (a + x) == (x + x) or (b + x) == (x + x):
                sum_collisions += 1
    
    # Product collisions
    prod_collisions = 0
    for x in A:
        if x != a and x != b:
            if (a * b) == (a * x) or (a * b) == (b * x) or (a * b) == (x * x):
                prod_collisions += 1
            if (a * x) == (b * x) or (a * x) == (x * x) or (b * x) == (x + x):
                prod_collisions += 1
    
    weight = sum_collisions + prod_collisions
    return weight

def branch_and_bound_clique(nodes, weights, k, max_weight, timeout=5.0):
    """
    Find a clique of size k with maximum total weight using branch and bound.
    nodes: list of node indices
    weights: dict mapping (i,j) to weight between nodes i and j
    k: desired clique size
    max_weight: upper bound on weight for pruning
    """
    start_time = time.time()
    best_clique = []
    best_weight = -1
    
    def dfs(current_clique, current_weight, candidates):
        nonlocal best_clique, best_weight
        
        if time.time() - start_time > timeout:
            return
        
        if len(current_clique) == k:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique[:]
            return
        
        if len(current_clique) + len(candidates) < k:
            return
        
        # Upper bound: current weight + max possible weight from remaining
        remaining_max = max_weight * (k - len(current_clique))
        if current_weight + remaining_max <= best_weight:
            return
        
        for i in range(len(candidates) - 1, -1, -1):
            node = candidates[i]
            new_candidates = [c for c in candidates[:i] if (node, c) in weights or (c, node) in weights]
            
            new_weight = current_weight
            for n in current_clique:
                w = weights.get((node, n), weights.get((n, node), 0))
                new_weight += w
            
            dfs(current_clique + [node], new_weight, new_candidates)
    
    dfs([], 0, nodes)
    return best_clique, best_weight

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate smooth numbers as candidate pool
    smooth_pool = generate_smooth_numbers(10000, 7)
    
    # Ensure we have enough candidates
    if len(smooth_pool) < 50:
        # Add some additional structured numbers
        for i in range(1, 30):
            if i not in smooth_pool:
                smooth_pool.append(i)
        smooth_pool = sorted(smooth_pool)
    
    # Limit pool size for computational tractability
    pool_size = min(len(smooth_pool), 100)
    pool = smooth_pool[:pool_size]
    
    # Compute pairwise weights
    n_pool = len(pool)
    weights = {}
    
    # For each pair, estimate collision potential
    # Higher weight = more likely to create collisions (smaller sum/product sets)
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            a, b = pool[i], pool[j]
            
            # Smooth numbers with similar magnitude create more collisions
            ratio = max(a, b) / min(a, b)
            
            # Base weight from smoothness (both have small prime factors)
            base_weight = 10.0
            
            # Bonus for close values (more sum collisions)
            if ratio < 1.5:
                base_weight += 5.0
            elif ratio < 2.0:
                base_weight += 3.0
            
            # Bonus for multiplicative relationships (more product collisions)
            if a % b == 0 or b % a == 0:
                base_weight += 8.0
            elif math.gcd(a, b) > 1:
                base_weight += 4.0
            
            weights[(i, j)] = base_weight
            weights[(j, i)] = base_weight
    
    # Phase 1: Use branch and bound to find a good initial clique
    # Start with a reasonable k (we want 20 elements)
    k_target = 20
    
    # Try multiple random orderings to get diverse starting points
    best_clique = None
    best_clique_weight = -1
    best_set = None
    best_score = float('inf')
    
    # Initial heuristic: take the first k smooth numbers
    initial_set = pool[:k_target]
    initial_score = compute_metrics(initial_set)
    best_score = initial_score
    best_set = initial_set[:]
    
    # Try branch and bound with different subsets
    if time.time() < deadline - 30.0:
        # Try a few random subsets of the pool
        for attempt in range(5):
            if time.time() > deadline - 25.0:
                break
            
            # Random subset of pool
            if n_pool > 40:
                subset_size = min(40, n_pool)
                subset_indices = random.sample(range(n_pool), subset_size)
            else:
                subset_indices = list(range(n_pool))
            
            # Run branch and bound
            clique, weight = branch_and_bound_clique(
                subset_indices, weights, k_target, 20.0, timeout=5.0
            )
            
            if clique and len(clique) == k_target:
                candidate_set = [pool[i] for i in clique]
                score = compute_metrics(candidate_set)
                
                if score < best_score:
                    best_score = score
                    best_set = candidate_set[:]
    
    # Phase 2: Local search refinement
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated annealing with focus on smooth number replacements
    T_start = 5.0
    T_end = 0.1
    num_steps = 100000
    
    step = 0
    while step < num_steps:
        if time.time() > deadline:
            break
        
        step += 1
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # Choose a random index to replace
        idx = random.randint(0, 19)
        old_val = current_set[idx]
        
        # Choose a new value from smooth pool
        new_val = random.choice(pool)
        
        if new_val in current_set:
            continue
        
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val
    
    return sorted(best_set)
