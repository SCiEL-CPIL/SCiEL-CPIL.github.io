import time
import random

def construct():
    N = 512
    # Indices 0 to 511 correspond to numbers 1 to 512
    
    # Precompute 4-APs involving each element as the "new" element?
    # Or better: Precompute for each element z, the set of "forbidden triples" in S.
    # If S contains any of these triples, z cannot be added.
    # Actually, it's easier to check: Does S contain a 4-AP?
    # For adding z, we check if there exists d such that {z-d, z-2d, z-3d} subset S (z is 4th), etc.
    # The current is_valid_addition does this in O(N) per check.
    # With N=512, 512 checks * 512 elements = 262k checks per full scan.
    # In Python, 262k * (loop overhead) might be ~1-2 seconds per full scan.
    # 60s allows ~30-60 full scans.
    
    # Let's optimize is_valid_addition using precomputed lists of potential conflicts.
    # For each z, precompute list of (i, j, k) indices such that {i, j, k, z} is a 4-AP.
    # There are at most ~N/2 such APs per z? No, number of APs containing z is ~N.
    # Total APs is ~N^2/12. Total (z, triple) pairs is ~N^2/3.
    # 512^2 / 3 ~ 87k pairs.
    # Store conflicts[z] = list of (i, j, k) tuples.
    
    conflicts = [[] for _ in range(N)]
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3*d + 1):
            a0 = a - 1
            a1 = a + d - 1
            a2 = a + 2*d - 1
            a3 = a + 3*d - 1
            
            # z is a0: conflict is (a1, a2, a3)
            conflicts[a0].append((a1, a2, a3))
            # z is a1: conflict is (a0, a2, a3)
            conflicts[a1].append((a0, a2, a3))
            # z is a2: conflict is (a0, a1, a3)
            conflicts[a2].append((a0, a1, a3))
            # z is a3: conflict is (a0, a1, a2)
            conflicts[a3].append((a0, a1, a2))
            
    def is_valid_addition(S, z_idx):
        """
        Check if adding z_idx to S creates a 4-AP.
        S is a bitmask.
        """
        for i, j, k in conflicts[z_idx]:
            # Check if i, j, k are all in S
            # Bit check: (S >> idx) & 1
            if ((S >> i) & 1) and ((S >> j) & 1) and ((S >> k) & 1):
                return False
        return True

    def get_size(S):
        return bin(S).count('1')

    start_time = time.time()
    deadline = start_time + 58.0 # Safety margin

    best_S = 0
    best_size = 0

    # 1. Greedy Initialization with multiple random orders
    # We will run several greedy passes to get a good starting point.
    for _ in range(10):
        order = list(range(N))
        random.shuffle(order)
        S = 0
        size = 0
        for z_idx in order:
            if is_valid_addition(S, z_idx):
                S |= (1 << z_idx)
                size += 1
        if size > best_size:
            best_size = size
            best_S = S

    # 2. Simulated Annealing
    # Current state: S
    # Move: Remove a random element from S, then try to add a random element from ~S.
    # Or: Try to swap.
    # Let's use a simple local search with restarts and randomization.
    # "Iterated Local Search": 
    #   Local Search: Greedy Refill (add any valid element).
    #   Perturbation: Remove k random elements.
    #   Repeat.
    
    current_S = best_S
    current_size = best_size
    
    # Greedy Refill Function
    def greedy_refill(S):
        size = get_size(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element not in S
            # To speed up, we can iterate in random order or fixed order.
            # Fixed order is deterministic, random might find different fillings.
            for b_idx in range(N):
                if not ((S >> b_idx) & 1):
                    if is_valid_addition(S, b_idx):
                        S |= (1 << b_idx)
                        size += 1
                        improved = True
        return S, size

    # Run Greedy Refill on best_S first
    best_S, best_size = greedy_refill(best_S)

    # Iterated Local Search
    # We will try to improve best_S.
    # Strategy: 
    # 1. Take current best.
    # 2. Remove a small number of random elements (perturbation).
    # 3. Greedy Refill.
    # 4. If better, update best.
    
    # How many to remove? 1 to 5.
    # How many iterations? Until deadline.
    
    # Optimization: is_valid_addition is the bottleneck.
    # With precomputed conflicts, it's O(num_conflicts[z]).
    # Average conflicts per z is ~ (N^2/3)/N = N/3 ~ 170.
    # So ~170 bit checks per validity check.
    # Greedy refill scans 512 elements. 512 * 170 ~ 87k ops.
    # One refill pass is fast (~0.01s in Py if optimized, maybe 0.05s).
    # We can do hundreds of iterations.
    
    while time.time() < deadline:
        # Perturb
        S_trial = best_S
        # Remove 1 to 3 random elements
        num_remove = random.randint(1, 3)
        S_bits = [i for i in range(N) if (S_trial >> i) & 1]
        if len(S_bits) <= num_remove:
            continue
        to_remove = random.sample(S_bits, num_remove)
        for r_idx in to_remove:
            S_trial &= ~(1 << r_idx)
        
        # Refill
        S_trial, size_trial = greedy_refill(S_trial)
        
        if size_trial > best_size:
            best_size = size_trial
            best_S = S_trial
            # Reset perturbation strength? No, keep it.
            
        # If we are stuck for a while, maybe try larger perturbation?
        # For now, fixed small perturbation is robust.

    # Convert bitmask to list of values (1-indexed)
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
