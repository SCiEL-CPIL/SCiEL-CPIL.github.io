"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Precompute primes up to 1000
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Map prime to index for quick lookup
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Helper functions
    def get_coverage(classes):
        """
        Returns the largest L such that every integer 1..L is covered.
        classes: list of (p, r)
        """
        if not classes:
            return 0
        # We only need to check up to a reasonable bound. 
        # If we cover 1..10000, that's already very good. 
        # The task caps score at 10^5, but 999 is the target.
        # Let's check up to 20000 to be safe for "large L".
        max_check = 20000
        
        # Optimization: If we have many classes, checking each n in 1..L is O(L * len(classes)).
        # L=1000, len(classes)~100 -> 100k ops. Very fast.
        # L=10000, len(classes)~100 -> 1M ops. Still fast in Python if optimized? 
        # Better: Precompute coverage bitset? 
        # Let's just loop. Python loops are slow, so we need to be careful.
        # However, we can short-circuit.
        
        # Alternative: For each class (p, r), it covers r, r+p, r+2p...
        # We want the smallest n >= 1 NOT covered.
        # We can maintain a boolean array `covered` of size max_check+1.
        # But re-initializing and filling it every time is O(L * len(classes)).
        
        # Let's try a direct check loop first, as L is likely smallish (< 2000) for this budget.
        # If L grows large, this becomes the bottleneck.
        
        # Efficient check:
        # For n in 1..max_check:
        #   is n covered?
        #   if not, return n-1
        
        # To speed up "is n covered", we can precompute for each n which primes could cover it?
        # No, p varies.
        
        # Let's stick to the direct loop but optimize it.
        # We can break early.
        
        # Actually, let's use a bitset approach with integers? 
        # Or just a list of booleans.
        
        # Given the time limit, let's assume L < 5000.
        # 5000 * 100 checks = 500k operations. ~0.1s in Python.
        # If we do 1000 iterations, that's 100s. Too slow.
        
        # We need a faster coverage check.
        # Idea: Maintain a global `covered` array and update it incrementally?
        # Adding a class: mark r, r+p, ... as covered.
        # Removing a class: unmark. But if another class covers it, we shouldn't unmark.
        # This requires reference counting.
        
        # Let's implement reference counting coverage.
        # State: 
        #   current_classes: set of indices into `primes` or list of (p,r)
        #   ref_count: array of size MAX_N, where ref_count[n] is how many classes cover n.
        
        MAX_N = 20000
        ref_count = [0] * (MAX_N + 1)
        
        def add_class(p, r):
            for n in range(r if r > 0 else p, MAX_N + 1, p):
                # Note: residue r is in 0..p-1.
                # The numbers covered are r, r+p, r+2p...
                # But we only care about n >= 1.
                # If r == 0, the numbers are p, 2p, ... (since 0 is not in 1..L)
                # If r > 0, the numbers are r, r+p, ...
                # Wait, if r=0, it covers multiples of p. The first one >=1 is p.
                # If r>0, the first one is r (if r>=1).
                if n >= 1:
                    ref_count[n] += 1

        def remove_class(p, r):
            for n in range(r if r > 0 else p, MAX_N + 1, p):
                if n >= 1:
                    ref_count[n] -= 1

        def get_L():
            for n in range(1, MAX_N + 1):
                if ref_count[n] == 0:
                    return n - 1
            return MAX_N

        # 3. Initialization: Start with an empty set or a greedy start?
        # Greedy start: Pick the class that covers the most new numbers in 1..1000?
        # Or just start with the seed solution and improve?
        # The seed solution covers 1..~1000? 
        # Seed: p_i covers i. So 1..100 are covered by first 100 primes?
        # Actually seed maps i to (i+1)th prime. 
        # p_0=13 covers 1. p_1=17 covers 2. ...
        # Sum 1/p for primes 13..~1000 is < 1.
        # Let's start with the seed solution to have a valid baseline.
        
        seed_classes = []
        seed_primes = [p for p in primes if p >= 13]
        # Seed uses primes from 13 up.
        # Let's just construct the seed exactly as in the prompt.
        # "i-th prime from 13 upward covers integer i"
        # primes_list = [13, 17, 19, ...]
        # class i: [p_i, (i+1) % p_i]
        
        for i, p in enumerate(seed_primes):
            seed_classes.append((p, (i + 1) % p))
            
        # Initialize ref_count with seed classes
        for p, r in seed_classes:
            add_class(p, r)
            
        current_L = get_L()
        best_L = current_L
        best_classes = seed_classes.copy()
        
        # Calculate current sum of reciprocals
        current_sum = sum(Fraction(1, p) for p, _ in seed_classes)
        
        # 4. Local Search / Simulated Annealing
        # Moves:
        # A. Add a random prime p and residue r if sum + 1/p <= 1.
        # B. Remove a random class.
        # C. Change residue of an existing class.
        # D. Swap a class with another.
        
        # To make it faster, we will limit the search.
        # We have ~140s. 
        # Each step involves:
        # 1. Update ref_count (O(MAX_N/p) ~ 20000/10 ~ 2000 ops)
        # 2. Calculate new L (O(MAX_N) ~ 20000 ops)
        # 3. Check sum (O(1) if tracked)
        
        # Total per step ~ 22k ops.
        # In Python, 22k ops might take ~5-10ms.
        # 140s / 0.01s = 14,000 steps.
        # This is a decent amount of local search.
        
        # We can optimize get_L by only checking up to current_L + some delta?
        # No, adding a class might cover a large gap far away? Unlikely if we want contiguous prefix.
        # But removing a class might break the prefix early.
        # So we must check from 1 upwards.
        
        # Optimization: 
        # Instead of full ref_count update and full get_L, 
        # can we just check the affected range?
        # Hard to do robustly. Let's stick to full update but optimize the loops.
        
        # Let's try to reduce MAX_N to 10000 to speed up.
        # The target is 999. 10000 is plenty of headroom.
        
        # Reset MAX_N for the internal logic if needed, but we already allocated.
        # Let's just run the loop.
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline:
            if T < T_min:
                break
                
            # Choose a move
            move_type = random.random()
            
            new_sum = current_sum
            # We need to be careful with Fraction arithmetic speed.
            # It's slow. 
            # Alternative: Use float for quick check, verify with Fraction only if close to limit?
            # Or just use Fraction, it's exact.
            
            if move_type < 0.3:
                # Add a class
                # Find a prime not in current classes
                current_primes_set = set(p for p, _ in current_classes)
                available_primes = [p for p in primes if p not in current_primes_set]
                if not available_primes:
                    continue
                    
                p = random.choice(available_primes)
                r = random.randint(0, p - 1)
                
                # Check budget
                # current_sum + 1/p <= 1
                # To avoid slow Fraction addition every time, 
                # we can track sum as float for rejection, and use Fraction for acceptance?
                # Risk: float precision. 1/p is small.
                # Let's use Fraction. It's the safe way.
                
                new_sum = current_sum + Fraction(1, p)
                if new_sum > 1:
                    continue
                    
                # Apply move
                add_class(p, r)
                new_L = get_L()
                
                # Acceptance
                if new_L >= best_L:
                    # Accept and update best
                    best_L = new_L
                    best_classes = current_classes.copy()
                    best_classes.append((p, r))
                    current_classes = best_classes
                    current_sum = new_sum
                    current_L = new_L
                else:
                    # Simulated annealing acceptance
                    delta = new_L - current_L
                    if delta < 0:
                        prob = pow(2.71828, delta / T)
                        if random.random() < prob:
                            # Accept worse move
                            current_classes.append((p, r))
                            current_sum = new_sum
                            current_L = new_L
                        else:
                            # Reject
                            remove_class(p, r)
                            # current_sum, current_L unchanged
                    else:
                        # Accept better (but not new best)
                        current_classes.append((p, r))
                        current_sum = new_sum
                        current_L = new_L
                        
            elif move_type < 0.6:
                # Remove a class
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                p, r = current_classes[idx]
                
                remove_class(p, r)
                new_L = get_L()
                new_sum = current_sum - Fraction(1, p)
                
                # Removing a class frees up budget, so we can always accept if L doesn't drop too much?
                # Actually, we want to maximize L. Removing usually decreases L.
                # We only accept if it leads to a better state later? 
                # In SA, we accept worse moves to escape local minima.
                
                delta = new_L - current_L
                if delta >= 0:
                    # Improved or same
                    current_classes.pop(idx)
                    current_sum = new_sum
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = current_classes.copy()
                else:
                    prob = pow(2.71828, delta / T)
                    if random.random() < prob:
                        current_classes.pop(idx)
                        current_sum = new_sum
                        current_L = new_L
                    else:
                        add_class(p, r) # Revert
            else:
                # Change residue
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                p, r_old = current_classes[idx]
                r_new = random.randint(0, p - 1)
                if r_new == r_old:
                    continue
                
                # Remove old, add new
                remove_class(p, r_old)
                add_class(p, r_new)
                new_L = get_L()
                # Sum doesn't change
                
                delta = new_L - current_L
                if delta >= 0:
                    current_classes[idx] = (p, r_new)
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = current_classes.copy()
                else:
                    prob = pow(2.71828, delta / T)
                    if random.random() < prob:
                        current_classes[idx] = (p, r_new)
                        current_L = new_L
                    else:
                        # Revert
                        remove_class(p, r_new)
                        add_class(p, r_old)
                        
            T *= cooling
            
        # Final check: Return the best classes found
        # Ensure the best classes satisfy the constraint
        # (They should, by construction, but double check)
        final_sum = sum(Fraction(1, p) for p, _ in best_classes)
        if final_sum > 1:
            # Fallback to seed if something went wrong
            return [[p, r] for p, r in seed_classes]
            
        return [[p, r] for p, r in best_classes]

# EVOLVE-BLOCK-END
