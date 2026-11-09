import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    # Precompute divisors for numbers 1..1000
    # divisors[n] = list of indices of primes that divide n
    # Note: We only care about primes <= 1000.
    divisors = [[] for _ in range(1001)]
    for idx, p in enumerate(primes):
        for multiple in range(p, 1001, p):
            divisors[multiple].append(idx)
            
    # Strategy: 
    # We want to cover 1..L. 
    # The "bottleneck" numbers are those with few prime factors.
    # If we select a prime p, we must choose a residue r.
    # For a number n to be covered by p, we need r == n % p.
    # 
    # This is a hitting set / covering problem with a budget constraint.
    # 
    # Let's try a different angle: 
    # Since sum(1/p) <= 1, we can afford roughly the first 10-15 small primes?
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 + 1/37
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 + 0.052 + 0.043 + 0.034 + 0.032 + 0.027
    # Sum ~ 1.58. Too high.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 + 0.052 + 0.043 = 1.49. Still high.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 + 0.052 = 1.447.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 = 1.395.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 = 1.337.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 = 1.26.
    # 1/2 + 1/3 + 1/5 + 1/7
    # 0.5 + 0.33 + 0.2 + 0.14 = 1.17.
    # 1/2 + 1/3 + 1/5
    # 0.5 + 0.33 + 0.2 = 1.03.
    # 1/2 + 1/3
    # 0.83.
    # So we can definitely use {2, 3}. Remaining budget 1/6.
    # We can add 7 (1/7 = 0.14 > 0.16? No, 1/6 = 0.166).
    # 1/2 + 1/3 + 1/7 = 0.976. Remaining 0.024.
    # Next smallest prime 11 (1/11 = 0.09). Can't add.
    # So {2, 3, 7} is a strong base.
    # Or {2, 5, 7, 11, 13, ...}?
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31
    # 0.5 + 0.2 + 0.142 + 0.09 + 0.076 + 0.058 + 0.052 + 0.043 + 0.034 + 0.032 = 1.227.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29
    # 1.195.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23
    # 1.161.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19
    # 1.127.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17
    # 1.075.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13
    # 1.017.
    # 1/2 + 1/5 + 1/7 + 1/11
    # 0.941. Remaining 0.059.
    # Can add 17 (0.058). 
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/17 = 0.999.
    # This set {2, 5, 7, 11, 17} is very efficient.
    
    # Let's use a randomized local search on the set of selected primes and their residues.
    # State: 
    #   selected_primes: list of prime indices
    #   residues: dict mapping prime index to residue
    #   
    # Objective: Maximize L.
    # 
    # We will try several random sets of primes that satisfy the budget, and for each, 
    # solve for the best residues using a greedy/iterative method.
    
    # Generate candidate sets of primes
    # We want sets where sum(1/p) <= 1.
    # We'll generate random subsets by shuffling primes and adding them if budget allows.
    
    best_L = 0
    best_classes = []
    
    # Precompute masks for all primes/residues? 
    # Too much memory if we store all, but we can compute on the fly or cache.
    # Let's compute coverage on the fly for the specific set of primes.
    
    def evaluate_solution(prime_indices, residues):
        """
        prime_indices: list of indices into 'primes' list
        residues: dict {index: residue}
        Returns L (largest prefix covered)
        """
        if not prime_indices:
            return 0
        
        # Build coverage bitset
        covered = 0
        for idx in prime_indices:
            p = primes[idx]
            r = residues[idx]
            # Add mask for this class
            # Mask for (p, r): bits set for n in 1..1000 where n % p == r
            # We can compute this quickly.
            # Optimization: Precompute masks for all (p, r) is 168*avg(500) ~ 84k masks.
            # Each mask is an int (1000 bits ~ 125 bytes). 84k * 125 ~ 10MB. Acceptable.
            pass
            
        # Actually, let's precompute all masks once.
        return 0 # Placeholder

    # Let's precompute masks globally to speed up evaluation
    masks = {}
    for idx, p in enumerate(primes):
        masks[idx] = [0] * p
        for r in range(p):
            mask = 0
            if r == 0:
                start = p
            else:
                start = r
            n = start
            while n <= 1000:
                mask |= (1 << (n - 1))
                n += p
            masks[idx][r] = mask

    def calculate_L(covered_mask):
        L = 0
        # Check prefix
        # We want largest L such that bits 0..L-1 are 1.
        # If bit k is 0, L <= k.
        # We can iterate.
        for i in range(1000):
            if not (covered_mask & (1 << i)):
                break
            L = i + 1
        return L

    # Generate diverse prime sets
    # Strategy: 
    # 1. Always include small primes? Not necessarily. 
    # 2. Random subsets.
    
    # Let's try a specific heuristic: 
    # Sort primes by 1/p (cost). 
    # Greedy add primes that maximize "potential coverage" per cost?
    # But residues matter.
    
    # Let's run a loop of random restarts.
    # In each restart:
    # 1. Generate a random set of primes S such that sum(1/p) <= 1.
    # 2. For this set S, find optimal residues.
    #    This is a constraint satisfaction problem.
    #    We can use a simple greedy: 
    #    Start with no residues assigned.
    #    Iteratively assign residues to primes to cover the most new numbers.
    #    Or, use local search on residues.
    
    # Let's implement a function to optimize residues for a fixed set of primes.
    def optimize_residues(prime_indices):
        """
        Given a list of prime indices, find residues that maximize L.
        Uses local search / hill climbing.
        """
        # Initial assignment: random
        current_residues = {idx: random.randint(0, primes[idx]-1) for idx in prime_indices}
        
        # Calculate initial coverage
        covered = 0
        for idx in prime_indices:
            covered |= masks[idx][current_residues[idx]]
        
        current_L = calculate_L(covered)
        best_L_local = current_L
        best_residues_local = current_residues.copy()
        
        # Hill climbing
        # For each prime, try changing its residue to any other residue.
        # Keep track of improvements.
        
        # To speed up, we can precompute the "contribution" of each prime-residue.
        # But interactions matter (overlap).
        
        # Simple local search:
        # Repeat until no improvement or time limit.
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Shuffle order to avoid bias
            indices_list = prime_indices.copy()
            random.shuffle(indices_list)
            
            for idx in indices_list:
                p = primes[idx]
                current_r = current_residues[idx]
                current_mask = masks[idx][current_r]
                
                # Try all other residues? 
                # p can be up to 1000. 
                # If p is large, this is expensive.
                # Optimization: Only try residues that cover at least one currently uncovered number?
                
                # Calculate uncovered numbers
                # We can't easily get a list of uncovered numbers without scanning.
                # Let's just try a random subset of residues if p is large, or all if small.
                
                candidates = []
                if p < 50:
                    candidates = [r for r in range(p) if r != current_r]
                else:
                    # Sample
                    candidates = [random.randint(0, p-1) for _ in range(min(10, p))]
                    # Ensure current_r is not the only one, but we skip it anyway
                    candidates = [r for r in candidates if r != current_r]
                
                best_r = current_r
                best_gain = 0
                best_new_cover = covered
                
                # We need to calculate new cover for each candidate.
                # new_cover = (covered ^ current_mask) | masks[idx][r_new]
                # Note: covered includes current_mask.
                # So covered_without_current = covered & ~current_mask
                
                covered_without_current = covered & ~current_mask
                
                for r in candidates:
                    new_mask = masks[idx][r]
                    new_cover = covered_without_current | new_mask
                    # Calculate L for new_cover
                    # This is O(1000) per candidate. 
                    # If we have 168 primes * 10 candidates * 1000 ops = 1.6M ops per iteration.
                    # This might be slow in Python.
                    
                    # Optimization: Estimate L gain.
                    # We can check if the new cover covers more of the prefix.
                    # Find first gap in new_cover.
                    # Instead of full calculate_L, we can check specific bits.
                    
                    # Let's just do a quick check: 
                    # If new_cover has more bits set in the prefix than current_cover, it's likely better.
                    # But L is about the prefix.
                    
                    # Let's compute L for the best few candidates.
                    # Actually, let's just compute L for all candidates if p is small.
                    
                    # To make this faster, let's precompute a "prefix score" function?
                    # Or just accept the cost if we limit restarts.
                    
                    # Let's try to optimize calculate_L.
                    # It's a loop of 1000.
                    
                    # Alternative: 
                    # We can maintain a bitset of "uncovered in prefix".
                    
                    # For now, let's just run it and see if it fits in time.
                    # If too slow, we reduce candidates.
                    
                    # Let's calculate L
                    l_val = 0
                    # Unroll? No.
                    # We can break early if we find a gap before current_L.
                    # But we want to know if it's better.
                    
                    # Let's just use a helper that is slightly faster.
                    # Actually, the bottleneck is the inner loop.
                    
                    # Let's try a different metric: 
                    # Gain in coverage of numbers 1..1000.
                    # But L is the metric.
                    
                    # Let's stick to calculate_L but only for the best 3 candidates by "raw gain".
                    
                    raw_gain = (new_mask & ~covered).bit_count()
                    if raw_gain > best_gain:
                        best_gain = raw_gain
                        best_r = r
                        best_new_cover = new_cover
                        # Don't break, might find better L with same gain? Unlikely.
                        # But L depends on position.
                        # A gain at the start is better than at the end.
                        
                # If best_r != current_r, update
                if best_r != current_r:
                    # Verify L improvement
                    new_L = calculate_L(best_new_cover)
                    if new_L > current_L:
                        current_residues[idx] = best_r
                        covered = best_new_cover
                        current_L = new_L
                        if new_L > best_L_local:
                            best_L_local = new_L
                            best_residues_local = current_residues.copy()
                        improved = True
                    else:
                        # No improvement in L, but maybe raw gain? 
                        # If L didn't improve, don't update to avoid cycles?
                        # Or update if raw gain is significantly higher?
                        # Let's stick to L improvement for stability.
                        pass
                        
        return best_L_local, best_residues_local

    # Main Loop: Random Prime Set Selection
    # We will generate 50 random sets of primes.
    
    # To ensure we explore good sets, we can bias towards small primes.
    # Sort primes by value.
    # Generate subsets by picking a random "cutoff" or random inclusion probability decreasing with size.
    
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random prime set
        # Method: 
        # 1. Shuffle primes.
        # 2. Iterate, adding if budget allows.
        # To get diversity, we can randomize the "greediness".
        
        shuffled_primes = primes.copy()
        random.shuffle(shuffled_primes)
        
        # Or, pick a random subset of size K?
        # Let's try: 
        # Pick a random number of primes N (between 5 and 20).
        # Pick N random primes.
        # Check budget. If over, remove largest cost primes until under.
        
        N = random.randint(5, 25)
        # Ensure we don't pick too many large primes which are cheap but low density?
        # Actually large primes are cheap (1/p small) but cover few numbers (1/p density).
        # Small primes are expensive but cover many numbers.
        
        # Let's sample primes with probability proportional to 1/p? No, that favors small ones.
        # We want a mix.
        
        # Simple approach:
        # Create a weighted list where weight = 1 (uniform) or weight = 1/p?
        # Let's try uniform random selection of N primes, then filter by budget.
        
        selected_indices = set()
        # Pick N random indices
        candidates = random.sample(range(num_primes), min(N, num_primes))
        
        # Sort candidates by cost (1/p) descending to remove expensive ones first if needed?
        # Or just add all and remove until sum <= 1.
        
        sum_cost = 0
        for idx in candidates:
            p = primes[idx]
            sum_cost += Fraction(1, p)
            
        # If sum_cost > 1, remove primes with largest 1/p (smallest p) until sum <= 1?
        # Wait, removing small primes reduces coverage density significantly.
        # Maybe remove largest p (smallest cost) first? That keeps the "strong" primes.
        
        # Let's sort candidates by p (ascending).
        # If budget exceeded, remove from the end (largest p, smallest cost)?
        # No, if we have budget 1, and sum is 1.2.
        # Removing a large prime (cost 0.01) frees up 0.01.
        # Removing a small prime (cost 0.5) frees up 0.5.
        # We want to keep the small primes because they cover more numbers.
        # So we should remove large primes first.
        
        candidates_sorted = sorted(candidates, key=lambda i: primes[i], reverse=True)
        
        final_indices = []
        current_cost = Fraction(0)
        
        # We want to keep small primes.
        # So we should try to include all small primes first?
        # Let's invert: 
        # Sort candidates by p ascending.
        # Add them one by one if budget allows.
        
        candidates_sorted_asc = sorted(candidates, key=lambda i: primes[i])
        
        for idx in candidates_sorted_asc:
            p = primes[idx]
            if current_cost + Fraction(1, p) <= 1:
                final_indices.append(idx)
                current_cost += Fraction(1, p)
        
        if not final_indices:
            continue
            
        # Optimize residues for this set
        L, residues = optimize_residues(final_indices)
        
        if L > best_L:
            best_L = L
            # Construct result
            best_classes = []
            for idx in final_indices:
                p = primes[idx]
                r = residues[idx]
                best_classes.append([p, r])
                
        # Early exit if we hit 999 or 1000
        if best_L >= 999:
            break

    # If no valid solution found (shouldn't happen), return empty or a trivial one
    if not best_classes:
        # Fallback: Use prime 2, residue 1 (covers odds)
        # Cost 1/2.
        # L will be 1 (covers 1).
        return [[2, 1]]
        
    return best_classes

# EVOLVE-BLOCK-END
