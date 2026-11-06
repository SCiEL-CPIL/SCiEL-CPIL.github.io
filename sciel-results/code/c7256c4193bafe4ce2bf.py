import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    target_low = 128
    target_high = 256
    
    def compute_multiplicities(A_set):
        """Compute representation counts for all targets in [128,256]."""
        mult = {}
        for s in range(target_low, target_high + 1):
            count = 0
            for a in A_set:
                b = s - a
                if b in A_set and a <= b:
                    count += 1
            mult[s] = count
        return mult
    
    def peak_and_coverage(A_set):
        """Return (peak_mult, uncovered_count)."""
        mult = compute_multiplicities(A_set)
        peak = 0
        uncovered = 0
        for s in range(target_low, target_high + 1):
            m = mult[s]
            if m == 0:
                uncovered += 1
            elif m > peak:
                peak = m
        return peak, uncovered
    
    # Initial solution: use a structured approach
    # Start with a dense set and then optimize
    random.seed(42)
    
    # Initial: try to get coverage first with a reasonable set
    # Use a greedy approach to ensure coverage
    A = set()
    # Add elements to cover all targets
    # For target s, we need a,b in A with a+b=s
    # Strategy: add elements from both low and high ranges
    
    # Start with a base that ensures coverage
    # If A contains [0,64] ∪ [64,128], then any s in [0,256] can be written
    # But that's too many elements. Let's be smarter.
    
    # Try: A = {0,1,...,k} ∪ {n-k,...,n} for some k
    # Or use a more balanced approach
    
    best_A = None
    best_peak = float('inf')
    best_uncovered = float('inf')
    
    # Generate initial candidate using a parametric family
    # A = {i for i in range(0, n+1) if i % 2 == 0} ∪ some extras
    # Let's try: even numbers plus some odd numbers
    
    # Actually, let's start with a random set of size ~40-50 and optimize
    
    # First, ensure we have a valid starting point
    # Use the seed as fallback
    seed_A = set(range(64, 129))
    seed_peak, seed_uncovered = peak_and_coverage(seed_A)
    best_A = set(seed_A)
    best_peak = seed_peak
    best_uncovered = seed_uncovered
    
    # Try several random initializations
    for trial in range(10):
        if time.time() > deadline:
            break
        # Random set of size 35-55
        size = random.randint(35, 55)
        candidate = set(random.sample(range(0, n + 1), size))
        # Ensure 0 and n are included for better coverage
        candidate.add(0)
        candidate.add(n)
        peak, uncov = peak_and_coverage(candidate)
        score = (uncov, peak)
        best_score = (best_uncovered, best_peak)
        if score < best_score:
            best_uncovered = uncov
            best_peak = peak
            best_A = set(candidate)
    
    if best_uncovered > 0:
        # Need to fix coverage - add elements greedily
        for _ in range(100):
            if time.time() > deadline:
                break
            mult = compute_multiplicities(best_A)
            uncovered_targets = [s for s in range(target_low, target_high + 1) if mult[s] == 0]
            if not uncovered_targets:
                break
            # Pick a random uncovered target and add an element to help
            s = random.choice(uncovered_targets)
            # Find a,b not both in A such that a+b=s, add one
            candidates = []
            for a in range(0, n + 1):
                b = s - a
                if 0 <= b <= n and a <= b:
                    if a not in best_A and b not in best_A:
                        candidates.append((a, b))
            if candidates:
                a, b = random.choice(candidates)
                best_A.add(a)
                # Also consider adding b
                if random.random() < 0.5:
                    best_A.add(b)
    
    # Now do local search to minimize peak
    if best_uncovered == 0:
        current_A = set(best_A)
        current_peak, _ = peak_and_coverage(current_A)
        
        # Local search: try adding/removing/swapping elements
        temp = 5.0
        cooling = 0.995
        max_iters = 100000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if temp < 0.1:
                break
            
            # Generate a move
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.3 and len(new_A) > 20:
                # Remove a random element
                elem = random.choice(list(new_A))
                new_A.discard(elem)
            elif move_type < 0.6:
                # Add a random element
                elem = random.randint(0, n)
                new_A.add(elem)
            elif move_type < 0.8:
                # Swap: remove one, add one
                if len(new_A) > 20:
                    elem_out = random.choice(list(new_A))
                    elem_in = random.randint(0, n)
                    new_A.discard(elem_out)
                    new_A.add(elem_in)
            else:
                # Perturb: change multiple elements
                for _ in range(random.randint(2, 5)):
                    if len(new_A) > 20 and random.random() < 0.5:
                        elem = random.choice(list(new_A))
                        new_A.discard(elem)
                    else:
                        elem = random.randint(0, n)
                        new_A.add(elem)
            
            new_peak, new_uncov = peak_and_coverage(new_A)
            
            # Objective: heavily penalize uncovered, then minimize peak
            curr_obj = (1000 * best_uncovered) + best_peak
            new_obj = (1000 * new_uncov) + new_peak
            
            if new_obj < curr_obj or random.random() < math.exp(-(new_obj - curr_obj) / temp):
                current_A = new_A
                best_peak = new_peak
                best_uncovered = new_uncov
                if new_obj < (1000 * best_uncovered) + best_peak:
                    best_A = set(new_A)
                    best_peak = new_peak
                    best_uncovered = new_uncov
            
            temp *= cooling
        
        # Final polish: try to remove elements that don't increase peak
        if best_uncovered == 0:
            for elem in list(best_A):
                if time.time() > deadline:
                    break
                test_A = set(best_A)
                test_A.discard(elem)
                p, u = peak_and_coverage(test_A)
                if u == 0 and p <= best_peak:
                    best_A = test_A
                    best_peak = p
    
    # Return as sorted list
    return sorted(best_A)
