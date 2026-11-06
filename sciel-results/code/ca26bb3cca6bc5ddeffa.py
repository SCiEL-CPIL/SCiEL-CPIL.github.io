"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    # Generate all primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    all_primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Precompute 1/p as Fraction for exact budget checking
    reciprocals = {p: Fraction(1, p) for p in all_primes}
    
    # We want to cover 1..L where L is as large as possible
    # Strategy: Use local search with targeted residue flips
    # Start with a greedy construction, then optimize
    
    # Greedy initialization: try to cover consecutive integers
    # For each integer n, find a prime p that doesn't cover it yet and assign residue n % p
    # But we need to be careful about the budget
    
    # Better approach: Start with a known good covering and improve via local search
    
    # Let's try a different strategy: 
    # 1. Start with empty set
    # 2. For each prime in some order, decide whether to include it and what residue to use
    # 3. Use simulated annealing or hill climbing to maximize coverage
    
    # First, let's compute the maximum possible coverage with all primes
    # The sum of 1/p for all primes up to 1000 is about log(log(1000)) + M ≈ 2.2 + 0.26 ≈ 2.46
    # So we can afford a subset with sum <= 1
    
    # Let's try a greedy approach:
    # Sort primes by some heuristic, add primes that maximize marginal coverage
    
    def compute_coverage(classes, max_L=100000):
        """Compute the largest L such that every integer 1..L is covered."""
        if not classes:
            return 0
        
        # Create a boolean array for coverage
        covered = [False] * (max_L + 1)
        
        for p, r in classes:
            # Mark all numbers ≡ r (mod p) as covered
            if r == 0:
                # Numbers p, 2p, 3p, ...
                for n in range(p, max_L + 1, p):
                    covered[n] = True
            else:
                # Numbers r, r+p, r+2p, ...
                for n in range(r, max_L + 1, p):
                    covered[n] = True
        
        # Find the first gap
        for i in range(1, max_L + 1):
            if not covered[i]:
                return i - 1
        return max_L
    
    def compute_coverage_fast(classes, current_L):
        """Check if coverage extends beyond current_L, return new max L."""
        if not classes:
            return 0
        
        # Only check up to current_L + some margin
        check_limit = min(current_L + 1000, 100000)
        covered = [False] * (check_limit + 1)
        
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, check_limit + 1, p):
                covered[n] = True
        
        # Find the first gap starting from 1
        for i in range(1, check_limit + 1):
            if not covered[i]:
                return i - 1
        return check_limit
    
    # Greedy construction:
    # Try to add primes that maximize the extension of coverage
    
    # Start with a small set of primes
    # We know from the seed that primes from 13 upward work
    # Let's try to be more systematic
    
    # Initialize with primes that have small 1/p (large primes) to save budget
    # Actually, we want to cover small numbers, so we need small primes
    
    # Let's try: for each prime, the best residue is the one that covers the most uncovered small numbers
    
    # Start with no classes
    best_classes = []
    best_L = 0
    
    # Try adding primes in order of increasing size (small primes cover more densely)
    # But we need to balance: small primes have large 1/p
    
    # Let's use a greedy algorithm:
    # At each step, try each unused prime and each possible residue
    # Pick the one that maximizes the new coverage L
    # Stop when we can't improve or run out of budget
    
    used_primes = set()
    current_classes = []
    current_L = 0
    
    # Precompute primes sorted by size
    primes_sorted = sorted(all_primes)
    
    # Time-limited greedy
    while time.time() < deadline - 5:
        best_gain = 0
        best_prime = None
        best_residue = None
        
        for p in primes_sorted:
            if p in used_primes:
                continue
            
            # Check if adding this prime would exceed budget
            # We need sum of 1/p for all classes <= 1
            # Current sum:
            current_sum = sum(reciprocals[cp] for cp, _ in current_classes)
            if current_sum + reciprocals[p] > 1:
                continue
            
            # Try different residues
            # We want to maximize coverage extension
            # Try residue 0 first (covers multiples of p), then others
            # Actually, let's try a few promising residues
            
            # The best residue is likely one that covers a gap near current_L + 1
            # Let's try residue (current_L + 1) % p and a few others
            candidate_residues = set()
            candidate_residues.add((current_L + 1) % p)
            candidate_residues.add(0)
            candidate_residues.add(1)
            if p > 2:
                candidate_residues.add(p - 1)
            
            for r in candidate_residues:
                # Test this candidate
                test_classes = current_classes + [[p, r]]
                test_L = compute_coverage_fast(test_classes, current_L)
                
                if test_L > current_L + best_gain:
                    best_gain = test_L - current_L
                    best_prime = p
                    best_residue = r
        
        if best_prime is None:
            break
        
        # Add the best prime
        current_classes.append([best_prime, best_residue])
        used_primes.add(best_prime)
        current_L += best_gain
    
    # Now try local search: flip residues of existing primes to improve coverage
    while time.time() < deadline - 2:
        improved = False
        
        for idx, (p, r) in enumerate(current_classes):
            # Try changing the residue for this prime
            best_new_r = r
            best_new_L = current_L
            
            # Try a few candidate residues
            candidates = [r]
            candidates.append((current_L + 1) % p)
            candidates.append(0)
            candidates.append(1)
            if p > 2:
                candidates.append(p - 1)
            
            for new_r in candidates:
                if new_r == r:
                    continue
                test_classes = current_classes.copy()
                test_classes[idx] = [p, new_r]
                test_L = compute_coverage_fast(test_classes, current_L)
                
                if test_L > best_new_L:
                    best_new_L = test_L
                    best_new_r = new_r
            
            if best_new_r != r:
                current_classes[idx] = [p, best_new_r]
                current_L = best_new_L
                improved = True
        
        if not improved:
            break
    
    # Final coverage computation
    final_L = compute_coverage_fast(current_classes, current_L)
    
    return current_classes
# EVOLVE-BLOCK-END
