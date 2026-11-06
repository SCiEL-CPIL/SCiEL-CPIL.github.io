"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    
    # We'll work with a coverage array up to some limit
    # Start with a reasonable limit, extend as needed
    MAX_L = 5000
    
    def compute_L(residues, max_check=MAX_L):
        """Compute the largest L such that every integer 1..L has coverage >= 2."""
        coverage = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            for x in range(r if r > 0 else p, max_check + 1, p):
                coverage[x] += 1
        # Find the first position with coverage < 2
        for x in range(1, max_check + 1):
            if coverage[x] < 2:
                return x - 1
        return max_check
    
    # Greedy approach: process primes in order, choose residue to maximize coverage
    # in the critical region
    residues = [0] * n_primes
    
    # Initial coverage array
    coverage = [0] * (MAX_L + 1)
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        
        best_r = 0
        best_score = -1
        
        # For each possible residue, compute how well it helps
        # Focus on positions that are currently under-covered
        # Score: number of positions in [1, current_L+margin] that go from 1 to 2
        # plus a small bonus for extending the prefix
        
        # First, determine current L
        current_L = 0
        for x in range(1, min(MAX_L + 1, current_L + p * 2 + 10)):
            if coverage[x] < 2:
                current_L = x - 1
                break
        else:
            current_L = min(MAX_L, current_L + p * 2 + 10)
        
        # Look at the critical region: positions near the frontier
        # We want to maximize coverage in [1, current_L + some window]
        window_start = 1
        window_end = min(MAX_L, current_L + p + 50)
        
        # Try a subset of residues for speed, or all if p is small
        if p <= 200:
            candidates = range(p)
        else:
            # For large primes, sample strategically
            # Focus on residues that align with under-covered positions
            candidates = list(range(0, p, max(1, p // 50)))
            # Always include 0 and some near the frontier
            frontier_residues = set()
            for x in range(max(1, current_L - 10), min(window_end + 1, current_L + p + 10)):
                frontier_residues.add(x % p)
            for r in frontier_residues:
                if r not in candidates:
                    candidates.append(r)
        
        for r in candidates:
            # Compute the effect of adding this residue
            # Count how many positions in the critical window have coverage == 1
            # and would become 2
            gain = 0
            # Also consider positions that are 0 and would become 1 (less important)
            gain_zero = 0
            
            # The positions covered by x ≡ r (mod p) in [1, window_end]
            first = r if r >= 1 else p
            for x in range(first, window_end + 1, p):
                if x <= MAX_L:
                    if coverage[x] == 1:
                        gain += 1
                    elif coverage[x] == 0:
                        gain_zero += 1
            
            # Score: prioritize converting 1->2, then 0->1
            # Also add a small penalty for not covering early positions
            score = gain * 10 + gain_zero
            
            # Bonus for covering the immediate frontier
            for x in range(max(1, current_L + 1), min(current_L + p + 1, window_end + 1)):
                if x % p == r:
                    score += 5
                    break
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        
        # Update coverage
        r = best_r
        first = r if r >= 1 else p
        for x in range(first, MAX_L + 1, p):
            coverage[x] += 1
    
    # Now try local search / hill climbing to improve
    if time.time() < deadline - 20:
        current_L = compute_L(residues, MAX_L)
        
        # Hill climbing: try changing individual residues
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(n_primes):
                if time.time() > deadline - 10:
                    break
                p = primes[idx]
                old_r = residues[idx]
                best_r = old_r
                best_L = current_L
                
                # Try changing this residue
                # Sample some alternatives
                if p <= 100:
                    try_rs = [r for r in range(p) if r != old_r]
                else:
                    # For large p, try a few random alternatives
                    import random
                    try_rs = [random.randint(0, p - 1) for _ in range(min(20, p - 1))]
                    if old_r in try_rs:
                        try_rs.remove(old_r)
                
                for new_r in try_rs:
                    if time.time() > deadline - 10:
                        break
                    
                    # Temporarily change
                    residues[idx] = new_r
                    new_L = compute_L(residues, MAX_L)
                    residues[idx] = old_r
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                
                if best_r != old_r:
                    residues[idx] = best_r
                    current_L = best_L
                    improved = True
    
    return residues
# EVOLVE-BLOCK-END
