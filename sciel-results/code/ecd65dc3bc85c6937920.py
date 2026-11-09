"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168 primes

    # We want to maximize L such that every x in 1..L is covered by at least 2 congruences.
    # Strategy: Greedy CSP. Process primes one by one (smallest to largest).
    # For each prime p_i, choose residue r_i to maximize the current "safe prefix" L.
    # 
    # However, a pure greedy on the prefix might get stuck. 
    # A better heuristic: 
    # The "bottleneck" for coverage is usually the small numbers.
    # If we ensure that for every small x, there are at least 2 primes p such that x % p == r_p,
    # we extend the prefix.
    #
    # Let's try a greedy approach that explicitly looks at the current boundary L.
    # State: residues for primes[0..k-1] are fixed. We need to pick residue for primes[k].
    # Goal: Extend the prefix as far as possible.
    #
    # To evaluate a candidate residue r for the current prime p, we need to know:
    # 1. Does it help cover the current "gap" at position L+1?
    # 2. Does it maintain coverage for 1..L?
    #
    # Actually, let's look at the structure of the problem.
    # The score is the largest L.
    # If we use a greedy method that always tries to cover the *current* first uncovered point,
    # we might fail because covering x doesn't guarantee covering x+1.
    #
    # Alternative: "Lookahead" Greedy.
    # For each prime p, try all possible residues? No, p can be up to 997.
    # But we only care about residues that affect the prefix 1..L_max.
    #
    # Let's refine the "Greedy Prefix Extension" idea.
    # Maintain an array `coverage` of length L_max (say 2000 or 5000).
    # Initially coverage is 0.
    # We process primes in order.
    # For prime p, we want to choose r to maximize the new prefix length.
    # The new prefix length is determined by the first x where total_coverage(x) < 2.
    #
    # Efficient check:
    # When we add a new congruence x ≡ r (mod p), it adds 1 to coverage at positions r, r+p, r+2p...
    # We can precompute the "impact" of each possible residue?
    # For a fixed p, the positions affected are periodic.
    #
    # Let's try a different angle: **Simulated Annealing with a better evaluation function**.
    # The current program's evaluation is O(L * N) which is slow if done naively, but vectorized it's okay.
    # However, the search strategy (hill climbing) is weak.
    #
    # Let's try **Greedy with Backtracking** on the first few primes, then heuristic for the rest.
    # The first few primes (2, 3, 5, 7, 11, ...) have the most impact on small x.
    #
    # Actually, let's look at Inspiration 1 (score 1.5176). It used Simulated Annealing.
    # Let's try to improve upon that by using a **Constructive Heuristic** first to get a good starting point,
    # then refine with Simulated Annealing.
    #
    # Constructive Heuristic:
    # 1. Start with L_target = 1000.
    # 2. For each prime p, choose r_p such that it covers as many "critical" points as possible.
    #    Critical points are those currently covered by exactly 1 congruence.
    #
    # Let's implement a "Max-Coverage" greedy:
    # For each prime p, calculate for each residue r the number of x in 1..L_max
    # that are currently covered by exactly 1 congruence and would be covered by this new one.
    # Pick r that maximizes this count.
    #
    # This is O(N * L_max * N_primes) if done naively, but we can optimize.
    #
    # Let's try a simpler, robust strategy: **Randomized Greedy with Local Repair**.
    # 1. Initialize residues randomly.
    # 2. Calculate score L.
    # 3. Identify the first gap x where coverage < 2.
    # 4. Try to fix this gap by changing residues of primes that divide x or are close to x.
    #    Specifically, if x is not covered by p, and we change r_p to x % p, we cover x.
    #    But we must ensure we don't break coverage for y < x.
    #
    # Let's implement a **Greedy Repair** algorithm.
    #
    # 1. Start with a random solution.
    # 2. Evaluate coverage for 1..L_max.
    # 3. Find the first x where coverage[x] < 2.
    # 4. If no such x, we are done (or L_max is reached).
    # 5. Otherwise, try to increase coverage at x.
    #    We can do this by changing the residue of some prime p.
    #    Which prime? Ideally one that covers x and doesn't hurt others.
    #    Let's pick a prime p randomly from those that don't currently cover x.
    #    Set r_p = x % p.
    #    Check if this improves the score (or at least doesn't make the first gap earlier).
    #    If it improves, keep it.
    #    If not, revert.
    # 6. Repeat until time runs out or no improvement.
    #
    # This is essentially a local search, but guided by the "first gap".
    #
    # Let's try to make this more efficient.
    #
    # Improved Strategy: **Prefix-Greedy with Lookahead**
    # Process primes in order. For each prime, pick the residue that maximizes the
    # length of the contiguous prefix where coverage >= 2.
    #
    # To do this efficiently, we maintain the coverage array.
    # When considering prime p, we test residues.
    # Instead of testing all p residues, we can note that only residues that cover
    # the current "frontier" are interesting.
    #
    # Let's implement this.

    L_MAX_CHECK = 2000  # We only care about extending the prefix up to this for speed
    # Actually, the score can be higher, but the prefix is what matters for the greedy step.
    # We will evaluate the full score at the end.

    def evaluate_score(residues):
        """Calculate the exact score (largest L)."""
        residues = np.array(residues)
        primes_arr = np.array(primes)
        
        # We need to find the largest L.
        # Let's check up to 10000 first. If it's 10000, check more.
        # But for speed, let's assume the score is < 10000 for most iterations.
        # We can do a binary search or linear scan.
        
        # Vectorized coverage for 1..10000
        check_limit = 10000
        coverage = np.zeros(check_limit + 1, dtype=np.int32)
        for i in range(n):
            p = primes_arr[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > check_limit:
                continue
            idx = np.arange(start, check_limit + 1, p)
            coverage[idx] += 1
        
        valid = coverage[1:] >= 2
        if not valid[0]:
            return 0
        false_positions = np.where(~valid)[0]
        if len(false_positions) == 0:
            # Check further if needed? For now, return check_limit
            # If the task requires > 1000, and we hit 10000, we might need to check more.
            # But 10000 is likely high enough for this heuristic phase.
            return check_limit
        return false_positions[0]

    def greedy_construct():
        """
        Greedy construction: Process primes in order.
        For each prime, choose the residue that maximizes the current prefix length.
        """
        residues = [0] * n
        coverage = np.zeros(L_MAX_CHECK + 1, dtype=np.int32)
        
        # We need to track the current prefix length
        # prefix_len = largest L such that coverage[1..L] >= 2
        
        # Helper to update coverage and prefix_len
        # Instead of recalculating prefix_len from scratch, we can update it.
        # But for simplicity in the greedy step, we might just calculate the "potential"
        # prefix length for each candidate residue.
        
        # Let's precompute the "delta" coverage for each prime and residue?
        # That's too much memory.
        
        # Let's just iterate.
        for i in range(n):
            p = primes[i]
            best_r = 0
            best_score = -1
            
            # We don't need to try all p residues.
            # We only care about residues that help cover the current frontier.
            # But to be safe, let's try a subset.
            # Actually, for small p, p is small, so trying all is cheap.
            # For large p, p is large, but the number of affected points in 1..L_MAX_CHECK is small.
            
            # Let's try all residues for p <= 50, and a random subset for larger p.
            # Or better: try residues that cover the current "gap".
            
            # First, find the current gap (first x where coverage[x] < 2)
            # If coverage is all >= 2 up to L_MAX_CHECK, we are good.
            
            # Let's calculate the current prefix length
            valid = coverage[1:L_MAX_CHECK+1] >= 2
            if not valid[0]:
                current_prefix = 0
            else:
                false_pos = np.where(~valid)[0]
                if len(false_pos) == 0:
                    current_prefix = L_MAX_CHECK
                else:
                    current_prefix = false_pos[0]
            
            # If current_prefix is 0, we have a problem at x=1.
            # We need to cover x=1.
            
            # Strategy: Try residues that cover the current "bottleneck".
            # The bottleneck is the first x where coverage < 2.
            # If no such x, we are at L_MAX_CHECK.
            
            bottleneck_x = current_prefix + 1 if current_prefix < L_MAX_CHECK else L_MAX_CHECK
            
            # Candidate residues:
            # 1. The residue that covers bottleneck_x: r = bottleneck_x % p
            # 2. A few random residues.
            # 3. If p is small, try all.
            
            candidates = set()
            if p <= 20:
                candidates = set(range(p))
            else:
                # Add the bottleneck residue
                candidates.add(bottleneck_x % p)
                # Add a few random ones
                for _ in range(10):
                    candidates.add(np.random.randint(0, p))
                # Add 0
                candidates.add(0)
                
            for r in candidates:
                # Calculate new coverage if we set residues[i] = r
                # We need to see how this changes the prefix.
                # Instead of updating the global coverage array (which is destructive),
                # we can estimate the new prefix.
                
                # The new coverage at x is coverage[x] + (1 if x % p == r else 0)
                # We want the largest L such that for all x in 1..L, new_coverage[x] >= 2.
                
                # This is equivalent to finding the first x where:
                # coverage[x] + (1 if x % p == r else 0) < 2
                # => coverage[x] < 2 AND x % p != r
                # OR
                # coverage[x] < 1 AND (x % p == r OR x % p != r) -> always < 2 if coverage[x] < 1?
                # Wait.
                # new_cov[x] = cov[x] + delta(x)
                # We need new_cov[x] >= 2.
                # If cov[x] >= 2, delta doesn't matter.
                # If cov[x] == 1, we need delta(x) = 1 => x % p == r.
                # If cov[x] == 0, we need delta(x) = 2 => Impossible for a single prime.
                # So if cov[x] == 0, the new coverage is at most 1, so x is a gap.
                
                # So the new prefix is the smallest x such that:
                # (cov[x] == 0) OR (cov[x] == 1 AND x % p != r)
                
                # We can precompute the "bad" positions for each r?
                # No, let's just scan.
                
                # Optimization: We only need to scan up to current_prefix + some margin.
                # Because if cov[x] >= 2 for all x <= current_prefix, and we add a positive delta,
                # they stay >= 2.
                # The only way the prefix shrinks is if we *remove* coverage?
                # No, we are *adding* a prime. So coverage only increases.
                # Therefore, the new prefix is >= current_prefix.
                
                # So we just need to find the first x > current_prefix where the condition fails.
                # Actually, we need to check if the condition holds for x in current_prefix+1 .. L_MAX_CHECK.
                
                # Let's find the first x > current_prefix such that:
                # cov[x] == 0 OR (cov[x] == 1 AND x % p != r)
                
                # We can precompute the next "0" and next "1" positions?
                
                # For speed, let's just check the next few points.
                # If current_prefix is close to L_MAX_CHECK, we check up to L_MAX_CHECK.
                
                # Let's define a function to check the extension.
                # We scan x from current_prefix + 1 to L_MAX_CHECK.
                # Stop at the first x where the condition fails.
                
                # To speed this up, we can precompute the arrays of positions where cov[x] == 0 and cov[x] == 1.
                # But cov changes with each prime.
                
                # Let's just do a vectorized check on the slice.
                # slice = coverage[current_prefix+1 : L_MAX_CHECK+1]
                # We need to find the first index j in the slice where:
                # slice[j] == 0 OR (slice[j] == 1 AND (current_prefix + 1 + j) % p != r)
                
                # This is a bit complex to vectorize with the modulo.
                # Let's just loop in Python for the candidate check?
                # L_MAX_CHECK is 2000. Looping 2000 times is fast.
                
                new_prefix = current_prefix
                for x in range(current_prefix + 1, L_MAX_CHECK + 1):
                    c = coverage[x]
                    if c == 0:
                        break
                    if c == 1:
                        if x % p != r:
                            break
                    # if c >= 2, it's fine
                    new_prefix = x
                    
                if new_prefix > best_score:
                    best_score = new_prefix
                    best_r = r
            
            residues[i] = best_r
            
            # Update coverage array
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_MAX_CHECK:
                idx = np.arange(start, L_MAX_CHECK + 1, p)
                coverage[idx] += 1

        return residues

    # Run greedy construction
    best_residues = greedy_construct()
    best_score = evaluate_score(best_residues)
    
    # Refine with local search (Hill Climbing)
    # Use a fast evaluation for local search
    def fast_evaluate(residues, limit=5000):
        residues = np.array(residues)
        primes_arr = np.array(primes)
        coverage = np.zeros(limit + 1, dtype=np.int32)
        for i in range(n):
            p = primes_arr[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > limit:
                continue
            idx = np.arange(start, limit + 1, p)
            coverage[idx] += 1
        valid = coverage[1:] >= 2
        if not valid[0]:
            return 0
        false_positions = np.where(~valid)[0]
        if len(false_positions) == 0:
            return limit
        return false_positions[0]

    # Hill climbing
    current = list(best_residues)
    current_score = fast_evaluate(current)
    
    while time.time() < deadline - 10:
        improved = False
        # Randomly shuffle the order of primes to try
        order = list(range(n))
        np.random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 10:
                break
            p = primes[i]
            old_val = current[i]
            
            # Try a few random new values
            # Also try the value that covers the current gap
            # Find current gap
            # For speed, let's just try randoms and the gap-covering value
            
            # Get current gap
            # We can approximate the gap by evaluating a small prefix
            # Or just try randoms
            
            candidates = [np.random.randint(0, p) for _ in range(5)]
            # Add the value that covers x=1, x=2, etc?
            # Let's add the value that covers the current best_score + 1
            target_x = current_score + 1
            if target_x <= 10000:
                candidates.append(target_x % p)
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                current[i] = new_val
                new_score = fast_evaluate(current)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current[i] = old_val
            
            if not improved:
                # If we didn't improve in this pass, we might be stuck.
                # Try a random restart or a larger jump?
                # For now, just break and restart the loop with a new shuffle
                pass

        if not improved:
            # Stagnation. Try a random perturbation
            for _ in range(10):
                i = np.random.randint(0, n)
                p = primes[i]
                current[i] = np.random.randint(0, p)
            new_score = fast_evaluate(current)
            if new_score < current_score - 100:
                # Revert if it got much worse
                # We don't have the old state saved, so this is tricky.
                # Let's just continue.
                pass

    # Final evaluation
    final_score = evaluate_score(current)
    if final_score > best_score:
        best_residues = current
        best_score = final_score

    return best_residues
# EVOLVE-BLOCK-END
