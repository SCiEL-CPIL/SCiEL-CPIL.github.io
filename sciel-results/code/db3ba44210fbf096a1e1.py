"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    # Candidate pool: 2..256
    candidates = list(range(2, N + 1))
    
    # Precompute prime factors for divisibility checks
    # divisors[x] = set of divisors of x in range(2, N+1)
    # Actually, we need to check if a | b*c.
    # a | b*c iff for all prime powers p^k || a, p^k divides b*c.
    # Simpler: just use modulo. But it's slow.
    # Let's precompute prime factorization to make checks faster?
    # Or just use modulo since N is small (256).
    # Modulo is O(1) per check.
    # Total checks per pair (a,b,c) is O(1).
    # Number of triples is O(S^3).
    
    def is_valid_triple(a, b, c):
        # a does not divide b*c
        return (b * c) % a != 0

    def count_violations(S):
        """Count the number of invalid triples (a,b,c) in S."""
        # S is a list of distinct ints
        n = len(S)
        if n < 3:
            return 0
        violations = 0
        # To speed up, we can iterate.
        # Symmetry: (a,b,c) is distinct members.
        # The condition is asymmetric in a: a | b*c.
        # So for each a in S, count pairs (b,c) from S\{a} such that a | b*c.
        # Since b and c are distinct and from S\{a}, order matters?
        # "a does not divide b*c". b and c are just values. b*c is commutative.
        # But b and c are distinct members.
        # So we iterate a, then b != a, then c != a, c != b.
        # This is O(S^3).
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i == j: continue
                for k, c in enumerate(S):
                    if i == k or j == k: continue
                    if (b * c) % a == 0:
                        violations += 1
        return violations

    # Initial State: Random subset of size ~100
    # We want to maximize cardinality.
    # Start with a medium size, try to grow or swap.
    # Let's start with all primes first as a good baseline, then randomize/swap.
    # Primes are safe: if a is prime, a | b*c implies a|b or a|c.
    # Since b,c != a and distinct, if a is prime, a cannot divide b or c (unless b or c is multiple of a).
    # If S contains primes, and we add composites, it might break.
    
    # Strategy:
    # 1. Start with a set of primes.
    # 2. Try to add non-primes if they don't increase violations (or keep it 0).
    # 3. Use local search to swap elements to allow larger sets.
    
    # Let's use a simpler objective: Maximize size, subject to 0 violations.
    # If we can't get 0 violations, we minimize violations but that's not the goal.
    # The goal is MAX CARDINALITY with 0 violations.
    # So we want to find the largest S such that V(S) == 0.
    
    # Heuristic:
    # Start with primes.
    # Try to add each composite number. If adding it creates violations, try to remove one of the conflicting elements to fix it?
    # Or just use Simulated Annealing on the state space of subsets.
    
    # State: Set of integers.
    # Move: Add a random element from candidates not in S.
    # Move: Remove a random element from S.
    # Move: Swap an element in S with one not in S.
    
    # Cost function:
    # We want 0 violations.
    # Primary goal: Minimize violations.
    # Secondary goal: Maximize size (if violations are 0).
    # Actually, if violations > 0, size doesn't matter as much as reducing violations.
    # Let's define Cost = violations * 1000 - size.
    # We want to minimize Cost.
    
    def get_cost(S):
        v = count_violations(S)
        return v * 10000 - len(S)
        
    # Initial S: All primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    S = set(primes)
    best_S = set(S)
    best_cost = get_cost(S)
    
    # If best_cost is negative (violations=0), we have a valid set.
    # We try to increase size while keeping violations 0.
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.99
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        if len(S) < len(candidates):
            # Try Add
            if random.random() < 0.3:
                new_elem = random.choice([x for x in candidates if x not in S])
                S_new = S | {new_elem}
            else:
                # Try Swap
                if S:
                    old_elem = random.choice(list(S))
                    new_elem = random.choice([x for x in candidates if x not in S])
                    S_new = (S - {old_elem}) | {new_elem}
                else:
                    S_new = S
        else:
            S_new = S
            
        # Also try Remove to escape local minima?
        # If we are stuck with high violations, removing might help.
        if random.random() < 0.1 and S:
            old_elem = random.choice(list(S))
            S_new = S - {old_elem}

        cost_new = get_cost(S_new)
        cost_cur = get_cost(S)
        
        if cost_new < cost_cur:
            S = S_new
            if cost_new < best_cost:
                best_cost = cost_new
                best_S = set(S)
        else:
            delta = cost_new - cost_cur
            if delta < 0:
                S = S_new
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_S = set(S)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.71828, -delta / T):
                    S = S_new
        
        T *= alpha
        
        # Early exit if we have a large valid set?
        # If best_cost < 0, we have a valid set.
        # We can try to greedily add more if possible.
        # But SA should handle it.
        
    # Final greedy attempt to expand best_S if it's valid
    if best_cost < 0:
        # best_S has 0 violations
        # Try to add any remaining candidates that keep it valid
        S_final = set(best_S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in candidates:
                if x not in S_final:
                    S_test = S_final | {x}
                    if count_violations(S_test) == 0:
                        S_final = S_test
                        changed = True
        if len(S_final) > len(best_S):
            best_S = S_final
            
    # Return list of ints
    return list(best_S)

# EVOLVE-BLOCK-END
