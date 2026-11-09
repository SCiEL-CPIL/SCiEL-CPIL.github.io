"""Rewritten program for the admissible tuple maximization task."""
import math
from collections import defaultdict

N = 5000

# EVOLVE-BLOCK-START

def _primes_up_to(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for multiple in range(p * p, limit + 1, p):
                sieve[multiple] = False
    return [p for p, is_prime in enumerate(sieve) if is_prime]


def _is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(num)) + 1, 2):
        if num % i == 0:
            return False
    return True


def _compute_missed_residues(points, primes):
    """
    For each prime p in primes, determine which residue classes mod p are missed.
    Returns a dict: p -> set of missed residues (0 to p-1).
    """
    missed = {}
    for p in primes:
        present = set()
        for v in points:
            present.add(v % p)
        # If all residues are present, none are missed
        if len(present) == p:
            missed[p] = set()
        else:
            all_res = set(range(p))
            missed[p] = all_res - present
    return missed


def construct():
    """
    Construct the largest admissible tuple in [0, N] by greedy addition.
    
    Strategy:
    1. Start with a core set of large primes (primes > sqrt(N)) plus 1.
       This core is admissible because for any prime q <= len(core),
       we can find a missed residue.
    2. Identify the missed residue for each relevant prime.
    3. Greedily add candidates from [0, N] that do not "complete" any 
       currently missed residue class for any prime q <= len(tuple) + 1.
    4. After adding, if a prime q now has all residues covered, we must
       remove elements. However, with greedy addition, we only add if
       safe, so we maintain admissibility by construction.
    
    Key insight: A set S is admissible if for every prime p <= |S|,
    S does not cover all residue classes mod p.
    We maintain a set S. Initially, S is small and admissible.
    We try to add x. After adding x, |S| becomes k.
    For each prime p <= k, we check if S covers all residues mod p.
    If yes, adding x is invalid. If no, it's valid.
    
    To speed up:
    - Precompute primes up to N (max possible tuple size is N+1, but realistically much smaller).
    - For each candidate x, check against all primes p <= current_size + 1.
    - Since current_size grows, the number of primes to check grows slowly.
    - For each prime p, we track the count of residues covered.
    
    Efficient implementation:
    - Maintain a dictionary: p -> set of residues covered in S.
    - Also maintain: p -> list of elements in S that map to each residue? 
      Not needed for just checking if a residue is missed.
      But when we add x, we update the covered set for each p.
      If covered set becomes full, we must reject x.
    - However, if we reject x, we don't add it, so no state change.
    
    Algorithm:
    1. Initialize S with core elements.
    2. Compute covered_res[p] for all relevant primes p.
    3. For each candidate x in [0, N] \ S:
         Check if adding x is safe.
         Safe means: for all primes p <= |S|+1, x does not fill a 
         previously missed residue class mod p.
         Wait, that's not quite right.
         
         Let's rephrase:
         S is admissible if for all primes p <= |S|, there exists a 
         residue r mod p such that no element of S is congruent to r mod p.
         
         When we add x to S, forming S' = S ∪ {x}, |S'| = |S| + 1.
         S' is admissible if for all primes p <= |S'|, there exists a 
         residue r mod p such that no element of S' is congruent to r mod p.
         
         For primes p <= |S|:
             S already had a missed residue r_p.
             If x % p != r_p, then r_p is still missed in S'.
             If x % p == r_p, then we need to check if there is another 
             missed residue in S' for p. But S had exactly one missed 
             residue? No, S could have multiple missed residues.
             
         Actually, the condition is just: does S' cover all residues mod p?
         So for each prime p <= |S|+1, we need to check if S' covers 
         all residues mod p.
         
         Since we are adding one element, the only primes p for which 
         coverage might change are those where x % p was a previously 
         missed residue.
         
         So:
         For each prime p <= |S|+1:
             If p <= |S|:
                 S has at least one missed residue. Let M_p be the set of 
                 missed residues in S.
                 If x % p is in M_p, then M_p' = M_p \ {x % p}.
                 If M_p' is empty, then S' covers all residues mod p -> invalid.
                 If M_p' is not empty, then S' still has a missed residue -> OK.
             If p == |S|+1 and p is prime:
                 S' has size p. We need S' to not cover all residues mod p.
                 Check if S' covers all residues mod p.
         
         So we need to track M_p (missed residues) for each prime p.
         
    4. If safe, add x to S, update M_p for all primes p where x % p 
       changes the missed set.
    
    Let's implement this.
    """
    # Step 1: Generate primes up to N
    primes = _primes_up_to(N)
    prime_set = set(primes)
    
    # Step 2: Initialize S with core elements
    # Core: 1 and primes p where p * p > N (i.e., p > sqrt(N))
    sqrt_N = int(math.isqrt(N))
    core = [1]
    for p in primes:
        if p > sqrt_N:
            core.append(p)
    
    S = set(core)
    
    # Step 3: Compute missed residues for all primes p <= len(S)
    # We need primes up to len(S) + some margin, since |S| will grow.
    # Max |S| can be at most N+1, but let's compute primes up to N.
    
    # For each prime p, track missed residues
    # missed[p] = set of residues r in [0, p-1] such that no element in S is r mod p
    missed = {}
    
    def update_missed_for_new_elements(new_elements, current_S_size):
        """
        Update missed residues when adding new elements to S.
        """
        max_p_to_check = current_S_size + 1
        # Ensure we have primes up to max_p_to_check
        # We already have primes list, but let's filter
        relevant_primes = [p for p in primes if p <= max_p_to_check]
        
        for p in relevant_primes:
            if p not in missed:
                # Initialize missed[p] for current S
                present = set()
                for v in S:
                    present.add(v % p)
                all_res = set(range(p))
                missed[p] = all_res - present
            else:
                # Update with new elements
                for v in new_elements:
                    r = v % p
                    if r in missed[p]:
                        missed[p].remove(r)
    
    # Initialize missed for initial S
    initial_size = len(S)
    update_missed_for_new_elements([], initial_size)
    
    # Step 4: Greedy addition
    # Candidates: all numbers in [0, N] not in S
    # We iterate in some order. Let's try ascending order.
    # To make it faster, we can skip numbers that are clearly bad.
    
    candidates = [x for x in range(N + 1) if x not in S]
    # Shuffle? No, let's try ascending.
    
    added = 0
    for x in candidates:
        new_size = len(S) + 1
        # Check if adding x is safe
        safe = True
        relevant_primes = [p for p in primes if p <= new_size]
        
        for p in relevant_primes:
            if p not in missed:
                # This shouldn't happen, but just in case
                present = set()
                for v in S:
                    present.add(v % p)
                all_res = set(range(p))
                missed[p] = all_res - present
            
            r = x % p
            if r in missed[p]:
                # Adding x would fill this missed residue
                # Check if there are other missed residues
                if len(missed[p]) == 1:
                    # This was the only missed residue, so adding x makes it complete
                    safe = False
                    break
            # If r not in missed[p], then it was already covered, so no change
            # If r in missed[p] but len > 1, then still safe
        
        if safe:
            S.add(x)
            added += 1
            # Update missed residues
            # Only need to update for primes p where x % p is in missed[p]
            # But simpler: just update for all relevant primes
            # Since we only remove from missed, we can do:
            for p in relevant_primes:
                if p in missed:
                    r = x % p
                    if r in missed[p]:
                        missed[p].remove(r)
    
    # Step 5: Return sorted list
    return sorted(S)

# EVOLVE-BLOCK-END
