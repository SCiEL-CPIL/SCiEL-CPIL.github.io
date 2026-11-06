"""Stronger covering-set search with deadline guard."""
import time
import math
import random
from fractions import Fraction
from collections import defaultdict

# Precompute primes up to 1000
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

ALL_PRIMES = _sieve(1000)
PRIME_SET = set(ALL_PRIMES)
# Index primes for quick lookup
PRIME_INDEX = {p: i for i, p in enumerate(ALL_PRIMES)}

def _is_prime(n):
    return n in PRIME_SET

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    CAP = 100000
    MAX_L = 100000

    # Precompute reciprocals as Fractions for exact budget
    recip = {p: Fraction(1, p) for p in ALL_PRIMES}
    
    # We'll track coverage using a boolean array up to some max we care about
    # Since we cap at 10^5, let's use a bytearray for speed
    # But we need to know L, so let's track the maximum covered continuously
    
    def compute_L(covered_set, cap=MAX_L):
        """Compute largest L such that 1..L all in covered_set."""
        # covered_set is a set of integers
        L = 0
        for x in range(1, cap + 1):
            if x in covered_set:
                L = x
            else:
                break
        return L
    
    # Faster: use a sorted list or just track min uncovered
    # For speed during search, maintain a coverage array up to a reasonable max
    # Let's use a bytearray of size MAX_L+1
    cov_size = MAX_L + 1
    cov_count = bytearray(cov_size)  # count how many classes cover each number
    # Actually, we just need boolean, but for add/remove we need counts
    # Let's use an array of ints (or just track the set)
    
    # Alternative: maintain a set of covered numbers
    covered = set()
    # And a way to quickly compute L
    # We'll maintain a min-heap or just scan
    # For efficiency, let's maintain the current L and a count of uncovered in [1, L]
    
    # Let's use a different approach: maintain an array 'covered_arr' of booleans
    # and a variable 'current_L' and 'uncovered_in_1_to_L'
    covered_arr = bytearray(cov_size)  # 0 or 1
    current_L = 0
    uncovered_count = 0  # number of x in 1..current_L with covered_arr[x]==0
    
    def recompute_L_from_scratch():
        nonlocal current_L, uncovered_count
        current_L = 0
        for x in range(1, cov_size):
            if covered_arr[x]:
                current_L = x
            else:
                break
        # Count uncovered in 1..current_L
        uncovered_count = 0
        for x in range(1, current_L + 1):
            if not covered_arr[x]:
                uncovered_count += 1
        return current_L
    
    def add_class(p, r):
        """Add class (p, r). Returns True if budget allows."""
        nonlocal current_L, uncovered_count
        # Mark coverage
        # x ≡ r (mod p) means x = r + k*p
        # For x in 1..MAX_L
        for x in range(r if r >= 1 else p, cov_size, p):
            if x <= 0:
                continue
            if covered_arr[x]:
                continue
            covered_arr[x] = 1
            if x <= current_L:
                uncovered_count -= 1
            else:
                # x > current_L, but if all 1..x are covered, extend L
                # Check if 1..x all covered
                # This is expensive; let's just extend L optimistically
                pass
        # After adding, recompute L efficiently
        # Try to extend L
        while current_L + 1 < cov_size and covered_arr[current_L + 1]:
            current_L += 1
        # Update uncovered_count for the new range
        # We need to recount for the new L
        # For simplicity in the search loop, let's just recompute L when needed
        # But for the search, we want speed. Let's use a simpler metric:
        # Just track the set and compute L at the end of each move.
        # For now, let's just return and compute L separately when evaluating.
        pass
    
    # The above incremental approach is getting complex. Let's use a simpler approach:
    # Maintain the set of classes, and for evaluation, compute coverage up to some limit.
    # Since we need speed, let's precompute for each prime p and residue r, 
    # the list of numbers covered. But that's too much memory.
    
    # Better approach: Use a numpy array for coverage, and for each move, 
    # update the coverage array in O(MAX_L/p) time.
    # With MAX_L=100000 and p>=2, this is at most 50000 ops per move.
    # With 150s and maybe 10000 moves, that's 500 million ops, too slow in Python.
    
    # Let's use a smarter approach: only track coverage up to, say, 2000 initially,
    # and expand if we get close to the target.
    
    # Actually, the key insight: we want to maximize L. The current seed gets ~1000.
    # Let's focus on covering 1..2000 well, then try to extend.
    
    # Let's use a different strategy:
    # 1. Greedy construction: start with empty set, repeatedly add the class (p,r) 
    #    that maximizes the number of newly covered numbers in 1..L_target, 
    #    subject to budget.
    # 2. Then local search: try swaps and reassignments.
    
    # For speed, let's limit the coverage tracking to L_target = 2000 initially.
    
    L_TARGET = 2000
    cov = bytearray(L_TARGET + 1)  # 1-indexed, cov[x] = 1 if x covered
    used_primes = set()
    budget = Fraction(0, 1)
    classes = []  # list of (p, r)
    
    def compute_L_from_cov(cov, max_check=2000):
        L = 0
        for x in range(1, max_check + 1):
            if cov[x]:
                L = x
            else:
                break
        return L
    
    def add_class_to_cov(p, r, cov):
        """Add coverage for class (p,r) to cov array."""
        # x ≡ r (mod p), x in 1..L_TARGET
        # First x >= 1 with x ≡ r (mod p):
        # If r == 0, first x is p (if p <= L_TARGET)
        # If r > 0, first x is r (if r <= L_TARGET)
        if r == 0:
            start = p
        else:
            start = r
        if start > L_TARGET:
            return
        for x in range(start, L_TARGET + 1, p):
            cov[x] = 1
    
    def remove_class_from_cov(p, r, cov):
        """Remove coverage for class (p,r) from cov array.
        Note: this only works if no other class covers the same numbers.
        For correctness, we'd need reference counting. Let's use a count array."""
        pass
    
    # Let's use a count array for reference counting
    cov_count = bytearray(L_TARGET + 1)
    
    def add_class(p, r):
        nonlocal budget
        if p in used_primes:
            return False
        if budget + recip[p] > 1:
            return False
        # Add coverage
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_TARGET:
            for x in range(start, L_TARGET + 1, p):
                cov_count[x] += 1
        used_primes.add(p)
        budget += recip[p]
        classes.append((p, r))
        return True
    
    def remove_class(p, r):
        nonlocal budget
        if p not in used_primes:
            return False
        # Remove coverage
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_TARGET:
            for x in range(start, L_TARGET + 1, p):
                cov_count[x] -= 1
        used_primes.discard(p)
        budget -= recip[p]
        classes.remove((p, r))
        return True
    
    def get_current_L():
        L = 0
        for x in range(1, L_TARGET + 1):
            if cov_count[x] > 0:
                L = x
            else:
                break
        return L
    
    # Phase 1: Greedy construction
    # Try all possible (p, r) and pick the one that maximizes newly covered numbers
    # To speed up, we can limit r to 0..p-1 but that's up to 1000 options per prime
    # With 168 primes, that's ~168*500 = 84000 options per step. Too slow.
    
    # Better: for each prime p, try a few candidate residues.
    # Or: use a randomized greedy with limited residues.
    
    # Let's try a simpler greedy: for each prime p (in some order), 
    # pick the residue r that maximizes coverage.
    
    # Order primes by 1/p descending (smaller primes first) to use budget efficiently?
    # Or ascending? Smaller primes cover more numbers, so use them first.
    
    primes_sorted = sorted(ALL_PRIMES, key=lambda p: p)  # ascending
    
    for p in primes_sorted:
        if time.time() > deadline - 10.0:
            break
        if budget + recip[p] > 1:
            continue
        # Try all residues 0..p-1
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Compute gain: number of x in 1..L_TARGET with cov_count[x]==0 and x≡r (mod p)
            gain = 0
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_TARGET:
                for x in range(start, L_TARGET + 1, p):
                    if cov_count[x] == 0:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            add_class(p, best_r)
    
    # Phase 2: Local search with swaps
    # Try to improve by swapping a class or reassigning residues
    
    current_L = get_current_L()
    best_L = current_L
    best_classes = [(p, r) for p, r in classes]
    best_budget = budget
    best_used = set(used_primes)
    
    # Local search: try reassigning residues of existing primes
    # And try swapping primes
    
    num_iterations = 0
    while time.time() < deadline - 5.0:
        num_iterations += 1
        
        # Try random reassignment of one prime's residue
        if classes:
            idx = random.randint(0, len(classes) - 1)
            p, r = classes[idx]
            old_r = r
            # Try a random new residue
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                # Remove old, add new
                remove_class(p, old_r)
                add_class(p, new_r)
                L = get_current_L()
                if L > best_L:
                    best_L = L
                    best_classes = [(pp, rr) for pp, rr in classes]
                    best_budget = budget
                    best_used = set(used_primes)
                else:
                    # Revert
                    remove_class(p, new_r)
                    add_class(p, old_r)
        
        # Try swap: remove a prime, add a different one
        if classes and len(used_primes) < len(ALL_PRIMES):
            # Pick a prime to remove
            idx = random.randint(0, len(classes) - 1)
            p_remove, r_remove = classes[idx]
            # Pick a prime to add
            available = [p for p in ALL_PRIMES if p not in used_primes]
            if available:
                p_add = random.choice(available)
                if budget - recip[p_remove] + recip[p_add] <= 1:
                    # Try it
                    remove_class(p_remove, r_remove)
                    # Find best residue for p_add
                    best_r = 0
                    best_gain = -1
                    for r in range(p_add):
                        gain = 0
                        if r == 0:
                            start = p_add
                        else:
                            start = r
                        if start <= L_TARGET:
                            for x in range(start, L_TARGET + 1, p_add):
                                if cov_count[x] == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    add_class(p_add, best_r)
                    L = get_current_L()
                    if L > best_L:
                        best_L = L
                        best_classes = [(pp, rr) for pp, rr in classes]
                        best_budget = budget
                        best_used = set(used_primes)
                    else:
                        # Revert
                        remove_class(p_add, best_r)
                        add_class(p_remove, r_remove)
    
    # Phase 3: Try to extend L beyond L_TARGET if we're doing well
    if best_L >= L_TARGET - 10:
        # Extend coverage tracking
        L_TARGET_2 = min(5000, MAX_L)
        cov_count_2 = bytearray(L_TARGET_2 + 1)
        for p, r in best_classes:
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_TARGET_2:
                for x in range(start, L_TARGET_2 + 1, p):
                    cov_count_2[x] += 1
        # Compute L in extended range
        L_ext = 0
        for x in range(1, L_TARGET_2 + 1):
            if cov_count_2[x] > 0:
                L_ext = x
            else:
                break
        if L_ext > best_L:
            best_L = L_ext
    
    # Return the best classes found
    # Convert to the required format: [[p, r], ...]
    result = [[p, r] for p, r in best_classes]
    return result

# Test
if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Number of classes: {len(result)}")
    # Verify budget
    from fractions import Fraction
    budget = sum(Fraction(1, p) for p, r in result)
    print(f"Budget: {budget} = {float(budget):.6f}")
    # Verify distinct primes
    primes_used = [p for p, r in result]
    assert len(primes_used) == len(set(primes_used)), "Duplicate primes!"
    # Verify coverage
    MAX_CHECK = 2000
    covered = [False] * (MAX_CHECK + 1)
    for p, r in result:
        if r == 0:
            start = p
        else:
            start = r
        for x in range(start, MAX_CHECK + 1, p):
            covered[x] = True
    L = 0
    for x in range(1, MAX_CHECK + 1):
        if covered[x]:
            L = x
        else:
            break
    print(f"Covered up to L = {L}")
