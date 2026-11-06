"""Rewritten program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute: for each triple (a,b,c) with a<b<c, check if all pairwise gcds are equal
    # We need to find all "bad" triples where gcd(a,b) == gcd(a,c) == gcd(b,c)
    
    # For efficient checking, precompute for each pair (a,b) the set of c that form a bad triple
    # Actually, let's think about this differently.
    
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c) = d
    # This means a, b, c are all multiples of d, and after dividing by d, the three numbers are pairwise coprime.
    # i.e., a/d, b/d, c/d are pairwise coprime.
    
    # So for a given d, the bad triples are triples of multiples of d where the quotients are pairwise coprime.
    
    # Let's precompute for each number x, which other numbers y have gcd(x,y) = d for various d.
    # Actually, let's precompute conflict sets.
    
    # For each pair (a,b), compute g = gcd(a,b). Then a third number c creates a bad triple
    # if gcd(a,c) == g and gcd(b,c) == g.
    # So c must be a multiple of g, and c/g must be coprime to both a/g and b/g.
    
    # Let's precompute: for each pair (a,b) with a<b, the list of c > b such that (a,b,c) is bad.
    # This is expensive but n=128 is small.
    
    # Alternative approach: use a greedy + local search approach with fast validity checking.
    
    # Precompute for each x, a bitset of numbers that conflict with x in some sense.
    # Actually, the conflict is triple-based, not pair-based. So we need to track triples.
    
    # Let's use a different approach: 
    # 1. Greedy construction with multiple orderings
    # 2. Local search (swap-based improvements)
    
    # Precompute: for each x, the set of pairs (a,b) in the current set such that adding x would create a bad triple.
    # This is O(|S|^2) per check, which is fine for |S| up to ~128.
    
    def is_valid_addition(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            a = selected[i]
            ga = gcd_mat[a][x]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gb = gcd_mat[b][x]
                gab = gcd_mat[a][b]
                if ga == gb == gab:
                    return False
        return True
    
    def is_valid_set(selected):
        """Check if the entire set is valid."""
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gab = gcd_mat[a][b]
                for k in range(j + 1, len(selected)):
                    c = selected[k]
                    if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                        return False
        return True
    
    def greedy_add(order):
        """Greedy add numbers in the given order."""
        selected = []
        for x in order:
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected
    
    def greedy_remove(start_set):
        """Try to remove elements to fix conflicts, then re-add."""
        selected = list(start_set)
        # Find all bad triples
        bad_triples = []
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gab = gcd_mat[a][b]
                for k in range(j + 1, len(selected)):
                    c = selected[k]
                    if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                        bad_triples.append((a, b, c))
        
        # Remove one element from each bad triple (greedy)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for a, b, c in bad_triples:
                if a in selected and b in selected and c in selected:
                    # Remove the one that causes the most conflicts
                    # Just remove c for simplicity
                    selected.remove(c)
                    changed = True
                    break
        
        return selected
    
    # Multiple greedy strategies
    best_set = []
    
    # Strategy 1: Greedy in natural order
    s1 = greedy_add(numbers)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Greedy in reverse order
    s2 = greedy_add(numbers[::-1])
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: Greedy by number of divisors (fewer divisors first)
    def num_divisors(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    s3 = greedy_add(sorted(numbers, key=num_divisors))
    if len(s3) > len(best_set):
        best_set = s3
    
    # Strategy 4: Greedy by largest prime factor
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        lp = 1
        for p in range(2, int(x**0.5) + 1):
            while x % p == 0:
                lp = p
                x //= p
        if x > 1:
            lp = x
        return lp
    
    s4 = greedy_add(sorted(numbers, key=largest_prime_factor, reverse=True))
    if len(s4) > len(best_set):
        best_set = s4
    
    # Strategy 5: Randomized greedy with multiple restarts
    import random
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > DEADLINE - 5:
            break
        order = numbers[:]
        rng.shuffle(order)
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try 1-for-1 swaps
    def local_search(start_set):
        selected = list(start_set)
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            for i in range(len(selected)):
                if time.time() > DEADLINE - 2:
                    break
                for x in numbers:
                    if x in selected:
                        continue
                    # Try swapping selected[i] with x
                    trial = selected[:]
                    trial.remove(selected[i])
                    trial.append(x)
                    if is_valid_set(trial):
                        selected = trial
                        improved = True
                        break
                if improved:
                    break
        return selected
    
    # Apply local search to best set
    if len(best_set) > 0 and time.time() < DEADLINE - 5:
        improved_set = local_search(best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Try to add more elements
    if time.time() < DEADLINE - 3:
        for x in numbers:
            if x not in best_set and is_valid_addition(x, best_set):
                best_set.append(x)
    
    # Final local search
    if len(best_set) > 0 and time.time() < DEADLINE - 2:
        improved_set = local_search(best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
