import time
import random
from math import gcd
from collections import defaultdict

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    deadline = time.time() + 85.0  # Safety margin

    # Precompute GCD table
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(129)]
    for n in range(2, 129):
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                prime_factors[n].add(p)
                while temp % p == 0:
                    temp //= p
            else:
                p += 1
        if temp > 1:
            prime_factors[n].add(temp)

    def is_bad_triple(a, b, c):
        """Check if three elements form a bad triple."""
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def count_conflicts(x, selected):
        """Count how many bad triplets would be formed if x were added to selected."""
        conflicts = 0
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            g_ax = gcd_table[x][a]
            for j in range(i + 1, n):
                b = sel_list[j]
                if g_ax == gcd_table[x][b] == gcd_table[a][b]:
                    conflicts += 1
        return conflicts

    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        return selected

    def local_search(selected, max_iters=50):
        """Try to improve the set by removing/adding elements."""
        selected = set(selected)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try adding any excluded element
            excluded = [x for x in range(1, 129) if x not in selected]
            for x in excluded:
                if count_conflicts(x, selected) == 0:
                    selected.add(x)
                    improved = True
                    break
            
            # If no add, try 1-for-2 exchange
            if not improved:
                for x in list(selected):
                    temp = selected - {x}
                    added = []
                    for y in excluded:
                        if y == x:
                            continue
                        if count_conflicts(y, temp | set(added)) == 0:
                            added.append(y)
                    if len(added) >= 2:
                        selected = temp | set(added)
                        improved = True
                        break

        return selected

    # Strategy 1: Structural construction using prime multipliers
    # The key insight: if we pick a set S of numbers, and for each number n in S,
    # we can represent it as n = d * m where d is a "core" and m is a multiplier,
    # then bad triples occur when three numbers share the same core d and their
    # multipliers m1, m2, m3 form a triple with all pairwise gcds = 1.
    
    # Approach: Group numbers by their "reduced form" after removing small primes.
    # For each group, we want to select a subset that avoids coprime triples.
    
    # Let's try a different structural approach:
    # 1. Start with all numbers 1-128
    # 2. Identify "dangerous" triples (those with common gcd)
    # 3. Use a constraint satisfaction approach to remove minimal elements
    
    # Actually, let's use the insight from the problem structure:
    # A triple {a,b,c} is bad iff gcd(a,b) = gcd(a,c) = gcd(b,c) = d
    # This means a = d*a', b = d*b', c = d*c' where gcd(a',b') = gcd(a',c') = gcd(b',c') = 1
    # So the "reduced" triple {a',b',c'} must be pairwise coprime.
    
    # Strategy: For each possible gcd value d, consider the set of numbers divisible by d.
    # Within that set, we must avoid triples whose reduced values are pairwise coprime.
    
    # Let's implement a smarter greedy that prioritizes numbers with many prime factors
    # (since they're less likely to form coprime triples with others)
    
    def degree_ordered_greedy():
        """Greedy that prioritizes numbers with higher 'conflict degree'."""
        # Precompute for each number, how many pairs (i,j) would form a bad triple with it
        # if we had all numbers
        conflict_degree = [0] * 129
        for x in range(1, 129):
            for i in range(1, 129):
                if i == x:
                    continue
                for j in range(i + 1, 129):
                    if j == x:
                        continue
                    if is_bad_triple(x, i, j):
                        conflict_degree[x] += 1
        
        # Sort by conflict degree (highest first - these are "most constrained")
        # Actually, we want to add numbers that have FEWER conflicts first
        # So sort by conflict degree ascending
        order = sorted(range(1, 129), key=lambda x: conflict_degree[x])
        return greedy_build(order)

    # Let's also try a set cover / hitting set approach
    # Find all bad triples, then find a minimum hitting set (remove minimum elements to hit all bad triples)
    # This is NP-hard, but we can use a greedy approximation
    
    def find_all_bad_triples():
        """Find all bad triples in [1, 128]."""
        triples = []
        for i in range(1, 129):
            for j in range(i + 1, 129):
                for k in range(j + 1, 129):
                    if is_bad_triple(i, j, k):
                        triples.append((i, j, k))
        return triples

    def greedy_hitting_set(triples, elements):
        """Find a hitting set: remove minimum elements to hit all triples."""
        # Greedy: repeatedly pick the element that appears in the most remaining triples
        remaining_triples = set(triples)
        removed = set()
        
        while remaining_triples:
            # Count occurrences of each element in remaining triples
            counts = defaultdict(int)
            for t in remaining_triples:
                for e in t:
                    counts[e] += 1
            
            if not counts:
                break
            
            # Pick the element with highest count
            best_elem = max(counts.keys(), key=lambda x: counts[x])
            removed.add(best_elem)
            
            # Remove all triples containing best_elem
            remaining_triples = {t for t in remaining_triples if best_elem not in t}
        
        return removed

    # Strategy 2: Use the structural insight to build a valid set
    # Key idea: If we ensure that for any common gcd d, the set of "reduced" values
    # (n/d for n in S divisible by d) contains NO pairwise coprime triple, then S is valid.
    
    def structural_build():
        """Build a set using structural constraints."""
        # Start with all numbers
        S = set(range(1, 129))
        
        # For each possible gcd d, check the reduced set
        # and remove elements to avoid coprime triples
        # This is complex, so let's use a simpler approach:
        # Iteratively remove elements that are part of bad triples,
        # prioritizing removal of elements that are in many bad triples
        
        # Precompute bad triples
        all_triples = find_all_bad_triples()
        
        if not all_triples:
            return S
        
        # Greedy hitting set
        removed = greedy_hitting_set(all_triples, set(range(1, 129)))
        return S - removed

    # Run multiple strategies and keep the best
    best_set = set()
    best_size = 0

    # Strategy A: Degree-ordered greedy
    s_a = degree_ordered_greedy()
    s_a = local_search(s_a)
    if len(s_a) > best_size:
        best_size = len(s_a)
        best_set = s_a

    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]

    # Strategy B: Structural hitting set
    s_b = structural_build()
    s_b = local_search(s_b)
    if len(s_b) > best_size:
        best_size = len(s_b)
        best_set = s_b

    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]

    # Strategy C: Randomized restarts with local search
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        
        if best_set and random.random() < 0.5:
            current = set(best_set)
            num_remove = random.randint(1, min(3, len(current)))
            to_remove = random.sample(list(current), num_remove)
            for x in to_remove:
                current.remove(x)
        else:
            order = list(range(1, 129))
            random.shuffle(order)
            current = greedy_build(order)
        
        s_c = local_search(current)
        if len(s_c) > best_size:
            best_size = len(s_c)
            best_set = s_c

    # Final polish
    s_final = local_search(best_set, max_iters=100)
    if len(s_final) > best_size:
        best_size = len(s_final)
        best_set = s_final

    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
