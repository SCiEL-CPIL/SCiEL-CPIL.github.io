"""Rewritten program for maximizing sum of pairwise-coprime integers from 1..256."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    def get_sum(s):
        return sum(s)
    
    # Initial solution: Greedy by value (larger numbers first, prefer those with fewer/smaller prime factors)
    # Start with 1, then add numbers in descending order if compatible
    solution = [1]
    # Sort candidates: prefer larger numbers, then those with fewer distinct prime factors, then smaller prime factors
    candidates = list(range(N, 1, -1))
    for a in candidates:
        if all(is_compatible(a, b) for b in solution):
            solution.append(a)
    
    current_sum = get_sum(solution)
    solution_set = set(solution)
    
    # Local search with 1-for-1, 2-for-1, and 1-for-2 moves
    # We'll use a simulated annealing-like approach with multiple restarts
    
    def try_add(x):
        """Try to add x to solution if compatible."""
        if x in solution_set:
            return False
        if all(is_compatible(x, b) for b in solution):
            solution.append(x)
            solution_set.add(x)
            return True
        return False
    
    def remove(x):
        """Remove x from solution."""
        solution.remove(x)
        solution_set.discard(x)
    
    # Precompute compatibility matrix for speed
    compat = [[True] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            compat[i][j] = is_compatible(i, j)
            compat[j][i] = compat[i][j]
    
    def is_compatible_with_all(x, s):
        """Check if x is compatible with all elements in set s."""
        for b in s:
            if not compat[x][b]:
                return False
        return True
    
    # Local search phase
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1-for-1 swaps: replace a in solution with b not in solution
        for a in list(solution):
            for b in range(N, max(N - 50, 1), -1):  # Try larger b first
                if b in solution_set:
                    continue
                if not compat[a][b]:
                    continue
                # Check if b is compatible with all other elements in solution
                others = [x for x in solution if x != a]
                if all(compat[b][x] for x in others):
                    # Check if this improves sum
                    if b > a:
                        remove(a)
                        solution.append(b)
                        solution_set.discard(a)
                        solution_set.add(b)
                        current_sum += b - a
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
        
        # 2-for-1: replace two elements a1, a2 with one element b
        # This is useful when a1+a2 < b and b is compatible with rest
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                a1, a2 = solution[i], solution[j]
                for b in range(N, max(N - 30, 1), -1):
                    if b in solution_set:
                        continue
                    if b <= a1 + a2:
                        break
                    # Check b compatible with all other elements
                    others = [x for k, x in enumerate(solution) if k != i and k != j]
                    if all(compat[b][x] for x in others):
                        # Remove a1, a2 and add b
                        solution.remove(a1)
                        solution.remove(a2)
                        solution.append(b)
                        solution_set.discard(a1)
                        solution_set.discard(a2)
                        solution_set.add(b)
                        current_sum += b - a1 - a2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 1-for-2: replace one element a with two elements b, c
        for a in list(solution):
            for b in range(N, max(N - 30, 1), -1):
                if b in solution_set:
                    continue
                if not compat[a][b]:
                    continue
                others = [x for x in solution if x != a]
                if not all(compat[b][x] for x in others):
                    continue
                for c in range(N, max(N - 30, 1), -1):
                    if c in solution_set or c == b:
                        continue
                    if not compat[a][c]:
                        continue
                    if not compat[b][c]:
                        continue
                    if not all(compat[c][x] for x in others):
                        continue
                    if b + c > a:
                        # Remove a, add b and c
                        solution.remove(a)
                        solution.append(b)
                        solution.append(c)
                        solution_set.discard(a)
                        solution_set.add(b)
                        solution_set.add(c)
                        current_sum += b + c - a
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final validation
    result = list(solution)
    assert len(result) == len(set(result)), "Duplicates found"
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result), "Non-int found"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) == 1, f"GCD not 1: {result[i]}, {result[j]}"
    
    return result
# EVOLVE-BLOCK-END
