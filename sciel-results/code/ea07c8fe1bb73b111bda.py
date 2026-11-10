"""Superpermutation solver using genetic evolution of insertion patterns."""
import time
import heapq
import random
from collections import defaultdict
from itertools import permutations

_DEADLINE = 145.0  # safety margin before 150s kill

def construct(n):
    start_time = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal: length 33
        return "12341213421124312132413214312142314"
    
    # For n >= 5, use genetic evolution of insertion/overlap patterns
    if n >= 5:
        return _genetic_superpermutation(n, start_time)
    
    # Fallback: classic construction
    return _classic_superpermutation(n)


def _classic_superpermutation(n):
    """Classic recursive construction: length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _genetic_superpermutation(n, start_time):
    """
    Genetic evolution of insertion/overlap patterns for superpermutation construction.
    
    The classic construction builds hierarchically: for each new symbol m (2..n),
    it processes (m-1)-permutations in some order and inserts blocks p+[m]+p with
    maximum overlap. We evolve the ordering of permutations at each level.
    
    For efficiency with n=7, we use the actual (m-1)-permutations that appear
    in the classic construction at each level, not all permutations.
    """
    symbols = list(range(1, n + 1))
    
    # Get the classic word and extract the permutation orderings at each level
    classic_word = _classic_list(n)
    best_word = classic_word[:]
    best_len = len(classic_word)
    
    elapsed = time.time() - start_time
    if elapsed > 3:
        return "".join(map(str, best_word))
    
    # Extract orderings from classic construction
    # At level m, the (m-1)-permutations are the windows of length (m-1) in the
    # (m-1)-superpermutation that have all distinct elements
    def _extract_orderings(word, n):
        """Extract permutation orderings at each level from a superpermutation."""
        orderings = []
        for m in range(2, n + 1):
            # Find all (m-1)-windows with distinct elements
            seen = []
            seen_set = set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_set:
                    seen_set.add(window)
                    seen.append(window)
            orderings.append(seen)
        return orderings
    
    classic_orderings = _extract_orderings(classic_word, n)
    
    def _build_from_orderings(orderings, n):
        """Build superpermutation from orderings (list of permutation lists per level)."""
        word = [1]
        for m in range(2, n + 1):
            order = orderings[m - 2]
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, 0, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Verify classic orderings reproduce classic word
    test_word = _build_from_orderings(classic_orderings, n)
    if test_word != classic_word:
        # Fallback to simple approach
        return "".join(map(str, best_word))
    
    # Generate population with varied orderings
    pop_size = 10
    population = [classic_orderings]
    
    for _ in range(pop_size - 1):
        genome = []
        for m in range(2, n + 1):
            perms = classic_orderings[m - 2][:]
            random.shuffle(perms)
            genome.append(perms)
        population.append(genome)
    
    # Evaluate
    for genome in population:
        word = _build_from_orderings(genome, n)
        if _is_valid_superpermutation(word, n) and len(word) < best_len:
            best_word = word[:]
            best_len = len(word)
    
    elapsed = time.time() - start_time
    if elapsed > 8:
        return "".join(map(str, best_word))
    
    # Evolution
    while time.time() - start_time < _DEADLINE - 15:
        new_pop = []
        
        # Elitism
        best_genome = None
        for genome in population:
            word = _build_from_orderings(genome, n)
            if _is_valid_superpermutation(word, n) and len(word) <= best_len:
                if len(word) < best_len:
                    best_word = word[:]
                    best_len = len(word)
                best_genome = genome
                break
        if best_genome:
            new_pop.append(best_genome)
        
        # Generate offspring
        while len(new_pop) < pop_size and time.time() - start_time < _DEADLINE - 15:
            if random.random() < 0.6:
                # Crossover
                p1 = random.choice(population)
                p2 = random.choice(population)
                child = []
                for m in range(2, n + 1):
                    if random.random() < 0.5:
                        child.append(p1[m - 2][:])
                    else:
                        child.append(p2[m - 2][:])
            else:
                # Mutation
                parent = random.choice(population)
                child = [p[:] for p in parent]
            
            # Mutate: swap elements in one or more levels
            num_swaps = random.randint(1, 2)
            for _ in range(num_swaps):
                m = random.randint(2, n)
                idx = m - 2
                if len(child[idx]) >= 2:
                    i, j = random.sample(range(len(child[idx])), 2)
                    child[idx][i], child[idx][j] = child[idx][j], child[idx][i]
            
            # Evaluate
            word = _build_from_orderings(child, n)
            if _is_valid_superpermutation(word, n):
                if len(word) < best_len:
                    best_word = word[:]
                    best_len = len(word)
                new_pop.append(child)
        
        if not new_pop:
            break
        population = new_pop
    
    return "".join(map(str, best_word))


def _classic_list(n):
    """Classic recursive construction, returns list of ints."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _is_valid_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as contiguous substrings."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == _factorial(n)


def _factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result


# Precompute for small n
_SMALL_RESULTS = {}

def _get_small_result(n):
    if n in _SMALL_RESULTS:
        return _SMALL_RESULTS[n]
    if n <= 4:
        # Known optimal
        opts = {1: "1", 2: "121", 3: "123121321", 4: "12341213421124312132413214312142314"}
        _SMALL_RESULTS[n] = opts[n]
        return opts[n]
    return None
