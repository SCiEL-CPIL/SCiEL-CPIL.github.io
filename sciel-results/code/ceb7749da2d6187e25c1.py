"""Superpermutation construction with deadline guard."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _standard_superpermutation(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _perm_overlap(p1, p2, n):
    """Maximum k such that p1[-k:] == p2[:k] and k < n."""
    for k in range(n - 1, -1, -1):
        if p1[-k:] == p2[:k]:
            return k
    return 0

def _word_to_perms(word, n):
    """Extract all unique permutations in order from the word."""
    perms = []
    seen = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if len(set(p)) == n and p not in seen:
            seen.add(p)
            perms.append(p)
    return perms

def _total_cost(s, n):
    """Total length of concatenated permutations with overlaps."""
    if not s:
        return 0
    cost = n
    for i in range(len(s) - 1):
        cost += n - _perm_overlap(s[i], s[i+1], n)
    return cost

def _reconstruct_word(seq, n):
    """Build the word from a sequence of permutations."""
    if not seq:
        return []
    result = list(seq[0])
    for i in range(1, len(seq)):
        ov = _perm_overlap(seq[i-1], seq[i], n)
        result.extend(seq[i][ov:])
    return result

def _construct_sa(n, deadline):
    """Simulated annealing on permutation ordering to minimize superpermutation length."""
    start_time = time.time()
    
    # Get the permutation order from standard construction
    std_word = _standard_superpermutation(n)
    seq = _word_to_perms(std_word, n)
    N = len(seq)
    
    n_fact = math.factorial(n)
    if N != n_fact:
        return std_word
    
    def overlap(p1, p2):
        for k in range(n - 1, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0
    
    def total_cost(s):
        c = n
        for i in range(len(s) - 1):
            c += n - overlap(s[i], s[i+1])
        return c
    
    current = seq[:]
    current_cost = total_cost(current)
    best = current[:]
    best_cost = current_cost
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    
    random.seed(42)
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        
        # Compute delta in O(n) for the affected edges
        def edge(a, b):
            return n - overlap(a, b)
        
        old_cost = 0
        new_cost = 0
        
        if abs(i - j) > 1:
            # Positions i and j are not adjacent
            if i > 0:
                old_cost += edge(current[i-1], current[i])
                new_cost += edge(current[i-1], current[j])
            if i < N-1:
                old_cost += edge(current[i], current[i+1])
                new_cost += edge(current[j], current[i+1])
            if j > 0:
                old_cost += edge(current[j-1], current[j])
                new_cost += edge(current[j-1], current[i])
            if j < N-1:
                old_cost += edge(current[j], current[j+1])
                new_cost += edge(current[i], current[j+1])
        else:
            # Adjacent: just i and i+1
            k = min(i, j)
            l = max(i, j)
            if k > 0:
                old_cost += edge(current[k-1], current[k])
                new_cost += edge(current[k-1], current[l])
            old_cost += edge(current[k], current[l])
            new_cost += edge(current[l], current[k])
            if l < N-1:
                old_cost += edge(current[l], current[l+1])
                new_cost += edge(current[k], current[l+1])
        
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current[i], current[j] = current[j], current[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best = current[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return _reconstruct_word(best, n)

def construct(n):
    start = time.time()
    deadline = start + 135.0  # Leave 5s safety margin
    
    if n <= 1:
        return "1"
    
    if n <= 5:
        return "".join(map(str, _standard_superpermutation(n)))
    
    # For n >= 6, try SA
    try:
        result = _construct_sa(n, deadline)
        if result and len(result) <= math.factorial(n) * n:
            return "".join(map(str, result))
    except Exception:
        pass
    
    # Fallback
    return "".join(map(str, _standard_superpermutation(n)))

# EVOLVE-BLOCK-END
