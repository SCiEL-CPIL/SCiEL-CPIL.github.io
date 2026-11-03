import time
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation and extract permutation sequence
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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Extract the sequence of permutations from the generated word
    # The word contains every permutation exactly once in the standard construction
    perm_sequence = []
    L = len(word)
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perm_sequence.append(window)
    
    # Verify we have all n! permutations
    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
        
    if len(perm_sequence) != total_perms:
        # Fallback to original if extraction failed (should not happen)
        return "".join(map(str, word))

    # 2. Local Search (2-Opt) to minimize total length
    # Length = n + sum(cost(P[i], P[i+1]))
    # cost(a, b) = n - overlap(a, b)
    
    def get_cost(a, b):
        # a, b are tuples of length n
        # Find max k such that a[n-k:] == b[:k]
        # k can be 0 to n-1 (if k=n, a==b, but perms are distinct in path? 
        # Actually, in a superpermutation, consecutive windows can be the same? 
        # No, if window i is P and window i+1 is P, then P is a shift of itself, 
        # which implies periodicity, impossible for distinct symbols.
        # So k < n always for distinct perms.
        for k in range(n - 1, -1, -1):
            if a[n-k:] == b[:k]:
                return n - k
        return n

    P = perm_sequence
    M = len(P)
    
    # Precompute cost matrix for all pairs of permutations
    # This allows O(1) cost lookups during 2-opt
    cost_matrix = {}
    for i in range(M):
        for j in range(M):
            if i != j:
                cost_matrix[(i, j)] = get_cost(P[i], P[j])
            else:
                cost_matrix[(i, j)] = n  # Should not happen in valid superperm path

    # Precompute initial cost
    current_cost = 0
    for i in range(M - 1):
        current_cost += cost_matrix[(i, i+1)]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try 2-opt
        # We iterate i from 0 to M-2, j from i+1 to M-1
        # Reversing P[i:j+1]
        for i in range(M - 1):
            if time.time() > deadline - 5:
                break
            # Optimization: skip if i is small and we just changed it? 
            # Simple restart strategy: if we make a change, break and restart outer loop.
            for j in range(i + 1, M):
                # Calculate delta cost
                # Edges involved:
                # (P[i-1], P[i]) -> (P[i-1], P[j])
                # (P[j], P[j+1]) -> (P[i], P[j+1])
                # Note: if i=0, no left edge. if j=M-1, no right edge.
                
                c1 = 0
                c2 = 0
                c3 = 0
                c4 = 0
                
                if i > 0:
                    c1 = cost_matrix[(i-1, i)]
                    c3 = cost_matrix[(i-1, j)]
                else:
                    # Start of path, cost is 0 (or n, but constant)
                    c1 = 0
                    c3 = 0
                    
                if j < M - 1:
                    c2 = cost_matrix[(j, j+1)]
                    c4 = cost_matrix[(i, j+1)]
                else:
                    c2 = 0
                    c4 = 0
                    
                delta = c3 + c4 - c1 - c2
                
                if delta < 0:
                    # Apply move
                    P[i:j+1] = P[i:j+1][::-1]
                    current_cost += delta
                    improved = True
                    break # Restart scan from i=0
    
    # 3. Construct final word from optimized path
    final_word = list(P[0])
    for i in range(1, M):
        a = P[i-1]
        b = P[i]
        # Find overlap
        ov = 0
        for k in range(n - 1, -1, -1):
            if a[n-k:] == b[:k]:
                ov = k
                break
        final_word.extend(b[ov:])
        
    return "".join(map(str, final_word))
