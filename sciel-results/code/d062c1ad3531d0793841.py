"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Build classic superpermutation for n=1..6
    word = [1]
    for m in range(2, 7):
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
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Extract all windows of length 6 in order of first appearance
    perms_6, seen = [], set()
    for i in range(len(word) - 5):
        window = tuple(word[i:i + 6])
        if window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    assert len(perms_6) == 720, f"Expected 720 perms, got {len(perms_6)}"
    
    # Precompute overlap matrix: overlap[i][j] = longest suffix of perms_6[i] that is prefix of perms_6[j]
    N = 720
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms_6[i]
        for j in range(N):
            pj = perms_6[j]
            max_k = 5  # since length is 6, max overlap is 5
            for k in range(max_k, -1, -1):
                if k == 0 or pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Simulated annealing
    deadline = time.time() + 100
    
    current_order = list(range(N))
    
    def compute_total_overlap(order):
        total = 0
        for i in range(N - 1):
            total += overlap[order[i]][order[i+1]]
        return total
    
    current_overlap = compute_total_overlap(current_order)
    best_order = current_order[:]
    best_overlap = current_overlap
    
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.005
    
    while temp > min_temp and time.time() < deadline:
        i = random.randint(0, N-2)
        j = random.randint(i+1, N-1)
        
        a, b = current_order[i], current_order[j]
        
        # Compute delta
        delta = 0
        # Remove old edges
        if i > 0:
            delta -= overlap[current_order[i-1]][a]
        if i < N-1:
            delta -= overlap[a][current_order[i+1]]
        if j > 0:
            delta -= overlap[current_order[j-1]][b]
        if j < N-1:
            delta -= overlap[b][current_order[j+1]]
        
        # Add new edges
        if i > 0:
            delta += overlap[current_order[i-1]][b]
        if i < N-1:
            delta += overlap[b][current_order[i+1]]
        if j > 0:
            delta += overlap[current_order[j-1]][a]
        if j < N-1:
            delta += overlap[a][current_order[j+1]]
        
        # Also need to handle the edge between i and i+1, and j-1 and j
        # Wait, I'm double counting. Let me be more careful.
        
        # Actually, the edges affected are:
        # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # After swap: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
        
        # Let me redo this properly
        delta = 0
        # Old edges to remove
        if i > 0:
            delta -= overlap[current_order[i-1]][a]
        if i < N-1:
            delta -= overlap[a][current_order[i+1]]
        if j > 0:
            delta -= overlap[current_order[j-1]][b]
        if j < N-1:
            delta -= overlap[b][current_order[j+1]]
        
        # New edges to add
        if i > 0:
            delta += overlap[current_order[i-1]][b]
        if i < N-1:
            delta += overlap[b][current_order[i+1]]
        if j > 0:
            delta += overlap[current_order[j-1]][a]
        if j < N-1:
            delta += overlap[a][current_order[j+1]]
        
        if delta >= 0 or random.random() < (2.718281828 ** (delta / temp)):
            current_order[i], current_order[j] = b, a
            current_overlap += delta
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_order = current_order[:]
        # else: don't swap
        
        temp *= cooling
    
    final_order = best_order
    
    # Build final word
    result = word[:]
    for idx in final_order:
        p = list(perms_6[idx])
        block = p + [7] + p
        overlap_len = 0
        max_overlap = min(len(result), len(block))
        for o in range(max_overlap, -1, -1):
            if o == 0 or result[len(result) - o:] == block[:o]:
                overlap_len = o
                break
        result += block[overlap_len:]
    
    return "".join(map(str, result))

# EVOLVE-BLOCK-END
