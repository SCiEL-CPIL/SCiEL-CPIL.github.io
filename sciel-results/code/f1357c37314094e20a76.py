import time
import random
import math
import sys

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341213421234132413214321412341213421324132143214"
    
    # For n >= 5, use the standard recursive construction
    # which gives length sum_{k=1}^{n} k!
    
    def get_standard_superperm(m):
        """Generate the standard recursive superpermutation for m symbols."""
        if m == 1:
            return [1]
        
        word = [1]
        for k in range(2, m + 1):
            # Extract all distinct (k-1)-permutations from current word
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Build new word by expanding each permutation
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                # Find max overlap with current end of new
                overlap = 0
                limit = min(len(new), len(block))
                for o in range(limit, -1, -1):
                    if o == 0:
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Get S_{n-1}
    word_prev = get_standard_superperm(n - 1)
    
    # Extract order of (n-1) permutations from word_prev
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - (n - 1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    N = len(perms_order)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that suffix of perms_order[i] of length k == prefix of perms_order[j]
    overlap_matrix = [[0]*N for _ in range(N)]
    max_overlap_len = n - 2
    
    for i in range(N):
        p1 = perms_order[i]
        for j in range(N):
            if i == j:
                continue
            p2 = perms_order[j]
            # Check overlaps from max down to 0
            for k in range(max_overlap_len, -1, -1):
                if k == 0:
                    break
                if p1[n-1-k:] == p2[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Use the natural order from the recursive construction
    current_order = list(range(N))
    
    # Try to improve with hill climbing (swap-based)
    start_time = time.time()
    timeout = 100  # Leave 20s margin
    
    def calc_total_overlap(order):
        total = 0
        for i in range(N - 1):
            total += overlap_matrix[order[i]][order[i+1]]
        return total
    
    current_overlap = calc_total_overlap(current_order)
    
    # Hill climbing with random swaps
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Try random swaps
        for _ in range(10000):
            if time.time() - start_time > timeout:
                break
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            u = current_order[i]
            v = current_order[j]
            
            # Calculate old overlap contribution
            old_ov = 0
            if i > 0: old_ov += overlap_matrix[current_order[i-1]][u]
            if i < N-1: old_ov += overlap_matrix[u][current_order[i+1]]
            if j > 0: old_ov += overlap_matrix[current_order[j-1]][v]
            if j < N-1: old_ov += overlap_matrix[v][current_order[j+1]]
            
            # Calculate new overlap contribution
            new_ov = 0
            if j == i + 1:
                # Adjacent
                if i > 0: new_ov += overlap_matrix[current_order[i-1]][v]
                new_ov += overlap_matrix[v][u]
                if j < N-1: new_ov += overlap_matrix[u][current_order[j+1]]
            else:
                # Non-adjacent
                if i > 0: new_ov += overlap_matrix[current_order[i-1]][v]
                if i < N-1: new_ov += overlap_matrix[v][current_order[i+1]]
                if j > 0: new_ov += overlap_matrix[current_order[j-1]][u]
                if j < N-1: new_ov += overlap_matrix[u][current_order[j+1]]
            
            if new_ov > old_ov:
                current_order[i], current_order[j] = v, u
                current_overlap += (new_ov - old_ov)
                improved = True
                break
    
    # Construct the final word
    word = []
    for idx in range(N):
        p = perms_order[current_order[idx]]
        block = list(p) + [n] + list(p)
        if not word:
            word = block
        else:
            # Find overlap
            max_k = n - 2
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    break
                if word[-k:] == block[:k]:
                    found = k
                    break
            word += block[found:]
    
    return "".join(map(str, word))
