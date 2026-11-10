import math
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # Generate standard superpermutation up to n-1
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if time.time() > deadline - 15:
        return "".join(map(str, word))
    
    # Now optimize the last step (m=n)
    # Get perms of n-1 from word
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    if num_perms == 0:
        return "".join(map(str, word))
    
    # Precompute blocks
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
    
    block_len = len(blocks[0])
    
    # Precompute overlap matrix
    # overlaps[i][j] = max overlap of block_i suffix and block_j prefix
    overlaps = [[0]*num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        bi = blocks[i]
        for j in range(num_perms):
            if i == j:
                continue
            bj = blocks[j]
            # Max possible overlap is block_len - 1 (can't be full since i != j)
            max_o = block_len - 1
            o = 0
            # Check from largest to smallest
            for k in range(max_o, 0, -1):
                if bi[len(bi)-k:] == bj[:k]:
                    o = k
                    break
            overlaps[i][j] = o
    
    if time.time() > deadline - 10:
        # Not enough time for optimization, use standard order
        std_word = []
        for idx in range(num_perms):
            block = blocks[idx]
            max_overlap = min(len(std_word), block_len)
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or std_word[len(std_word) - o:] == block[:o]:
                    overlap = o
                    break
            std_word += block[overlap:]
        if len(std_word) < len(word):
            word = std_word
        return "".join(map(str, word))
    
    def build_word_from_order(order):
        """Build superpermutation from a given order of permutation indices."""
        result = []
        for idx in order:
            block = blocks[idx]
            max_overlap = min(len(result), block_len)
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or result[len(result) - o:] == block[:o]:
                    overlap = o
                    break
            result += block[overlap:]
        return result
    
    def is_valid_superperm(word_candidate):
        """Check if word_candidate contains all n! permutations of 1..n."""
        if len(word_candidate) < n:
            return False
        target = math.factorial(n)
        seen = set()
        for i in range(len(word_candidate) - n + 1):
            w = tuple(word_candidate[i:i+n])
            if len(set(w)) == n:
                seen.add(w)
                if len(seen) == target:
                    return True
        return len(seen) == target
    
    # Standard order baseline
    std_order = list(range(num_perms))
    std_word = build_word_from_order(std_order)
    best_word = std_word
    best_len = len(std_word)
    
    # Nearest Neighbor heuristic
    def nearest_neighbor(start):
        visited = [False]*num_perms
        order = [start]
        visited[start] = True
        
        for _ in range(num_perms - 1):
            last = order[-1]
            best_next = -1
            best_overlap = -1
            for j in range(num_perms):
                if not visited[j]:
                    if overlaps[last][j] > best_overlap:
                        best_overlap = overlaps[last][j]
                        best_next = j
            if best_next == -1:
                break
            order.append(best_next)
            visited[best_next] = True
        
        return order
    
    # Try NN from different starting points
    for start in range(min(num_perms, 50)):
        if time.time() > deadline - 5:
            break
        order = nearest_neighbor(start)
        w = build_word_from_order(order)
        if len(w) < best_len:
            best_word = w
            best_len = len(w)
    
    # Random restarts with greedy construction
    random.seed(42)
    
    def random_greedy(start):
        """Greedy with some randomization."""
        visited = [False]*num_perms
        order = [start]
        visited[start] = True
        
        for _ in range(num_perms - 1):
            last = order[-1]
            # Get unvisited neighbors sorted by overlap
            candidates = []
            for j in range(num_perms):
                if not visited[j]:
                    candidates.append((overlaps[last][j], j))
            
            if not candidates:
                break
            
            # Sort by overlap descending
            candidates.sort(reverse=True)
            
            # Pick from top candidates with some randomness
            top_k = min(5, len(candidates))
            if top_k > 1 and random.random() < 0.3:
                # Random choice from top 3
                choice_idx = random.randint(0, min(2, top_k-1))
            else:
                choice_idx = 0
            
            best_next = candidates[choice_idx][1]
            order.append(best_next)
            visited[best_next] = True
        
        return order
    
    # Try random restarts
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline - 3:
            break
        start = random.randint(0, num_perms - 1)
        order = random_greedy(start)
        w = build_word_from_order(order)
        if len(w) < best_len:
            best_word = w
            best_len = len(w)
    
    # Local search: try swapping adjacent elements in the order
    if time.time() < deadline - 2:
        # Get the order that produced best_word
        # We need to reconstruct it - let's just do local search on the word directly
        # Actually, let's try 2-opt style improvements
        
        # Reconstruct order from best_word is complex, so let's try a different approach
        # Try to improve by checking if we can trim the end
        candidate = list(best_word)
        while len(candidate) >= n:
            # Check if removing last element still gives valid superperm
            test_candidate = candidate[:-1]
            if is_valid_superperm(test_candidate):
                candidate = test_candidate
                best_len = len(candidate)
            else:
                break
        best_word = candidate
    
    return "".join(map(str, best_word))
