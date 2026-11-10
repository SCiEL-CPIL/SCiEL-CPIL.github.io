import math
import time
import random
import sys
import numpy as np

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
    
    # Precompute overlap matrix using NumPy for speed
    # overlaps[i][j] = max overlap of block_i suffix and block_j prefix
    num_perms_np = num_perms
    max_o = block_len - 1
    
    # Convert blocks to numpy array
    blocks_np = np.array(blocks, dtype=np.int8)
    
    # For each possible overlap k (from max to 1), check which pairs match
    # overlaps[i][j] = k if blocks[i][-k:] == blocks[j][:k]
    overlaps_np = np.zeros((num_perms_np, num_perms_np), dtype=np.int16)
    
    # For each k from max_o down to 1, check if suffix of length k of block i
    # equals prefix of length k of block j
    for k in range(max_o, 0, -1):
        # suffixes: blocks_np[:, -k:] shape (num_perms, k)
        # prefixes: blocks_np[:, :k] shape (num_perms, k)
        # We need: for each i, j: blocks[i][-k:] == blocks[j][:k]
        # This is: (blocks_np[:, -k:, None, :] == blocks_np[None, :, :k, :]).all(axis=-1)
        # But that's O(num_perms^2 * k) which is expensive
        # Instead, use a hash-based approach or direct comparison
        
        # More efficient: for each k, compute suffix hashes and prefix hashes
        # Then match them
        suffixes = blocks_np[:, -k:]  # (num_perms, k)
        prefixes = blocks_np[:, :k]   # (num_perms, k)
        
        # Create a mapping from prefix tuple to index
        # For each i, find all j where suffixes[i] == prefixes[j]
        # Use np.unique or dict approach
        prefix_dict = {}
        for j in range(num_perms_np):
            key = tuple(prefixes[j].tolist())
            prefix_dict[key] = j
        
        for i in range(num_perms_np):
            key = tuple(suffixes[i].tolist())
            if key in prefix_dict:
                j = prefix_dict[key]
                if i != j:  # Can't overlap with self
                    overlaps_np[i, j] = k
    
    overlaps = overlaps_np.tolist()
    
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
    
    # Nearest Neighbor heuristic (optimized with numpy)
    overlaps_np = np.array(overlaps, dtype=np.int16)
    
    def nearest_neighbor(start):
        visited = np.zeros(num_perms, dtype=bool)
        order = [start]
        visited[start] = True
        
        for _ in range(num_perms - 1):
            last = order[-1]
            # Get overlaps from last to all unvisited
            row = overlaps_np[last]
            # Mask visited
            row_masked = row.copy()
            row_masked[visited] = -1
            best_next = int(np.argmax(row_masked))
            if row_masked[best_next] < 0:
                break
            order.append(best_next)
            visited[best_next] = True
        
        return order
    
    # Try NN from more starting points (we have more time now)
    num_nn_starts = min(num_perms, 200)
    for start in range(num_nn_starts):
        if time.time() > deadline - 5:
            break
        order = nearest_neighbor(start)
        w = build_word_from_order(order)
        if len(w) < best_len:
            best_word = w
            best_len = len(w)
    
    # Random restarts with greedy construction (optimized)
    random.seed(42)
    
    def random_greedy(start, temperature=0.3):
        """Greedy with some randomization, optimized with numpy."""
        visited = np.zeros(num_perms, dtype=bool)
        order = [start]
        visited[start] = True
        
        for _ in range(num_perms - 1):
            last = order[-1]
            row = overlaps_np[last].copy()
            row[visited] = -1
            
            # Get top candidates
            max_val = np.max(row)
            if max_val < 0:
                break
            
            # Find all with max overlap
            top_indices = np.where(row == max_val)[0]
            if len(top_indices) > 1 and random.random() < temperature:
                # Random choice among top
                best_next = int(top_indices[random.randint(0, len(top_indices)-1)])
            else:
                best_next = int(top_indices[0])
            
            order.append(best_next)
            visited[best_next] = True
        
        return order
    
    # Try more random restarts (we have more time now)
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline - 3:
            break
        start = random.randint(0, num_perms - 1)
        temp = random.uniform(0.1, 0.5)
        order = random_greedy(start, temperature=temp)
        w = build_word_from_order(order)
        if len(w) < best_len:
            best_word = w
            best_len = len(w)
    
    # Local search: try swapping adjacent elements in the order
    if time.time() < deadline - 2:
        # Try to trim the end
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
        
        # Try 2-opt style local search on the order
        # Reconstruct a good order, then try swaps
        # Use the best order we found
        # For simplicity, try a few random 2-opt moves
        best_order = None
        # Re-find the order that gave best_word by trying all starts
        # This is expensive, so just do a few random 2-opt attempts
        for _ in range(20):
            if time.time() > deadline - 1:
                break
            # Get a random good order
            start = random.randint(0, num_perms - 1)
            order = nearest_neighbor(start)
            w = build_word_from_order(order)
            if len(w) >= best_len:
                continue
            
            # Try 2-opt
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(len(order) - 2):
                    if time.time() > deadline - 1:
                        break
                    # Try reversing order[i+1:i+3]
                    new_order = order[:i+1] + [order[i+2], order[i+1]] + order[i+3:]
                    w2 = build_word_from_order(new_order)
                    if len(w2) < best_len:
                        best_word = w2
                        best_len = len(w2)
                        order = new_order
                        improved = True
                        break
    
    return "".join(map(str, best_word))
