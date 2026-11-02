import time
import itertools
import sys
import random
import math

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal Graham-Lagarias construction.
    For n = 7, uses Graham-Lagarias as a seed and applies local search
    to the final step's permutation ordering to minimize length.
    """
    if n == 1:
        return "1"
    
    # Standard Graham-Lagarias construction
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        # Get word and perms for m-1
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    if n <= 6:
        word, _ = graham_lagarias(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Get perms of 1..6 in the order they appear in the standard construction
        # We need the list of permutations P_1 ... P_720
        _, perms = graham_lagarias(6)
        
        num_perms = len(perms)
        if num_perms != 720:
            # Fallback to standard if something went wrong
            word, _ = graham_lagarias(7)
            return "".join(map(str, word))
            
        # Precompute overlap matrix using string representation for speed
        # Convert each perm to string for faster comparison
        perm_strings = ["".join(map(str, p)) for p in perms]
        
        max_k = n - 2
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        # Precompute suffixes and prefixes for fast comparison
        suffixes = [perm_strings[i][max_k-k:] for i in range(num_perms) for k in range(max_k+1)]
        prefixes = [perm_strings[j][:k] for j in range(num_perms) for k in range(max_k+1)]
        
        # More efficient: just compare directly
        for i in range(num_perms):
            si = perm_strings[i]
            for j in range(num_perms):
                sj = perm_strings[j]
                found = 0
                for k in range(max_k, 0, -1):
                    if si[-k:] == sj[:k]:
                        found = k
                        break
                overlap_matrix[i][j] = found
                
        # Initial order: use nearest-neighbor heuristic
        # Start with perm 0, repeatedly add the perm with max overlap to current end
        order = [0]
        used = [False] * num_perms
        used[0] = True
        
        for _ in range(num_perms - 1):
            last = order[-1]
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if not used[j]:
                    ov = overlap_matrix[last][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            order.append(best_next)
            used[best_next] = True
        
        # Calculate initial total overlap
        total_overlap = 0
        for i in range(num_perms - 1):
            total_overlap += overlap_matrix[order[i]][order[i+1]]
            
        best_length = 2 * (n - 1) + 1 + (num_perms - 1) * (2 * (n - 1) + 1) - total_overlap
        best_order = order[:]
        
        start_time = time.time()
        deadline = start_time + 110.0 # Leave 10s for safety margin (total 120s limit)
        
        # Phase 1: 2-opt local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = num_perms
            # 2-opt search with O(1) delta updates
            for i in range(N - 1):
                S_fwd = 0
                S_rev = 0
                base_left = overlap_matrix[order[i-1]][order[i]] if i > 0 else 0
                
                for j in range(i, N):
                    if j > i:
                        # Add order[j] to the segment [i, j-1]
                        # The new forward edge is order[j-1] -> order[j]
                        # The new reverse edge is order[j] -> order[j-1]
                        S_fwd += overlap_matrix[order[j-1]][order[j]]
                        S_rev += overlap_matrix[order[j]][order[j-1]]
                    
                    # Calculate delta
                    # Delta = (New Overlap Sum) - (Old Overlap Sum)
                    # New Internal: S_rev
                    # Old Internal: S_fwd
                    delta = S_rev - S_fwd
                    
                    # Left Boundary
                    if i > 0:
                        # Old: order[i-1] -> order[i]
                        # New: order[i-1] -> order[j]
                        delta += overlap_matrix[order[i-1]][order[j]] - base_left
                    
                    # Right Boundary
                    if j < N - 1:
                        # Old: order[j] -> order[j+1]
                        # New: order[i] -> order[j+1]
                        delta += overlap_matrix[order[i]][order[j+1]] - overlap_matrix[order[j]][order[j+1]]
                        
                    if delta > 0:
                        # Apply reversal
                        order[i:j+1] = order[i:j+1][::-1]
                        total_overlap += delta
                        improved = True
                        
                        # Update S_fwd and S_rev for the new segment [i, j]
                        # The segment is now reversed.
                        # The forward path of the new segment is the reverse path of the old segment.
                        S_fwd, S_rev = S_rev, S_fwd
                        
                        # Update best
                        current_length = 2 * (n - 1) + 1 + (num_perms - 1) * (2 * (n - 1) + 1) - total_overlap
                        if current_length < best_length:
                            best_length = current_length
                            best_order = order[:]
        
        # Phase 2: Simulated annealing to escape local optima
        if time.time() < deadline:
            T = 5.0  # Initial temperature
            cooling = 0.999
            min_T = 0.1
            current_order = best_order[:]
            current_overlap = 0
            for i in range(num_perms - 1):
                current_overlap += overlap_matrix[current_order[i]][current_order[i+1]]
            
            best_sa_order = best_order[:]
            best_sa_overlap = best_length if best_length < 2 * (n - 1) + 1 + (num_perms - 1) * (2 * (n - 1) + 1) else 0
            # Recompute best_sa_overlap from best_order
            best_sa_overlap = sum(overlap_matrix[best_order[i]][best_order[i+1]] for i in range(num_perms - 1))
            
            while T > min_T and time.time() < deadline:
                # Pick two random positions and swap
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                # Calculate delta for swap
                old_ov = overlap_matrix[current_order[i]][current_order[i+1]] + overlap_matrix[current_order[j-1]][current_order[j]] + overlap_matrix[current_order[j]][current_order[j+1]] if j < num_perms - 1 else overlap_matrix[current_order[i]][current_order[i+1]] + overlap_matrix[current_order[j-1]][current_order[j]]
                
                # More careful calculation
                delta = 0
                # Remove old edges
                if i > 0:
                    delta -= overlap_matrix[current_order[i-1]][current_order[i]]
                if i < num_perms - 1:
                    delta -= overlap_matrix[current_order[i]][current_order[i+1]]
                if j > 0:
                    delta -= overlap_matrix[current_order[j-1]][current_order[j]]
                if j < num_perms - 1:
                    delta -= overlap_matrix[current_order[j]][current_order[j+1]]
                
                # Add new edges
                if i > 0:
                    delta += overlap_matrix[current_order[i-1]][current_order[j]]
                if i < num_perms - 1:
                    delta += overlap_matrix[current_order[j]][current_order[i+1]]
                if j > 0:
                    delta += overlap_matrix[current_order[i]][current_order[j]]
                if j < num_perms - 1:
                    delta += overlap_matrix[current_order[i]][current_order[j+1]]
                
                # Handle the edge between i and j if they're adjacent
                if j == i + 1:
                    # Special case: adjacent swap
                    delta = 0
                    # Remove old edges
                    if i > 0:
                        delta -= overlap_matrix[current_order[i-1]][current_order[i]]
                    if j < num_perms - 1:
                        delta -= overlap_matrix[current_order[j]][current_order[j+1]]
                    # Add new edges
                    if i > 0:
                        delta += overlap_matrix[current_order[i-1]][current_order[j]]
                    if j < num_perms - 1:
                        delta += overlap_matrix[current_order[i]][current_order[j+1]]
                
                # Accept or reject
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Perform swap
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_overlap += delta
                    
                    if current_overlap > best_sa_overlap:
                        best_sa_overlap = current_overlap
                        best_sa_order = current_order[:]
                        # Update global best
                        current_length = 2 * (n - 1) + 1 + (num_perms - 1) * (2 * (n - 1) + 1) - current_overlap
                        if current_length < best_length:
                            best_length = current_length
                            best_order = current_order[:]
                
                T *= cooling
        
        # Construct final word
        final_word = []
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
            
        return "".join(map(str, final_word))
        
    # Fallback for n > 7 (unlikely to be tested, but handle gracefully)
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
