import time
import sys
import numpy as np
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    total_space = q ** n

    # Precompute powers of q
    q_powers = [1]
    for i in range(1, n):
        q_powers.append(q_powers[-1] * q)

    def int_to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits

    # Precompute neighbor indices using numpy for speed
    # For q=3, n=8, radius=1: neighborhood size = 1 + 8*(3-1) = 17
    # We'll build a list of lists of neighbor indices
    neighbor_indices = []
    for i in range(total_space):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q

        neighbors = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = digits[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    delta = (val - orig_val) * q_powers[pos]
                    new_index = i + delta
                    if 0 <= new_index < total_space:
                        neighbors.append(new_index)
        neighbor_indices.append(neighbors)

    # Build constraint matrix: each row is a word in the space, 
    # each column is a potential codeword
    # c[i][j] = 1 if word i is within radius of codeword j
    
    # For memory efficiency with 6561x6561 matrix, use sparse representation
    # But scipy linprog needs dense or sparse matrix. Let's use sparse.
    
    # Actually, for this problem size (6561x6561), let's use a different approach:
    # Instead of full LP, use a heuristic inspired by LP rounding:
    # 1. Compute fractional weights via iterative proportional fitting (sinkhorn-like)
    # 2. Use these to guide a greedy selection with better ordering
    
    # Alternative: Use a randomized approach with multiple restarts
    # Key insight: greedy with random perturbation + local search with 2-exchange
    
    import random
    
    def greedy_cover(uncovered_mask, order=None):
        """Greedy set cover with optional custom ordering"""
        if order is None:
            order = list(range(total_space))
        
        code = []
        uncovered = uncovered_mask
        remaining = set(order)
        
        while uncovered != 0 and remaining:
            best_word = -1
            max_cover = 0
            
            # Sample a subset if too many candidates
            candidates = list(remaining) if len(remaining) < 500 else random.sample(candidates, 500)
            
            for w in candidates:
                cover_count = 0
                for nb in neighbor_indices[w]:
                    if uncovered & (1 << nb):
                        cover_count += 1
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_word = w
            
            if best_word == -1:
                break
            
            code.append(best_word)
            remaining.discard(best_word)
            
            for nb in neighbor_indices[best_word]:
                uncovered &= ~(1 << nb)
        
        return code

    def try_remove_redundant(code_indices):
        """Try to remove redundant codewords"""
        if not code_indices:
            return code_indices
        
        coverage_count = [0] * total_space
        for idx in code_indices:
            for nb in neighbor_indices[idx]:
                coverage_count[nb] += 1
        
        removed = []
        for i, idx in enumerate(code_indices):
            redundant = True
            for nb in neighbor_indices[idx]:
                if coverage_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                removed.append(i)
                for nb in neighbor_indices[idx]:
                    coverage_count[nb] -= 1
        
        if removed:
            remove_set = set(removed)
            code_indices = [idx for i, idx in enumerate(code_indices) if i not in remove_set]
        
        return code_indices

    def two_exchange_search(code_indices, max_iterations=200):
        """Local search: try replacing pairs of codewords"""
        if len(code_indices) < 2:
            return code_indices
        
        best_code = list(code_indices)
        best_len = len(best_code)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            if len(best_code) == best_len and iteration > 20:
                break
            
            # Compute coverage
            coverage_count = [0] * total_space
            for idx in best_code:
                for nb in neighbor_indices[idx]:
                    coverage_count[nb] += 1
            
            # Try 2-exchange: remove two codewords, add one that covers what they did
            # Or remove one, add one that covers more
            improved = False
            
            # Try single replacement first (cheaper)
            for i in range(len(best_code)):
                if time.time() > deadline - 10:
                    break
                
                # What would be uncovered if we remove best_code[i]?
                temp_coverage = coverage_count[:]
                for nb in neighbor_indices[best_code[i]]:
                    temp_coverage[nb] -= 1
                
                uncovered_if_removed = [nb for nb in range(total_space) if temp_coverage[nb] == 0]
                
                if not uncovered_if_removed:
                    # Can remove without adding
                    best_code.pop(i)
                    improved = True
                    break
                
                # Try to find a single codeword that covers all uncovered
                for w in range(total_space):
                    if w == best_code[i]:
                        continue
                    covers_all = all(coverage_count[nb] >= 1 for nb in uncovered_if_removed)
                    # Check if w covers all the newly uncovered positions
                    w_neighbors = set(neighbor_indices[w])
                    if all(nb in w_neighbors for nb in uncovered_if_removed):
                        # Check if replacing improves or maintains coverage
                        # and doesn't cause other issues
                        best_code[i] = w
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved and len(best_code) >= 2:
                # Try 2-for-1: remove two, add one
                for i in range(len(best_code)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i+1, len(best_code)):
                        # What would be uncovered if we remove both?
                        temp_coverage = coverage_count[:]
                        for nb in neighbor_indices[best_code[i]]:
                            temp_coverage[nb] -= 1
                        for nb in neighbor_indices[best_code[j]]:
                            temp_coverage[nb] -= 1
                        
                        uncovered_if_removed = [nb for nb in range(total_space) if temp_coverage[nb] == 0]
                        
                        if not uncovered_if_removed:
                            best_code.pop(j)
                            best_code.pop(i)
                            improved = True
                            break
                        
                        # Try to find a codeword covering all uncovered
                        for w in range(total_space):
                            if w in (best_code[i], best_code[j]):
                                continue
                            w_neighbors = set(neighbor_indices[w])
                            if all(nb in w_neighbors for nb in uncovered_if_removed):
                                best_code.pop(j)
                                best_code.pop(i)
                                best_code.append(w)
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_code

    # Multi-restart local search
    best_result = None
    best_size = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random perturbation for greedy
        random.seed(restart * 1000 + 42)
        
        # Build a random ordering
        order = list(range(total_space))
        random.shuffle(order)
        
        code = greedy_cover((1 << total_space) - 1, order)
        
        # Remove redundant
        code = try_remove_redundant(code)
        
        # 2-exchange local search
        code = two_exchange_search(code, max_iterations=100)
        
        # Final redundant removal
        code = try_remove_redundant(code)
        
        # Verify coverage
        covered = 0
        for idx in code:
            for nb in neighbor_indices[idx]:
                covered |= (1 << nb)
        
        if covered == (1 << total_space) - 1:
            if len(code) < best_size:
                best_size = len(code)
                best_result = code
                break  # Got a good result, stop
    
    if best_result is None:
        # Fallback: deterministic greedy
        code = greedy_cover((1 << total_space) - 1)
        code = try_remove_redundant(code)
        best_result = code
    
    # Convert to digits
    result = []
    for idx in best_result:
        result.append(int_to_digits(idx))
    
    return result
