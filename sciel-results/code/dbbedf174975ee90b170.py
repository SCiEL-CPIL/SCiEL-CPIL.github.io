# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations as _itertools_permutations

def construct(n):
    """
    Construct a superpermutation for n symbols.
    Returns a string of digits 1..n such that every permutation appears as a substring of length n.
    """
    deadline = time.time() + 115  # safety margin before 120s kill
    start_time = time.time()
    
    # For small n, use known optimal constructions
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return _standard_construction(4)
    if n == 5:
        return _standard_construction(5)
    if n == 6:
        return _standard_construction(6)
    
    # For n >= 7: build up to n-1 using standard construction,
    # then optimize the final stage with enhanced local search
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        word = _expand_step(word, m)
    
    # Final stage: add symbol n
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    if not perms:
        return "".join(map(str, word))
    
    # Precompute overlap costs between all pairs of permutations
    block_len = 2 * m - 1
    k_perms = len(perms)
    
    # Precompute overlap as a 2D list
    # overlap_matrix[i][j]: max k in 0..m-1 where perms[i][-k:] == perms[j][:k]
    overlap_matrix = [[0] * k_perms for _ in range(k_perms)]
    for i in range(k_perms):
        pi = perms[i]
        for j in range(k_perms):
            if i == j:
                overlap_matrix[i][j] = block_len
                continue
            pj = perms[j]
            # Find max overlap: suffix of pi matches prefix of pj
            best_k = 0
            for k in range(m - 1, 0, -1):
                if pi[-k:] == pj[:k]:
                    best_k = k
                    break
            overlap_matrix[i][j] = best_k
    
    # Multi-start greedy initialization
    best_order_indices = None
    best_len = float('inf')
    
    num_starts = min(5, k_perms)
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        order_indices = _greedy_order(k_perms, overlap_matrix, block_len, deadline, start_idx)
        cur_len = compute_length_fast(order_indices, overlap_matrix, block_len)
        if cur_len < best_len:
            best_len = cur_len
            best_order_indices = list(order_indices)
    
    if best_order_indices is None:
        best_order_indices = list(range(k_perms))
        best_len = compute_length_fast(best_order_indices, overlap_matrix, block_len)
    
    # Enhanced local search: adjacent swaps, 2-opt, 3-opt, and simulated annealing
    if n <= 7 and time.time() < deadline - 15:
        best_order_indices, best_len = _enhanced_local_search(
            best_order_indices, best_len, overlap_matrix, block_len, deadline
        )
    
    # Build final word
    final_order = [perms[idx] for idx in best_order_indices]
    return "".join(map(str, _build_final_stage(final_order, m)))


def compute_length_fast(ord_indices, overlap_matrix, block_len):
    if len(ord_indices) <= 1:
        return block_len
    total = block_len
    for t in range(len(ord_indices) - 1):
        total += block_len - overlap_matrix[ord_indices[t]][ord_indices[t + 1]]
    return total


def _greedy_order(k, overlap_matrix, block_len, deadline, start_idx=0):
    """Greedy nearest-neighbor starting from a specific permutation."""
    visited = [False] * k
    order_indices = [start_idx]
    visited[start_idx] = True
    current = start_idx
    
    for _ in range(k - 1):
        if time.time() > deadline - 10:
            break
        best_next = -1
        best_cost = float('inf')
        for j in range(k):
            if not visited[j]:
                cost = block_len - overlap_matrix[current][j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
        if best_next == -1:
            break
        order_indices.append(best_next)
        visited[best_next] = True
        current = best_next
    
    # Fill in any unvisited
    for j in range(k):
        if not visited[j]:
            order_indices.append(j)
    
    return order_indices


def _enhanced_local_search(order_indices, best_len, overlap_matrix, block_len, deadline):
    """Enhanced local search with multiple move types and simulated annealing."""
    current = list(order_indices)
    current_len = best_len
    n = len(current)
    
    # Precompute cost matrix for faster delta calculations
    cost_matrix = [[block_len - overlap_matrix[i][j] for j in range(len(overlap_matrix))] for i in range(len(overlap_matrix))]
    
    def get_delta_adjacent_swap(i):
        """Delta for swapping positions i and i+1."""
        a = current[i]
        b = current[i + 1]
        old_cost = cost_matrix[a][b]
        new_cost = cost_matrix[b][a]
        if i > 0:
            left = current[i - 1]
            old_cost += cost_matrix[left][a]
            new_cost += cost_matrix[left][b]
        if i + 2 < n:
            right = current[i + 2]
            old_cost += cost_matrix[b][right]
            new_cost += cost_matrix[a][right]
        return new_cost - old_cost
    
    def get_delta_2opt(i, j):
        """Delta for reversing segment [i..j]."""
        if i >= j:
            return 0
        old_cost = 0
        new_cost = 0
        if i > 0:
            left = current[i - 1]
            old_cost += cost_matrix[left][current[i]]
            new_cost += cost_matrix[left][current[j]]
        if j + 1 < n:
            right = current[j + 1]
            old_cost += cost_matrix[current[j]][right]
            new_cost += cost_matrix[current[i]][right]
        for t in range(i, j):
            old_cost += cost_matrix[current[t]][current[t + 1]]
            new_cost += cost_matrix[current[t + 1]][current[t]]
        return new_cost - old_cost
    
    # Phase 1: Hill climbing with adjacent swaps
    improved = True
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > deadline - 15:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 15:
                break
            delta = get_delta_adjacent_swap(i)
            if delta < 0:
                current[i], current[i + 1] = current[i + 1], current[i]
                current_len += delta
                improved = True
        if not improved:
            break
    
    # Phase 2: 2-opt moves
    improved = True
    max_rounds = 100
    seg_limit = min(50, n)
    for rnd in range(max_rounds):
        if time.time() > deadline - 15:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, min(i + seg_limit, n)):
                if time.time() > deadline - 15:
                    break
                delta = get_delta_2opt(i, j)
                if delta < 0:
                    current[i:j + 1] = current[i:j + 1][::-1]
                    current_len += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Phase 3: Simulated annealing with perturbation
    if time.time() < deadline - 20:
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        sa_improved = True
        while sa_improved and time.time() < deadline - 15:
            sa_improved = False
            # Try random perturbation + local search
            for _ in range(20):
                if time.time() > deadline - 15:
                    break
                perturbed = list(current)
                # Randomly swap 5-10 pairs
                num_swaps = random.randint(5, 10)
                for _ in range(num_swaps):
                    i, j = random.sample(range(n), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                
                perturbed_len = compute_length_fast(perturbed, overlap_matrix, block_len)
                
                # Quick local search on perturbed
                for _ in range(30):
                    if time.time() > deadline - 15:
                        break
                    imp = False
                    for i in range(n - 1):
                        a = perturbed[i]
                        b = perturbed[i + 1]
                        old_cost = cost_matrix[a][b]
                        new_cost = cost_matrix[b][a]
                        if i > 0:
                            left = perturbed[i - 1]
                            old_cost += cost_matrix[left][a]
                            new_cost += cost_matrix[left][b]
                        if i + 2 < n:
                            right = perturbed[i + 2]
                            old_cost += cost_matrix[b][right]
                            new_cost += cost_matrix[a][right]
                        if new_cost < old_cost:
                            perturbed[i], perturbed[i + 1] = perturbed[i + 1], perturbed[i]
                            perturbed_len += new_cost - old_cost
                            imp = True
                    if not imp:
                        break
                
                delta = perturbed_len - current_len
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
                    if delta < 0:
                        sa_improved = True
                    current = perturbed
                    current_len = perturbed_len
                
                temp *= cooling
                if temp < min_temp:
                    break
            
            if temp < min_temp:
                break
    
    return current, current_len


def _standard_construction(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        word = _expand_step(word, m)
    return "".join(map(str, word))


def _expand_step(word, m):
    """Expand word from level m-1 to level m using standard construction."""
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _build_final_stage(order, m):
    """Build the final stage word given an ordering of (m-1)-permutations."""
    new = []
    for p in order:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _find_overlap(current, block):
    """Find the maximum overlap between end of current and start of block."""
    if not current:
        return 0
    max_overlap = min(len(current), len(block))
    for o in range(max_overlap, -1, -1):
        if o == 0:
            return 0
        if current[len(current) - o:] == block[:o]:
            return o
    return 0
# EVOLVE-BLOCK-END
