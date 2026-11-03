"""Shortest superpermutation construction for n symbols."""
import time
import random
import sys
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    """Construct a superpermutation over symbols 1..n.
    
    Returns a string of digits 1..n such that every permutation of 1..n
    appears as n consecutive symbols.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    # For small n, use the classic recursive construction
    if n <= 4:
        return _classic_construct(n)
    
    # For n >= 5, use optimized construction
    # Strategy: Build word for n-1, then insert n's in optimal order
    
    # Step 1: Build base word for n-1
    base_word = _classic_construct(n - 1)
    base_list = [int(c) for c in base_word]
    
    # Extract all (n-1)-permutations from the base word in order of first appearance
    perms = []
    seen = set()
    for i in range(len(base_list) - (n - 2)):
        window = tuple(base_list[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Step 2: For each permutation, we create a block: perm + [n] + perm
    # The key is to order these blocks to maximize overlaps
    # 
    # When going from block A (perm a) to block B (perm b):
    # Block A ends with: ...a[-2], a[-1]
    # Block B starts with: b[0], b[1], ...
    # Overlap: find longest suffix of block A that matches prefix of block B
    #
    # The block for perm p is: p[0], p[1], ..., p[n-2], n, p[0], p[1], ..., p[n-2]
    # So block has length 2*(n-1) + 1 = 2n-1
    #
    # Suffix of block A of length k (k <= n-1) is: a[n-1-k], ..., a[n-2]
    # Prefix of block B of length k (k <= n-1) is: b[0], ..., b[k-1]
    #
    # So overlap is the longest k <= n-1 such that:
    #   a[n-1-k : n-1] == b[0 : k]
    
    m = n - 1  # size of each permutation
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = number of characters we can overlap when going from perm i to perm j
    overlap = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            # Find max k (1 <= k <= m) such that pi[m-k:] == pj[:k]
            best_k = 0
            for k in range(1, m + 1):
                if pi[m-k:] == pj[:k]:
                    best_k = k
                else:
                    break
            overlap[i][j] = best_k
    
    # Step 3: Find a good ordering of permutations to minimize total length
    # Total length = sum of (block_length - overlap) for consecutive pairs
    # = num_perms * (2*m + 1) - sum of overlaps
    # So we want to maximize sum of overlaps = TSP-like problem
    
    # Use greedy + local search
    order = _solve_tsp_greedy(overlap, deadline, start_time)
    
    # Step 4: Build the final word
    final = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        if final:
            # Find maximum overlap with current end of final
            prev = order[-1] if len(order) > 0 else idx
            # Actually we need to check overlap with what we've already placed
            # The last permutation placed is order[len(order)-2] if len > 1
            # But simpler: just find overlap between end of final and start of block
            max_overlap = min(len(final), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or final[len(final)-o:] == block[:o]:
                    best_overlap = o
                    break
            final.extend(block[best_overlap:])
        else:
            final.extend(block)
    
    result = "".join(map(str, final))
    
    # Verify and return
    return result


def _classic_construct(n):
    """Classic recursive superpermutation construction."""
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


def _solve_tsp_greedy(overlap, deadline, start_time):
    """Solve the ordering problem using greedy construction + 2-opt local search."""
    num_perms = len(overlap)
    
    # Greedy construction: start from random node, always go to node with max overlap
    random.seed(42)
    
    # Try multiple starting points and take the best
    best_order = None
    best_cost = float('inf')
    
    num_starts = min(20, num_perms)
    start_indices = random.sample(range(num_perms), num_starts)
    
    for start in start_indices:
        if time.time() > deadline:
            break
        order = [start]
        used = {start}
        
        while len(order) < num_perms:
            last = order[-1]
            # Find unused node with maximum overlap
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if j not in used and overlap[last][j] > best_ov:
                    best_ov = overlap[last][j]
                    best_next = j
            if best_next == -1:
                break
            order.append(best_next)
            used.add(best_next)
        
        if len(order) < num_perms:
            # Add remaining
            remaining = [j for j in range(num_perms) if j not in used]
            order.extend(remaining)
        
        cost = _compute_cost(overlap, order)
        if cost < best_cost:
            best_cost = cost
            best_order = order
    
    if best_order is None:
        return list(range(num_perms))
    
    # 2-opt local search
    order = best_order[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_perms):
                # Compute cost change for reversing segment i+1..j
                old_cost = _local_cost(overlap, order, i, j)
                new_cost = _local_cost_after_reverse(overlap, order, i, j)
                if new_cost < old_cost:
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    improved = True
                    if time.time() > deadline:
                        return order
    
    return order


def _compute_cost(overlap, order):
    """Compute total overlap sum for an ordering (higher is better)."""
    total = 0
    for i in range(len(order) - 1):
        total += overlap[order[i]][order[i+1]]
    return total


def _local_cost(overlap, order, i, j):
    """Compute the cost contribution of edges affected by reversing segment i+1..j."""
    # Edges affected: (i, i+1), (j, j+1) [if j < len-1]
    # and internal edges within i+1..j
    cost = 0
    if i > 0:
        cost += overlap[order[i-1]][order[i]]
    cost += overlap[order[i]][order[i+1]]
    if j < len(order) - 1:
        cost += overlap[order[j]][order[j+1]]
    # Internal
    for k in range(i+1, j):
        cost += overlap[order[k]][order[k+1]]
    return cost


def _local_cost_after_reverse(overlap, order, i, j):
    """Compute cost after reversing segment i+1..j."""
    cost = 0
    if i > 0:
        cost += overlap[order[i-1]][order[j]]  # edge (i-1, i) unchanged, but (i, i+1) becomes (i, j)
    cost += overlap[order[i]][order[j]]  # new edge (i, j)
    if j < len(order) - 1:
        cost += overlap[order[i+1]][order[j+1]]  # new edge (i+1, j+1)
    # Internal edges reversed
    for k in range(i+1, j):
        cost += overlap[order[k+1]][order[k]]
    return cost


# EVOLVE-BLOCK-END
