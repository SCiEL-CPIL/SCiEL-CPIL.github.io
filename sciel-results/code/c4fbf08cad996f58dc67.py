"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    if n <= 6:
        return _classic_construct(n)
    
    # For n=7, use optimized ordering of the final step
    word = _classic_construct(n - 1)
    return _optimized_step(word, n)

def _classic_construct(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _optimized_step(word_str, n):
    word = [int(c) for c in word_str]
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Create blocks
    blocks = [list(p) + [m] + list(p) for p in perms]
    num_blocks = len(blocks)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that suffix_k(blocks[i]) == prefix_k(blocks[j])
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        b1 = blocks[i]
        for j in range(num_blocks):
            if i == j:
                continue
            b2 = blocks[j]
            max_k = min(len(b1), len(b2))
            # Since blocks are of form p + [m] + p where p is a permutation of 1..m-1
            # The overlap can be at most 2*(m-1) = 12 for m=7
            # But actually, suffix of b1 and prefix of b2
            # b1 ends with p1 (the permutation part), b2 starts with p2
            # So we need suffix of (p1 + [m] + p1) == prefix of (p2 + [m] + p2)
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if b1[-k:] == b2[:k]:
                    overlap[i][j] = k
                    break
    
    def calc_total_overlap(ord):
        total = 0
        for i in range(len(ord) - 1):
            total += overlap[ord[i]][ord[i+1]]
        return total
    
    def greedy_construct():
        # Start with a random block
        unvisited = set(range(num_blocks))
        order = []
        start = random.choice(list(unvisited))
        order.append(start)
        unvisited.remove(start)
        
        while unvisited:
            last = order[-1]
            # Find the unvisited block with maximum overlap
            best_next = None
            best_ov = -1
            for j in unvisited:
                ov = overlap[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            if best_next is None:
                break
            order.append(best_next)
            unvisited.remove(best_next)
        
        return order
    
    def two_opt(ord):
        improved = True
        while improved:
            improved = False
            for i in range(num_blocks - 1):
                for j in range(i + 1, num_blocks):
                    # Try reversing segment i..j
                    # Old edges affected: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    # Internal edges are reversed
                    
                    prev_i = ord[i-1] if i > 0 else None
                    next_j = ord[j+1] if j < num_blocks - 1 else None
                    
                    old_ov = 0
                    if prev_i is not None:
                        old_ov += overlap[prev_i][ord[i]]
                    if next_j is not None:
                        old_ov += overlap[ord[j]][next_j]
                    # Internal edges
                    for k in range(i, j):
                        old_ov += overlap[ord[k]][ord[k+1]]
                    
                    new_ov = 0
                    if prev_i is not None:
                        new_ov += overlap[prev_i][ord[j]]
                    if next_j is not None:
                        new_ov += overlap[ord[i]][next_j]
                    # Internal edges reversed
                    for k in range(i, j):
                        new_ov += overlap[ord[j-(k-i)]] [ord[j-(k-i)+1]]
                    
                    if new_ov > old_ov:
                        # Reverse segment i..j
                        ord[i:j+1] = ord[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return ord
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds for search
    
    # Multiple greedy restarts with 2-opt refinement
    best_order = None
    best_overlap = -1
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = greedy_construct()
        cur_ov = calc_total_overlap(order)
        if cur_ov > best_overlap:
            best_overlap = cur_ov
            best_order = order[:]
        
        # 2-opt refinement
        order = two_opt(order)
        cur_ov = calc_total_overlap(order)
        if cur_ov > best_overlap:
            best_overlap = cur_ov
            best_order = order[:]
    
    # Use best order found
    order = best_order
    
    # Construct final word
    new_word = list(blocks[order[0]])
    for k in range(1, len(order)):
        prev_idx = order[k-1]
        curr_idx = order[k]
        b_curr = blocks[curr_idx]
        ov = overlap[prev_idx][curr_idx]
        new_word += b_curr[ov:]
        
    return "".join(map(str, new_word))

# EVOLVE-BLOCK-END
