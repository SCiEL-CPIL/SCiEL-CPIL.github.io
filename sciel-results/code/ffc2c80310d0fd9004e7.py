import sys
import time
import random
from itertools import permutations

_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def construct(n):
    global _start_time
    _start_time = time.time()
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    # For n < 7, use standard recursive construction
    if n < 7:
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
    
    # Special handling for n=7
    # Get all permutations of 1..6
    prev_syms = list(range(1, n))
    perms = list(permutations(prev_syms))
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    overlap = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        if _elapsed() > _DEADLINE - 20:
            break
        pi = perms[i]
        for j in range(num_perms):
            if i == j:
                continue
            pj = perms[j]
            max_k = min(len(pi), len(pj))
            for k in range(max_k, -1, -1):
                if k == 0 or pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Greedy construction to get initial good solutions
    def greedy_order(start_idx):
        visited = [False] * num_perms
        order = [start_idx]
        visited[start_idx] = True
        curr = start_idx
        total_len = len(perms[start_idx]) * 2 + 1  # block length = 2*(n-1) + 1 = 13
        
        for _ in range(num_perms - 1):
            if _elapsed() > _DEADLINE - 10:
                return None, float('inf')
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap[curr][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            if best_next == -1:
                break
            order.append(best_next)
            visited[best_next] = True
            total_len += 13 - best_ov
            curr = best_next
        return order, total_len
    
    # Try multiple greedy starts
    best_order = None
    best_len = float('inf')
    
    starts_to_try = list(range(min(20, num_perms)))
    random.seed(42)
    random.shuffle(starts_to_try)
    
    for s in starts_to_try:
        if _elapsed() > _DEADLINE - 30:
            break
        order, length = greedy_order(s)
        if order is not None and length < best_len:
            best_len = length
            best_order = order
    
    if best_order is None:
        best_order = list(range(num_perms))
    
    # Calculate overlap sum for an order
    def calc_overlap_sum(order):
        total = 0
        for i in range(len(order) - 1):
            total += overlap[order[i]][order[i + 1]]
        return total
    
    # Simulated annealing to improve the ordering
    current_order = best_order[:]
    current_score = calc_overlap_sum(current_order)
    best_sa_order = current_order[:]
    best_sa_score = current_score
    
    T = 50.0  # initial temperature
    cooling_rate = 0.9997
    
    while _elapsed() < _DEADLINE - 5:
        # Pick a random adjacent swap
        i = random.randint(0, num_perms - 2)
        j = i + 1
        
        # Calculate delta in score
        old_score = 0
        new_score = 0
        
        if i > 0:
            old_score += overlap[current_order[i-1]][current_order[i]]
            new_score += overlap[current_order[i-1]][current_order[j]]
        old_score += overlap[current_order[i]][current_order[j]]
        new_score += overlap[current_order[j]][current_order[i]]
        if j + 1 < num_perms:
            old_score += overlap[current_order[j]][current_order[j+1]]
            new_score += overlap[current_order[i]][current_order[j+1]]
        
        delta = new_score - old_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            # Accept the move
            current_order[i], current_order[j] = current_order[j], current_order[i]
            current_score += delta
            if current_score > best_sa_score:
                best_sa_score = current_score
                best_sa_order = current_order[:]
        
        T *= cooling_rate
        if T < 0.1:
            break
    
    # Use the best order found
    final_order = best_sa_order
    
    # Construct the word
    word = []
    for idx, p_idx in enumerate(final_order):
        p = perms[p_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            ov = 0
            max_ov = min(len(word), len(block))
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
    
    return "".join(map(str, word))
