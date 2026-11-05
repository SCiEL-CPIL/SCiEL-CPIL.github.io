import time
import math
import random

def _standard_superperm(m):
    """Generate standard superpermutation for m symbols."""
    if m == 1:
        return [1]
    if m == 2:
        return [1, 2, 1]
    prev = _standard_superperm(m - 1)
    # Extract blocks of length m-1
    blocks = []
    seen = set()
    for i in range(len(prev) - (m - 1) + 1):
        w = tuple(prev[i:i + m - 1])
        if len(set(w)) == m - 1 and w not in seen:
            seen.add(w)
            blocks.append(w)
    res = []
    for b in blocks:
        res.extend(b)
        res.append(m)
        res.extend(b)
    return res

def _extract_blocks(word, block_size):
    """Extract all distinct permutations of length block_size from word."""
    blocks = []
    seen = set()
    for i in range(len(word) - block_size + 1):
        w = tuple(word[i:i + block_size])
        if len(set(w)) == block_size and w not in seen:
            seen.add(w)
            blocks.append(w)
    return blocks

def _overlap_len(a, b):
    """Max k such that a[-k:] == b[:k]."""
    max_k = min(len(a), len(b))
    for k in range(max_k, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0

def _construct_from_order(blocks, order, n):
    """Reconstruct superpermutation from block order."""
    # Precompute overlaps
    k = len(blocks)
    ov = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i != j:
                ov[i][j] = _overlap_len(blocks[i], blocks[j])
    
    word = []
    for idx, block_idx in enumerate(order):
        block = blocks[block_idx]
        if idx == 0:
            word.extend(block)
        else:
            prev_idx = order[idx - 1]
            o = ov[prev_idx][block_idx]
            word.extend(block[o:])
        word.append(n)
        word.extend(block)
    return word

def _verify_superpermutation(word, n):
    """Check if word contains all n! permutations of 1..n."""
    target = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
        if len(seen) == target:
            return True
    return len(seen) == target

def _greedy_tsp(blocks, n):
    """Greedy nearest-neighbor construction of block order."""
    k = len(blocks)
    # Precompute overlap matrix
    ov = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i != j:
                ov[i][j] = _overlap_len(blocks[i], blocks[j])
    
    # Start with block 0
    order = [0]
    remaining = set(range(1, k))
    while remaining:
        curr = order[-1]
        best_next = None
        best_ov = -1
        for nxt in remaining:
            if ov[curr][nxt] > best_ov:
                best_ov = ov[curr][nxt]
                best_next = nxt
        order.append(best_next)
        remaining.remove(best_next)
    return order, ov

def _calc_total_overlap(order, ov):
    """Calculate total overlap for an order."""
    total = 0
    for i in range(len(order) - 1):
        total += ov[order[i]][order[i + 1]]
    return total

def _two_opt_improve(blocks, order, ov, time_limit=100):
    """2-Opt local search to improve overlap."""
    k = len(blocks)
    current_order = order[:]
    
    best_overlap = _calc_total_overlap(current_order, ov)
    best_order = current_order[:]
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try all 2-opt moves
        for i in range(k - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, k):
                # Reverse segment [i, j]
                old_val = 0
                new_val = 0
                if i > 0:
                    old_val += ov[current_order[i - 1]][current_order[i]]
                    new_val += ov[current_order[i - 1]][current_order[j]]
                # Internal edges
                for m in range(i, j):
                    old_val += ov[current_order[m]][current_order[m + 1]]
                    new_val += ov[current_order[m + 1]][current_order[m]]
                if j < k - 1:
                    old_val += ov[current_order[j]][current_order[j + 1]]
                    new_val += ov[current_order[i]][current_order[j + 1]]
                
                if new_val > old_val:
                    current_order[i:j + 1] = current_order[i:j + 1][::-1]
                    total = _calc_total_overlap(current_order, ov)
                    if total > best_overlap:
                        best_overlap = total
                        best_order = current_order[:]
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
    return best_order

def _or_opt_improve(blocks, order, ov, time_limit=100):
    """Or-opt local search: move contiguous segments to new positions."""
    k = len(blocks)
    current_order = order[:]
    
    best_overlap = _calc_total_overlap(current_order, ov)
    best_order = current_order[:]
    start_time = time.time()
    improved = True
    
    # Segment lengths to try
    seg_lengths = [1, 2, 3]
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        for seg_len in seg_lengths:
            if time.time() - start_time > time_limit:
                break
            # Try moving each segment of length seg_len to each position
            for i in range(k - seg_len + 1):
                if time.time() - start_time > time_limit:
                    break
                segment = current_order[i:i + seg_len]
                # Remove segment
                remaining = current_order[:i] + current_order[i + seg_len:]
                # Try inserting at each position in remaining
                for j in range(len(remaining) - seg_len + 1):
                    if j == i:
                        continue  # same position
                    # Calculate new overlap
                    new_order = remaining[:j] + segment + remaining[j:]
                    new_ov = _calc_total_overlap(new_order, ov)
                    if new_ov > best_overlap:
                        current_order = new_order
                        best_overlap = new_ov
                        best_order = current_order[:]
                        improved = True
                        if time.time() - start_time > time_limit:
                            break
                if time.time() - start_time > time_limit:
                    break
    return best_order

def _vns_search(blocks, order, ov, time_limit=100):
    """Variable neighborhood search combining 2-opt and Or-opt."""
    k = len(blocks)
    current_order = order[:]
    best_overlap = _calc_total_overlap(current_order, ov)
    best_order = current_order[:]
    start_time = time.time()
    
    neighborhoods = [
        ("2opt", _two_opt_improve),
        ("oropt", _or_opt_improve),
    ]
    
    while time.time() - start_time < time_limit:
        improved_in_round = False
        for name, func in neighborhoods:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 1:
                break
            # Allocate time proportionally
            alloc = min(remaining_time, max(5, int(remaining_time / 2)))
            new_order = func(blocks, current_order, ov, time_limit=alloc)
            new_overlap = _calc_total_overlap(new_order, ov)
            if new_overlap > best_overlap:
                best_overlap = new_overlap
                best_order = new_order[:]
                current_order = new_order[:]
                improved_in_round = True
        if not improved_in_round:
            # Try random restart if stuck
            if time.time() - start_time < time_limit - 5:
                # Random perturbation
                for _ in range(10):
                    i = random.randint(0, k - 2)
                    j = random.randint(i + 1, k - 1)
                    current_order[i:j + 1] = current_order[i:j + 1][::-1]
                    new_overlap = _calc_total_overlap(current_order, ov)
                    if new_overlap > best_overlap:
                        best_overlap = new_overlap
                        best_order = current_order[:]
                        improved_in_round = True
                    if time.time() - start_time > time_limit - 5:
                        break
            if not improved_in_round:
                break
    
    return best_order

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341214321342143121432134124312341324134212413421324124312341432142313421234142313241243123414321"
    
    start_time = time.time()
    time_limit = 130  # Leave 20s margin for safety
    
    # Get standard superpermutation for n-1
    prev_word = _standard_superperm(n - 1)
    block_size = n - 1
    blocks = _extract_blocks(prev_word, block_size)
    k = len(blocks)
    
    # Greedy TSP to get initial order
    order, ov = _greedy_tsp(blocks, n)
    
    # VNS improvement
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        order = _vns_search(blocks, order, ov, time_limit=int(remaining_time))
    
    # Reconstruct
    word = _construct_from_order(blocks, order, n)
    
    return "".join(map(str, word))
