import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1"
    
    # We are constructing S_n from S_{n-1} permutations
    # The permutations are of length n-1
    # For n=7, perms are of length 6
    
    # Generate all permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    m = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # L[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
    # Overlap is used to determine how much we can skip when concatenating
    # Block i: perms[i] + [n] + perms[i]
    # Block j: perms[j] + [n] + perms[j]
    # Overlap between Block i and Block j is L(perms[i], perms[j])
    # Note: The suffix of Block i is perms[i]. The prefix of Block j is perms[j].
    
    overlaps = [[0] * m for _ in range(m)]
    for i, p_i in enumerate(perms):
        for j, p_j in enumerate(perms):
            # Find max k such that p_i[-k:] == p_j[:k]
            # k can be 0 to n-1
            # Since p_i and p_j are permutations, if k > 1, it's a strong constraint
            max_k = 0
            # Check from largest possible k down
            # Max possible is n-1
            for k in range(n - 1, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    max_k = k
                    break
            overlaps[i][j] = max_k
    
    # Initialize ordering
    # Use the standard recursive order if possible, or just lexicographic
    # Lexicographic is simple.
    order = list(range(m))
    
    # Calculate initial score (sum of overlaps)
    def calc_score(ord):
        total = 0
        for k in range(len(ord) - 1):
            total += overlaps[ord[k]][ord[k+1]]
        return total
    
    best_order = order[:]
    best_score = calc_score(order)
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 130.0 # Leave 20s for construction and safety
    
    # Search: 2-opt and Or-opt (insertion)
    # We want to MAXIMIZE score, so we accept moves that INCREASE score.
    
    # Helper to get delta for swapping i and i+1
    def delta_swap(ord, i):
        # Remove i, i+1
        a = ord[i-1] if i > 0 else None
        b = ord[i]
        c = ord[i+1]
        d = ord[i+2] if i+2 < len(ord) else None
        
        old = 0
        if a is not None: old += overlaps[a][b]
        old += overlaps[b][c]
        if d is not None: old += overlaps[c][d]
        
        new = 0
        if a is not None: new += overlaps[a][c]
        new += overlaps[c][b]
        if d is not None: new += overlaps[b][d]
        
        return new - old

    # Helper to get delta for moving item at i to position j
    def delta_move(ord, i, j):
        # Move ord[i] to ord[j]
        # Assume i < j
        item = ord[i]
        prev = ord[i-1] if i > 0 else None
        next_ = ord[i+1]
        prev_j = ord[j-1]
        next_j = ord[j]
        
        old = 0
        if prev is not None: old += overlaps[prev][item]
        old += overlaps[item][next_]
        old += overlaps[prev_j][next_j]
        
        new = 0
        if prev is not None: new += overlaps[prev][next_]
        new += overlaps[prev_j][item]
        new += overlaps[item][next_j]
        
        return new - old

    # 2-opt loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(order) - 1):
            if time.time() > deadline: break
            d = delta_swap(order, i)
            if d > 0:
                order[i], order[i+1] = order[i+1], order[i]
                best_score += d
                best_order = order[:]
                improved = True
    
    # Or-opt (Insertion) loop
    # Try moving each element to every other position
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(order)):
            if time.time() > deadline: break
            item = order[i]
            for j in range(i + 1, len(order)):
                # Check if moving i to j improves
                # We need a more robust delta calc for arbitrary i, j
                # For simplicity, we might just do local moves in the first pass
                # Let's implement a generic move delta
                pass # Skip complex move for now, 2-opt is often sufficient for TSP-like
    
    # Add a simple random restart / SA if time permits
    # Given the structure, 2-opt on the initial standard order might be enough.
    # Let's try to run a simple SA for the remaining time if 2-opt stalls.
    
    current_score = calc_score(order)
    if current_score > best_score:
        best_score = current_score
        best_order = order[:]
        
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    while time.time() < deadline - 5.0:
        # Pick random i
        i = random.randint(0, len(order) - 2)
        # Swap i, i+1
        d = delta_swap(order, i)
        if d > 0 or random.random() < pow(2.71828, d / T):
            order[i], order[i+1] = order[i+1], order[i]
            current_score += d
            if current_score > best_score:
                best_score = current_score
                best_order = order[:]
        T *= cooling
        
    # Construct the word
    # Start with first block
    word = list(perms[best_order[0]]) + [n] + list(perms[best_order[0]])
    
    for k in range(1, len(best_order)):
        prev_idx = best_order[k-1]
        curr_idx = best_order[k]
        prev_perm = perms[prev_idx]
        curr_perm = perms[curr_idx]
        
        # Overlap
        ov = overlaps[prev_idx][curr_idx]
        
        # The suffix of the current word is prev_perm (plus potentially more if we didn't overlap fully? No, we always overlap max)
        # Actually, the word ends with prev_perm.
        # We append curr_perm[ov:] + [n] + curr_perm
        # Wait, the block is curr_perm + [n] + curr_perm.
        # We overlap the suffix of the previous block (which is prev_perm) with the prefix of the current block (curr_perm).
        # So we append curr_perm[ov:] + [n] + curr_perm.
        
        word += curr_perm[ov:]
        word.append(n)
        word += curr_perm
        
    return "".join(map(str, word))
