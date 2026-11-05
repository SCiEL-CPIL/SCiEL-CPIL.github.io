import time
import math
from itertools import permutations

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses a local search heuristic to improve upon the standard recursive construction.
    """
    start_time = time.time()
    deadline = 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
        
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    
    # Initial ordering: Standard lexicographic (or recursive)
    # The current code uses the order of appearance in the recursive build.
    # For simplicity, we start with lexicographic and try to improve.
    # Actually, the recursive order is often better.
    # Let's generate the recursive order.
    
    def get_recursive_order(n):
        if n == 1:
            return [(1,)]
        prev_order = get_recursive_order(n - 1)
        current_order = []
        for p in prev_order:
            current_order.append(p + (n,))
            current_order.append(p)
            # Wait, the standard construction order is specific.
            # The code extracts perms from the word.
            # Let's just use lexicographic for now and rely on local search.
        return current_order

    # To match the current code's behavior closely, we should use the same initial order.
    # The current code builds the word iteratively.
    # Let's just use lexicographic order for the initial state of the search.
    # The local search should find a good order.
    
    initial_order = perms
    
    def get_overlap(p1, p2):
        # p1, p2 are tuples
        # Max overlap of suffix of p1 and prefix of p2
        max_ov = 0
        for k in range(min(len(p1), len(p2)), -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    def calc_total_overlap(order):
        total = 0
        for i in range(len(order) - 1):
            total += get_overlap(order[i], order[i + 1])
        return total

    # Local Search
    current_order = initial_order[:]
    current_overlap = calc_total_overlap(current_order)
    best_order = current_order[:]
    best_overlap = current_overlap
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Random move: swap two adjacent elements or reverse a small segment
        i = int(time.time() * 1000000) % (len(current_order) - 1)
        # Or use random
        import random
        i = random.randint(0, len(current_order) - 2)
        
        # Try swapping i and i+1
        new_order = current_order[:]
        new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
        
        # Calculate change in overlap
        # Old: O(i-1, i) + O(i, i+1) + O(i+1, i+2)
        # New: O(i-1, i+1) + O(i+1, i) + O(i, i+2)
        
        ov_old = 0
        ov_new = 0
        
        if i > 0:
            ov_old += get_overlap(current_order[i-1], current_order[i])
            ov_new += get_overlap(new_order[i-1], new_order[i])
            
        ov_old += get_overlap(current_order[i], current_order[i+1])
        ov_new += get_overlap(new_order[i], new_order[i+1])
        
        if i + 1 < len(current_order) - 1:
            ov_old += get_overlap(current_order[i+1], current_order[i+2])
            ov_new += get_overlap(new_order[i+1], new_order[i+2])
            
        delta = ov_new - ov_old
        
        if delta > 0:
            current_order = new_order
            current_overlap += delta
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_order = current_order[:]
        else:
            # Accept with probability
            r = random.random()
            if math.exp(delta / T) > r:
                current_order = new_order
                current_overlap += delta
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Reconstruct word from best_order
    word = []
    for p in best_order:
        block = list(p) + [n] + list(p)
        max_overlap = min(len(word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if word[len(word) - o:] == block[:o]:
                overlap = o
                break
        word.extend(block[overlap:])
        
    return "".join(map(str, word))
