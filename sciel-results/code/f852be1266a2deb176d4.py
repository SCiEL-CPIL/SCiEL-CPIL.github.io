import random
import time
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the standard superpermutation for n-1 to get the initial order of permutations
    # This is the "seed" order.
    def standard_superperm_order(k):
        if k == 1:
            return [[1]]
        prev_perms = standard_superperm_order(k-1)
        # The standard construction order is the order of appearance in the superpermutation string
        # We need to simulate the construction to get the order of windows.
        
        # Actually, we can just generate the superpermutation string for k-1 and extract the windows.
        word = [1]
        for m in range(2, k + 1):
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
        return [tuple(w) for w in word[k-1:]] # This is wrong, we need the order of perms of size k-1 used to build k.
        
    # Let's just generate the order for n-1 directly using the recursive logic.
    def get_order(k):
        if k == 1:
            return [(1,)]
        prev_order = get_order(k-1)
        # The order of permutations of k-1 in the superpermutation of k-1 is prev_order.
        # When building k, we iterate through these.
        # So the order of "blocks" for k is determined by prev_order.
        # But we need the order of permutations of k-1 for the TSP of k.
        # The TSP nodes are permutations of k-1.
        # The initial order is the order they appear in S_{k-1}.
        # So for n=7, nodes are perms of 6. Initial order is order in S_6.
        return prev_order

    # Get the list of all permutations of 1..n-1
    import itertools
    perms = list(itertools.permutations(range(1, n)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Get the initial order from the standard construction
    # We need the order of permutations of n-1 as they appear in S_{n-1}.
    # We can generate S_{n-1} and extract them.
    
    def generate_superperm_string(k):
        word = [1]
        for m in range(2, k + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    if n == 2:
        return "121"
        
    # Get initial order of permutations of n-1
    s_prev = generate_superperm_string(n-1)
    initial_order = []
    seen = set()
    for i in range(len(s_prev) - (n-1) + 1):
        w = tuple(s_prev[i:i+n-1])
        if w not in seen:
            seen.add(w)
            initial_order.append(w)
            
    # Map to indices
    initial_idx_order = [perm_to_idx[p] for p in initial_order]
    
    # Precompute overlap matrix
    # L[i][j] = overlap between perm[i] and perm[j]
    N = len(perms)
    L = [[0]*N for _ in range(N)]
    for i in range(N):
        p_i = perms[i]
        for j in range(N):
            p_j = perms[j]
            max_k = min(len(p_i), len(p_j))
            for k in range(max_k, 0, -1):
                if p_i[len(p_i)-k:] == p_j[:k]:
                    L[i][j] = k
                    break
    
    # Simulated Annealing
    current_order = initial_idx_order[:]
    current_cost = -sum(L[current_order[i]][current_order[i+1]] for i in range(len(current_order)-1))
    best_order = current_order[:]
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 110 # seconds
    T = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        # Move: Swap two random positions
        i, j = random.sample(range(len(current_order)), 2)
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Calculate delta cost
        # Only neighbors of i and j are affected
        # Original edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # New edges: (i-1, j), (j, i+1), (j-1, i), (i, j+1) ... depends on indices
        
        def get_cost(order):
            return -sum(L[order[k]][order[k+1]] for k in range(len(order)-1))
            
        # For efficiency, calculate delta
        old_cost_segment = 0
        new_cost_segment = 0
        
        # Indices involved: i-1, i, i+1, j-1, j, j+1
        # We need to be careful with boundaries
        
        # Simpler: Just recalculate cost for the small affected region
        # But N is 720, sum is 719 terms. Recalculating full cost is O(N).
        # 110s / O(720) is ~100k iterations. Might be slow in Python.
        # Let's try to optimize delta calculation.
        
        # Edges to remove:
        edges_to_remove = []
        if i > 0: edges_to_remove.append((i-1, i))
        if i < N-1: edges_to_remove.append((i, i+1))
        if j > 0: edges_to_remove.append((j-1, j))
        if j < N-1: edges_to_remove.append((j, j+1))
        
        # Edges to add:
        edges_to_add = []
        if i > 0: edges_to_add.append((i-1, j))
        if i < N-1: edges_to_add.append((j, i+1)) # Wait, i takes j's place? No, swap.
        # Swap i and j.
        # Pos i now has old j. Pos j now has old i.
        
        # Let's just use a full recalculation if N is small enough.
        # 720 * 10000 iterations = 7.2 million ops. Python can do ~10M simple ops/sec.
        # So 10000 iterations might take 1 sec. 110s -> 100k iterations.
        # This is feasible.
        
        new_cost = get_cost(current_order)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = current_order[:]
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        T *= cooling
        if T < 0.001:
            T = 0.001

    # Construct the final word using best_order
    # best_order contains indices of permutations of 1..n-1
    # We need to build the superpermutation of n.
    
    word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        # Find max overlap with current word
        max_overlap = 0
        # The overlap is between the end of `word` and start of `block`
        # Max possible overlap is len(block)
        # But we know it's at most len(p) = n-1
        for o in range(min(len(word), len(block)), -1, -1):
            if word[len(word)-o:] == block[:o]:
                max_overlap = o
                break
        word += block[max_overlap:]
        
    return "".join(map(str, word))
