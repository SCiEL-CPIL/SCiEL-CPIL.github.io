import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    total = q ** n
    
    # 1. Precompute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1: neighbors are the word itself + 2*6 = 13 words
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        # Iterate through positions
        temp = idx
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)
    
    # 2. Build Distance-2 Graph for Independent Set Initialization
    # Two words are connected if their Hamming distance is exactly 2.
    # A maximal independent set in this graph gives a set of words where no two are at distance 2.
    # This is a strong structural property for covering codes.
    dist2_adj = [set() for _ in range(total)]
    
    # Optimization: Only build the graph if we have time. 
    # For 729 nodes, this is fast.
    for idx in range(total):
        # To avoid duplicates, only consider pairs (i, j) with i < j? 
        # Or just build full adjacency. 729 * (6*5/2) * (2*2) is small.
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                # Change digit i and j
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
                            dist2_adj[new_idx].add(idx)
    
    rng = random.Random(42)
    
    # 3. Greedy Independent Set Initialization
    # We want a maximal independent set in the distance-2 graph.
    # Strategy: Randomly shuffle vertices, then greedily pick a vertex if it doesn't conflict with already picked vertices.
    # To get a *good* (smaller) independent set, we can use a "minimum degree" heuristic on the remaining graph, 
    # but simple random greedy is often robust and fast. 
    # Let's try a weighted greedy: pick vertex with fewest neighbors in the remaining candidate set?
    # Actually, for covering codes, we want the independent set to also be a *cover*.
    # A maximal independent set in the distance-2 graph is NOT necessarily a cover.
    # We will generate the independent set, then repair it to be a cover.
    
    candidates = list(range(total))
    rng.shuffle(candidates)
    
    code_set = set()
    in_set = [False] * total
    
    # Greedy Independent Set
    for v in candidates:
        if in_set[v]:
            continue
        # Check if v conflicts with any node in code_set
        # Conflict if u in code_set and u is in dist2_adj[v]
        is_conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                is_conflict = True
                break
        if not is_conflict:
            code_set.add(v)
            in_set[v] = True
    
    # Convert to list for easier manipulation in SA
    code = list(code_set)
    
    # 4. Repair to Cover
    # The independent set might not cover the whole space.
    # We need to add words until it's a valid cover.
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    # Find uncovered words
    uncovered = [i for i in range(total) if counts[i] == 0]
    
    # Greedily add words to cover uncovered space
    # We prioritize words that cover the most uncovered nodes.
    # Since we want to maintain the "independent set" property loosely, we just add best covering words.
    # Note: Adding words might violate the distance-2 independence, but coverage is a hard constraint.
    while uncovered and time.time() < deadline - 5.0:
        best_c = -1
        best_gain = 0
        # Sample some candidates to speed up? Or iterate all? 729 is small.
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1:
            break
            
        code_set.add(best_c)
        code.append(best_c)
        for u in neigh[best_c]:
            counts[u] += 1
            if counts[u] == 1:
                # Remove from uncovered list? 
                # It's easier to just recalculate uncovered or keep a set.
                pass
        
        # Recalculate uncovered for next iteration (cheap for 729)
        uncovered = [i for i in range(total) if counts[i] == 0]
        
    # If still uncovered (shouldn't happen with greedy), force add
    if uncovered:
        for u in uncovered:
            # Add a random neighbor? Or just add u itself?
            # Adding u itself is safe.
            if u not in code_set:
                code_set.add(u)
                code.append(u)
                for v in neigh[u]:
                    counts[v] += 1
    
    best_code = code[:]
    best_size = len(code)
    
    # 5. Simulated Annealing
    # Moves:
    # 1. Remove: If a codeword is redundant (all its neighbors are covered by others), remove it.
    # 2. Swap: Replace a codeword with another if the new one covers all unique neighbors of the old one.
    
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c):
        # Check if removing c leaves any neighbor uncovered by new_c
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    # Perform removal
                    code.pop(idx)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
        elif move_type < 0.9:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Perform swap
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

        else:
            # Try Add (Perturbation) - allows escaping local minima by temporarily increasing size
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                # Accept with probability exp(-delta_E / T)
                # Here delta_E is effectively +1 (increase in size)
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate

    # Use the best code found
    final_code = best_code
    
    # Convert back to list of lists
    result = []
    for idx in final_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result

# EVOLVE-BLOCK-END
