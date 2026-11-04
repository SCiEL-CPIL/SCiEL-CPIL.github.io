import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over base-B binary expansion full sets and local improvements
    # U consists of all numbers with digits 0 or 1 in base B, up to n digits.
    
    for B in range(2, 20):
        for n in range(2, 12):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = list(all_vals)
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
            
            # Local search: try removing elements
            # If n is small enough, we can do multiple passes
            if n <= 8: # 2^8 = 256 elements, 256^2 = 65k ops per score calc. 
                       # 256 removals * 65k = 16M ops per pass. 
                       # Python might be slow. Let's limit to n <= 6 for local search.
                if n <= 6:
                    current_U = list(all_vals)
                    current_score = calc_score(current_U)
                    improved = True
                    max_iters = 5
                    iter_count = 0
                    while improved and iter_count < max_iters:
                        improved = False
                        iter_count += 1
                        for i in range(len(current_U)):
                            # Try removing current_U[i]
                            new_U = current_U[:i] + current_U[i+1:]
                            if not new_U or 0 not in new_U:
                                continue
                            new_score = calc_score(new_U)
                            if new_score > current_score:
                                current_U = new_U
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best = list(current_U)
                
                # Also try adding elements from a slightly larger space?
                # Or start with a subset and add?
                # Let's stick to removal for now as it reduces max_u potentially? No, max_u stays same unless max element removed.
                
            # Try a random subset approach for larger n?
            # If n is 7 or 8, random subsets might find better structures.
            if 6 < n <= 8:
                # Try a few random subsets of the full set
                for _ in range(10):
                    size = random.randint(len(all_vals)//2, len(all_vals))
                    U_rand = random.sample(all_vals, size)
                    if 0 not in U_rand:
                        U_rand.append(0)
                    U_rand = list(set(U_rand))
                    s_rand = calc_score(U_rand)
                    if s_rand > best_score:
                        best_score = s_rand
                        best = U_rand
                    # Local search on this random subset
                    if len(U_rand) <= 50:
                        current_U = U_rand
                        current_score = s_rand
                        improved = True
                        max_iters = 3
                        iter_count = 0
                        while improved and iter_count < max_iters:
                            improved = False
                            iter_count += 1
                            for i in range(len(current_U)):
                                new_U = current_U[:i] + current_U[i+1:]
                                if not new_U or 0 not in new_U:
                                    continue
                                new_score = calc_score(new_U)
                                if new_score > current_score:
                                    current_U = new_U
                                    current_score = new_score
                                    improved = True
                                    if current_score > best_score:
                                        best_score = current_score
                                        best = list(current_U)
                                # Try adding an element from all_vals not in current_U
                                # This is expensive, skip for now

    return best
