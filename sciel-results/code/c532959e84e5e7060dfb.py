import time
from math import log, floor
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Strategy: Base-b expansion with variable digit sets.
    # U = { sum_{k=0}^{m-1} d_k b^k | d_k in D_k }
    # We search for bases b and digit sets D_k that maximize the score.
    # This structure allows for controlled "gaps" in the sumset and denser difference sets.
    
    # We will perform a multi-start local search over the space of (base, digit_sets).
    # To keep it efficient, we limit the number of "levels" (powers of b) to a small number (e.g., 3-6).
    
    # Helper to generate a set from base and digit lists
    def gen_set(b, digit_lists):
        U = [0]
        # Use iterative construction to avoid huge intermediate sets if b is large
        # But for small b and few levels, direct expansion is fine.
        
        # Start with {0}
        current = {0}
        for d_list in digit_lists:
            next_current = set()
            for x in current:
                for d in d_list:
                    next_current.add(x + d * (b ** len(current))) # Wait, this logic is wrong for standard positional notation.
            # Let's use standard positional notation: U = { sum d_k b^k }
            # It's easier to just generate all combinations.
            pass
        
        # Simpler generation:
        U = set()
        n_levels = len(digit_lists)
        # If n_levels is small (<= 6) and digit sets are small (<= 5), total size <= 5^6 = 15625, which is manageable.
        
        # Use product
        from itertools import product
        for combo in product(*[range(len(dl)) for dl in digit_lists]):
            val = 0
            for k, idx in enumerate(combo):
                val += digit_lists[k][idx] * (b ** k)
            U.add(val)
        
        U_list = sorted(list(U))
        if 0 not in U_list:
            U_list.append(0)
        return U_list

    # We need a fast enough generation. The above product approach is O(|U| * m).
    # For m=5, |U|~1000, it's fast.
    
    # Local search on the digit sets and base.
    # State: (b, [D_0, D_1, ..., D_{m-1}])
    # Perturbation:
    # 1. Change b slightly (b-1, b+1)
    # 2. Add/remove/modify digits in D_k
    
    # We will try a few random initial configurations and then refine.
    
    random.seed(42)
    
    # Candidate bases
    bases_to_try = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]
    
    # For each base, try a few random digit set configurations
    for b in bases_to_try:
        if time.time() > deadline - 30:
            break
            
        # Try different numbers of levels
        for m_levels in [3, 4, 5]:
            if time.time() > deadline - 10:
                break
            
            # Initialize random digit sets
            # D_k should contain 0 to ensure 0 is in U (though we add 0 explicitly, it helps structure)
            # Let's force 0 in D_0.
            
            # Run several random restarts for this (b, m_levels)
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Generate random digit sets
                # D_0 must contain 0
                # Other D_k can be random subsets of {0, 1, ..., b-1} or small sets like {0, k}
                
                digit_lists = []
                for k in range(m_levels):
                    if k == 0:
                        # Small set for first level to control density
                        size = random.randint(1, 3)
                        ds = random.sample(range(b), size)
                        if 0 not in ds:
                            ds[0] = 0
                        ds = sorted(set(ds))
                    else:
                        # For higher levels, often sparse sets like {0, 1} or {0, b-1} work well for sumset gaps
                        # Let's try a mix
                        if random.random() < 0.5:
                            ds = [0, 1]
                        elif random.random() < 0.5:
                            ds = [0, b-1]
                        else:
                            size = random.randint(1, 4)
                            ds = random.sample(range(b), size)
                            if 0 not in ds:
                                ds[0] = 0
                            ds = sorted(set(ds))
                    digit_lists.append(ds)
                
                U = gen_set(b, digit_lists)
                if len(U) < 2:
                    continue
                    
                curr_score = score_set(U)
                
                # Local search on this configuration
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    n = len(U)
                    
                    # Try perturbing digit sets
                    # 1. Change a digit in some D_k
                    for k in range(m_levels):
                        if time.time() > deadline - 2:
                            break
                        old_ds = digit_lists[k]
                        
                        # Try adding a new digit
                        for d in range(b):
                            if d in old_ds:
                                continue
                            new_ds = sorted(old_ds + [d])
                            new_digit_lists = digit_lists.copy()
                            new_digit_lists[k] = new_ds
                            new_U = gen_set(b, new_digit_lists)
                            if len(new_U) < 2:
                                continue
                            new_score = score_set(new_U)
                            if new_score > curr_score:
                                curr_score = new_score
                                U = new_U
                                digit_lists = new_digit_lists
                                improved = True
                                break
                        if improved:
                            break
                        
                        # Try removing a digit (except if it leaves set empty or 0 is removed and k==0? 0 is always in U via construction if D_0 has 0, but if we remove 0 from D_0, 0 might still be in U if other combos make 0? No, only all zeros make 0. So if we remove 0 from D_0, 0 is lost unless we add it back explicitly. The gen_set adds 0 explicitly if missing? No, my gen_set adds 0 if missing. So it's safe.)
                        for d in old_ds:
                            if len(old_ds) <= 1:
                                continue
                            new_ds = [x for x in old_ds if x != d]
                            new_digit_lists = digit_lists.copy()
                            new_digit_lists[k] = new_ds
                            new_U = gen_set(b, new_digit_lists)
                            if len(new_U) < 2:
                                continue
                            new_score = score_set(new_U)
                            if new_score > curr_score:
                                curr_score = new_score
                                U = new_U
                                digit_lists = new_digit_lists
                                improved = True
                                break
                        if improved:
                            break
                        
                        # Try swapping a digit
                        # This is more complex, maybe skip for now or just random replacement
                        if random.random() < 0.3:
                            d_old = random.choice(old_ds)
                            d_new = random.randint(0, b-1)
                            if d_new == d_old:
                                continue
                            new_ds = sorted(set([x for x in old_ds if x != d_old] + [d_new]))
                            new_digit_lists = digit_lists.copy()
                            new_digit_lists[k] = new_ds
                            new_U = gen_set(b, new_digit_lists)
                            if len(new_U) < 2:
                                continue
                            new_score = score_set(new_U)
                            if new_score > curr_score:
                                curr_score = new_score
                                U = new_U
                                digit_lists = new_digit_lists
                                improved = True
                                break
                        if improved:
                            break
                    
                    # 2. Change base?
                    # Changing base is a big jump. Maybe try b-1, b+1 occasionally.
                    if not improved and random.random() < 0.1:
                        for db in [-1, 1]:
                            new_b = b + db
                            if new_b < 2:
                                continue
                            # Regenerate with same digit structures? Digits must be < new_b.
                            # This is tricky. If b increases, digits are still valid. If b decreases, some digits might be invalid.
                            # Let's just restart with new base if we get stuck? No, that's expensive.
                            # Let's skip base change in local search to keep it simple and fast.
                            pass
                    
                    # 3. Add a level?
                    if not improved and m_levels < 6:
                        # Try adding a new level with small digit set
                        new_ds = [0, 1]
                        new_digit_lists = digit_lists + [new_ds]
                        new_U = gen_set(b, new_digit_lists)
                        if len(new_U) < 2:
                            continue
                        new_score = score_set(new_U)
                        if new_score > curr_score:
                            curr_score = new_score
                            U = new_U
                            digit_lists = new_digit_lists
                            m_levels += 1
                            improved = True
                    
                    # 4. Remove a level?
                    if not improved and m_levels > 3:
                        # Remove last level
                        new_digit_lists = digit_lists[:-1]
                        new_U = gen_set(b, new_digit_lists)
                        if len(new_U) < 2:
                            continue
                        new_score = score_set(new_U)
                        if new_score > curr_score:
                            curr_score = new_score
                            U = new_U
                            digit_lists = new_digit_lists
                            m_levels -= 1
                            improved = True

                if curr_score > best_score:
                    best_score = curr_score
                    best_set = U

    # Final polish: Element-wise local search on the best set found
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 2 and iteration < 50:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old_val = current[i]
                # Try small deltas
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 30 and time.time() < deadline - 2:
                max_val = max(current)
                # Try values near the gap or max
                # Sample some candidates
                for _ in range(20):
                    new_val = random.randint(1, max_val + 20)
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 2:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
