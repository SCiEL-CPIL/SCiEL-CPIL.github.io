import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    candidates = list(range(1, 512))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Precompute prime factors
    primes = []
    for i in range(2, 512):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Bitmask of prime factors for each number
    factor_masks = [0] * 512
    for a in range(1, 512):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    # Build adjacency as bitmasks for fast conflict checking
    # For each candidate, store a bitmask of all candidates that conflict with it
    n_cands = len(candidates)
    cand_idx = {a: i for i, a in enumerate(candidates)}
    
    # conflict_mask[i] is a bitmask where bit j is set if candidates[i] conflicts with candidates[j]
    conflict_mask = [0] * n_cands
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            if factor_masks[candidates[i]] & factor_masks[candidates[j]]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute weight arrays for fast access
    weight_arr = [weights[a] for a in candidates]
    
    def get_score(selected_mask):
        """Compute score from a bitmask of selected candidates."""
        s = 0.0
        for i in range(n_cands):
            if selected_mask & (1 << i):
                s += weight_arr[i]
        return s
    
    def is_valid(selected_mask):
        """Check if selection is pairwise coprime."""
        for i in range(n_cands):
            if selected_mask & (1 << i):
                # Check no conflict with other selected
                conflicts = conflict_mask[i] & selected_mask
                # Remove self
                conflicts &= ~(1 << i)
                if conflicts:
                    return False
        return True
    
    def greedy_select(strategy='weight'):
        """Greedy selection with different strategies."""
        available = (1 << n_cands) - 1
        selected = 0
        score = 0.0
        
        while available:
            # Find best candidate based on strategy
            best_i = -1
            best_val = -1
            
            for i in range(n_cands):
                if not (available & (1 << i)):
                    continue
                
                if strategy == 'weight':
                    val = weight_arr[i]
                elif strategy == 'weight_degree':
                    degree = bin(conflict_mask[i] & available).count('1')
                    val = weight_arr[i] / (degree + 1)
                elif strategy == 'weight_conflict':
                    # Prefer nodes that conflict with fewer high-weight nodes
                    conflict_weight = 0.0
                    cm = conflict_mask[i] & available
                    for j in range(n_cands):
                        if cm & (1 << j):
                            conflict_weight += weight_arr[j]
                    val = weight_arr[i] / (1 + conflict_weight)
                else:
                    val = weight_arr[i]
                
                if val > best_val:
                    best_val = val
                    best_i = i
            
            if best_i == -1:
                break
            
            selected |= (1 << best_i)
            score += weight_arr[best_i]
            available &= ~(1 << best_i)
            # Remove all conflicts
            available &= ~conflict_mask[best_i]
        
        return score, selected
    
    # Try multiple greedy strategies
    best_score = 0.0
    best_mask = 0
    
    for strat in ['weight', 'weight_degree', 'weight_conflict']:
        s, m = greedy_select(strat)
        if s > best_score:
            best_score = s
            best_mask = m
    
    # Local search: try swapping one selected with one unselected
    def local_search(initial_mask, max_iters=5000):
        """Improve solution via local search (add/remove/swap)."""
        current_mask = initial_mask
        current_score = get_score(current_mask)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding a node
            for i in range(n_cands):
                if current_mask & (1 << i):
                    continue
                # Check if i conflicts with any selected
                if conflict_mask[i] & current_mask:
                    continue
                new_mask = current_mask | (1 << i)
                new_score = current_score + weight_arr[i]
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            
            # Try removing a node (rarely helps, but for completeness)
            if not improved:
                for i in range(n_cands):
                    if not (current_mask & (1 << i)):
                        continue
                    new_mask = current_mask & ~(1 << i)
                    # Check if removing i allows adding a better node
                    # For now, just try simple removal
                    # Actually, removal alone never helps for max weight, so skip
            
            # Try swap: remove one selected, add one unselected
            if not improved:
                for i in range(n_cands):
                    if not (current_mask & (1 << i)):
                        continue
                    for j in range(n_cands):
                        if current_mask & (1 << j):
                            continue
                        # Check if j conflicts with any selected except i
                        cm_j = conflict_mask[j] & (current_mask & ~(1 << i))
                        if cm_j:
                            continue
                        new_mask = (current_mask & ~(1 << i)) | (1 << j)
                        new_score = current_score - weight_arr[i] + weight_arr[j]
                        if new_score > current_score:
                            current_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        
        return current_score, current_mask
    
    # Run local search from each greedy solution
    for strat in ['weight', 'weight_degree', 'weight_conflict']:
        s, m = greedy_select(strat)
        s_ls, m_ls = local_search(m, max_iters=10000)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
    
    # Random restarts with local search
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initial solution: randomly select coprime nodes
        random_mask = 0
        random_score = 0.0
        available = (1 << n_cands) - 1
        
        # Shuffle and greedily pick
        indices = list(range(n_cands))
        random.shuffle(indices)
        
        for i in indices:
            if not (available & (1 << i)):
                continue
            if conflict_mask[i] & random_mask:
                continue
            random_mask |= (1 << i)
            random_score += weight_arr[i]
            available &= ~conflict_mask[i]
        
        # Local search from random start
        s_ls, m_ls = local_search(random_mask, max_iters=5000)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
    
    # Convert best_mask to list of candidates
    result = []
    for i in range(n_cands):
        if best_mask & (1 << i):
            result.append(candidates[i])
    
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    # Verify pairwise coprime
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    if valid:
        print("Valid")
    print("Done")
