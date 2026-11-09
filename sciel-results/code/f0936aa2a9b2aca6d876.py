import time
import random

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def build_sumsets_bitset(A):
    """Build cardinality-indexed subset sum bitsets for set A."""
    n = len(A)
    # Max possible sum is sum(A), but we can use a fixed max for bitset size if needed
    # However, Python ints grow dynamically.
    S = [0] * (n + 1)
    S[0] = 1  # Bit 0 is set, sum 0 with 0 elements
    
    for x in A:
        # Update in reverse to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            if S[k]:
                S[k+1] |= (S[k] << x)
    return S

def check_validity_bitset(S):
    """Check if cardinality-indexed sum bitsets have no cross-cardinality collisions."""
    n = len(S) - 1
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def validate(A):
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    S = build_sumsets_bitset(A)
    return check_validity_bitset(S)

def get_ap_solution(n=31):
    """Generate the arithmetic progression solution of size n."""
    # We derived a=226, d=1 for n=31
    # Let's calculate generally
    # M = max k(n-k-1)
    M = 0
    for k in range(n):
        val = k * (n - k - 1)
        if val > M:
            M = val
    # d=1
    a = M + 1
    # Check if a + (n-1) <= 256
    if a + (n - 1) > 256:
        # Try smaller n or different d?
        # For now, return empty or partial
        return []
    return [a + i for i in range(n)]

def local_search_bitset(initial, time_budget=40.0):
    """Local search using bitset validity checks."""
    start = time.time()
    best = list(initial)
    best_S = build_sumsets_bitset(best)
    
    if not check_validity_bitset(best_S):
        return initial
        
    all_values = set(range(1, 257))
    current_set = set(best)
    
    while not is_done() and (time.time() - start) < time_budget:
        improved = False
        
        # 1. Try adding elements
        candidates_add = list(all_values - current_set)
        random.shuffle(candidates_add)
        for x in candidates_add:
            if len(best) >= 60:
                break
            if (time.time() - start) > time_budget:
                break
            
            new_set = best + [x]
            new_S = build_sumsets_bitset(new_set)
            if check_validity_bitset(new_S):
                best = new_set
                best_S = new_S
                current_set.add(x)
                improved = True
                break # Restart loop to re-shuffle or continue
        
        # 2. Try swaps (remove 1, add 1)
        if not improved:
            candidates_swap_out = list(best)
            random.shuffle(candidates_swap_out)
            for x_out in candidates_swap_out:
                if (time.time() - start) > time_budget:
                    break
                remaining = [x for x in best if x != x_out]
                candidates_swap_in = list(all_values - set(remaining))
                random.shuffle(candidates_swap_in)
                
                for x_in in candidates_swap_in:
                    new_set = remaining + [x_in]
                    new_S = build_sumsets_bitset(new_set)
                    if check_validity_bitset(new_S):
                        best = new_set
                        best_S = new_S
                        current_set = set(best)
                        improved = True
                        break
                if improved:
                    break
        
        # 3. Try 2-swaps (remove 2, add 2)
        if not improved and len(best) < 60:
            # Sample a few pairs to avoid O(n^2) explosion
            n_samples = min(20, len(best))
            indices = random.sample(range(len(best)), n_samples)
            for i in indices:
                for j in random.sample([k for k in range(len(best)) if k != i], min(10, len(best)-1)):
                    if (time.time() - start) > time_budget:
                        break
                    remaining = [x for k, x in enumerate(best) if k != i and k != j]
                    candidates_add = list(all_values - set(remaining))
                    random.shuffle(candidates_add)
                    
                    # Try adding 2 from candidates
                    if len(candidates_add) >= 2:
                        x1, x2 = random.sample(candidates_add, 2)
                        new_set = remaining + [x1, x2]
                        new_S = build_sumsets_bitset(new_set)
                        if check_validity_bitset(new_S):
                            best = new_set
                            best_S = new_S
                            current_set = set(best)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            # Stagnation: try perturbation
            if len(best) > 0 and (time.time() - start) < time_budget - 5:
                perturbed = list(best)
                # Remove 1-2 random elements
                for _ in range(random.randint(1, 2)):
                    if perturbed:
                        idx = random.randint(0, len(perturbed) - 1)
                        perturbed.pop(idx)
                
                # Try to rebuild by adding
                current_set = set(perturbed)
                candidates_add = list(all_values - current_set)
                random.shuffle(candidates_add)
                for x in candidates_add:
                    if len(perturbed) >= len(best):
                        break
                    if (time.time() - start) > time_budget:
                        break
                    new_set = perturbed + [x]
                    new_S = build_sumsets_bitset(new_set)
                    if check_validity_bitset(new_S):
                        perturbed = new_set
                        current_set.add(x)
                
                if len(perturbed) > len(best):
                    best = perturbed
                    best_S = build_sumsets_bitset(best)
                    current_set = set(best)
                    improved = True
            else:
                # No improvement, break to save time or try random restart
                break

    return best

def construct():
    # Start with AP solution
    base_solution = get_ap_solution(31)
    if not base_solution:
        base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    best = list(base_solution)
    
    # Run local search
    time_left = DEADLINE - elapsed()
    if time_left > 5:
        ls_time = min(time_left * 0.8, 70.0)
        ls_result = local_search_bitset(best, time_budget=ls_time)
        if len(ls_result) > len(best):
            best = ls_result
            
    # Final validation
    if validate(best):
        return best
    
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
