"""Optimized 4-AP-free subset of [1,256] maximizing 3-AP count via simulated annealing."""
import time
import random
import sys

N = 256

def _count_3aps(A):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in set A."""
    A_set = set(A)
    count = 0
    for a in A:
        for d in range(1, (N - a + 1) // 2 + 1):
            if a + d in A_set and a + 2 * d in A_set:
                count += 1
    return count

def _has_4ap(A):
    """Check if set A contains a nonconstant 4-term AP."""
    A_set = set(A)
    for a in A:
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            if a + d in A_set and a + 2 * d in A_set and a + 3 * d in A_set:
                return True
    return False

def _delta_3ap_add(x, A):
    """Change in 3-AP count when adding x to A."""
    A_set = set(A)
    delta = 0
    # x as first term: (x, x+d, x+2d)
    for d in range(1, (N - x + 1) // 2 + 1):
        if x + d in A_set and x + 2 * d in A_set:
            delta += 1
    # x as middle term: (x-d, x, x+d)
    for d in range(1, x):
        if x - d in A_set and x + d in A_set:
            delta += 1
    # x as last term: (x-2d, x-d, x)
    for d in range(1, x // 2 + 1):
        if x - 2 * d in A_set and x - d in A_set:
            delta += 1
    return delta

def _delta_4ap_add(x, A):
    """Check if adding x to A creates a 4-AP. Returns True if it does."""
    A_set = set(A)
    # x as first term: (x, x+d, x+2d, x+3d)
    for d in range(1, (N - x + 1) // 3 + 1):
        if x + d in A_set and x + 2 * d in A_set and x + 3 * d in A_set:
            return True
    # x as second term: (x-d, x, x+d, x+2d)
    for d in range(1, x):
        if x - d in A_set and x + d in A_set and x + 2 * d in A_set:
            return True
    # x as third term: (x-2d, x-d, x, x+d)
    for d in range(1, (x) // 2 + 1):
        if x - 2 * d in A_set and x - d in A_set and x + d in A_set:
            return True
    # x as fourth term: (x-3d, x-2d, x-d, x)
    for d in range(1, x // 3 + 1):
        if x - 3 * d in A_set and x - 2 * d in A_set and x - d in A_set:
            return True
    return False

def _delta_4ap_remove(x, A):
    """Check if removing x from A breaks all 4-APs containing x.
    We need to check: after removal, are there still 4-APs? 
    Actually for SA we just need to know if adding x would create a 4-AP.
    For removal, we just remove and the set is still valid if it was valid before.
    """
    pass  # Not needed directly; we only check validity on add

def construct():
    deadline = time.time() + 80  # 80s to be safe under 90s limit
    random.seed(42)
    
    # Greedy initialization: build a 4-AP-free set
    A = set()
    for x in range(1, N + 1):
        if not _delta_4ap_add(x, A):
            A.add(x)
    
    current_3ap = _count_3aps(A)
    best_A = set(A)
    best_3ap = current_3ap
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    cooling = 0.9995
    
    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4 and len(A) < N:
            # Try to add a random element
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                break
            x = random.choice(candidates)
            if _delta_4ap_add(x, A):
                T *= cooling
                continue
            d_3ap = _delta_3ap_add(x, A)
            A.add(x)
            current_3ap += d_3ap
            if current_3ap > best_3ap:
                best_3ap = current_3ap
                best_A = set(A)
                no_improve = 0
            else:
                no_improve += 1
        elif move_type < 0.7 and len(A) > 1:
            # Try to remove a random element
            x = random.choice(list(A))
            A.discard(x)
            # Recompute 3ap (expensive but rare)
            current_3ap = _count_3aps(A)
            if current_3ap > best_3ap:
                best_3ap = current_3ap
                best_A = set(A)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Swap: remove one, add another
            if len(A) < 2 or len(A) == N:
                T *= cooling
                continue
            x_remove = random.choice(list(A))
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                T *= cooling
                continue
            x_add = random.choice(candidates)
            
            # Check if adding x_add is valid (after removing x_remove)
            temp_A = A - {x_remove}
            if _delta_4ap_add(x_add, temp_A):
                T *= cooling
                continue
            
            # Compute delta
            old_3ap = current_3ap
            temp_A.add(x_add)
            new_3ap = _count_3aps(temp_A)
            
            if new_3ap > old_3ap or random.random() < (2.718281828 ** ((new_3ap - old_3ap) / T)):
                A = temp_A
                current_3ap = new_3ap
                if current_3ap > best_3ap:
                    best_3ap = current_3ap
                    best_A = set(A)
                    no_improve = 0
                else:
                    no_improve += 1
            # else: reject, A unchanged
        
        T *= cooling
        if T < T_end:
            T = T_end
        
        if no_improve > max_no_improve:
            # Restart with perturbation
            no_improve = 0
            # Perturb best solution
            A = set(best_A)
            current_3ap = best_3ap
            # Remove a few random elements
            for _ in range(random.randint(3, 8)):
                if A:
                    A.discard(random.choice(list(A)))
            current_3ap = _count_3aps(A)
            T = T_start * 0.5
    
    return sorted(best_A)

if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"3-AP count: {_count_3aps(result)}")
    print(f"Has 4-AP: {_has_4ap(result)}")
