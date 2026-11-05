import time
import random
import sys

def compute_sizes(A):
    """Compute sizes of sumset and product set."""
    n = len(A)
    # Using sets for distinct values
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(i, n):  # i <= j is sufficient for commutative + and *
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    return len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial configuration: Grid in (2,3)-lattice
    # i in [0, 3], j in [0, 4] gives 20 numbers
    # Max val: 2^3 * 3^4 = 8 * 81 = 648
    base_i = [0, 1, 2, 3]
    base_j = [0, 1, 2, 3, 4]
    
    initial_A = []
    for i in base_i:
        for j in base_j:
            initial_A.append((2 ** i) * (3 ** j))
            
    # Ensure distinct and sorted (distinctness guaranteed by unique factorization)
    initial_A = list(set(initial_A))
    if len(initial_A) != 20:
        # Fallback if something went wrong, though it shouldn't
        return list(range(1, 21))
        
    best_A = sorted(initial_A)
    best_s, best_p = compute_sizes(best_A)
    best_score = max(best_s, best_p)
    
    # Local Search
    # We will try to replace elements with other integers to reduce the max set size.
    # Since the product set is already very small (approx 25-30), the sum set (approx 100-200) is likely the bottleneck.
    # We can try to make the sum set smaller by creating arithmetic progressions or clustering.
    # However, changing numbers away from the 2^i 3^j lattice increases |A*A| quickly.
    # So we must be careful: only accept moves that keep |A*A| low or if |A+A| drops significantly enough to offset the |A*A| increase.
    
    rng = random.Random(42)
    num_iterations = 0
    
    while time.time() < deadline:
        num_iterations += 1
        
        # Pick a random index to perturb
        idx = rng.randint(0, 19)
        orig_val = best_A[idx]
        
        # Generate candidate
        # Strategy: Try values in the same lattice region or nearby integers
        # 1. Swap with another lattice point not in set? 
        # 2. Try nearby integer?
        
        # Let's try a mix:
        # 50% chance: try a nearby integer (orig_val +/- delta)
        # 50% chance: try another grid point (random i, j)
        
        if rng.random() < 0.5:
            # Nearby integer
            delta = rng.randint(-50, 50)
            new_val = orig_val + delta
            if new_val < 1 or new_val > 10000:
                continue
        else:
            # Random grid point in a wider range
            # i in [0, 5], j in [0, 5] -> max 32 * 243 = 7776 < 10000
            i = rng.randint(0, 5)
            j = rng.randint(0, 5)
            new_val = (2 ** i) * (3 ** j)
            if new_val > 10000:
                continue
        
        # Check distinctness
        if new_val in best_A:
            if new_val == orig_val:
                continue
            else:
                # Collision with another element, reject
                continue
        
        # Create new set
        new_A = best_A.copy()
        new_A[idx] = new_val
        new_A.sort()
        
        new_s, new_p = compute_sizes(new_A)
        new_score = max(new_s, new_p)
        
        # Acceptance criterion
        # Simple hill climbing
        if new_score < best_score:
            best_A = new_A
            best_s = new_s
            best_p = new_p
            best_score = new_score
        elif new_score == best_score:
            # Tie-breaking: prefer smaller product set if sum set is the bottleneck?
            # Or just accept to move in the landscape
            if rng.random() < 0.1:
                best_A = new_A
                best_s = new_s
                best_p = new_p
                # best_score remains same
        # else: reject

        # Optional: Periodic restart if stuck?
        # For now, let it run.
        
        # Throttle if necessary, but compute_sizes is O(400) so it's very fast.
        # We can do ~10k-100k iterations in 85s easily in Python if optimized.
        # Let's check speed.
        
        if num_iterations % 1000 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break

    return best_A

# Self-test to ensure it runs quickly and produces valid output
if __name__ == "__main__":
    import time
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")
    print(f"Result: {res}")
    s, p = compute_sizes(res)
    print(f"Sumset size: {s}, Prodset size: {p}, Max: {max(s,p)}")
