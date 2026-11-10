"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (2a[i])."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Find the size of the largest Sidon subset (with diagonals) of arr using branch and bound."""
    n = len(arr)
    if n <= 1:
        return n
    
    arr_sorted = sorted(arr)
    best = 0
    
    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best
        
        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return False
        
        # Update best
        if current_size > best:
            best = current_size
            # If we reached the theoretical max (n), we can stop entirely
            if best == n:
                return True # Signal to stop
        
        # If we've considered all elements
        if idx == n:
            return False
        
        # Option 1: Exclude arr_sorted[idx]
        if dfs(idx + 1, current_size, current_sums, current_elements):
            return True
            
        # Option 2: Include arr_sorted[idx]
        x = arr_sorted[idx]
        new_sums = set()
        can_add = True
        
        # Check sums involving x against existing sums
        for y in current_elements:
            s = x + y
            if s in current_sums:
                can_add = False
                break
            new_sums.add(s)
        
        if can_add:
            s_diag = 2 * x
            if s_diag in current_sums:
                can_add = False
            else:
                new_sums.add(s_diag)
        
        if can_add:
            # Add x
            if dfs(idx + 1, current_size + 1, current_sums | new_sums, current_elements + [x]):
                return True
        
        return False

    dfs(0, 0, set(), [])
    return best

def _evaluate(arr):
    """Evaluate the quality: return the max Sidon subset size (lower is better)."""
    return _max_sidon_subset_size(arr)

def construct():
    global _START_TIME, _DEADLINE
    
    # Ensure valid output: 12 distinct integers in [0, 128]
    # Start with a clustered set to encourage many collisions
    # Numbers close together create more sum collisions
    
    best_arr = list(range(12))  # initial: 0..11
    best_score = _evaluate(best_arr)
    
    # Try some initial structured sets
    candidates = []
    
    # Clustered at low end
    candidates.append([0,1,2,3,4,5,6,7,8,9,10,11])
    # Two clusters
    candidates.append([0,1,2,3,4,5, 64,65,66,67,68,69])
    # Arithmetic progression with small step
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    # Squares
    candidates.append([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121])
    # Powers
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 128])
    # More clustered
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    for c in candidates:
        if len(c) != 12 or len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        s = _evaluate(c)
        if s < best_score:
            best_score = s
            best_arr = c[:]
    
    # Hill climbing
    rng = random.Random(42)
    
    def _neighbor(arr):
        """Generate a random neighbor by changing one element."""
        new_arr = arr[:]
        idx = rng.randint(0, 11)
        old_val = new_arr[idx]
        # Try to pick a value close to old_val to maintain structure?
        # Or random? Random is better for escaping local minima.
        # But let's try a mix: 50% random, 50% local
        if rng.random() < 0.5:
            new_val = rng.randint(0, 128)
        else:
            step = rng.choice([-2, -1, 1, 2, 5, 10])
            new_val = old_val + step
            if new_val < 0: new_val = 0
            if new_val > 128: new_val = 128
        
        if new_val == old_val:
            return None
        
        # Check distinctness
        # Efficient check:
        if new_val in new_arr[:idx] or new_val in new_arr[idx+1:]:
            return None
            
        new_arr[idx] = new_val
        return new_arr
    
    # Run hill climbing until deadline
    # Since _evaluate is fast, we can do many iterations
    while time.time() < _DEADLINE - 2.0:  # 2s safety margin
        neighbor = _neighbor(best_arr)
        if neighbor is None:
            continue
        score = _evaluate(neighbor)
        if score < best_score:
            best_score = score
            best_arr = neighbor
            if best_score <= 2:
                break  # can't do much better than 2
        elif score == best_score:
            # Accept with some probability for diversity
            if rng.random() < 0.1:
                best_arr = neighbor
    
    # Final validation
    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(0 <= x <= 128 for x in best_arr)
    
    return best_arr

# EVOLVE-BLOCK-START
# (construct is defined above; this block is for SCiEL evolution markers)
# EVOLVE-BLOCK-END
