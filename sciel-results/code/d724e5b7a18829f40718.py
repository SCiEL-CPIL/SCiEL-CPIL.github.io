import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    base = 51975
    # Find all divisors > 1 of base
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors
    divisors_sorted = sorted(divisors)
    num_divisors = len(divisors_sorted)
    
    # Precompute indices for each divisor for faster slicing if needed, 
    # but slicing r::m is fast in numpy.
    
    def evaluate_greedy(order):
        """
        Greedily assign residues to moduli in the given order.
        Returns (classes, cover_count, uncovered_count)
        """
        classes = []
        cover_count = np.zeros(base, dtype=np.int32)
        uncovered = base
        
        for m in order:
            best_r = 0
            best_gain = -1
            
            # Find residue that maximizes newly covered residues
            # newly covered = residues where cover_count is 0
            # We want to maximize sum(cover_count[r::m] == 0)
            
            # Optimization: If m is small, check all. If m is large, check all is still fine.
            # base is 51975. Max m is 51975.
            # Looping m times and doing numpy sum is O(m * (base/m)) = O(base).
            # Total for all m is O(base * num_divisors) which is ~ 50k * 48 ~ 2.4M ops. Fast.
            
            for r in range(m):
                # Count zeros in slice
                gain = np.sum(cover_count[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            classes.append([m, best_r])
            
            # Update cover_count and uncovered
            # Decrement uncovered for those that become covered
            # Actually, just update cover_count and recalculate uncovered at the end of greedy?
            # No, we need uncovered to guide greedy.
            # Update cover_count
            cover_count[best_r::m] += 1
            # Update uncovered: for indices in best_r::m, if cover_count was 0, it becomes 1, so uncovered decreases.
            # We already calculated gain. So uncovered -= gain.
            uncovered -= best_gain
            
        return classes, cover_count, uncovered

    def local_search(classes, cover_count, uncovered):
        """
        Hill climbing local search.
        """
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Shuffle class order to escape local minima? Or just iterate.
            # Let's iterate through all classes.
            for i in range(len(classes)):
                m, r_old = classes[i]
                best_r = r_old
                best_uncovered = uncovered
                
                # Try all other residues
                # To speed up, we can skip r_old
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    
                    # Calculate delta in uncovered
                    # Indices affected: old_slice = r_old::m, new_slice = r_new::m
                    # We need to know:
                    # 1. How many in old_slice have cover_count == 1 (will become 0, increasing uncovered)
                    # 2. How many in new_slice have cover_count == 0 (will become 1, decreasing uncovered)
                    
                    # Note: old_slice and new_slice are disjoint if r_old != r_new mod m?
                    # Yes, because they are different residues modulo m.
                    # So no index is in both slices.
                    
                    old_slice = cover_count[r_old::m]
                    new_slice = cover_count[r_new::m]
                    
                    # Count indices in old_slice with count 1
                    lose = np.sum(old_slice == 1)
                    # Count indices in new_slice with count 0
                    gain = np.sum(new_slice == 0)
                    
                    delta = lose - gain
                    new_uncovered = uncovered + delta
                    
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new
                
                if best_r != r_old:
                    # Apply change
                    classes[i][1] = best_r
                    cover_count[r_old::m] -= 1
                    cover_count[best_r::m] += 1
                    uncovered = best_uncovered
                    improved = True
                    
        return classes, cover_count, uncovered

    best_classes = []
    best_uncovered = base + 1
    
    # Phase 1: Multiple Greedy Initializations
    # 1. Ascending order
    # 2. Descending order
    # 3. Random shuffles
    
    orders = [
        divisors_sorted,
        sorted(divisors_sorted, reverse=True)
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(10):
        order = divisors_sorted[:]
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline - 20:
            break
            
        classes, cover_count, uncovered = evaluate_greedy(order)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c[:] for c in classes]
            best_cover_count = cover_count.copy()
        
        # Phase 2: Local Search on this solution
        if time.time() < deadline - 10:
            classes_ls, cover_count_ls, uncovered_ls = local_search(classes, cover_count, uncovered)
            
            if uncovered_ls < best_uncovered:
                best_uncovered = uncovered_ls
                best_classes = [c[:] for c in classes_ls]
                best_cover_count = cover_count_ls.copy()
                
    # Phase 3: More local search on the global best
    if best_classes and time.time() < deadline - 5:
        best_classes, best_cover_count, best_uncovered = local_search(best_classes, best_cover_count, best_uncovered)

    if not best_classes:
        # Fallback
        best_classes = [[3, 0]]
        
    return best_classes
