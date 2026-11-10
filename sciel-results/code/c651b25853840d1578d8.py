"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 256
    
    def is_sidon(s):
        """Check if set s is Sidon (all pair sums including a+a are unique)."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diffs(s):
        """Get all nonzero differences from set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(s, x, sum_set):
        """Check if x can be added to Sidon set s while maintaining Sidon property."""
        # Check x+x against existing sums
        xx = x + x
        if xx in sum_set:
            return False
        # Check x+a against existing sums
        for a in s:
            if (x + a) in sum_set:
                return False
        return True
    
    def get_sum_set(s):
        """Get all pair sums (including a+a) from set s."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def greedy_construct(order, max_size=64):
        """Greedy construction using given element order."""
        sets = [[], []]
        diff_sets = [set(), set()]
        sum_sets = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            best_side = -1
            best_priority = None
            
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                
                # Check Sidon property
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                
                # Check disjoint differences
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                
                if new_diffs & diff_sets[1 - side]:
                    continue
                
                # Priority: prefer the side with fewer elements to balance
                priority = (-len(sets[side]), x)
                
                if best_priority is None or priority < best_priority:
                    best_priority = priority
                    best_side = side
            
            if best_side >= 0:
                side = best_side
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                for a in sets[side]:
                    if a != x:
                        diff_sets[side].add(x - a)
                        diff_sets[side].add(a - x)
        
        return sets
    
    def local_improve(sets, order):
        """Try to add more elements by removing some and re-adding."""
        sets = [list(s) for s in sets]
        diff_sets = [get_diffs(s) for s in sets]
        sum_sets = [get_sum_set(s) for s in sets]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in order:
                if x in sets[0] or x in sets[1]:
                    continue
                for side in (0, 1):
                    if len(sets[side]) >= 64:
                        continue
                    if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                        continue
                    new_diffs = set()
                    for a in sets[side]:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    new_diffs.discard(0)
                    if new_diffs & diff_sets[1 - side]:
                        continue
                    sets[side].append(x)
                    sum_sets[side].add(x + x)
                    for a in sets[side]:
                        if a != x:
                            sum_sets[side].add(x + a)
                    diff_sets[side] |= new_diffs
                    improved = True
                    break
        return sets
    
    def spectral_guided_construct():
        """Use spectral clustering on difference-conflict graph to guide construction.
        
        Build a graph where vertices are elements 1..n. Edge weight between x,y is
        the number of difference conflicts they would create if placed in opposite sets.
        Use Fiedler vector to partition into two groups that minimize cross-set conflicts.
        """
        sets = [[], []]
        diff_sets = [set(), set()]
        sum_sets = [set(), set()]
        
        # Build difference-conflict graph
        # For each pair (x,y), compute how many differences {x-y, y-x} conflict
        # with differences that would be "natural" for the other set
        # Simpler: weight = 1 if |x-y| is a "common" difference (appears in many pairs)
        
        # Compute frequency of each difference
        diff_freq = {}
        for d in range(1, n):
            diff_freq[d] = n - d
        
        # Build adjacency matrix for spectral clustering
        # Weight between x and y: higher if they have "compatible" differences
        # We want to put elements with similar difference profiles together
        # Use: weight = 1 / (1 + |freq(x-y) - avg_freq|) to group elements with similar diff magnitudes
        
        avg_freq = n / 2.0
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                x, y = i + 1, j + 1
                d = abs(x - y)
                # Elements with rare differences are more flexible
                # Put elements with similar difference frequencies together
                weight = 1.0 / (1.0 + abs(diff_freq[d] - avg_freq))
                W[i, j] = weight
                W[j, i] = weight
        
        # Compute degree matrix
        deg = W.sum(axis=1)
        
        # Compute normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        
        # Find Fiedler vector (eigenvector of second-smallest eigenvalue)
        # Use scipy for efficient eigenvalue computation
        try:
            from scipy.linalg import eigh
            evals, evecs = eigh(L, subset_by_index=[1, 1])
            fiedler = evecs[:, 0]
        except Exception:
            # Fallback: use numpy
            evals, evecs = np.linalg.eigh(L)
            fiedler = evecs[:, 1]
        
        # Partition based on sign of Fiedler vector
        partition = (fiedler >= 0).astype(int)
        
        # Get elements for each side
        side0_elements = [i + 1 for i in range(n) if partition[i] == 0]
        side1_elements = [i + 1 for i in range(n) if partition[i] == 1]
        
        # Greedily add elements from each side, respecting Sidon and difference constraints
        for side, elements in [(0, side0_elements), (1, side1_elements)]:
            for x in elements:
                if time.time() > deadline:
                    break
                if len(sets[side]) >= 64:
                    continue
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if new_diffs & diff_sets[1 - side]:
                    continue
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                diff_sets[side] |= new_diffs
        
        # Now try to add remaining elements to either side
        used = set(sets[0]) | set(sets[1])
        remaining = [x for x in range(1, n + 1) if x not in used]
        
        # Score by conflict
        def conflict_score(x):
            score = 0
            for side in (0, 1):
                if len(sets[side]) >= 64:
                    continue
                for a in sets[side]:
                    if (x + a) in sum_sets[side] or (x + x) in sum_sets[side]:
                        score += 1
                    d1 = x - a
                    d2 = a - x
                    if d1 in diff_sets[1 - side] or d2 in diff_sets[1 - side]:
                        score += 1
            return score
        
        remaining.sort(key=conflict_score)
        
        for x in remaining:
            if time.time() > deadline:
                break
            
            best_side = -1
            best_conflict = float('inf')
            for side in (0, 1):
                if len(sets[side]) >= 64:
                    continue
                if not can_add_to_sidon(sets[side], x, sum_sets[side]):
                    continue
                new_diffs = set()
                for a in sets[side]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if new_diffs & diff_sets[1 - side]:
                    continue
                # Count conflicts
                c = 0
                for a in sets[side]:
                    if (x + a) in sum_sets[side]:
                        c += 1
                    if (x + x) in sum_sets[side]:
                        c += 1
                if c < best_conflict:
                    best_conflict = c
                    best_side = side
            
            if best_side >= 0:
                side = best_side
                sets[side].append(x)
                sum_sets[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        sum_sets[side].add(x + a)
                for a in sets[side]:
                    if a != x:
                        diff_sets[side].add(x - a)
                        diff_sets[side].add(a - x)
        
        return sets
    
    # Strategy: Try multiple approaches
    best_sets = [[], []]
    best_score = 0
    
    # Approach 1: Spectral-guided construction (primary)
    if time.time() < deadline - 30:
        sets = spectral_guided_construct()
        sets = local_improve(sets, list(range(1, n + 1)))
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Approach 2: Deterministic smaller-first ordering
    if time.time() < deadline - 25:
        base_order = list(range(1, n + 1))
        sets = greedy_construct(base_order)
        sets = local_improve(sets, base_order)
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Approach 3: Reverse ordering
    if time.time() < deadline - 20:
        rev_order = list(range(n, 0, -1))
        sets = greedy_construct(rev_order)
        sets = local_improve(sets, rev_order)
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Approach 4: Random orderings
    random.seed(42)
    for trial in range(300):
        if time.time() > deadline - 8:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        sets = greedy_construct(order)
        sets = local_improve(sets, order)
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Final improvement pass on the best sets
    final_order = list(range(1, n + 1))
    best_sets = local_improve(best_sets, final_order)
    
    # Additional local search: try moving elements between sets to free up space
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            improved = False
            for side in (0, 1):
                for i, x in enumerate(list(best_sets[side])):
                    if time.time() > deadline - 3:
                        break
                    # Try moving x to the other side
                    other = 1 - side
                    if len(best_sets[other]) >= 64:
                        continue
                    
                    # Temporarily remove x
                    best_sets[side].remove(x)
                    d0 = get_diffs(best_sets[0])
                    d1 = get_diffs(best_sets[1])
                    s0 = get_sum_set(best_sets[0])
                    s1 = get_sum_set(best_sets[1])
                    
                    # Check if x can be added to other side
                    if can_add_to_sidon(best_sets[other], x, s1):
                        new_diffs = set()
                        for a in best_sets[other]:
                            new_diffs.add(x - a)
                            new_diffs.add(a - x)
                        new_diffs.discard(0)
                        if not (new_diffs & d0):
                            best_sets[other].append(x)
                            improved = True
                    
                    # If not improved, put x back
                    if x not in best_sets[other]:
                        best_sets[side].append(x)
            
            if not improved:
                break
    
    # Verify validity
    for s in best_sets:
        assert is_sidon(s), "Sidon property violated"
    
    d0 = get_diffs(best_sets[0])
    d1 = get_diffs(best_sets[1])
    assert not (d0 & d1), "Difference sets not disjoint"
    
    # Ensure both sets are nonempty
    if len(best_sets[0]) == 0 and len(best_sets[1]) > 0:
        # Move one element from set 1 to set 0 if possible
        for x in list(best_sets[1]):
            best_sets[1].remove(x)
            d0_new = get_diffs(best_sets[0])
            s0_new = get_sum_set(best_sets[0])
            if can_add_to_sidon(best_sets[0], x, s0_new):
                new_diffs = set()
                for a in best_sets[0]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if not (new_diffs & d1):
                    best_sets[0].append(x)
                    break
            else:
                best_sets[1].append(x)
                break
    
    if len(best_sets[1]) == 0 and len(best_sets[0]) > 0:
        for x in list(best_sets[0]):
            best_sets[0].remove(x)
            d1_new = get_diffs(best_sets[1])
            s1_new = get_sum_set(best_sets[1])
            if can_add_to_sidon(best_sets[1], x, s1_new):
                new_diffs = set()
                for a in best_sets[1]:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if not (new_diffs & d0):
                    best_sets[1].append(x)
                    break
            else:
                best_sets[0].append(x)
                break
    
    # Final verification
    for s in best_sets:
        assert is_sidon(s), "Sidon property violated"
    d0_final = get_diffs(best_sets[0])
    d1_final = get_diffs(best_sets[1])
    assert not (d0_final & d1_final), "Difference sets not disjoint"
    
    return best_sets

# EVOLVE-BLOCK-END
