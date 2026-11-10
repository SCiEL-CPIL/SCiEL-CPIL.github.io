import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Precompute all triples
    all_triples = []
    mask_to_idx = {}
    idx_to_mask = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(all_triples)
                all_triples.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                
    num_triples = len(all_triples)
    
    # Precompute conflicts
    # conflicts[i] is a set of indices j such that triple i and j share >= 2 points
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = all_triples[i]
        for j in range(i + 1, num_triples):
            mj = all_triples[j]
            if bin(mi & mj).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute 6-sets and 5-sets for alpha check
    # We only care if there exists an independent set of size 6 or 5.
    # Since min alpha is 5, we check 6 first. If none, check 5.
    
    # For speed, we can check subsets of a candidate independent set.
    # But let's just iterate 6-sets and 5-sets.
    # Number of 6-sets: 8008. Number of 5-sets: 4368.
    # Checking all is fast enough if done efficiently.
    
    def is_independent(mask, edges_set):
        """Check if 'mask' contains no triple from edges_set."""
        # Iterate over triples in edges_set? No, too slow.
        # Better: for each triple in edges_set, check if it is subset of mask.
        # But edges_set can be large.
        # Alternative: Precompute for each 6-set which triples are contained in it.
        # Or, for each mask, check if any triple in edges_set is a subset.
        
        # Let's optimize: 
        # If edges_set is small, iterate edges.
        # If edges_set is large, iterate subsets of mask?
        # Mask size is 6. Subsets of size 3 are 20.
        # Check if any of these 20 triples is in edges_set.
        
        # Precompute a boolean array or set for edges_set for O(1) lookup.
        # But building the set takes O(E).
        # Let's pass the set of edge indices.
        
        # For a 6-set mask, there are 20 triples.
        # We check if any of them is in edges_set.
        
        # This is fast: 20 lookups in a set.
        
        # Get all triples contained in mask
        # We can precompute list of triple indices for each 6-set?
        # 8008 * 20 = 160k entries. Fine.
        
        pass # Placeholder

    # Precompute triples contained in each 6-set and 5-set
    triples_in_6set = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == 6:
            # Find all triples in mask
            ts = []
            # Iterate subsets of size 3
            # Simple way: iterate all triples and check subset? 5120 checks. 8008 * 5120 ~ 40M. Might be slow.
            # Better: iterate combinations of points in mask.
            pts = [i for i in range(n) if mask & (1 << i)]
            for c in combinations(pts, 3):
                m = (1 << c[0]) | (1 << c[1]) | (1 << c[2])
                if m in mask_to_idx:
                    ts.append(mask_to_idx[m])
            triples_in_6set[mask] = ts
            
    triples_in_5set = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == 5:
            pts = [i for i in range(n) if mask & (1 << i)]
            ts = []
            for c in combinations(pts, 3):
                m = (1 << c[0]) | (1 << c[1]) | (1 << c[2])
                if m in mask_to_idx:
                    ts.append(mask_to_idx[m])
            triples_in_5set[mask] = ts

    def calc_alpha(edges_set):
        # Check 6-sets
        for mask, ts in triples_in_6set.items():
            # If none of ts is in edges_set, then mask is independent
            independent = True
            for t_idx in ts:
                if t_idx in edges_set:
                    independent = False
                    break
            if independent:
                return 6
                
        # Check 5-sets
        for mask, ts in triples_in_5set.items():
            independent = True
            for t_idx in ts:
                if t_idx in edges_set:
                    independent = False
                    break
            if independent:
                return 5
                
        return 4 # Should not happen if linear constraint holds and n=16? 
                 # Actually, if no 5-set is independent, alpha <= 4.
                 # But we proved alpha >= 5. So this branch is unreachable for valid linear graphs?
                 # Wait, if alpha=5, there IS a 5-set independent.
                 # If we eliminate all 5-sets, alpha < 5, which is impossible.
                 # So if we reach here, it means we found a contradiction or error.
                 # But we check 5-sets. If one is independent, return 5.
                 # If none are independent, return 4.

    def is_valid_add(triple_idx, edges_set):
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True

    def greedy_pack():
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        return current_triples

    def random_swap(current_triples):
        if not current_triples:
            return None
        to_remove = random.choice(list(current_triples))
        new_triples = set(current_triples)
        new_triples.remove(to_remove)
        
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        for idx in candidates:
            if idx in new_triples:
                continue
            if is_valid_add(idx, new_triples):
                new_triples.add(idx)
                return new_triples
        return None

    best_alpha = n
    best_triples = None
    current_triples = greedy_pack()
    current_alpha = calc_alpha(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = set(current_triples)
        
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            if best_alpha == 5:
                break # Optimal found
                
        new_triples = random_swap(current_triples)
        if new_triples is None:
            # Perturb
            if current_triples:
                for _ in range(2):
                    if current_triples:
                        current_triples.remove(random.choice(list(current_triples)))
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        delta = new_alpha - current_alpha
        if delta <= 0:
            current_triples = new_triples
            current_alpha = new_alpha
            if delta < 0:
                T = max(T, 0.5)
        else:
            if random.random() < (2.71828 ** (-delta / T)):
                current_triples = new_triples
                current_alpha = new_alpha
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    result = []
    for idx in best_triples:
        mask = idx_to_mask[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
