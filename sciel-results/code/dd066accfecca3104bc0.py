"""Linear triple systems on 16 points with small independence number."""
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    points = list(range(16))
    n_points = 16
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute bitmask for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
    
    # Precompute compatibility: two triples are compatible if they share at most 1 point
    # i.e., popcount(mask_i & mask_j) <= 1
    compat = [[False] * n_triples for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            inter = mi & triple_masks[j]
            # popcount
            pc = bin(inter).count('1')
            if pc <= 1:
                compat[i][j] = True
                compat[j][i] = True
    
    def get_alpha(triples):
        """Compute the independence number (largest set with no triple fully contained)."""
        if not triples:
            return n_points
        # A set S is independent if no triple from 'triples' is fully contained in S.
        # We want the largest such S.
        # Brute force over all subsets is 2^16 = 65536, which is fine.
        # For each subset, check if any triple is fully contained.
        best = 0
        for mask in range(1 << n_points):
            # Count bits
            sz = bin(mask).count('1')
            if sz <= best:
                continue
            # Check if any triple is fully in this set
            independent = True
            for t in triples:
                tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                best = sz
        return best
    
    # Faster alpha check using precomputed triple masks
    def get_alpha_fast(triple_indices):
        """Compute independence number given a list of triple indices."""
        if not triple_indices:
            return n_points
        tm_list = [triple_masks[i] for i in triple_indices]
        best = 0
        for mask in range(1 << n_points):
            sz = bin(mask).count('1')
            if sz <= best:
                continue
            independent = True
            for tm in tm_list:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                best = sz
        return best
    
    # Greedy construction: build a large set of mutually compatible triples
    def greedy_build():
        """Greedy: repeatedly pick a random compatible triple."""
        chosen = []
        remaining = set(range(n_triples))
        random.shuffle(remaining)
        
        for idx in list(remaining):
            if idx not in remaining:
                continue
            # Check compatibility with all chosen
            compatible = True
            for c in chosen:
                if not compat[idx][c]:
                    compatible = False
                    break
            if compatible:
                chosen.append(idx)
                remaining.discard(idx)
        
        return chosen
    
    # Local search: try to swap out a triple for another to reduce alpha
    def local_search(triple_indices, max_iters=5000):
        """Try to improve by swapping triples."""
        current = list(triple_indices)
        current_alpha = get_alpha_fast(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing one triple and adding another
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try adding a random triple
                candidate = random.randint(0, n_triples - 1)
                if candidate in current:
                    continue
                # Check compatibility
                ok = True
                for j in range(len(current)):
                    if j != i and not compat[candidate][current[j]]:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_list = current[:i] + [candidate] + current[i+1:]
                new_alpha = get_alpha_fast(new_list)
                if new_alpha < current_alpha:
                    current = new_list
                    current_alpha = new_alpha
                    improved = True
                    break
        
        return current, current_alpha
    
    # Try to build a good initial solution
    best_triples = None
    best_alpha = n_points
    
    # Multiple random restarts
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Greedy build
        triples = greedy_build()
        alpha = get_alpha_fast(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(triples)
        
        # Local search
        improved_triples, improved_alpha = local_search(triples, max_iters=2000)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved_triples
        
        # Additional local search passes
        for _ in range(3):
            if time.time() > deadline:
                break
            further_triples, further_alpha = local_search(best_triples, max_iters=3000)
            if further_alpha < best_alpha:
                best_alpha = further_alpha
                best_triples = further_triples
        
        if best_alpha <= 5:
            break
    
    # Convert indices back to triples
    if best_triples is None:
        # Fallback: return the initial seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return [list(all_triples[i]) for i in best_triples]
