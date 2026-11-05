import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(12345)
    
    # Precompute triple lists for various n
    def get_triples(n):
        """Return list of all triples (a,b,c) with a<b<c in lexicographic order."""
        return list(combinations(range(n), 3))
    
    def build_index_map(n, triples):
        """Map triple -> index."""
        return {t: i for i, t in enumerate(triples)}
    
    def count_red_k4(colors, n, triples, idx_map):
        """Count red K4 violations. Red = 0."""
        # For each pair (a,b), build set of c where (a,b,c) is red
        # Use numpy for speed
        color_arr = np.array(colors, dtype=np.uint8)
        
        # For efficiency, iterate over 4-sets
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                # c values where (a,b,c) is red
                red_cs_ab = set()
                for c in range(b+1, n):
                    if color_arr[idx_map[(a,b,c)]] == 0:
                        red_cs_ab.add(c)
                if len(red_cs_ab) < 2:
                    continue
                for c in red_cs_ab:
                    # Need d > c such that (a,c,d), (b,c,d), (a,b,d) all red
                    for d in range(c+1, n):
                        if d in red_cs_ab:  # (a,b,d) red
                            if color_arr[idx_map[(a,c,d)]] == 0 and color_arr[idx_map[(b,c,d)]] == 0:
                                count += 1
        return count
    
    def count_blue_k5(colors, n, triples, idx_map):
        """Count blue K5 violations. Blue = 1."""
        color_arr = np.array(colors, dtype=np.uint8)
        
        count = 0
        # For each 5-set, check if all 10 triples are blue
        for a in range(n):
            for b in range(a+1, n):
                # Build set of c > b where (a,b,c) is blue
                blue_cs_ab = set()
                for c in range(b+1, n):
                    if color_arr[idx_map[(a,b,c)]] == 1:
                        blue_cs_ab.add(c)
                if len(blue_cs_ab) < 3:
                    continue
                for c in blue_cs_ab:
                    # Need d, e > c such that all triples among {a,b,c,d,e} are blue
                    blue_cs_ac = set()
                    blue_cs_bc = set()
                    for d in range(c+1, n):
                        if color_arr[idx_map[(a,c,d)]] == 1:
                            blue_cs_ac.add(d)
                        if color_arr[idx_map[(b,c,d)]] == 1:
                            blue_cs_bc.add(d)
                    # d must be in blue_cs_ab ∩ blue_cs_ac ∩ blue_cs_bc
                    valid_d = blue_cs_ab & blue_cs_ac & blue_cs_bc
                    if len(valid_d) < 2:
                        continue
                    valid_d_list = sorted(valid_d)
                    for i, d in enumerate(valid_d_list):
                        # Need e > d such that (a,d,e), (b,d,e), (c,d,e) all blue
                        # and e in blue_cs_ab
                        for e in valid_d_list[i+1:]:
                            if color_arr[idx_map[(a,d,e)]] == 1 and color_arr[idx_map[(b,d,e)]] == 1 and color_arr[idx_map[(c,d,e)]] == 1:
                                count += 1
        return count
    
    def total_violations(colors, n, triples, idx_map):
        return count_red_k4(colors, n, triples, idx_map) + count_blue_k5(colors, n, triples, idx_map)
    
    def greedy_local_search(colors, n, triples, idx_map, deadline, rng, max_iters=2000):
        """
        Greedy local search: find the most conflicting triple and flip it.
        Prioritize red K4 violations (rarer, harder to satisfy).
        """
        color_arr = np.array(colors, dtype=np.uint8)
        triples_arr = np.array(triples, dtype=np.int32)  # shape (n_t, 3)
        
        # Precompute for each triple, which 4-sets it belongs to (for red K4)
        # and which 5-sets it belongs to (for blue K5)
        # This is expensive, so we'll use a lighter approach
        
        # For each triple index, compute a "conflict score"
        # = number of red K4s containing it (if red) + number of blue K5s containing it (if blue)
        
        # Efficient approach: for each pair, maintain counts
        # For red K4: triple (a,b,c) is in a red K4 if there exists d>c with all 4 triples red
        # For blue K5: triple (a,b,c) is in a blue K5 if there exist d,e>c with all 10 triples blue
        
        # Let's use a sampling-based conflict score for speed
        def compute_conflict_scores():
            """Compute conflict score for each triple."""
            scores = np.zeros(len(triples), dtype=np.int32)
            
            # For red K4: for each triple (a,b,c) that is red, count how many d>c make K4
            for i, (a, b, c) in enumerate(triples):
                if color_arr[i] == 0:  # red
                    # Count d > c such that (a,b,d), (a,c,d), (b,c,d) all red
                    cnt = 0
                    for d in range(c+1, n):
                        if color_arr[idx_map[(a,b,d)]] == 0 and color_arr[idx_map[(a,c,d)]] == 0 and color_arr[idx_map[(b,c,d)]] == 0:
                            cnt += 1
                    scores[i] = cnt
                else:  # blue
                    # Count (d,e) pairs with d,e > c such that all triples among {a,b,c,d,e} are blue
                    cnt = 0
                    # For speed, limit to a few candidates
                    for d in range(c+1, min(c+5, n)):
                        if color_arr[idx_map[(a,b,d)]] == 1 and color_arr[idx_map[(a,c,d)]] == 1 and color_arr[idx_map[(b,c,d)]] == 1:
                            for e in range(d+1, min(d+5, n)):
                                if color_arr[idx_map[(a,b,e)]] == 1 and color_arr[idx_map[(a,c,e)]] == 1 and color_arr[idx_map[(b,c,e)]] == 1 and \
                                   color_arr[idx_map[(a,d,e)]] == 1 and color_arr[idx_map[(b,d,e)]] == 1 and color_arr[idx_map[(c,d,e)]] == 1:
                                    cnt += 1
                    scores[i] = cnt * 2  # Weight blue K5 contributions more since they're 5-sets
        
            return scores
        
        best_total = total_violations(colors, n, triples, idx_map)
        best_colors = list(colors)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute conflict scores
            scores = compute_conflict_scores()
            
            if scores.max() == 0:
                break
            
            # Pick the triple with highest conflict score (prioritize red K4)
            # Among ties, prefer red triples (since red K4 is rarer)
            max_score = scores.max()
            candidates = np.where(scores == max_score)[0]
            
            # Prefer red triples (color 0)
            red_candidates = [i for i in candidates if color_arr[i] == 0]
            if red_candidates:
                flip_idx = red_candidates[rng.randrange(len(red_candidates))]
            else:
                flip_idx = candidates[rng.randrange(len(candidates))]
            
            # Flip
            old_color = color_arr[flip_idx]
            color_arr[flip_idx] = 1 - old_color
            colors[flip_idx] = int(color_arr[flip_idx])
            
            # Check improvement
            new_total = total_violations(colors, n, triples, idx_map)
            if new_total < best_total:
                best_total = new_total
                best_colors = list(colors)
            else:
                # Revert
                color_arr[flip_idx] = old_color
                colors[flip_idx] = int(old_color)
                # Random perturbation
                for _ in range(3):
                    ri = rng.randrange(len(triples))
                    color_arr[ri] = 1 - color_arr[ri]
                    colors[ri] = int(color_arr[ri])
                best_total = total_violations(colors, n, triples, idx_map)
                best_colors = list(colors)
        
        return best_colors, best_total
    
    # Try to find valid colorings for n from 34 down to 10
    best_result = None
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
        
        triples = get_triples(n)
        idx_map = build_index_map(n, triples)
        n_t = len(triples)
        
        found_valid = False
        
        # Try multiple random initializations with local search
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            # Random initialization with p=0.5
            colors = [rng.randint(0, 1) for _ in range(n_t)]
            
            # Check if already valid
            tv = total_violations(colors, n, triples, idx_map)
            if tv == 0:
                found_valid = True
                break
            
            # Run greedy local search
            colors, tv = greedy_local_search(colors, n, triples, idx_map, deadline - 3, rng)
            
            if tv == 0:
                found_valid = True
                break
        
        # Also try structured initializations
        if not found_valid:
            # Sum mod 3
            colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
            tv = total_violations(colors, n, triples, idx_map)
            if tv == 0:
                found_valid = True
            else:
                colors, tv = greedy_local_search(colors, n, triples, idx_map, deadline - 3, rng)
                if tv == 0:
                    found_valid = True
        
        if found_valid:
            # Verify
            tv = total_violations(colors, n, triples, idx_map)
            if tv == 0:
                best_result = {'n': n, 'colors': list(colors)}
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: small valid coloring
    # n=9 with sum mod 3
    n = 9
    triples = get_triples(n)
    colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples]
    return {'n': n, 'colors': colors}
