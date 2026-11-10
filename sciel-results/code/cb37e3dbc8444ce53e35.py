"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # We aim for n=34, which is the known published construction size.
    # If we can't solve it, we fall back to smaller n.
    # However, the task asks to maximize n. 
    # The seed gives n=9.
    # Let's try to construct a valid coloring for n=34.
    # If local search fails to find a valid one for 34, we can try smaller n?
    # No, the evaluator scores 'n'. Higher is better.
    # If we return an invalid coloring, score is 0.
    # So we MUST return a valid coloring.
    # Strategy:
    # 1. Try to find a valid coloring for n=34 using local search.
    # 2. If time runs out or we fail, try n=33, 32, etc.
    # 3. If all fail, return the seed (n=9) or a known small valid one.
    
    # Helper to check validity and count violations
    # A red K4 is a set of 4 vertices where all 4 triples are red (0).
    # A blue K5 is a set of 5 vertices where all 10 triples are blue (1).
    
    # For efficiency, we can precompute the list of all K4s and K5s.
    # For n=34:
    # Number of K4s: C(34,4) = 46376
    # Number of K5s: C(34,5) = 278256
    # Total constraints ~ 324k.
    
    # We represent the coloring as a flat array of bits.
    # Index mapping: for triple (a,b,c) with a<b<c, index = ?
    # We can generate all triples and their indices once.
    
    def get_triples(n):
        triples = []
        idx_map = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx_map[(a,b,c)] = idx
                    triples.append((a,b,c))
                    idx += 1
        return triples, idx_map, idx

    def build_constraints(n, triples, idx_map):
        # Returns list of (type, vertices, triple_indices)
        # type 0: Red K4 (all 4 triples must not be 0) -> at least one must be 1
        # type 1: Blue K5 (all 10 triples must not be 1) -> at least one must be 0
        
        k4s = []
        for combo in combinations(range(n), 4):
            t_indices = []
            # 4 triples in a K4: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            # combo is (a,b,c,d)
            t_indices.append(idx_map[(combo[0], combo[1], combo[2])])
            t_indices.append(idx_map[(combo[0], combo[1], combo[3])])
            t_indices.append(idx_map[(combo[0], combo[2], combo[3])])
            t_indices.append(idx_map[(combo[1], combo[2], combo[3])])
            k4s.append(t_indices)
            
        k5s = []
        for combo in combinations(range(n), 5):
            t_indices = []
            # 10 triples in a K5
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t_indices.append(idx_map[(combo[i], combo[j], combo[k])])
            k5s.append(t_indices)
            
        return k4s, k5s

    def count_violations(colors, k4s, k5s):
        # colors is a numpy array of 0s and 1s
        # Red K4 violation: all 4 are 0
        # Blue K5 violation: all 10 are 1
        
        # We can vectorize this somewhat, but looping over 300k constraints in Python is slow.
        # Let's try to be smart.
        # For each K4, check if sum(colors[idx]) == 0.
        # For each K5, check if sum(colors[idx]) == 10.
        
        # This is still heavy. 
        # Alternative: Use a different representation or incremental updates.
        # But for simplicity and robustness within time, let's see if we can just do a few iterations.
        # 324k constraints * 34 vertices? No, 324k constraints total.
        # Checking each constraint takes ~4 or ~10 array lookups.
        # 324k * 10 = 3.2M operations. In Python, this might take ~0.1-0.5s per full scan.
        # If we do 1000 scans, that's 100-500s. Too slow.
        
        # We need a faster way to count violations or update them.
        # Or, we can use a randomized approach with a heuristic that is very likely to be valid?
        # No, we need certainty.
        
        # Let's try to optimize the violation count using NumPy.
        # For K4s: We have a matrix M4 where each row is a K4's 4 triple indices.
        # We want to count rows where sum(colors[M4[i]]) == 0.
        # This can be done with fancy indexing.
        
        pass

    # Let's refine the violation counting.
    # Precompute index arrays for K4s and K5s.
    
    def prepare_constraints(n):
        triples, idx_map, total_triples = get_triples(n)
        k4_indices = []
        for combo in combinations(range(n), 4):
            k4_indices.append([
                idx_map[(combo[0], combo[1], combo[2])],
                idx_map[(combo[0], combo[1], combo[3])],
                idx_map[(combo[0], combo[2], combo[3])],
                idx_map[(combo[1], combo[2], combo[3])]
            ])
        k4_indices = np.array(k4_indices, dtype=np.int32)
        
        k5_indices = []
        for combo in combinations(range(n), 5):
            t_indices = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t_indices.append(idx_map[(combo[i], combo[j], combo[k])])
            k5_indices.append(t_indices)
        k5_indices = np.array(k5_indices, dtype=np.int32)
        
        return k4_indices, k5_indices

    def count_violations_fast(colors, k4_indices, k5_indices):
        # colors: np.array of 0/1
        # K4 violation: sum == 0
        sums4 = np.sum(colors[k4_indices], axis=1)
        v4 = np.sum(sums4 == 0)
        
        # K5 violation: sum == 10
        sums5 = np.sum(colors[k5_indices], axis=1)
        v5 = np.sum(sums5 == 10)
        
        return v4 + v5

    # Local Search
    def local_search(n, max_time=10.0):
        # Initialize randomly
        # Heuristic: Start with 50% red, 50% blue?
        # Or use the modular one as a start? The modular one is invalid for n>9 probably.
        # Random is safer.
        
        triples, idx_map, total_triples = get_triples(n)
        k4_indices, k5_indices = prepare_constraints(n)
        
        # Initial coloring
        # Let's try a few random starts if the first one doesn't work quickly.
        # But we have a tight time budget.
        
        # We will try to solve n=34.
        # If we don't succeed in 10s, we return None.
        
        start = time.time()
        
        # Try 3 random initializations
        for attempt in range(3):
            colors = np.random.randint(0, 2, total_triples).astype(np.int8)
            
            # Calculate initial violations
            v4, v5 = count_violations_fast(colors, k4_indices, k5_indices)
            current_violations = v4 + v5
            
            # If already valid, return
            if current_violations == 0:
                return colors.tolist()
            
            # Simulated Annealing / Greedy
            # We flip bits that reduce violations.
            # To speed up, we can pick a random violated constraint and flip one of its triples?
            # Or pick a random triple and see if flipping it helps?
            # Flipping a triple affects C(n-3, 1) K4s? No, it affects K4s containing that triple.
            # A triple (a,b,c) is in C(n-3, 1) K4s? No, a K4 is 4 vertices.
            # A triple is part of C(n-3, 1) K4s?
            # If we have vertices {a,b,c}, we can add any d from the remaining n-3 to form a K4.
            # So each triple is in n-3 K4s.
            # Similarly, each triple is in C(n-3, 2) K5s.
            
            # This dependency makes incremental updates complex but possible.
            # Given the time limit, a full re-calculation of violations after each flip might be too slow if we do many flips.
            # 3.2M ops per full count. If we do 1000 flips, that's 3.2B ops. Too slow in Python/NumPy overhead.
            
            # We need a faster heuristic or a better algorithm.
            
            # Alternative: Use a SAT solver? Not available.
            # Alternative: Use the known construction?
            # The prompt says "published construction has 34 points".
            # Can we find it online? No.
            # Can we derive it?
            # It might be based on a finite field or group.
            # Z_34? No, 34 is not prime.
            # Maybe Z_17 x Z_2?
            # Or maybe it's a specific known Ramsey graph.
            
            # Let's try a different strategy:
            # If local search is too slow, we might just have to return a smaller n that we CAN solve quickly.
            # What is the smallest n we can solve quickly with local search?
            # n=20? C(20,4)=4845, C(20,5)=15504. Total ~20k constraints.
            # Full count: 20k * 10 = 200k ops. Very fast.
            # We can do many iterations.
            # So we can likely solve n=20, 25, 30 quickly.
            # n=34 is the bottleneck.
            
            # Let's try to solve n=30 first, then 34?
            # The score is n. So n=30 is better than n=9.
            # Let's try to maximize n within the time limit.
            
            pass

    # Revised Strategy:
    # 1. Try n=34 with a very limited time (e.g., 5s). If valid, return it.
    # 2. Try n=33 with 5s.
    # 3. Try n=30 with 5s.
    # 4. Try n=25 with 5s.
    # 5. Return n=9 (seed) if all fail.
    
    # But wait, if I can solve n=30 easily, I should just do that.
    # Let's implement a robust local search that works for n up to 30 or 34.
    
    def solve_for_n(n, time_budget):
        triples, idx_map, total_triples = get_triples(n)
        k4_indices, k5_indices = prepare_constraints(n)
        
        start = time.time()
        
        # Try multiple random starts
        for attempt in range(10):
            if time.time() - start > time_budget:
                break
                
            # Heuristic initialization: 
            # Maybe start with a pattern? 
            # Let's just use random.
            colors = np.random.randint(0, 2, total_triples).astype(np.int8)
            
            v4, v5 = count_violations_fast(colors, k4_indices, k5_indices)
            current_violations = v4 + v5
            
            if current_violations == 0:
                return colors.tolist()
            
            # Greedy local search
            # We will try to flip bits that reduce violations.
            # To avoid full re-count, we can sample.
            # But let's try a simple greedy: 
            # Pick a random violated K4 or K5, flip a random bit in it, check if total violations decrease.
            # This is "random walk with drift".
            
            # However, checking total violations is expensive.
            # Let's assume we can afford ~100-500 full counts for n=30.
            # For n=34, maybe ~50 full counts.
            
            max_iterations = 200
            for i in range(max_iterations):
                if time.time() - start > time_budget:
                    break
                
                # Pick a random violated constraint
                # We need to know which ones are violated.
                # We can re-calculate sums to find them.
                sums4 = np.sum(colors[k4_indices], axis=1)
                v4_indices = np.where(sums4 == 0)[0]
                
                sums5 = np.sum(colors[k5_indices], axis=1)
                v5_indices = np.where(sums5 == 10)[0]
                
                if len(v4_indices) == 0 and len(v5_indices) == 0:
                    return colors.tolist()
                
                # Randomly choose between fixing a K4 or a K5
                # Weight by number of violations?
                if np.random.random() < len(v4_indices) / (len(v4_indices) + len(v5_indices)):
                    # Fix a K4
                    idx = v4_indices[np.random.randint(len(v4_indices))]
                    triple_idx = k4_indices[idx, np.random.randint(4)]
                else:
                    # Fix a K5
                    idx = v5_indices[np.random.randint(len(v5_indices))]
                    triple_idx = k5_indices[idx, np.random.randint(10)]
                
                # Flip the bit
                old_color = colors[triple_idx]
                colors[triple_idx] = 1 - old_color
                
                # Check if it improved
                new_v4, new_v5 = count_violations_fast(colors, k4_indices, k5_indices)
                new_violations = new_v4 + new_v5
                
                if new_violations < current_violations:
                    current_violations = new_violations
                else:
                    # Revert
                    colors[triple_idx] = old_color
                    # If stuck, try a random flip (simulated annealing step)
                    # Or just break and try a new start?
                    # Let's try a random flip 10% of the time to escape local minima
                    if np.random.random() < 0.1:
                        rand_triple = np.random.randint(total_triples)
                        colors[rand_triple] = 1 - colors[rand_triple]
                        # Don't check improvement for random flips, just accept and see next iter
                        # But we need to update current_violations?
                        # For simplicity, let's just re-calculate next iter.
                        # This is inefficient but okay for small n.
                        pass

        return None

    # Try n=34
    result = solve_for_n(34, 20.0)
    if result is not None:
        return {'n': 34, 'colors': result}
    
    # Try n=33
    if time.time() < deadline - 20:
        result = solve_for_n(33, 10.0)
        if result is not None:
            return {'n': 33, 'colors': result}
            
    # Try n=30
    if time.time() < deadline - 10:
        result = solve_for_n(30, 5.0)
        if result is not None:
            return {'n': 30, 'colors': result}
            
    # Try n=25
    if time.time() < deadline - 5:
        result = solve_for_n(25, 3.0)
        if result is not None:
            return {'n': 25, 'colors': result}
            
    # Fallback to seed
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-END
