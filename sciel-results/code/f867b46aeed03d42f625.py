import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    
    def get_triples(n):
        return list(combinations(range(n), 3))

    def triple_index(a, b, c, n):
        # Precomputed map is better
        pass

    # Precompute triples and indices for n=10, 11
    # We will try n=10 first
    
    def solve(n, time_limit):
        triples = list(combinations(range(n), 3))
        t_map = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # Precompute for each 4-set, the 4 triples
        k4s = []
        for quad in combinations(range(n), 4):
            k4s.append([t_map[(quad[0], quad[1], quad[2])], 
                        t_map[(quad[0], quad[1], quad[3])],
                        t_map[(quad[0], quad[2], quad[3])],
                        t_map[(quad[1], quad[2], quad[3])]])
                        
        # Precompute for each 5-set, the 10 triples
        k5s = []
        for penta in combinations(range(n), 5):
            k5s.append([t_map[t] for t in combinations(penta, 3)])

        colors = [-1] * num_triples
        
        def is_consistent():
            # Check all K4s
            for k4 in k4s:
                c0, c1, c2, c3 = colors[k4[0]], colors[k4[1]], colors[k4[2]], colors[k4[3]]
                # If all 4 are 0, invalid
                if c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0:
                    return False
            # Check all K5s
            for k5 in k5s:
                # If all 10 are 1, invalid
                # Optimization: if any is 0, skip
                all_blue = True
                for idx in k5:
                    if colors[idx] == 0:
                        all_blue = False
                        break
                    if colors[idx] == -1:
                        # Not decided yet, can't fail now
                        all_blue = False 
                        break
                if all_blue:
                    return False
            return True

        def backtrack(i):
            if time.time() - start_time > time_limit:
                return False
            
            if i == num_triples:
                return is_consistent()
            
            # Try 0 then 1
            for val in [0, 1]:
                colors[i] = val
                if is_consistent():
                    if backtrack(i + 1):
                        return True
                colors[i] = -1
            
            return False

        # Randomize order of variables? Or just lexicographic?
        # Lexicographic might be slow.
        # Let's try a simple random restart with greedy assignment?
        # For n=10, simple backtracking might work.
        
        return backtrack(0)

    # Try n=10
    if solve(10, 5):
        # Need to return colors in the required format
        # The solver uses a local 'colors' list. 
        # I need to capture it.
        pass
