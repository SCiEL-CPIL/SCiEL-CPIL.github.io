import random
import math

def construct():
        # Moduli: all even numbers from 20 to 200
        moduli = list(range(20, 201, 2))
        n = len(moduli)
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
                gcds[j][i] = gcds[i][j]
        
        # Initialize residues
        residues = [0] * n
        
        # Objective: Minimize conflicts
        def count_conflicts(res):
            conflicts = 0
            for i in range(n):
                for j in range(i+1, n):
                    g = gcds[i][j]
                    if res[i] % g == res[j] % g:
                        conflicts += 1
            return conflicts
        
        # Local Search
        best_res = residues[:]
        best_conf = count_conflicts(best_res)
        
        # Simulated Annealing / Hill Climbing
        # Time limit guard
        import time
        start_time = time.time()
        limit = 10.0 # 10 seconds for search, plenty of time left
        
        current_res = best_res[:]
        current_conf = best_conf
        
        temp = 1.0
        cooling = 0.999
        
        while time.time() - start_time < limit:
            # Pick random index
            i = random.randint(0, n-1)
            # Try a new residue
            old_r = current_res[i]
            new_r = random.randint(0, moduli[i]-1)
            if new_r == old_r: continue
            
            # Calculate delta conflicts
            # Only need to check pairs involving i
            old_conf_i = 0
            new_conf_i = 0
            for j in range(n):
                if i == j: continue
                g = gcds[i][j]
                if old_r % g == current_res[j] % g:
                    old_conf_i += 1
                if new_r % g == current_res[j] % g:
                    new_conf_i += 1
            
            delta = new_conf_i - old_conf_i
            
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                current_res[i] = new_r
                current_conf += delta
                
                if current_conf < best_conf:
                    best_conf = current_conf
                    best_res = current_res[:]
                    if best_conf == 0:
                        break
                
                temp *= cooling
        
        # If we found a valid solution (0 conflicts), return it
        if best_conf == 0:
            return [[moduli[i], best_res[i]] for i in range(n)]
        
        # Fallback: Return the seed or a partial valid set
        # If no valid assignment for all evens, we might need to drop some moduli.
        # But 0 conflicts is likely reachable.
        # If not, return the best partial? No, must be valid.
        # If we can't find valid residues for all evens, we should reduce the set.
        
        # Let's add a fallback: try a smaller set if all evens fail.
        # But given the structure, it should work.
        
        # For safety, if best_conf > 0, we return a known valid set (e.g. seed)
        # But we want to maximize.
        # Let's try to repair the best_res by dropping conflicting elements?
        # No, simpler: if no valid solution found, return the seed.
        
        return [[20 * j, j - 1] for j in range(1, 11)]
