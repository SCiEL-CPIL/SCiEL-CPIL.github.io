import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for 60s limit

    def get_products(seq):
        """Get all block products for a sequence."""
        n = len(seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        """Check if sequence is valid (strictly increasing, in [2,128], unique block products)."""
        n = len(seq)
        if n == 0:
            return True
        # Check strictly increasing
        for i in range(1, n):
            if seq[i] <= seq[i-1]:
                return False
        # Check range
        for x in seq:
            if x < 2 or x > 128:
                return False
        # Check unique block products
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def try_insert(seq, val):
        """Try to insert val into sorted seq, return new seq if valid, else None."""
        pos = 0
        while pos < len(seq) and seq[pos] < val:
            pos += 1
        if pos < len(seq) and seq[pos] == val:
            return None
        new_seq = seq[:pos] + [val] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None

    def greedy_construct():
        """Greedy construction from 2 to 128."""
        values = []
        for x in range(2, 129):
            new_seq = try_insert(values, x)
            if new_seq is not None:
                values = new_seq
        return values

    def polish(seq, max_iter=100):
        """Polish by trying to insert excluded elements."""
        all_values = set(range(2, 129))
        current = seq[:]
        for _ in range(max_iter):
            improved = False
            excluded = sorted(all_values - set(current))
            for e in excluded:
                new_seq = try_insert(current, e)
                if new_seq is not None:
                    current = new_seq
                    improved = True
                    break
            if not improved:
                break
        return current

    def count_collisions(seq):
        """Count number of product collisions in seq."""
        n = len(seq)
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return len(prods) - len(set(prods))

    def simulated_annealing(init_seq, T_init=20.0, cooling_rate=0.995, max_steps=5000):
        """Simulated annealing with collision-count surrogate."""
        current = init_seq[:]
        current_collisions = count_collisions(current)
        best = current[:]
        best_score = len(current) - current_collisions * 10
        T = T_init
        
        all_values = set(range(2, 129))
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Generate neighbor: swap one element with an excluded one
            if len(current) == 0:
                break
            
            i = random.randint(0, len(current) - 1)
            excluded = list(all_values - set(current))
            if not excluded:
                break
            e = random.choice(excluded)
            
            neighbor = current[:]
            neighbor[i] = e
            neighbor.sort()
            
            # Check if valid (strictly increasing is guaranteed by sort, range guaranteed)
            if not is_valid(neighbor):
                # Try to make it valid by removing duplicates or out-of-range
                # Actually, since we're swapping with an excluded value, it should be fine
                # But let's just skip invalid neighbors
                continue
            
            neighbor_collisions = count_collisions(neighbor)
            neighbor_score = len(neighbor) - neighbor_collisions * 10
            
            delta = neighbor_score - (len(current) - current_collisions * 10)
            
            if delta > 0 or random.random() < min(1.0, math.exp(delta / T) if T > 0 else 0.0):
                current = neighbor
                current_collisions = neighbor_collisions
                
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best = list(neighbor)
            
            # Cool down
            T *= cooling_rate
            
            # Periodic polish
            if step % 500 == 0 and time.time() < deadline:
                polished = polish(current)
                p_score = len(polished) - count_collisions(polished) * 10
                if p_score > best_score:
                    best_score = p_score
                    best = list(polished)
                if len(polished) > len(current):
                    current = polished
                    current_collisions = count_collisions(current)
        
        return best

    # 1. Initial Greedy Construction
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # 2. Polish the initial solution
    polished = polish(best_seq)
    if len(polished) >= best_len:
        best_seq = polished
        best_len = len(best_seq)
    
    # 3. Simulated Annealing
    if time.time() < deadline:
        sa_result = simulated_annealing(best_seq, 20.0)
        if len(sa_result) > best_len:
            best_seq = sa_result
            best_len = len(best_seq)
    
    # 4. Final polish
    if time.time() < deadline:
        final_polished = polish(best_seq)
        if len(final_polished) >= best_len:
            best_seq = final_polished
            best_len = len(best_seq)
    
    return best_seq
