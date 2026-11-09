import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    def evaluate(S):
        """Count distinct d > 0 such that a, a+d, a+2d are all in S."""
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def random_set():
        return set(random.sample(range(1, 513), 32))
    
    def structured_set():
        """Create a set with block structure to encourage many progressions."""
        base = random.randint(1, 300)
        s = set()
        configs = [
            (4, 8),
            (8, 4),
            (16, 2),
            (32, 1),
        ]
        nblocks, bsize = random.choice(configs)
        if nblocks * bsize > 32:
            bsize = 32 // nblocks
        spacing = random.randint(max(1, (512 - base) // max(nblocks, 1) - bsize), max(2, (512 - base) // max(nblocks, 1)))
        for i in range(nblocks):
            start = base + i * spacing
            for k in range(bsize):
                val = start + k
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    def greedy_init():
        """Greedy construction: add elements that maximize new progressions."""
        s = set()
        s.add(random.randint(1, 512))
        for _ in range(31):
            best_elem = None
            best_score = -1
            candidates = random.sample(range(1, 513), min(200, 512))
            for cand in candidates:
                if cand in s:
                    continue
                new_s = s | {cand}
                sc = evaluate(new_s)
                if sc > best_score or (sc == best_score and random.random() < 0.1):
                    best_score = sc
                    best_elem = cand
            if best_elem is not None:
                s.add(best_elem)
            else:
                while True:
                    val = random.randint(1, 512)
                    if val not in s:
                        s.add(val)
                        break
        return s
    
    # Generate diverse initial candidates
    candidates = [
        set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48]),  # Known good
        set(range(1, 33)),
        random_set(),
        random_set(),
        structured_set(),
        structured_set(),
        greedy_init(),
    ]
    
    # Evaluate all candidates
    scored = []
    for c in candidates:
        if len(c) == 32:
            sc = evaluate(c)
            scored.append((sc, c))
    
    scored.sort(key=lambda x: -x[0])
    
    best_score = scored[0][0]
    best_set = scored[0][1].copy()
    
    # Simulated Annealing with multi-start
    def simulated_annealing(initial_set, max_iters=5000):
        """SA with temperature cooling and diverse moves."""
        current = set(initial_set)
        current_score = evaluate(current)
        best = set(current)
        best_sc = current_score
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        for it in range(max_iters):
            if time.time() - start_time > timeout - 3.0:
                break
            
            # Choose move type
            r = random.random()
            if r < 0.5:
                # Single swap
                new_set = set(current)
                to_remove = random.choice(list(new_set))
                new_set.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in new_set:
                        new_set.add(to_add)
                        break
            elif r < 0.8:
                # Two swap
                new_set = set(current)
                elems = list(new_set)
                idx1 = random.randint(0, 31)
                idx2 = random.randint(0, 31)
                if idx1 == idx2:
                    continue
                new_set.discard(elems[idx1])
                new_set.discard(elems[idx2])
                for _ in range(2):
                    while True:
                        to_add = random.randint(1, 512)
                        if to_add not in new_set:
                            new_set.add(to_add)
                            break
            else:
                # Three swap
                new_set = set(current)
                elems = list(new_set)
                idxs = random.sample(range(32), 3)
                for idx in idxs:
                    new_set.discard(elems[idx])
                for _ in range(3):
                    while True:
                        to_add = random.randint(1, 512)
                        if to_add not in new_set:
                            new_set.add(to_add)
                            break
            
            new_score = evaluate(new_set)
            delta = new_score - current_score
            
            if delta > 0:
                current = new_set
                current_score = new_score
                if new_score > best_sc:
                    best_sc = new_score
                    best = set(new_set)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(delta / T):
                    current = new_set
                    current_score = new_score
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best, best_sc
    
    # Multi-start SA
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 5.0:
            break
        
        # Choose initial point: mix of best found and random
        if restart == 0:
            init = best_set
        elif restart < 5:
            # Perturb best
            init = set(best_set)
            elems = list(init)
            n_mut = random.randint(2, 5)
            for _ in range(n_mut):
                to_remove = random.choice(list(init))
                init.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in init:
                        init.add(to_add)
                        break
        else:
            # Random or structured
            if random.random() < 0.5:
                init = random_set()
            else:
                init = structured_set()
        
        sa_best, sa_score = simulated_annealing(init, max_iters=3000)
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_best
    
    return list(best_set)
