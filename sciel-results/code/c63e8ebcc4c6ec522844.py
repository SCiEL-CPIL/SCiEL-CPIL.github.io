import numpy as np
import random
import time
import math

def construct():
    start_time = time.time()
    
    def get_score(c):
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)

    def init_population(size=100):
        pop = []
        for _ in range(size):
            if random.random() < 0.7:
                c = [random.choice([-1, 1]) for _ in range(12)]
            else:
                c = [random.randint(-2, 2) for _ in range(12)]
                c = [1 if x == 0 else x for x in c]
            pop.append(np.array(c, dtype=int))
        return pop

    def evolve(pop, max_time=55):
        scored = [(get_score(c), c) for c in pop]
        scored.sort(key=lambda x: x[0])
        best_c = scored[0][1].copy()
        best_score = scored[0][0]
        
        while time.time() - start_time < max_time:
            half = len(pop) // 2
            survivors = [c for _, c in scored[:half]]
            
            new_pop = survivors.copy()
            while len(new_pop) < len(pop):
                p1, p2 = random.sample(survivors, 2)
                child = np.zeros(12, dtype=int)
                for i in range(12):
                    if random.random() < 0.5:
                        child[i] = p1[i]
                    else:
                        child[i] = p2[i]
                    
                    if random.random() < 0.2:
                        if random.random() < 0.8:
                            child[i] = random.choice([-3, -2, -1, 1, 2, 3])
                        else:
                            child[i] = random.randint(-5, 5)
                            if child[i] == 0:
                                child[i] = 1
                    if child[i] == 0:
                        child[i] = 1
                        
                new_pop.append(child)
                
            pop = new_pop
            scored = [(get_score(c), c) for c in pop]
            scored.sort(key=lambda x: x[0])
            
            if scored[0][0] < best_score:
                best_score = scored[0][0]
                best_c = scored[0][1].copy()
                
            if best_score <= 13:
                break

        c = best_c.copy()
        curr_score = best_score
        for _ in range(2000):
            if time.time() - start_time > 58:
                break
            idx = random.randint(0, 11)
            old_val = c[idx]
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue
            c[idx] = new_val
            new_score = get_score(c)
            if new_score < curr_score:
                curr_score = new_score
            else:
                c[idx] = old_val
            if curr_score < best_score:
                best_score = curr_score
                best_c = c.copy()
            if best_score <= 12:
                break

        return best_c, best_score

    pop = init_population(100)
    best_c, best_score = evolve(pop)
    
    if best_c is None:
        best_c = np.array([1]*12, dtype=int)
        
    return [int(x) for x in best_c]
