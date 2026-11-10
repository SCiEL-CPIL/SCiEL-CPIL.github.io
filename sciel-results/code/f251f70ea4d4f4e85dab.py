import random
import time
from itertools import combinations

# Global deadline
_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    def get_time_left():
        return 120.0 - (time.time() - _START_TIME)

    def solve(n, time_budget):
        # Generate all triples
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Generate all K4s and map them to their triples
        k4s = list(combinations(range(n), 4))
        triple_to_k4 = [[] for _ in range(len(triples))]
        k4_triples = []
        for k4 in k4s:
            a, b, c, d = k4
            t1 = triple_index[(a, b, c)]
            t2 = triple_index[(a, b, d)]
            t3 = triple_index[(a, c, d)]
            t4 = triple_index[(b, c, d)]
            k4_triples.append((t1, t2, t3, t4))
            triple_to_k4[t1].append(len(k4_triples)-1)
            triple_to_k4[t2].append(len(k4_triples)-1)
            triple_to_k4[t3].append(len(k4_triples)-1)
            triple_to_k4[t4].append(len(k4_triples)-1)
            
        # Generate all K5s and map them to their triples
        k5s = list(combinations(range(n), 5))
        triple_to_k5 = [[] for _ in range(len(triples))]
        k5_triples = []
        for k5 in k5s:
            t_idx = []
            for t in combinations(k5, 3):
                t_idx.append(triple_index[t])
            k5_triples.append(tuple(t_idx))
            for t in t_idx:
                triple_to_k5[t].append(len(k5_triples)-1)
                
        def count_bad_colors(colors):
            r4_count = 0
            b5_count = 0
            
            for k4_idx in range(len(k4s)):
                t1, t2, t3, t4 = k4_triples[k4_idx]
                if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
                    r4_count += 1
                    
            for k5_idx in range(len(k5s)):
                t_idx = k5_triples[k5_idx]
                all_blue = True
                for t in t_idx:
                    if colors[t] != 1:
                        all_blue = False
                        break
                if all_blue:
                    b5_count += 1
                    
            return r4_count, b5_count

        # Local Search
        attempts = 0
        while get_time_left() > 1.0 and attempts < 100:
            attempts += 1
            colors = [random.randint(0, 1) for _ in range(len(triples))]
            
            r4, b5 = count_bad_colors(colors)
            if r4 == 0 and b5 == 0:
                return colors, n
                
            max_iters = 5000
            for it in range(max_iters):
                if get_time_left() < 1.0:
                    break
                    
                # Pick a random triple
                t_idx = random.randint(0, len(triples) - 1)
                old_color = colors[t_idx]
                
                # Calculate delta
                delta_r4 = 0
                delta_b5 = 0
                
                # Check K4s containing t_idx
                for k4_idx in triple_to_k4[t_idx]:
                    t1, t2, t3, t4 = k4_triples[k4_idx]
                    reds_before = 0
                    if colors[t1] == 0: reds_before += 1
                    if colors[t2] == 0: reds_before += 1
                    if colors[t3] == 0: reds_before += 1
                    if colors[t4] == 0: reds_before += 1
                    
                    # t_idx is one of these
                    if old_color == 0:
                        reds_after = reds_before - 1
                    else:
                        reds_after = reds_before + 1
                        
                    if reds_before == 4 and reds_after < 4:
                        delta_r4 -= 1
                    elif reds_before < 4 and reds_after == 4:
                        delta_r4 += 1
                        
                # Check K5s containing t_idx
                for k5_idx in triple_to_k5[t_idx]:
                    t_idx_list = k5_triples[k5_idx]
                    blues_before = 0
                    for t in t_idx_list:
                        if colors[t] == 1:
                            blues_before += 1
                            
                    if old_color == 1:
                        blues_after = blues_before - 1
                    else:
                        blues_after = blues_before + 1
                        
                    if blues_before == 10 and blues_after < 10:
                        delta_b5 -= 1
                    elif blues_before < 10 and blues_after == 10:
                        delta_b5 += 1
                        
                # Accept if improvement
                if delta_r4 + delta_b5 < 0:
                    colors[t_idx] = 1 - old_color
                    r4 += delta_r4
                    b5 += delta_b5
                    if r4 == 0 and b5 == 0:
                        return colors, n
                else:
                    # Random walk with low probability to escape local minima
                    if random.random() < 0.05:
                        colors[t_idx] = 1 - old_color
                        # Recount
                        r4, b5 = count_bad_colors(colors)
                        if r4 == 0 and b5 == 0:
                            return colors, n

    # Try n=34
    res = solve(34, 100)
    if res:
        colors, n = res
        return {'n': n, 'colors': colors}
        
    # Fallbacks
    for n in [30, 25, 20]:
        if get_time_left() > 5:
             res = solve(n, 5)
             if res:
                 colors, n = res
                 return {'n': n, 'colors': colors}
                 
    # Final fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
