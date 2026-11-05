import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def search(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        sets4 = []
        for quad in combinations(range(n), 4):
            edges = [triple_to_idx[tri] for tri in combinations(quad, 3)]
            sets4.append(edges)
            
        sets5 = []
        for quint in combinations(range(n), 5):
            edges = [triple_to_idx[tri] for tri in combinations(quint, 3)]
            sets5.append(edges)
            
        edge_to_sets4 = [[] for _ in range(num_triples)]
        for i, edges in enumerate(sets4):
            for e in edges:
                edge_to_sets4[e].append(i)
                
        edge_to_sets5 = [[] for _ in range(num_triples)]
        for i, edges in enumerate(sets5):
            for e in edges:
                edge_to_sets5[e].append(i)

        def count_bad(colors):
            bad4 = sum(1 for edges in sets4 if all(colors[e] == 0 for e in edges))
            bad5 = sum(1 for edges in sets5 if all(colors[e] == 1 for e in edges))
            return bad4, bad5

        # Try a few initializations
        inits = []
        inits.append([random.randint(0,1) for _ in range(num_triples)])
        inits.append([0 if (a+b+c)%3==0 else 1 for a,b,c in triples])
        
        for init_colors in inits:
            colors = init_colors[:]
            bad4, bad5 = count_bad(colors)
            if bad4 == 0 and bad5 == 0:
                return {'n': n, 'colors': colors}
                
            T = 1.0
            min_T = 0.01
            cooling = 0.9995
            
            while time.time() < deadline:
                if bad4 == 0 and bad5 == 0:
                    return {'n': n, 'colors': colors}
                    
                e = random.randint(0, num_triples - 1)
                old_c = colors[e]
                new_c = 1 - old_c
                
                delta_bad4 = 0
                for s_idx in edge_to_sets4[e]:
                    edges = sets4[s_idx]
                    other_red = 0
                    for oe in edges:
                        if oe != e:
                            if colors[oe] == 0:
                                other_red += 1
                            else:
                                break
                    if other_red == 3:
                        if new_c == 0:
                            delta_bad4 += 1
                        else:
                            delta_bad4 -= 1
                            
                delta_bad5 = 0
                for s_idx in edge_to_sets5[e]:
                    edges = sets5[s_idx]
                    other_blue = 0
                    for oe in edges:
                        if oe != e:
                            if colors[oe] == 1:
                                other_blue += 1
                            else:
                                break
                    if other_blue == 9:
                        if new_c == 1:
                            delta_bad5 += 1
                        else:
                            delta_bad5 -= 1
                            
                delta = delta_bad4 + delta_bad5
                
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
                if accept:
                    colors[e] = new_c
                    bad4 += delta_bad4
                    bad5 += delta_bad5
                    
                T *= cooling
                if T < min_T:
                    T = min_T
                    
                if time.time() > deadline - 5:
                    break
                    
        return None

    # Try from 34 down to 1
    for n in range(34, 0, -1):
        res = search(n)
        if res:
            return res
            
    # Fallback
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
