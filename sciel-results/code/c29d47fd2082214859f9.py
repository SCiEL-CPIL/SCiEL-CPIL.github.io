import time
import random
import itertools
import numpy as np

def calc_alpha(triples):
    if not triples:
        return 16
    
    n = 16
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    for k in range(16, 0, -1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            
            if is_indep:
                return k
                
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_triples = []
    best_alpha = 16
    best_density = 0
    
    def is_linear(triple, existing_triples):
        a, b, c = triple
        new_set = {a, b, c}
        for t in existing_triples:
            old_set = {t[0], t[1], t[2]}
            if len(new_set & old_set) > 1:
                return False
        return True
        
    def try_add_random_triple(triples, used_pairs):
        # Pick a random pair that is not used
        # We can pick random a, b and check if (a,b) in used_pairs
        # If not, try to find c
        for _ in range(10): # Try a few random pairs
            a, b = random.sample(range(16), 2)
            if (a, b) in used_pairs or (b, a) in used_pairs:
                continue
                
            candidates = [c for c in range(16) if c != a and c != b]
            random.shuffle(candidates)
            
            for c in candidates:
                if (a, c) in used_pairs or (c, a) in used_pairs:
                    continue
                if (b, c) in used_pairs or (c, b) in used_pairs:
                    continue
                    
                new_triple = (a, b, c)
                if is_linear(new_triple, triples):
                    return list(new_triple)
        return None

    def generate_initial_graph():
        triples = []
        used_pairs = set()
        attempts = 0
        while attempts < 1000:
            attempts += 1
            t = try_add_random_triple(triples, used_pairs)
            if t:
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b)); used_pairs.add((b, a))
                used_pairs.add((a, c)); used_pairs.add((c, a))
                used_pairs.add((b, c)); used_pairs.add((c, b))
        return triples, used_pairs

    triples, used_pairs = generate_initial_graph()
    current_density = len(triples)
    current_alpha = calc_alpha(triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = [list(t) for t in triples]
        best_density = current_density
        
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if random.random() < 0.8:
            new_triple = try_add_random_triple(triples, used_pairs)
            if new_triple:
                a, b, c = new_triple
                triples.append(new_triple)
                used_pairs.add((a, b)); used_pairs.add((b, a))
                used_pairs.add((a, c)); used_pairs.add((c, a))
                used_pairs.add((b, c)); used_pairs.add((c, b))
                
                current_density = len(triples)
                
                if current_density > best_density:
                    current_alpha = calc_alpha(triples)
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = [list(t) for t in triples]
                        best_density = current_density
        else:
            if triples:
                idx = random.randint(0, len(triples) - 1)
                t_to_remove = triples[idx]
                a, b, c = t_to_remove
                triples.pop(idx)
                used_pairs.discard((a, b)); used_pairs.discard((b, a))
                used_pairs.discard((a, c)); used_pairs.discard((c, a))
                used_pairs.discard((b, c)); used_pairs.discard((c, b))
                current_density = len(triples)
                
        temperature *= cooling
        if temperature < 0.1:
            temperature = 1.0

    return best_triples
