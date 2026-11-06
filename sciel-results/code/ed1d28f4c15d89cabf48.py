"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 16
    rng = random.Random(42)
    
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute: for each triple, its pair set and mask
    triple_pairs = []
    triple_masks = []
    for t in all_triples:
        a, b, c = t
        pairs = frozenset({(min(a,b),max(a,b)), (min(a,c),max(a,c)), (min(b,c),max(b,c))})
        mask = (1 << a) | (1 << b) | (1 << c)
        triple_pairs.append(pairs)
        triple_masks.append(mask)
    
    # Precompute conflict graph: conflict[i] = set of triple indices that conflict with triple i
    conflict = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        for j in range(i+1, num_triples):
            if triple_pairs[i] & triple_pairs[j]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def independence_number_fast(triple_indices):
        """Compute independence number using bitmask representation."""
        if not triple_indices:
            return n
        masks = [triple_masks[i] for i in triple_indices]
        
        # Check from largest possible independent set size down
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained in this subset
                found = False
                for m in masks:
                    if (m & mask) == m:
                        found = True
                        break
                if not found:
                    return size
        return 0
    
    # Greedy construction: add triple that covers most uncovered pairs
    def greedy_construct():
        triples = []
        triple_set = set()
        used_pairs = set()
        
        # Build list of available triples
        available = list(range(num_triples))
        
        while available:
            best_idx = -1
            best_new_pairs = -1
            
            for idx in available:
                if idx in triple_set:
                    continue
                ps = triple_pairs[idx]
                new_pairs = ps - used_pairs
                new_count = len(new_pairs)
                if new_count > best_new_pairs:
                    best_new_pairs = new_count
                    best_idx = idx
            
            if best_idx == -1 or best_new_pairs <= 0:
                break
            
            triples.append(best_idx)
            triple_set.add(best_idx)
            used_pairs |= triple_pairs[best_idx]
            
            # Remove conflicting triples from available
            available = [i for i in available if i not in conflict[best_idx] and i != best_idx]
        
        return triples
    
    # Local search with conflict graph
    def local_search(triple_indices, time_limit):
        best = list(triple_indices)
        best_set = set(best)
        best_alpha = independence_number_fast(best)
        
        while time.time() < time_limit:
            improved = False
            
            # Try 1-swap: remove one, add one
            for i in best:
                if time.time() >= time_limit:
                    break
                candidate_set = best_set - {i}
                
                # Find best replacement
                for j in range(num_triples):
                    if j in candidate_set:
                        continue
                    if j in conflict[i] and i in best_set:
                        # j conflicts with i, but i is being removed
                        # Check if j conflicts with any remaining triple
                        if candidate_set & conflict[j]:
                            continue
                    else:
                        if candidate_set & conflict[j]:
                            continue
                    
                    new_candidate = list(candidate_set) + [j]
                    new_alpha = independence_number_fast(new_candidate)
                    if new_alpha < best_alpha:
                        best = new_candidate
                        best_set = set(best)
                        best_alpha = new_alpha
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-swap: remove two, add two
            if not improved:
                found_2swap = False
                for i in range(len(best)):
                    if time.time() >= time_limit:
                        break
                    for j in range(i+1, len(best)):
                        if time.time() >= time_limit:
                            break
                        candidate_set = best_set - {best[i], best[j]}
                        
                        # Find two replacements
                        for a in range(num_triples):
                            if a in candidate_set or a == best[i] or a == best[j]:
                                continue
                            if candidate_set & conflict[a]:
                                continue
                            temp_set = candidate_set | {a}
                            
                            for b in range(num_triples):
                                if b in temp_set or b == a:
                                    continue
                                if temp_set & conflict[b]:
                                    continue
                                if conflict[a] & {b}:
                                    continue
                                
                                new_candidate = list(candidate_set) + [a, b]
                                new_alpha = independence_number_fast(new_candidate)
                                if new_alpha < best_alpha:
                                    best = new_candidate
                                    best_set = set(best)
                                    best_alpha = new_alpha
                                    found_2swap = True
                                    break
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                
                if not found_2swap:
                    # Try adding a triple
                    added = False
                    for j in range(num_triples):
                        if j in best_set:
                            continue
                        if best_set & conflict[j]:
                            continue
                        new_candidate = best + [j]
                        new_alpha = independence_number_fast(new_candidate)
                        if new_alpha <= best_alpha:
                            best = new_candidate
                            best_set = set(best)
                            best_alpha = new_alpha
                            added = True
                            break
                    
                    if not added:
                        break
        
        return best, best_alpha
    
    # Initial greedy construction
    triples = greedy_construct()
    
    # Run local search
    triples, alpha = local_search(triples, deadline - 5)
    
    return [list(all_triples[t]) for t in triples]
