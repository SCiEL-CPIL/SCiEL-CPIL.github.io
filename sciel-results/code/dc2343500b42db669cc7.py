"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    
    def get_pairs(triple):
        a, b, c = triple
        return frozenset({(a, b), (a, c), (b, c)})
    
    def independence_number(triples):
        """Find largest subset of points containing no triple."""
        triple_set = set(triples)
        best = 0
        # Check all subsets by size, largest first
        for size in range(n, best, -1):
            found = False
            for subset in itertools.combinations(all_points, size):
                s = set(subset)
                has_triple = False
                for t in triple_set:
                    if t[0] in s and t[1] in s and t[2] in s:
                        has_triple = True
                        break
                if not has_triple:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        used_pairs = set()
        for t in triples:
            a, b, c = t
            pairs = {(a, b), (a, c), (b, c)}
            for p in pairs:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True
    
    def greedy_construct(seed=None):
        """Greedy construction with randomization."""
        if seed is not None:
            random.seed(seed)
        
        triples = []
        used_pairs = set()
        point_coverage = [0] * n
        
        # Build a list of candidate triples, prioritizing under-covered points
        candidates = list(all_triples)
        random.shuffle(candidates)
        
        for triple in candidates:
            a, b, c = triple
            # Check linearity
            pairs = {(a, b), (a, c), (b, c)}
            if any(p in used_pairs for p in pairs):
                continue
            # Check if this triple adds value
            # Prefer triples that cover under-covered points
            triples.append(triple)
            used_pairs.update(pairs)
            point_coverage[a] += 1
            point_coverage[b] += 1
            point_coverage[c] += 1
            
            # Stop if we've added enough triples
            if len(triples) >= 40:
                break
        
        return triples
    
    def local_search(triples, max_iters=500):
        """Improve independence number via local search."""
        triples = list(triples)
        best_indep = independence_number(triples)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try removing a random triple and replacing with a better one
            if triples:
                idx = random.randrange(len(triples))
                old_triple = triples[idx]
                triples.remove(old_triple)
                
                # Find a replacement that improves independence number
                best_new_indep = best_indep
                best_replacement = None
                
                # Try adding back the old triple or finding a better one
                candidates = []
                for t in all_triples:
                    if t == old_triple:
                        continue
                    # Check linearity with current triples
                    a, b, c = t
                    pairs = {(a, b), (a, c), (b, c)}
                    valid = True
                    for ot in triples:
                        oa, ob, oc = ot
                        shared = set((a, b, c)) & set((oa, ob, oc))
                        if len(shared) > 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(t)
                
                if not candidates:
                    triples.append(old_triple)
                    continue
                
                # Try a few random candidates
                random.shuffle(candidates)
                for t in candidates[:20]:
                    test_triples = triples + [t]
                    test_indep = independence_number(test_triples)
                    if test_indep < best_new_indep:
                        best_new_indep = test_indep
                        best_replacement = t
                
                if best_replacement is not None:
                    triples.append(best_replacement)
                    best_indep = best_new_indep
                    improved = True
                else:
                    triples.append(old_triple)
            
            if not improved:
                break
        
        return triples, best_indep
    
    # Run multiple greedy constructions and local searches
    best_triples = None
    best_indep = n + 1
    
    # First, try a deterministic greedy approach
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        triples = greedy_construct(seed)
        
        if not triples:
            continue
        
        # Quick check of independence number
        indep = independence_number(triples)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = list(triples)
        
        # Try local search if we have time
        if time.time() < deadline - 30:
            improved_triples, improved_indep = local_search(triples, max_iters=100)
            if improved_indep < best_indep:
                best_indep = improved_indep
                best_triples = improved_triples
    
    # If we haven't found anything good, use a simple construction
    if best_triples is None:
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
