import time
import random

def _count_comparable(sets_list):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets_list)
    # Convert to frozensets for subset checks
    fsets = [frozenset(s) for s in sets_list]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if fsets[i] < fsets[j]:
                count += 1
            elif fsets[j] < fsets[i]:
                count += 1
    return count

def _greedy_local_search(sets_list, deadline):
    """Perform greedy local search: try replacing each set with any other subset."""
    n = len(sets_list)
    all_masks = list(range(1 << 12))  # All 4096 possible subsets
    best_sets = list(sets_list)
    best_score = _count_comparable(best_sets)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            current_mask = 0
            for v in best_sets[i]:
                current_mask |= (1 << v)
            best_new_mask = current_mask
            best_new_score = best_score
            for mask in all_masks:
                if mask == current_mask:
                    continue
                # Try replacing set i with this mask
                new_sets = list(best_sets)
                new_sets[i] = [b for b in range(12) if (mask >> b) & 1]
                # Quick check: must be distinct from all others
                new_fset = frozenset(new_sets[i])
                valid = True
                for j in range(n):
                    if j != i and frozenset(new_sets[j]) == new_fset:
                        valid = False
                        break
                if not valid:
                    continue
                score = _count_comparable(new_sets)
                if score > best_new_score:
                    best_new_score = score
                    best_new_mask = mask
            if best_new_mask != current_mask:
                best_sets[i] = [b for b in range(12) if (best_new_mask >> b) & 1]
                best_score = best_new_score
                improved = True
    
    return best_sets, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline (10s safety margin)
    
    # Initial solution: chain of 13 sets + 27 random sets
    chain = []
    for size in range(13):
        s = list(range(size))
        chain.append(s)
    
    # Add 27 more sets - try to extend the chain structure
    # Use sets that are subsets of the top of the chain or supersets of the bottom
    random.seed(42)
    extra_sets = []
    used_masks = set()
    for s in chain:
        used_masks.add(frozenset(s))
    
    # Generate candidate sets that might be comparable
    candidates = []
    # All subsets of {0,...,11}
    for mask in range(4096):
        s = frozenset([b for b in range(12) if (mask >> b) & 1])
        if s not in used_masks:
            candidates.append(s)
    
    # Greedily add 27 sets that maximize comparable pairs
    current_sets = list(chain)
    for _ in range(27):
        if time.time() > deadline:
            break
        best_candidate = None
        best_score = -1
        # Sample a subset of candidates for speed
        sample_size = min(len(candidates), 500)
        sample = random.sample(candidates, sample_size)
        for cand in sample:
            new_sets = current_sets + [list(cand)]
            score = _count_comparable(new_sets)
            if score > best_score:
                best_score = score
                best_candidate = cand
        if best_candidate is not None:
            current_sets.append(list(best_candidate))
            candidates.remove(best_candidate)
    
    # Run greedy local search to polish
    final_sets, final_score = _greedy_local_search(current_sets, deadline)
    
    # Ensure proper types
    result = []
    for s in final_sets:
        result.append([int(x) for x in s])
    
    return result
