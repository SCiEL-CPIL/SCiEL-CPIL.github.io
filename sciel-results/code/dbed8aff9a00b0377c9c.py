import itertools
import random
import time

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2)s
    k_list = []
    triple_to_ks = [[] for _ in range(num_triples)]
    k_to_triples = []  # List of list of triple indices for each K
    
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3))
    ]
    
    for subset in itertools.combinations(range(n), 6):
        for m in matchings:
            P1 = (subset[m[0][0]], subset[m[0][1]])
            P2 = (subset[m[1][0]], subset[m[1][1]])
            P3 = (subset[m[2][0]], subset[m[2][1]])
            
            k_triples = []
            for x in P1:
                for y in P2:
                    for z in P3:
                        t = tuple(sorted((x, y, z)))
                        k_triples.append(t)
            
            # Convert to indices
            k_triple_indices = [triple_to_idx[t] for t in k_triples]
            k_list.append(k_triple_indices)
            k_to_triples.append(k_triple_indices)
            
            for idx in k_triple_indices:
                triple_to_ks[idx].append(len(k_list) - 1)

    num_ks = len(k_list)

    def run_greedy_lsf():
        counts = [0] * num_ks
        blocked = [0] * num_triples
        used = [False] * num_triples
        # score[t] = number of Ks containing t that have count > 0
        scores = [0] * num_triples
        available = list(range(num_triples))
        result = []
        
        while available:
            # Find valid candidates with min score
            min_score = float('inf')
            candidates = []
            for i in available:
                if blocked[i] == 0:
                    if scores[i] < min_score:
                        min_score = scores[i]
                        candidates = [i]
                    elif scores[i] == min_score:
                        candidates.append(i)
            
            if not candidates:
                break
            
            # Random tie-break
            chosen_idx = random.choice(candidates)
            result.append(all_triples[chosen_idx])
            used[chosen_idx] = True
            available.remove(chosen_idx)
            
            # Update counts and scores
            for k_idx in triple_to_ks[chosen_idx]:
                old_count = counts[k_idx]
                counts[k_idx] += 1
                
                # If this K was empty (0), it becomes active (1)
                # So all other triples in this K get their score incremented
                if old_count == 0:
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx]:
                            scores[t_idx] += 1
                
                # If this K becomes saturated (7), block the remaining triple
                if counts[k_idx] == 7:
                    # Find the remaining triple
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx]:
                            blocked[t_idx] += 1
                            break

        return result

    best_result = []
    start_time = time.time()
    
    # Run as many restarts as possible
    while time.time() - start_time < 110:
        res = run_greedy_lsf()
        if len(res) > len(best_result):
            best_result = res

    return best_result
