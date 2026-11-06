import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs.
    # There are 15 ways to partition 6 items into 3 pairs.
    # Total configs: C(16,6) * 15
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    config_triples = [] # List of lists of triple indices
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # Generate 15 pairings
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            config_triples.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    def time_left():
        return deadline - time.time()

    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    def is_addable(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def get_blockers(t_idx, counts):
        """Returns list of config indices c where counts[c] >= 7"""
        blockers = []
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] >= 7:
                blockers.append(k_idx)
        return blockers

    def find_swap_partner(t_idx, active, counts):
        """
        Find an active triple t' that can be swapped with t_idx.
        Condition: t' must be in the intersection of all config_triples[c] 
        for c in get_blockers(t_idx, counts).
        """
        blockers = get_blockers(t_idx, counts)
        if not blockers:
            return None # Should be addable
        
        # Intersection of sets of triple indices
        # Start with the set of triples in the first blocker config
        common = set(config_triples[blockers[0]])
        for c in blockers[1:]:
            common.intersection_update(config_triples[c])
            
        if not common:
            return None
            
        # Check if any triple in 'common' is active
        for t_prime in common:
            if active[t_prime]:
                return t_prime
        return None

    def local_search(active, counts, current_size):
        improved = True
        while improved and time_left() > 2:
            improved = False
            
            # 1. Try to add any edge
            # Scan for addable edges. If found, add one and restart.
            addable_found = False
            for i in range(num_triples):
                if not active[i] and is_addable(i, counts):
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    improved = True
                    addable_found = True
                    break # Restart loop to check for more adds
            
            if addable_found:
                continue

            # 2. If no addable, try swaps
            # Pick a random inactive edge and try to find a swap partner
            inactive_edges = [i for i in range(num_triples) if not active[i]]
            if not inactive_edges:
                break # Full set (unlikely)
                
            # Try a few random swaps
            found_swap = False
            # We limit attempts per "step" to avoid getting stuck on one bad edge
            for _ in range(20):
                if time_left() < 1: break
                t_idx = random.choice(inactive_edges)
                t_prime = find_swap_partner(t_idx, active, counts)
                if t_prime is not None:
                    # Perform swap
                    # Remove t_prime
                    active[t_prime] = False
                    for k_idx in triple_to_k222s[t_prime]:
                        counts[k_idx] -= 1
                    
                    # Add t_idx
                    active[t_idx] = True
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    
                    improved = True
                    found_swap = True
                    break # Restart local search loop to find adds
                    
            if not found_swap:
                # No immediate swap found, break local search phase
                break

        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # ILS Loop
    # 1. Greedy Init
    active, counts, current_size = greedy_init()
    active, counts, current_size = local_search(active, counts, current_size)
    
    if current_size > best_size:
        best_size = current_size
        best_active = active[:]
        best_counts = counts[:]
        
    # 2. Iterated Local Search
    while time_left() > 5:
        # Perturb
        # Remove k random edges
        k = random.randint(5, 15)
        active_edges = [i for i in range(num_triples) if active[i]]
        if len(active_edges) < k:
            k = len(active_edges)
            
        for _ in range(k):
            if not active_edges: break
            t_idx = random.choice(active_edges)
            active_edges.remove(t_idx)
            active[t_idx] = False
            current_size -= 1
            for k_idx in triple_to_k222s[t_idx]:
                counts[k_idx] -= 1
                
        # Local Search on perturbed solution
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Final polish: ensure no addable edges remain
    if best_active:
        active = best_active[:]
        counts = best_counts[:]
        current_size = best_size
        # Run local search one last time to squeeze out any remaining gains
        active, counts, current_size = local_search(active, counts, current_size)
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Construct result from best_active
    if best_active is None:
        # Fallback
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
