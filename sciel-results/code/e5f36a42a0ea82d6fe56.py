import itertools
import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute transversals as bitmasks
    # A transversal is defined by 3 disjoint pairs covering 6 points.
    # The 8 transversal triples are all combinations of picking one from each pair.
    transversal_masks = []
    
    # Generate all sets of 6 points
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings of these 6 points
        # We can do this by fixing the partner of sp[0] to sp[1], sp[2], etc.
        # But a simpler robust way for n=6 is to generate all permutations and deduplicate
        # Or use a recursive generator.
        
        def gen_matchings(pts, depth=0, current=None):
            if current is None:
                current = []
            if depth == len(pts):
                if current:
                    yield tuple(current)
                return
            # Find the first unassigned point
            # Since we are building pairs, we just pick the first available
            # But 'pts' is fixed size 6. 
            # Let's use a standard algorithm for perfect matchings on 6 nodes.
            pass
            
        # Simpler: iterate over all ways to partition 6 items into 3 pairs.
        # There are 15 such partitions.
        # We can generate them by taking the first element, pairing it with one of the 5 others,
        # then recursively pairing the remaining 4.
        
        def get_matchings(pts):
            if len(pts) == 0:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                partner = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for sub_matchings in get_matchings(new_rest):
                    yield [(first, partner)] + sub_matchings
        
        for matching in get_matchings(sp):
            # matching is a list of 3 pairs: (a,b), (c,d), (e,f)
            (a, b), (c, d), (e, f) = matching
            # The 8 transversals are:
            # (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f)
            trans_triples = [
                tuple(sorted((a, c, e))),
                tuple(sorted((a, c, f))),
                tuple(sorted((a, d, e))),
                tuple(sorted((a, d, f))),
                tuple(sorted((b, c, e))),
                tuple(sorted((b, c, f))),
                tuple(sorted((b, d, e))),
                tuple(sorted((b, d, f)))
            ]
            mask = 0
            for t in trans_triples:
                idx = triple_to_idx[t]
                mask |= (1 << idx)
            transversal_masks.append(mask)
            
    num_transversals = len(transversal_masks)
    
    # For each triple, we might want to know which transversals it belongs to for re-adding logic
    # But for simple WalkSAT, we just check conflicts.
    
    # Local Search / WalkSAT
    def solve(seed):
        rng = random.Random(seed)
        # Start with all triples
        current_mask = (1 << num_triples) - 1
        
        # Calculate initial conflicts
        # We need to efficiently find conflicts.
        # Since num_transversals is small (~1400?), we can just iterate.
        # C(16,6) = 8006. 15 matchings each. 120090 transversals.
        # This is too many to iterate every step if we do it naively in Python loops.
        
        # Optimization:
        # Instead of checking all transversals every step, we can maintain a list of active transversals?
        # Or just rely on the fact that we start with many conflicts and remove them.
        
        # Let's estimate the cost.
        # 120,000 transversals.
        # Checking `if (mask & t_mask) == t_mask` is fast in C.
        # 120,000 checks per iteration.
        # If we do 10,000 iterations, that's 1.2 billion ops. Too slow in Python.
        
        # We need a better conflict detection or a different strategy.
        
        # Strategy: 
        # 1. Greedy removal to get a valid solution quickly.
        # 2. Local search to improve.
        
        # Greedy Phase:
        # While there is a conflict, remove a random triple from a random conflict.
        # To speed up, we can pre-compute for each triple, the list of transversals it is in.
        
        triple_to_transversals = [[] for _ in range(num_triples)]
        for ti, t_mask in enumerate(transversal_masks):
            # Extract set bits
            m = t_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                triple_to_transversals[idx].append(ti)
                m ^= lsb
        
        # Greedy
        conflicts = set()
        # Initialize conflicts
        for ti, t_mask in enumerate(transversal_masks):
            if (current_mask & t_mask) == t_mask:
                conflicts.add(ti)
        
        while conflicts:
            if time.time() > deadline - 2:
                break
            # Pick a random conflict
            ti = rng.choice(list(conflicts))
            t_mask = transversal_masks[ti]
            # Pick a random triple from this conflict to remove
            # Get the indices of the 8 triples
            m = t_mask
            indices = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                indices.append(idx)
                m ^= lsb
            
            # Strategy: Remove the triple that is in the most other conflicts?
            # Or just random? Random is faster.
            idx_to_remove = rng.choice(indices)
            
            # Remove it
            current_mask &= ~(1 << idx_to_remove)
            
            # Update conflicts
            # The removed triple was in `triple_to_transversals[idx_to_remove]`.
            # These transversals might no longer be conflicts.
            for affected_ti in triple_to_transversals[idx_to_remove]:
                if affected_ti in conflicts:
                    # Check if it's still a conflict
                    if (current_mask & transversal_masks[affected_ti]) == transversal_masks[affected_ti]:
                        pass # Still a conflict
                    else:
                        conflicts.discard(affected_ti)
            
            # No new conflicts are created by removal.
            
        # Local Search Phase: Try to add back triples
        # We have a valid solution. Try to add triples one by one if possible.
        # Or use a hill-climbing approach: flip a bit from 0 to 1 if it doesn't create a conflict.
        
        # Better: Use a simulated annealing or random restart WalkSAT style.
        # Given time constraints, a simple "add if possible" pass might be enough improvement.
        
        # Let's do a few passes of adding triples
        for _ in range(5): # 5 passes
            if time.time() > deadline - 5:
                break
            # Shuffle order of triples to add
            removed_indices = [i for i in range(num_triples) if not (current_mask & (1 << i))]
            rng.shuffle(removed_indices)
            
            for idx in removed_indices:
                if time.time() > deadline - 5:
                    break
                # Check if adding this triple creates a conflict
                # It creates a conflict if there is a transversal containing idx
                # where all other 7 triples are present.
                can_add = True
                for ti in triple_to_transversals[idx]:
                    t_mask = transversal_masks[ti]
                    # Check if (current_mask | (1<<idx)) & t_mask == t_mask
                    # This is equivalent to: (current_mask & t_mask) == (t_mask ^ (1<<idx))
                    # i.e., all other bits are set.
                    if (current_mask & t_mask) == (t_mask ^ (1 << idx)):
                        can_add = False
                        break
                
                if can_add:
                    current_mask |= (1 << idx)
        
        return current_mask.bit_count()

    best_count = 0
    best_mask = 0
    
    # Run multiple restarts
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        mask = solve(seed)
        count = mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = mask
    
    # Convert best_mask to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & (1 << i):
            result.append(list(all_triples[i]))
            
    return result

# EVOLVE-BLOCK-END
