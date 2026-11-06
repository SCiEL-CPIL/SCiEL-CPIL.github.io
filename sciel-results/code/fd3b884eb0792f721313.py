import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    all_numbers = list(range(2, 129))
    best_seq = []
    best_len = 0
    
    # Helper to check validity and find conflicts
    def find_conflicts(seq):
        n = len(seq)
        prods = {}
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    # Found a duplicate
                    # prods[prod] contains list of (start, end)
                    prods[prod].append((i, j))
                else:
                    prods[prod] = [(i, j)]
        
        conflicts = []
        for p, blocks in prods.items():
            if len(blocks) > 1:
                conflicts.append((p, blocks))
        return conflicts

    # Strategy 1: Greedy Ascending (Seed's base)
    def greedy_asc():
        values, products = [], set()
        for x in range(2, 129):
            # Check if adding x to end (since we add in increasing order, it's at the end)
            # Wait, if we add x, it goes to the end because values are sorted.
            # So we only check products ending at x.
            tails = []
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
            else:
                # Try to see if we can skip x? Greedy just skips.
                pass
        return values

    # Initial best from greedy
    best_seq = greedy_asc()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    
    # Strategies for pruning
    def prune_strategy(initial_set, strategy):
        S = set(initial_set)
        removed = []
        
        while time.time() < deadline - 1.0: # Leave 1s for other strategies
            A = sorted(S)
            conflicts = find_conflicts(A)
            if not conflicts:
                break
            
            # Pick a random conflict
            p, blocks = rng.choice(conflicts)
            # Pick two blocks from the conflict
            b1, b2 = rng.sample(blocks, 2)
            
            # Get elements in these blocks
            # Block (start, end) means A[start:end+1]
            elems_b1 = A[b1[0]:b1[1]+1]
            elems_b2 = A[b2[0]:b2[1]+1]
            
            set_b1 = set(elems_b1)
            set_b2 = set(elems_b2)
            
            # Candidates to remove
            if strategy == 'random':
                candidates = list(set_b1.union(set_b2))
                if not candidates: continue
                x_to_remove = rng.choice(candidates)
            elif strategy == 'largest':
                candidates = list(set_b1.union(set_b2))
                if not candidates: continue
                x_to_remove = max(candidates)
            elif strategy == 'smallest':
                candidates = list(set_b1.union(set_b2))
                if not candidates: continue
                x_to_remove = min(candidates)
            elif strategy == 'sym_diff':
                sym_diff = set_b1.symmetric_difference(set_b2)
                if sym_diff:
                    candidates = list(sym_diff)
                else:
                    candidates = list(set_b1.union(set_b2))
                if not candidates: continue
                x_to_remove = rng.choice(candidates)
            
            S.discard(x_to_remove)
            removed.append(x_to_remove)
            
        return sorted(S)
    
    # Run multiple strategies
    strategies = ['random', 'largest', 'smallest', 'sym_diff']
    
    for strat in strategies:
        if time.time() > deadline - 5.0:
            break
            
        # Start with all numbers
        seq = prune_strategy(all_numbers, strat)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    # Final Polish: Try to add back removed elements?
    # Since we don't track removed per run easily in the parallel strategy loop above, 
    # we can just do a local search on the best_seq found.
    
    # Local Search on best_seq
    # Try swapping an element in best_seq with one not in best_seq
    # Or just try adding elements from all_numbers not in best_seq
    
    current_seq = best_seq
    current_set = set(current_seq)
    
    # Try to add any element from all_numbers not in current_set
    # If it causes a conflict, we can't just add it. We need to remove something.
    # This is complex. Let's stick to the pruning result.
    # The pruning starts from the full set, so it finds a maximal subset.
    # Different heuristics find different maximal subsets.
    # Taking the best of several heuristics is a good bet.
    
    return best_seq
