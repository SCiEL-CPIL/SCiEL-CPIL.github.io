import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Beam Search
    W = 1000
    # State: (values_list, products_set)
    # Start with empty sequence
    beam = [([], set())]
    
    best_len = 0
    best_seq = []
    
    for x in range(2, 129):
        new_beam = []
        for values, products in beam:
            # Option 1: Skip x
            new_beam.append((values, products))
            
            # Option 2: Include x
            # Check if we have time
            if time.time() > deadline:
                break
            
            # Compute tails
            # tails[0] = x
            # tails[i] = x * values[-1] * ... * values[-i]
            # We need to check if any tail is in products
            
            # Optimization: if len(values) is large, computing tails is O(L)
            # We can do it incrementally?
            # For now, simple loop.
            
            tails = [x]
            prod = x
            # Multiply by elements in reversed(values)
            for a in reversed(values):
                prod *= a
                tails.append(prod)
                
            if not products.intersection(tails):
                new_products = set(products)
                new_products.update(tails)
                new_values = values + [x]
                new_beam.append((new_values, new_products))
                
        if time.time() > deadline:
            break
            
        # Select top W
        # Sort by length descending
        # If lengths are equal, we want diversity.
        # We can just shuffle the candidates with the same max length and take the first W.
        
        if not new_beam:
            break
            
        # Find max length
        max_len = max(len(v) for v, _ in new_beam)
        
        # Filter to max length?
        # If we filter to max length only, we might lose paths that are slightly shorter but have better potential?
        # Usually for maximizing length, keeping only max length is safe if the future depends only on the set of products.
        # But a shorter sequence might have a "cleaner" product set allowing more additions later.
        # Example: Short sequence with small products vs Long sequence with large products.
        # Actually, longer sequence means more products, which is "worse" for future collisions.
        # So a shorter sequence might be better!
        # Therefore, we should NOT just keep max length.
        # We should keep a diverse set of states.
        
        # Strategy: Keep top W states by a score.
        # Score: Length is primary.
        # But if we drop shorter states, we might miss the optimal path.
        # Let's keep states with length >= max_len - 2?
        # Or just keep W states from the entire new_beam, sorted by length.
        
        # Let's try keeping top W by length.
        new_beam.sort(key=lambda s: len(s[0]), reverse=True)
        
        # If we have more than W, we need to select W.
        # To ensure diversity, let's take W states.
        # We can take the first W (which are the longest).
        # But this might bias towards the greedy path.
        # Let's try: Take all states with length == max_len.
        # If count > W, take W randomly.
        # If count < W, add next best lengths.
        
        selected = []
        # Group by length
        from collections import defaultdict
        by_len = defaultdict(list)
        for state in new_beam:
            by_len[len(state[0])].append(state)
            
        for l in sorted(by_len.keys(), reverse=True):
            states_at_len = by_len[l]
            if len(selected) + len(states_at_len) <= W:
                selected.extend(states_at_len)
            else:
                # Need to pick some
                needed = W - len(selected)
                if len(states_at_len) > needed:
                    # Random sample
                    selected.extend(random.sample(states_at_len, needed))
                else:
                    selected.extend(states_at_len)
                break
            
        beam = selected
        
        # Track best
        if beam:
            current_max_len = len(beam[0][0])
            if current_max_len > best_len:
                best_len = current_max_len
                best_seq = beam[0][0]
                
    # If we ran out of time, return best found
    if time.time() > deadline:
        return best_seq if best_seq else [2] # Fallback
        
    # Return the best sequence found in the final beam
    if beam:
        # Find the longest in the final beam
        final_best = max(beam, key=lambda s: len(s[0]))[0]
        if len(final_best) > best_len:
            best_seq = final_best
            
    return best_seq if best_seq else [2]
