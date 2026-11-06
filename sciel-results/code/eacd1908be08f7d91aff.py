import time
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 50.0  # Safety margin
    
    # State: (sequence_tuple, blocked_count, blocked_set)
    # We want to minimize blocked_count.
    # Initial state: empty sequence
    # Actually, we start by adding the first number.
    # Let's start with beam of single elements.
    
    # To be efficient, we represent the sequence as a tuple.
    # We maintain a set of all sub-block products for the current sequence.
    
    def get_products(seq):
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid_extension(seq, x, existing_products):
        # Check if appending x creates a duplicate product
        # New products are x * (suffixes of seq)
        # Suffixes of seq: seq[-1], seq[-2]*seq[-1], ...
        # We need to check if any of these new products are in existing_products
        # OR if any of the new products are equal to each other?
        # The new products are x, x*seq[-1], x*seq[-2]*seq[-1], ...
        # Since seq is strictly increasing and x > seq[-1], the new products are strictly increasing.
        # So they are distinct among themselves.
        # We just need to check intersection with existing_products.
        
        # Optimization: 
        # The new products are { x * suffix_k for k in 1..len(seq) } U {x}
        # Wait, the sub-blocks ending at x are:
        # [x] -> x
        # [seq[-1], x] -> seq[-1]*x
        # [seq[-2], seq[-1], x] -> seq[-2]*seq[-1]*x
        # ...
        # [seq[0], ..., seq[-1], x] -> product(seq)*x
        
        # We can compute these iteratively.
        
        # Also, we must ensure x is not already in seq (strictly increasing).
        if x in seq:
            return False
            
        # Check collisions
        p = 1
        # Iterate from right to left
        for a in reversed(seq):
            p *= a
            if p * x in existing_products:
                return False
        if x in existing_products:
            return False
            
        return True

    def count_blocked(seq, existing_products):
        # Count how many x in [seq[-1]+1, 128] are blocked
        blocked = 0
        start = seq[-1] + 1 if seq else 2
        for x in range(start, 129):
            if not is_valid_extension(seq, x, existing_products):
                blocked += 1
        return blocked

    # Beam Search
    # State: (seq_tuple, products_set, blocked_count)
    # We want to keep states with LOW blocked_count.
    
    beam = []
    
    # Initialize with single elements? 
    # Or start with empty and expand?
    # Starting with empty:
    # Seq = (), Prods = set(), Blocked = count for empty?
    # For empty, any x in [2,128] is valid. Blocked = 0.
    
    # Let's initialize with the first few numbers to break symmetry?
    # No, beam search handles it.
    
    # State: (seq, prods, blocked)
    # We will expand the beam.
    
    # Initial beam: empty sequence
    # But empty sequence has 0 length. We want max length.
    # Let's just run beam search expanding from empty.
    
    current_beam = [ ((), set(), 0) ]
    
    best_seq = ()
    
    # We iterate by depth (length of sequence)
    # But beam search usually expands all nodes at depth d to get depth d+1.
    # Since we want to maximize length, we should go as deep as possible.
    
    # However, a state with low blocked count but short length might be better 
    # than a state with high blocked count but long length?
    # No, we want max length. 
    # The heuristic "minimize blocked" helps find a path that doesn't dead-end early.
    
    # Let's do iterative deepening or just expand the beam layer by layer.
    
    max_depth_reached = 0
    
    while True:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        next_beam = []
        
        for seq, prods, blocked in current_beam:
            # Try to extend this sequence
            # Candidates: x in [seq[-1]+1, 128]
            start_x = seq[-1] + 1 if seq else 2
            
            # Optimization: If we are at a high depth, maybe we can prune candidates
            # that are "too close" to previous numbers? No, strictly increasing is enough.
            
            for x in range(start_x, 129):
                if is_valid_extension(seq, x, prods):
                    # Create new state
                    new_seq = seq + (x,)
                    
                    # Update products
                    # New products are x * suffixes
                    new_prods = set(prods)
                    p = 1
                    for a in reversed(seq):
                        p *= a
                        new_prods.add(p * x)
                    new_prods.add(x)
                    
                    # Calculate new blocked count
                    # This is expensive. O(N * Cost of is_valid)
                    # is_valid is O(L). So count_blocked is O(N*L).
                    # N is up to 128. L is up to ~20. 128*20 = 2560 ops per candidate.
                    # Candidates per node: ~100. Nodes in beam: ~100.
                    # 100 * 100 * 2560 = 25.6M ops per layer.
                    # This might be too slow in Python if we do it naively.
                    
                    # Optimization: 
                    # We don't need to count ALL blocked numbers for sorting if we are just looking for the next step.
                    # But for the heuristic to work, we need the count.
                    
                    # Alternative heuristic: 
                    # Instead of counting blocked, maybe we just keep the top K sequences 
                    # based on "potential" which is hard to define without counting.
                    
                    # Let's try to speed up count_blocked.
                    # We can cache is_valid results? No, x changes.
                    
                    # Let's estimate runtime.
                    # If beam width is 50, and we do this for 20 layers.
                    # 50 * 100 (candidates) * 2560 (cost) = 12.8M ops per layer.
                    # 20 layers = 256M ops. In Python, 100M simple ops take ~10-20s.
                    # This is risky.
                    
                    # We can optimize is_valid_extension.
                    # We only need to check if p*x is in prods.
                    # The set lookup is O(1).
                    # The loop is O(L).
                    
                    # Let's try a smaller beam width first, e.g., 20.
                    # Or optimize the inner loop.
                    
                    new_blocked = count_blocked(new_seq, new_prods)
                    
                    next_beam.append((new_seq, new_prods, new_blocked))
            
            # If no extensions, this branch is dead.
        
        if not next_beam:
            break
            
        # Sort next_beam by blocked count (ascending)
        # If tie, prefer longer sequence? Or just blocked count.
        # Actually, we want to maximize length, so we should prefer states that allow further growth.
        # Low blocked count is the proxy for that.
        
        next_beam.sort(key=lambda x: x[2])
        
        # Keep top K
        K = 50
        current_beam = next_beam[:K]
        
        # Update best
        # The best sequence is the longest one in the current beam?
        # Or the one with lowest blocked count?
        # We want max length. 
        # Let's track the max length seen so far.
        for seq, prods, blocked in current_beam:
            if len(seq) > len(best_seq):
                best_seq = seq
                
        max_depth_reached = len(current_beam[0][0]) if current_beam else 0
        
        # If all sequences in the beam have the same length and are dead ends?
        # The loop will terminate when next_beam is empty.
        
    return list(best_seq)

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(res)
    
    # Verify
    prods = set()
    for i in range(len(res)):
        p = 1
        for j in range(i, len(res)):
            p *= res[j]
            if p in prods:
                print("Duplicate found!")
                break
            prods.add(p)
    print("Valid")
