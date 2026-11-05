import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    best_seq = []
    best_len = 0
    
    # Precompute: for a sequence, the new products when appending c are:
    # c, c*last, c*last*second_last, ...
    # We only need to check if any of these are in the existing product set.
    
    BEAM_WIDTH = 300
    MAX_CANDIDATES = 50  # Try up to 50 next values
    
    def get_new_products(seq, c):
        """Get the list of new products when appending c to seq."""
        prods = []
        prod = c
        prods.append(prod)
        for a in reversed(seq):
            prod *= a
            prods.append(prod)
        return prods
    
    def has_collision(prods, product_set):
        """Check if any product in prods is in product_set."""
        for p in prods:
            if p in product_set:
                return True
        return False
    
    # Beam search
    beams = [([], set())]
    
    while time.time() < deadline:
        new_beams = []
        
        for seq, prods in beams:
            if not seq:
                last_val = 1
            else:
                last_val = seq[-1]
            
            # Try candidates from last_val+1 up to min(last_val+MAX_CANDIDATES, 128)
            # But also consider that we might want to skip some values
            # to avoid collisions - beam search will handle this
            
            end = min(last_val + MAX_CANDIDATES + 1, 129)
            for c in range(last_val + 1, end):
                new_prods = get_new_products(seq, c)
                
                if not has_collision(new_prods, prods):
                    new_seq = seq + [c]
                    new_prod_set = prods.union(new_prods)
                    new_beams.append((new_seq, new_prod_set))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not new_beams:
            break
        
        # Sort beams: prefer longer, then smaller last value (more room to extend)
        # Also prefer fewer total products (less collision risk)
        def beam_sort_key(b):
            seq, prods = b
            return (-len(seq), seq[-1] if seq else 0, len(prods))
        
        new_beams.sort(key=beam_sort_key)
        beams = new_beams[:BEAM_WIDTH]
        
        # If we've reached a very long sequence, we can be more conservative
        if best_len >= 40:
            # Reduce beam width to save time, focus on best candidates
            BEAM_WIDTH = 100
    
    return best_seq
