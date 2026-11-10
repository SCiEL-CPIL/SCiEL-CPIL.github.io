"""Linear triple systems on 16 points with small independence number."""
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    random.seed(42)
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                if len(set(triples[i]) & set(triples[j])) > 1:
                    return False
        return True
    
    def independence_number(triples):
        """Find the largest subset of {0..15} containing no triple."""
        n = 16
        # Use bitmask representation
        triple_masks = [0] * len(triples)
        for i, t in enumerate(triples):
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks[i] = mask
        
        # For each subset size from n down to 1, check if any subset of that size has no triple
        # More efficient: check from large to small
        # Use a greedy approach with backtracking for small n
        
        # Precompute: for each point, which triples contain it
        point_to_triples = [[] for _ in range(n)]
        for i, t in enumerate(triples):
            for p in t:
                point_to_triples[p].append(i)
        
        # Find max independent set using a simple approach
        # For n=16, we can use a branch and bound
        best = 0
        
        # Try all subsets is 2^16 = 65536, which is feasible
        for mask in range(1 << n):
            # Check if this subset contains any triple
            has_triple = False
            for tm in triple_masks:
                if (mask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                popcount = bin(mask).count('1')
                if popcount > best:
                    best = popcount
                    if best == n:
                        return n
        return best
    
    def generate_structured_system():
        """Generate a structured linear triple system using Z_16 arithmetic."""
        triples = []
        points = list(range(16))
        
        # Use a cyclic construction: for each base triple, generate its orbit under +1 mod 16
        # But we need to ensure linearity
        
        # Start with a few base triples and their shifts
        # A triple {a,b,c} generates {a+k, b+k, c+k} mod 16 for k=0..15
        # But many will collide, so we need to be selective
        
        # Alternative: use a more systematic approach
        # Place triples to cover pairs as evenly as possible
        
        # Let's use a greedy approach to build a linear system
        used_pairs = set()
        triples = []
        
        # Generate candidate triples in a structured way
        candidates = []
        for a in range(16):
            for b in range(a+1, 16):
                for c in range(b+1, 16):
                    candidates.append([a, b, c])
        
        # Shuffle but with structure: prioritize triples with good spread
        random.shuffle(candidates)
        
        # Greedily add triples that don't conflict
        for t in candidates:
            if len(triples) >= 40:  # cap to keep search tractable
                break
            # Check if this triple conflicts with existing ones
            conflict = False
            for existing in triples:
                overlap = len(set(t) & set(existing))
                if overlap > 1:
                    conflict = True
                    break
            if not conflict:
                triples.append(t)
        
        return triples
    
    def local_search(triples, max_time=100):
        """Improve the system by swapping or replacing triples."""
        best_triples = triples[:]
        best_score = independence_number(best_triples)
        
        current_triples = triples[:]
        current_score = best_score
        
        timeout = time.time() + max_time
        
        # Try various improvement moves
        for iteration in range(20000):
            if time.time() > timeout:
                break
            
            # Random move: replace one triple with a random valid one
            idx = random.randint(0, len(current_triples) - 1)
            old_triple = current_triples[idx]
            
            # Generate a random replacement
            for attempt in range(50):
                new_triple = [random.randint(0, 15) for _ in range(3)]
                new_triple = sorted(set(new_triple))
                if len(new_triple) != 3:
                    continue
                
                # Check if it can replace old_triple (i.e., doesn't conflict with other triples)
                valid = True
                for j, t in enumerate(current_triples):
                    if j == idx:
                        continue
                    if len(set(new_triple) & set(t)) > 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Try the replacement
                current_triples[idx] = new_triple
                new_score = independence_number(current_triples)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = current_triples[:]
                else:
                    # Revert
                    current_triples[idx] = old_triple
                    break
        
        return best_triples, best_score
    
    # Generate initial system
    initial = generate_structured_system()
    
    if not is_linear(initial):
        # Fallback to a known good structure
        initial = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 0, 1]]
        # This isn't linear, let me fix
        initial = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 0, 3]]
        # Check
        if not is_linear(initial):
            initial = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 0, 1]]
            # Just use disjoint ones as fallback
            initial = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    best_triples = initial
    best_score = independence_number(best_triples)
    
    # Run local search with remaining time
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = local_search(best_triples, max_time=remaining_time)
    
    # Final check
    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
