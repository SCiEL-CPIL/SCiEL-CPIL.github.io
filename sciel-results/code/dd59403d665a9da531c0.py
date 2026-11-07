import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute all bad triples (a, b, c) with a < b < c, a+b <= n, a+c <= n, b+c <= n
    # A triple is "bad" for set A if a,b,c,a+b,a+c,b+c are all in A
    # We'll work with bitmasks for efficiency
    
    # Generate all candidate triples
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if a + b > n:
                break
            for c in range(b + 1, n + 1):
                if b + c > n:
                    break
                # a+c < b+c <= n, so a+c <= n automatically
                triples.append((a, b, c))
    
    num_triples = len(triples)
    
    # For each element x in 1..n, precompute which triples involve x
    # An element x is "involved" in triple (a,b,c) if x is one of a,b,c,a+b,a+c,b+c
    # We'll use a different approach: for each triple, store the 6 values
    # and for each value, which triples it participates in
    
    # Actually, let's use a simpler approach with sets and local search
    
    def count_violations(A):
        """Count number of bad triples in set A"""
        count = 0
        A_set = set(A) if not isinstance(A, set) else A
        for a, b, c in triples:
            if a in A_set and b in A_set and c in A_set:
                if (a+b) in A_set and (a+c) in A_set and (b+c) in A_set:
                    count += 1
        return count
    
    def is_valid(A):
        """Check if set A has no bad triples"""
        A_set = set(A) if not isinstance(A, set) else A
        for a, b, c in triples:
            if a in A_set and b in A_set and c in A_set:
                if (a+b) in A_set and (a+c) in A_set and (b+c) in A_set:
                    return False
        return True
    
    def greedy_construct():
        """Greedy construction: add elements one by one if they don't create violations"""
        A = set()
        # Try different orderings
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            A.add(x)
            if count_violations(A) > 0:
                A.remove(x)
        return A
    
    def local_search(A):
        """Improve set A via local search"""
        A = set(A)
        best_A = set(A)
        best_size = len(A)
        
        # Phase 1: Remove elements that are in many violations
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Count how many violations each element is part of
            viol_count = {}
            A_set = set(A)
            for a, b, c in triples:
                if a in A_set and b in A_set and c in A_set:
                    if (a+b) in A_set and (a+c) in A_set and (b+c) in A_set:
                        for x in [a, b, c, a+b, a+c, b+c]:
                            viol_count[x] = viol_count.get(x, 0) + 1
            
            if not viol_count:
                break
            
            # Remove element with max violation count
            max_elem = max(viol_count, key=viol_count.get)
            if max_elem in A:
                A.remove(max_elem)
        
        # Phase 2: Try to add elements back
        candidates = [x for x in range(1, n + 1) if x not in A]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            A.add(x)
            if is_valid(A):
                continue
            else:
                A.remove(x)
        
        # Phase 3: Try swapping elements
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Try removing one element and adding another
            removed = None
            added = None
            for x in list(A):
                A.remove(x)
                removed = x
                # Try adding a non-member
                for y in range(1, n + 1):
                    if y not in A:
                        A.add(y)
                        if is_valid(A) and len(A) > best_size:
                            best_A = set(A)
                            best_size = len(A)
                            added = y
                            break
                        A.remove(y)
                
                if added is not None:
                    break
                else:
                    A.add(x)
            
            if added is None:
                break
        
        return best_A
    
    # Main search: multiple restarts with different strategies
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy from full set, removing violations
    A = set(range(1, n + 1))
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        A = local_search(A)
        if len(A) > best_size:
            best_size = len(A)
            best_set = set(A)
        if is_valid(A) and len(A) > best_size:
            best_size = len(A)
            best_set = set(A)
        # Restart with slight perturbation
        if random.random() < 0.3:
            A = set(range(1, n + 1))
            # Remove a few random elements
            for _ in range(random.randint(1, 5)):
                elem = random.choice(list(A))
                A.remove(elem)
    
    # Strategy 2: Greedy construction from scratch
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        A = greedy_construct()
        A = local_search(A)
        if len(A) > best_size:
            best_size = len(A)
            best_set = set(A)
    
    # Strategy 3: Random restarts with random initial sets
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        # Start with random subset
        A = set()
        for x in range(1, n + 1):
            if random.random() < 0.7:
                A.add(x)
        A = local_search(A)
        if len(A) > best_size:
            best_size = len(A)
            best_set = set(A)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 5:
        for x in range(1, n + 1):
            if x not in best_set:
                best_set.add(x)
                if is_valid(best_set):
                    continue
                else:
                    best_set.remove(x)
    
    # Convert to list of Python ints
    result = [int(x) for x in best_set]
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
