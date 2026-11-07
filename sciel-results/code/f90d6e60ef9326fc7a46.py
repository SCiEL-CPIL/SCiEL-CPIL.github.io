import time
import random

def construct():
    n = 128
    A = set(range(1, n + 1))
    
    def get_bad_triples(A):
        bads = []
        elems = sorted(A)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                if a + b > n: 
                    break
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    if b + c > n: 
                        break
                    if a + c in A and b + c in A:
                        if a + b in A:
                            bads.append((a, b, c))
        return bads

    start_time = time.time()
    
    # Phase 1: Greedy Removal
    while time.time() - start_time < 30:
        bads = get_bad_triples(A)
        if not bads:
            break
        
        counts = {}
        for a, b, c in bads:
            for x in [a, b, c, a+b, a+c, b+c]:
                if x in A:
                    counts[x] = counts.get(x, 0) + 1
        
        if counts:
            # Remove element with max count
            # Random tie-breaking to escape local minima
            max_count = max(counts.values())
            candidates = [x for x, c in counts.items() if c == max_count]
            max_elem = random.choice(candidates)
            A.remove(max_elem)
        else:
            break

    # Phase 2: Try to add elements back
    removed = [x for x in range(1, n + 1) if x not in A]
    random.shuffle(removed)
    
    for x in removed:
        if time.time() - start_time > 50:
            break
        A.add(x)
        bads = get_bad_triples(A)
        if not bads:
            continue
        else:
            # Check if we can fix by removing one element
            # Find the element in bads that appears most often
            counts = {}
            for a, b, c in bads:
                for y in [a, b, c, a+b, a+c, b+c]:
                    if y in A:
                        counts[y] = counts.get(y, 0) + 1
            
            if counts:
                max_count = max(counts.values())
                # Only remove if it's not x, or if removing x is better?
                # If we remove x, we revert.
                # If we remove y != x, we keep x and lose y. Size constant.
                # But maybe y was critical?
                # Simplest: revert x.
                A.remove(x)
            else:
                A.remove(x)

    # Phase 3: Local Search / Swaps
    # Try to swap an element in A with an element not in A to increase size?
    # No, swap keeps size.
    # We want to increase size.
    # If we found a valid set of size S, can we find S+1?
    # This is hard.
    
    return list(A)
