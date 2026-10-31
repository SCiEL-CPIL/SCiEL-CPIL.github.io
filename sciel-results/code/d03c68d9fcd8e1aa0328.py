import numpy as np

N = 50

def construct():
    def find_sidon_size_k(n, k):
        # Returns a Sidon set of size k or None
        # Use backtracking
        current = []
        sums = set()
        
        def backtrack(start_idx):
            if len(current) == k:
                return True
            
            # Pruning: not enough indices left
            if n - start_idx < k - len(current):
                return False
                
            for i in range(start_idx, n):
                new_sums = [2*i]
                for j in current:
                    new_sums.append(i+j)
                    
                valid = True
                for s in new_sums:
                    if s in sums:
                        valid = False
                        break
                
                if valid:
                    current.append(i)
                    for s in new_sums:
                        sums.add(s)
                    
                    if backtrack(i+1):
                        return True
                    
                    current.pop()
                    for s in new_sums:
                        sums.remove(s)
                        
            return False
        
        if backtrack(0):
            return current
        return None

    # Try to find largest k
    # Theoretical max is around 14 for N=50
    best_sidon = []
    for k in range(14, 1, -1):
        s = find_sidon_size_k(N, k)
        if s:
            best_sidon = s
            break
    
    if not best_sidon:
        # Fallback to a known small set
        best_sidon = [0, 1, 4, 10]

    h = np.zeros(N)
    k = len(best_sidon)
    for idx in best_sidon:
        h[idx] = 1.0 / k
        
    return list(h)
