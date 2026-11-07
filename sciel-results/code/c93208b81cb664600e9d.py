import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Construct two orthogonal Latin squares of order 10
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 2*j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]

    # Verification functions
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    # Check validity of initial pair
    if not is_latin(L1) or not is_latin(L2) or not are_orthogonal(L1, L2):
        return [L1]

    squares = [L1, L2]

    # Search for a third orthogonal Latin square
    # Using backtracking with constraints
    L3 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Precompute constraints
    pos_by_l1 = {s: [] for s in range(n)}
    pos_by_l2 = {s: [] for s in range(n)}
    for i in range(n):
        for j in range(n):
            pos_by_l1[L1[i][j]].append((i, j))
            pos_by_l2[L2[i][j]].append((i, j))

    def find_third_square():
        # State
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        l1_used = [set() for _ in range(n)]
        l2_used = [set() for _ in range(n)]
        
        # MRV Heuristic: Pick the cell with the fewest candidates
        # To keep it simple and fast, we can just iterate linearly but with good pruning.
        # Or we can sort cells by constraint tightness.
        # Given the small size, linear iteration with strong pruning might be enough.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            # Candidates
            candidates = []
            for s in range(n):
                if s not in row_used[i] and s not in col_used[j] and \
                   s not in l1_used[L1[i][j]] and s not in l2_used[L2[i][j]]:
                    candidates.append(s)
            
            if not candidates:
                return False
            
            # Randomize candidates to help escape local minima if needed, 
            # but standard order is deterministic.
            # For speed, we try standard order first.
            
            for s in candidates:
                L3[i][j] = s
                row_used[i].add(s)
                col_used[j].add(s)
                l1_used[L1[i][j]].add(s)
                l2_used[L2[i][j]].add(s)
                
                if backtrack(idx + 1):
                    return True
                
                L3[i][j] = -1
                row_used[i].remove(s)
                col_used[j].remove(s)
                l1_used[L1[i][j]].remove(s)
                l2_used[L2[i][j]].remove(s)
                
            return False

        # Try standard backtracking
        if backtrack(0):
            return True
            
        # If failed, try randomized restarts
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Reset
            for i in range(n):
                for j in range(n):
                    L3[i][j] = -1
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            l1_used = [set() for _ in range(n)]
            l2_used = [set() for _ in range(n)]
            
            # Shuffle the order of cells? Or symbols?
            # Shuffling symbols in the candidate list is easy.
            # We can modify backtrack to take a random seed or shuffle candidates.
            
            def backtrack_rand(idx):
                if time.time() > deadline:
                    return False
                if idx == n * n:
                    return True
                i = idx // n
                j = idx % n
                candidates = []
                for s in range(n):
                    if s not in row_used[i] and s not in col_used[j] and \
                       s not in l1_used[L1[i][j]] and s not in l2_used[L2[i][j]]:
                        candidates.append(s)
                if not candidates:
                    return False
                random.shuffle(candidates)
                for s in candidates:
                    L3[i][j] = s
                    row_used[i].add(s)
                    col_used[j].add(s)
                    l1_used[L1[i][j]].add(s)
                    l2_used[L2[i][j]].add(s)
                    if backtrack_rand(idx + 1):
                        return True
                    L3[i][j] = -1
                    row_used[i].remove(s)
                    col_used[j].remove(s)
                    l1_used[L1[i][j]].remove(s)
                    l2_used[L2[i][j]].remove(s)
                return False

            if backtrack_rand(0):
                return True

        return False

    if find_third_square():
        squares.append(L3)
        # Verify the third square
        if not is_latin(L3) or not are_orthogonal(L1, L3) or not are_orthogonal(L2, L3):
            squares.pop()

    # Convert to ensure all values are Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return result
