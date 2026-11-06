import time
import itertools

def _has_c6_in_circulant(n, S):
    """
    Check if the circulant graph C_n(S) contains a simple 6-cycle.
    S is a set of steps. The graph has edges (i, i+s) for s in S.
    Since it's undirected, steps are S U -S.
    """
    steps = list(S) + [x - n if x > n // 2 else x for x in S]
    # Actually, S contains distances 1..8.
    # The steps in the graph are +/- d for d in S.
    # So steps = [d for d in S] + [-d for d in S]
    # We need to be careful with mod n.
    
    # Generate all sequences of 6 steps
    # There are 2*|S| possible steps.
    # If |S|=2, 4 steps. 4^6 = 4096 sequences.
    
    steps_list = []
    for d in S:
        steps_list.append(d)
        steps_list.append(-d)
        
    for seq in itertools.product(steps_list, repeat=6):
        # Check if sum is 0 mod n
        if sum(seq) % n != 0:
            continue
        
        # Check if partial sums are distinct
        # Partial sums p_0=0, p_1=s1, p_2=s1+s2, ...
        # Vertices are p_0, p_1, ..., p_5 (mod n)
        # They must be distinct.
        vertices = [0]
        current = 0
        for i in range(6):
            current = (current + seq[i]) % n
            vertices.append(current)
        
        # Check distinctness of vertices[0:6]
        # vertices[6] is 0, which is vertices[0], so we ignore it for distinctness check of the cycle nodes
        if len(set(vertices[:6])) == 6:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 17
    
    # Distances are 1 to 8
    distances = list(range(1, 9))
    
    # Generate all partitions of distances into 4 sets of 2
    # We can do this by choosing 2 for set 0, 2 for set 1, 2 for set 2, 2 for set 3.
    # To avoid duplicates, we can enforce an ordering, but 105 is small enough to just iterate and check.
    # Actually, we can just iterate over all combinations.
    
    found_valid = False
    valid_partition = None
    
    # Iterate over all ways to pick 2 elements for S0
    for s0 in itertools.combinations(distances, 2):
        rem1 = [d for d in distances if d not in s0]
        for s1 in itertools.combinations(rem1, 2):
            rem2 = [d for d in rem1 if d not in s1]
            for s2 in itertools.combinations(rem2, 2):
                rem3 = [d for d in rem2 if d not in s2]
                s3 = tuple(rem3)
                
                if time.time() > deadline:
                    break
                
                S_sets = [set(s0), set(s1), set(s2), set(s3)]
                
                valid = True
                for S in S_sets:
                    if _has_c6_in_circulant(n, S):
                        valid = False
                        break
                        
                if valid:
                    valid_partition = S_sets
                    found_valid = True
                    break
            if found_valid:
                break
        if found_valid:
            break
        if time.time() > deadline:
            break
            
    if found_valid:
        # Construct the colors
        # For each pair (i, j), i < j, distance d = j - i.
        # If d > n/2, distance is n - d.
        # Find which set S_k contains this distance.
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                if d > n // 2:
                    d = n - d
                # d is in 1..8
                for k in range(4):
                    if d in valid_partition[k]:
                        colors.append(k)
                        break
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=9
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
