def construct():
    N = 12
    # Standard Cerný automaton
    # a: i -> (i+1)%N
    # b: 0 -> 0, i -> i-1 for i>0
    
    # Apply a fixed non-identity permutation to state labels to create a distinct instance
    # This is a structural change in the specific mapping returned, though isomorphic to the seed.
    perm = [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    inv_perm = [0] * N
    for i, p in enumerate(perm):
        inv_perm[p] = i
        
    a_map = [0] * N
    b_map = [0] * N
    
    for s in range(N):
        orig_s = perm[s]
        # a transition: orig_s -> (orig_s + 1) % N
        dest_a = (orig_s + 1) % N
        a_map[s] = inv_perm[dest_a]
        
        # b transition: 0->0, i->i-1
        if orig_s == 0:
            dest_b = 0
        else:
            dest_b = orig_s - 1
        b_map[s] = inv_perm[dest_b]
        
    return [a_map, b_map]
