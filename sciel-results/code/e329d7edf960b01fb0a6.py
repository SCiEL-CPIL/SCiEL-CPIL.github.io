import math
import time

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17. Every element must share a factor with 255.
    The candidate set partitions into 7 classes by which subset of {3,5,17} divides them.
    Within each class, all elements form a clique. Between classes, edges depend
    on the smallest prime factor shared by at least one element in each class.
    This reduces to a weighted maximum clique on 7 nodes.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    # Build the 7 classes
    # Class mask: bit 0 = divisible by 3, bit 1 = divisible by 5, bit 2 = divisible by 17
    # Only non-empty masks (1-7)
    classes = {}  # mask -> list of elements
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            m = 0
            if x % 3 == 0: m |= 1
            if x % 5 == 0: m |= 2
            if x % 17 == 0: m |= 4
            if m not in classes:
                classes[m] = []
            classes[m].append(x)
    
    # Get the list of masks and their sizes
    masks = sorted(classes.keys())
    n_classes = len(masks)
    mask_to_idx = {m: i for i, m in enumerate(masks)}
    sizes = [len(classes[m]) for m in masks]
    
    # Find index of the class containing 255
    mask_255 = 7  # 255 is divisible by 3, 5, and 17
    idx_255 = mask_to_idx[mask_255]
    
    # Build the 7-node graph
    # Edge between class i and class j exists if there exist a in class_i, b in class_j with gcd(a,b) > 1
    # Since within a class all elements share the primes in that mask,
    # the minimum possible gcd between an element of class i and class j
    # is determined by the smallest prime factor that could be shared.
    
    # For two classes with masks mi and mj:
    # - If mi & mj != 0, they share a prime from {3,5,17}, so all pairs have gcd > 1. Edge exists.
    # - If mi & mj == 0, they don't share any of {3,5,17}. 
    #   But elements might share other primes. We need to check if ANY pair has gcd > 1.
    #   Actually, we need ALL pairs to have gcd > 1 for a clique.
    #   So for a clique, for every pair of classes (i,j) in the clique, 
    #   ALL elements in class i must be coprime-free with ALL elements in class j.
    
    # Wait, this is more subtle. For the full set to be a clique, we need:
    # For every pair of elements (a, b) in the set, gcd(a,b) > 1.
    # If we include ALL elements from class i and ALL from class j,
    # then we need gcd(a,b) > 1 for ALL a in class_i, b in class_j.
    
    # Let's check: for classes with disjoint masks (no shared prime from {3,5,17}),
    # can we have all pairs with gcd > 1?
    
    # Class 1 (div by 3 only): e.g., 3, 9, 15(no, 15 div by 5 too)... 
    # Actually let me just check the actual elements.
    
    # For each pair of classes, check if ALL cross-pairs have gcd > 1
    # If yes, we can include both full classes.
    # If no, we can't include both full classes in a clique.
    
    # But wait - we don't have to include ALL elements from a class. We could include a subset.
    # However, including all elements from a class is always optimal if the class is compatible
    # with all other classes in the clique, because within a class it's a clique.
    
    # Actually, the optimal solution will include all elements from certain classes.
    # Let me verify: if class i is in the clique, should we include all of it?
    # If we include some but not all, we lose size without gaining compatibility
    # (since all elements in class i have the same divisibility pattern for {3,5,17},
    #  their compatibility with other classes depends on other prime factors).
    
    # Hmm, this is where it gets tricky. Two elements in the same class might have
    # different compatibility with elements in other classes due to other prime factors.
    
    # Let me just check: for each pair of classes, is it true that either ALL cross-pairs
    # have gcd>1, or there exist some with gcd=1?
    
    # Let me compute this directly.
    
    # Build adjacency between classes: can[i][j] = True if all a in class_i, b in class_j have gcd(a,b)>1
    can = [[False] * n_classes for _ in range(n_classes)]
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                can[i][j] = True
                continue
            # Check if all cross-pairs have gcd > 1
            all_ok = True
            for a in classes[masks[i]]:
                for b in classes[masks[j]]:
                    if math.gcd(a, b) == 1:
                        all_ok = False
                        break
                if not all_ok:
                    break
            can[i][j] = all_ok
    
    # Now find the maximum weight clique in this 7-node graph, with the constraint
    # that node idx_255 must be included.
    
    best_weight = 0
    best_clique = None
    
    # Brute force all subsets of 7 nodes that include idx_255
    for subset in range(1 << n_classes):
        if not (subset & (1 << idx_255)):
            continue
        
        # Check if it's a clique
        is_clique = True
        for i in range(n_classes):
            if not (subset & (1 << i)):
                continue
            for j in range(i + 1, n_classes):
                if not (subset & (1 << j)):
                    continue
                if not can[i][j]:
                    is_clique = False
                    break
            if not is_clique:
                break
        
        if is_clique:
            weight = sum(sizes[i] for i in range(n_classes) if (subset & (1 << i)))
            if weight > best_weight:
                best_weight = weight
                best_clique = subset
    
    # Build the result set
    result_set = set()
    for i in range(n_classes):
        if best_clique & (1 << i):
            result_set.update(classes[masks[i]])
    
    # Ensure 255 is in the result
    if 255 not in result_set:
        result_set.add(255)
    
    res_list = sorted(result_set)
    
    # Verify validity
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                # Fallback to all multiples of 3
                return list(range(3, 256, 3))
    
    if 255 not in res_list:
        return list(range(3, 256, 3))
    
    return res_list


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
