"""Structural change: Use a valid B_3 skeleton (powers of 4) as a base, then greedily fill gaps."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 4096
    max_k = 80

    def get_triple_sums(elements):
        """Calculate set of all triple sums for a given list of elements."""
        sums = set()
        k = len(elements)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(elements[i] + elements[j] + elements[l])
        return sums

    def can_add(x, elements, used_sums):
        """
        Check if x can be added to elements without creating a triple sum collision.
        We only need to check sums involving x.
        Sums involving x are:
        1. x+x+x
        2. x+x+y for y in elements
        3. x+y+z for y,z in elements (y<=z)
        
        We must ensure these new sums are not in `used_sums` and are unique among themselves.
        """
        new_sums = set()
        
        # 1. x+x+x
        s = 3 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        
        # 2. x+x+y
        for y in elements:
            s = 2 * x + y
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
            
        # 3. x+y+z
        k = len(elements)
        for i in range(k):
            for j in range(i, k):
                s = x + elements[i] + elements[j]
                if s in used_sums or s in new_sums:
                    return False
                new_sums.add(s)
                
        return True

    def build_from_skeleton(skeleton, seed_val, randomize_order=False):
        """
        Start with a valid skeleton, compute its sums, then greedily add
        the smallest available numbers that don't break the property.
        """
        elements = list(skeleton)
        # Sort and remove duplicates just in case
        elements = sorted(set(elements))
        
        # Filter out elements > n
        elements = [x for x in elements if x <= n]
        
        if not elements:
            # If skeleton is empty, we start from scratch
            pass
        else:
            # Ensure no duplicates in skeleton
            elements = list(dict.fromkeys(elements))

        used_sums = get_triple_sums(elements)
        
        # Candidates to try: all numbers 1..n not in elements
        # We try them in ascending order to pack small numbers first
        candidates = [x for x in range(1, n + 1) if x not in elements]
        
        rng = random.Random(seed_val)
        if randomize_order:
            rng.shuffle(candidates)
        else:
            # Ascending
            pass 

        for x in candidates:
            if time.time() > deadline:
                break
            if len(elements) >= max_k:
                break
            
            if can_add(x, elements, used_sums):
                elements.append(x)
                # Update used_sums with new sums involving x
                # x+x+x
                used_sums.add(3 * x)
                # x+x+y
                for y in elements[:-1]: # y is existing, x is new
                    used_sums.add(2 * x + y)
                # x+y+z
                k = len(elements) - 1
                for i in range(k):
                    for j in range(i, k):
                        used_sums.add(x + elements[i] + elements[j])
                        
        return elements

    best_result = []
    best_len = 0

    # 1. Base skeleton: Powers of 4
    # 1, 4, 16, 64, 256, 1024, 4096
    base_skeleton = [4**i for i in range(7) if 4**i <= n]
    
    # Try building from this skeleton (Ascending)
    res = build_from_skeleton(base_skeleton, 0, randomize_order=False)
    if len(res) > best_len:
        best_len = len(res)
        best_result = res

    # 2. Try random order fills with the same skeleton
    for i in range(20):
        if time.time() > deadline - 5:
            break
        res = build_from_skeleton(base_skeleton, i*100, randomize_order=True)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res

    # 3. Try pure greedy from 1 (Empty skeleton) with Ascending
    res = build_from_skeleton([], 0, randomize_order=False)
    if len(res) > best_len:
        best_len = len(res)
        best_result = res

    # 4. Try pure greedy from 1 with Random Order
    for i in range(10):
        if time.time() > deadline - 2:
            break
        res = build_from_skeleton([], i*200, randomize_order=True)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res

    return best_result
