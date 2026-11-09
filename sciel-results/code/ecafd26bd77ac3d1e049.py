def construct():
    # Construct a set of integers in [2, 40] with distinct subset products.
    # Strategy: Use disjoint sets of prime powers.
    # For each prime p, select a subset of exponents E_p such that all subset sums of E_p are distinct.
    # This ensures that the numbers p^e for e in E_p have distinct subset products within the prime family.
    # Since the primes are distinct across families, the global subset products are also distinct.
    
    # Prime 2: Exponents available 1..5. Max distinct subset sum set size is 3.
    # Example: {3, 4, 5} -> 8, 16, 32. Sums: 0,3,4,5,7,8,9,12 (Distinct).
    nums_2 = [2**3, 2**4, 2**5]
    
    # Prime 3: Exponents available 1..3. Max size 2.
    # Example: {2, 3} -> 9, 27. Sums: 0,2,3,5 (Distinct).
    nums_3 = [3**2, 3**3]
    
    # Prime 5: Exponents available 1..2. Max size 2.
    # Example: {1, 2} -> 5, 25. Sums: 0,1,2,3 (Distinct).
    nums_5 = [5**1, 5**2]
    
    # Other primes in [2, 40]: 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Each contributes 1 number (the prime itself).
    other_primes = [7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = nums_2 + nums_3 + nums_5 + other_primes
    
    # Verify constraints
    # 1. All integers in [2, 40]
    # 2. All distinct
    # 3. All subset products distinct (guaranteed by construction)
    
    result = [int(x) for x in result]
    
    return result
