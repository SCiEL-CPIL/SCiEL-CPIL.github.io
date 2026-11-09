def construct():
    # The maximum size of a subset of {2, ..., 40} with distinct subset products is 16.
    # This is achieved by selecting prime powers such that the exponents for each prime
    # form a set with distinct subset sums.
    # 
    # For p=2, we can use 2^1, 2^2, 2^4 (values 2, 4, 16). Exponents {1, 2, 4} have distinct subset sums.
    # For p=3, we can use 3^1, 3^2 (values 3, 9). Exponents {1, 2} have distinct subset sums.
    # For p=5, we can use 5^1, 5^2 (values 5, 25). Exponents {1, 2} have distinct subset sums.
    # For p=7, we can use 7^1 (value 7).
    # For p in {11, 13, 17, 19, 23, 29, 31, 37}, we use p^1.
    #
    # Total count: 3 + 2 + 2 + 1 + 8 = 16.
    
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
