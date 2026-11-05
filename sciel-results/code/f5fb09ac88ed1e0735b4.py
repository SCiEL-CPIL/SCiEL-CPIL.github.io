"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 12! = 479001600
    # Factorization: 2^10 * 3^5 * 5^2 * 7
    # We need distinct integers > 12 with product = 479001600.
    # Minimize the largest factor.
    
    # A known good decomposition:
    # 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21 * 22 * 23 * 24 is too large (product is 12! * 13! / 12! ... wait, product of 13..24 is not 12!).
    # Let's construct a valid set.
    # 12! = 479001600
    
    # Let's try to find a set of distinct integers > 12.
    # If we use 13, 14, 15, ..., k, the product grows fast.
    # 13*14*15*16*17*18*19*20 = 87178291200 which is > 479001600.
    # So we can't use too many small factors.
    
    # Let's try 2 distinct factors: a * b = 479001600, a > 12, b > 12, a != b.
    # Minimize max(a, b). This happens when a and b are close to sqrt(479001600).
    # sqrt(479001600) ≈ 21886.
    # So we want factors near 21886.
    
    # Let's find divisors near 21886.
    # 479001600 / 21886 ≈ 21886.
    # Let's check if 21886 divides 479001600.
    # 21886 = 2 * 10943. 10943 is prime? 
    # 10943 / 13 = 841.7... no.
    # 10943 / 17 = 643.7... no.
    # 10943 / 19 = 575.9... no.
    # 10943 / 23 = 475.7... no.
    # 10943 / 29 = 377.3... no.
    # 10943 / 31 = 353.0... 31*353 = 10943.
    # So 21886 = 2 * 31 * 353.
    # 479001600 = 2^10 * 3^5 * 5^2 * 7.
    # 21886 has factor 31 and 353, which are not in 12!. So 21886 does not divide 12!.
    
    # We need a factor that only uses primes 2, 3, 5, 7.
    # Let's find the largest divisor of 12! that is <= sqrt(12!).
    # sqrt(12!) ≈ 21886.
    
    # Let's list divisors of 12! near 21886.
    # 12! = 479001600.
    # Divisors:
    # ...
    # Let's try to construct a set of 3 factors.
    # a * b * c = 12!, a,b,c distinct > 12.
    # Minimize max(a,b,c).
    # Ideally a,b,c ≈ (12!)^(1/3) ≈ 782.
    
    # Let's look for divisors near 782.
    # 782 = 2 * 17 * 23. No.
    # 783 = 3^3 * 29. No.
    # 784 = 2^4 * 7^2. Yes! 784 divides 12!.
    # 12! / 784 = 610971.4... no, 479001600 / 784 = 610971.428...
    # Wait. 479001600 / 784 = 610971.428...?
    # 784 = 2^4 * 7^2.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1.
    # 7^2 does not divide 12!. So 784 does not divide 12!.
    
    # Let's re-evaluate.
    # Primes in 12!: 2, 3, 5, 7.
    # Max power of 2: 10
    # Max power of 3: 5
    # Max power of 5: 2
    # Max power of 7: 1
    
    # So any factor must be of form 2^a * 3^b * 5^c * 7^d with a<=10, b<=5, c<=2, d<=1.
    
    # Let's try to find a factor close to 782.
    # 780 = 2^2 * 3 * 5 * 13. No (13 not in 12!).
    # 768 = 2^8 * 3. Yes.
    # 12! / 768 = 479001600 / 768 = 623698.75...
    # 768 = 2^8 * 3.
    # 12! / 768 = 2^2 * 3^4 * 5^2 * 7 = 4 * 81 * 25 * 7 = 56700.
    # So 768 * 56700 = 12!.
    # 768 > 12, 56700 > 12, 768 != 56700.
    # This is a valid 2-factor decomposition: [768, 56700].
    # Largest factor: 56700.
    
    # Can we do better with 3 factors?
    # We want a, b, c distinct > 12, product 12!, minimize max.
    # Let's fix a = 768.
    # b * c = 56700.
    # Minimize max(b, c). b, c close to sqrt(56700) ≈ 238.
    # Find divisors of 56700 near 238.
    # 56700 = 2^2 * 3^4 * 5^2 * 7.
    # Divisors near 238:
    # 240 = 2^4 * 3 * 5. Does 240 divide 56700?
    # 56700 / 240 = 236.25. No.
    # 225 = 3^2 * 5^2. 56700 / 225 = 252.
    # 225 > 12, 252 > 12, 225 != 252, 225 != 768, 252 != 768.
    # So [768, 225, 252] is a valid 3-factor decomposition.
    # Product: 768 * 225 * 252 = 43200 * 252 = 10886400. Wait.
    # 768 * 56700 = 43541760.
    # 225 * 252 = 56700.
    # So 768 * 225 * 252 = 12!.
    # Largest factor: 768.
    
    # This is much better than 56700.
    
    # Can we do better with 4 factors?
    # a * b * c * d = 12!.
    # Minimize max. Ideally (12!)^(1/4) ≈ 267.
    # Let's try to find 4 distinct factors > 12, product 12!, all < 768.
    # We already have 768, 225, 252.
    # Let's split 768.
    # 768 = 2^8 * 3.
    # We need two distinct factors > 12, not equal to 225 or 252, whose product is 768.
    # 768 = 16 * 48.
    # 16 > 12, 48 > 12, 16 != 48, 16 != 225, 48 != 225, 16 != 252, 48 != 252.
    # So [16, 48, 225, 252] is a valid 4-factor decomposition.
    # Product: 16 * 48 * 225 * 252 = 768 * 56700 = 12!.
    # Largest factor: 252.
    
    # This is even better.
    
    # Can we do better with 5 factors?
    # (12!)^(1/5) ≈ 136.
    # We have [16, 48, 225, 252].
    # Let's split 252.
    # 252 = 12 * 21. But 12 is not > 12.
    # 252 = 14 * 18.
    # 14 > 12, 18 > 12, 14 != 18, 14 != 16, 14 != 48, 14 != 225, 18 != 16, 18 != 48, 18 != 225.
    # So [14, 16, 18, 48, 225] is a valid 5-factor decomposition.
    # Product: 14 * 16 * 18 * 48 * 225 = 252 * 768 * 225? No.
    # 14 * 18 = 252.
    # 16 * 48 = 768.
    # So 14 * 16 * 18 * 48 * 225 = 252 * 768 * 225 = 12!.
    # Largest factor: 225.
    
    # Can we do better with 6 factors?
    # (12!)^(1/6) ≈ 100.
    # We have [14, 16, 18, 48, 225].
    # Let's split 225.
    # 225 = 15 * 15. Not distinct.
    # 225 = 9 * 25. 9 < 12.
    # 225 = 5 * 45. 5 < 12.
    # 225 = 3 * 75. 3 < 12.
    # So we cannot split 225 into two distinct integers > 12.
    
    # Let's try to split 48.
    # 48 = 12 * 4. 12 not > 12.
    # 48 = 16 * 3. 3 < 12.
    # 48 = 24 * 2. 2 < 12.
    # 48 = 48 * 1. 1 < 12.
    # So we cannot split 48 either.
    
    # Let's try a different 5-factor set.
    # Maybe we can get a max < 225.
    # Let's search for 6 distinct factors > 12 with product 12! and max < 225.
    # If max < 225, all factors are in [13, 224].
    # Product of 6 distinct integers from [13, 224] is 12!.
    
    # Let's try to find such a set.
    # 12! = 2^10 * 3^5 * 5^2 * 7.
    
    # Let's try to use small factors: 13, 14, 15, 16, 17, 18.
    # Product: 13*14*15*16*17*18 = 12,254,928.
    # 12! / 12254928 ≈ 39.08.
    # So we need a 6th factor ≈ 39.
    # But we already used 13..18.
    # We need 6 distinct factors.
    # Let's try 14, 15, 16, 17, 18, 19.
    # Product: 14*15*16*17*18*19 = 13,243,200.
    # 12! / 13243200 ≈ 36.17.
    # Not an integer.
    
    # Let's try to systematically find a set of 6 factors.
    # Factors must be divisors of 12! and > 12.
    # Let's pick 13, 14, 15, 16.
    # Product: 13*14*15*16 = 43680.
    # Remaining: 12! / 43680 = 10967.28...
    # 43680 = 2^6 * 3 * 5 * 7 * 13.
    # 12! = 2^10 * 3^5 * 5^2 * 7.
    # 12! / 43680 = 2^4 * 3^4 * 5 / 13.
    # Since 13 is not in 12!, 43680 does not divide 12!.
    # So we cannot use 13.
    
    # Let's pick 14, 15, 16.
    # Product: 14*15*16 = 3360.
    # 12! / 3360 = 142560.
    # 3360 = 2^5 * 3 * 5 * 7.
    # 12! / 3360 = 2^5 * 3^4 * 5.
    # We need 3 more distinct factors > 12, not equal to 14, 15, 16, whose product is 142560.
    # 142560 = 2^5 * 3^4 * 5.
    # Let's try to split 142560 into 3 distinct factors > 12.
    # Cube root of 142560 ≈ 52.
    # Let's try factors near 52.
    # 52 = 2^2 * 13. No.
    # 54 = 2 * 3^3. Yes.
    # 142560 / 54 = 2640.
    # 2640 = 2^4 * 3 * 5 * 11. 11 not in 12! factorization of 142560?
    # 142560 = 2^5 * 3^4 * 5.
    # 2640 has factor 11. So 54 does not divide 142560 in a way that leaves integer factors without new primes?
    # Wait. 142560 / 54 = 2640.
    # 2640 = 2^4 * 3 * 5 * 11.
    # But 142560 = 2^5 * 3^4 * 5.
    # 54 = 2 * 3^3.
    # 142560 / 54 = 2^4 * 3 * 5.
    # My calculation was wrong. 2640 = 2^4 * 3 * 5 * 11 is correct, but 142560 / 54 = 2640 is wrong.
    # 142560 / 54 = 2640.
    # 54 * 2640 = 142560.
    # 2^1 * 3^3 * 2^4 * 3 * 5 * 11 = 2^5 * 3^4 * 5 * 11.
    # This has an 11. So 142560 must have an 11.
    # 12! / 3360 = 142560.
    # 12! = 479001600.
    # 479001600 / 3360 = 142560.
    # 142560 = 2^5 * 3^4 * 5.
    # 2^5 = 32. 3^4 = 81. 5.
    # 32 * 81 * 5 = 12960.
    # So 12! / 3360 = 12960.
    # My previous calculation 142560 was wrong.
    
    # So remaining product is 12960.
    # 12960 = 2^5 * 3^4 * 5.
    # We need 3 distinct factors > 12, not 14, 15, 16, product 12960.
    # Cube root of 12960 ≈ 23.
    # Let's try 24, 25, 27.
    # 24 = 2^3 * 3.
    # 25 = 5^2.
    # 27 = 3^3.
    # Product: 24 * 25 * 27 = 600 * 27 = 16200.
    # 16200 != 12960.
    
    # Let's try 20, 24, 27.
    # 20 = 2^2 * 5.
    # 24 = 2^3 * 3.
    # 27 = 3^3.
    # Product: 20 * 24 * 27 = 480 * 27 = 12960.
    # Yes!
    # Factors: 14, 15, 16, 20, 24, 27.
    # All distinct? 14, 15, 16, 20, 24, 27. Yes.
    # All > 12? Yes.
    # Product: 14 * 15 * 16 * 20 * 24 * 27 = 3360 * 12960 = 43545600.
    # Wait. 12! = 479001600.
    # 3360 * 12960 = 43545600.
    # 43545600 != 479001600.
    # 479001600 / 3360 = 142560.
    # So 12! / 3360 = 142560.
    # My factorization of 142560 was wrong.
    # 142560 = 12960 * 11.
    # 11 is a prime not in 12!? No, 11 is in 12!.
    # 12! = 1 * 2 * ... * 12.
    # Primes: 2, 3, 5, 7, 11.
    # I forgot 11.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1.
    
    # Okay, restarting with correct factorization.
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11.
    
    # Set: [14, 15, 16, 20, 24, 27].
    # Product: 14*15*16*20*24*27 = 43545600.
    # 12! = 479001600.
    # 479001600 / 43545600 = 11.
    # So we are missing a factor of 11.
    # We can replace one factor with 11 * factor.
    # E.g., 14 -> 154.
    # Set: [154, 15, 16, 20, 24, 27].
    # Product: 154 * 15 * 16 * 20 * 24 * 27 = 11 * 43545600 = 479001600.
    # Largest factor: 154.
    # This is better than 225.
    
    # Can we do better?
    # Minimize max.
    # Let's try to balance the factors more.
    # 6 factors. (12!)^(1/6) ≈ 100.
    # We have [15, 16, 20, 24, 27, 154].
    # 154 is large.
    # Let's try to spread the 11.
    # Instead of 14*11=154, maybe 15*11=165? Worse.
    # 16*11=176? Worse.
    # 20*11=220? Worse.
    # 24*11=264? Worse.
    # 27*11=297? Worse.
    
    # Let's try different base factors.
    # We need product 12!.
    # Let's try to find 6 distinct factors > 12, all < 154.
    # Max < 154.
    # Let's try to use 11 as a factor? No, 11 is not > 12.
    # So 11 must be multiplied with something > 12.
    # The smallest such product is 11 * 13 = 143.
    # So at least one factor must be >= 143.
    # Can we get a set with max = 143?
    # One factor is 143 = 11 * 13.
    # Remaining product: 12! / 143 = 3350011.1...
    # 143 = 11 * 13.
    # 12! / 143 = (2^10 * 3^5 * 5^2 * 7 * 11) / (11 * 13) = (2^10 * 3^5 * 5^2 * 7) / 13.
    # 13 does not divide 12!.
    # So we cannot use 143.
    
    # Next smallest factor containing 11 is 11 * 14 = 154.
    # So the minimum possible largest factor is at least 154.
    # We found a solution with max 154.
    # So 154 is likely the optimal answer.
    
    return [14 * 11, 15, 16, 20, 24, 27]
# EVOLVE-BLOCK-END
