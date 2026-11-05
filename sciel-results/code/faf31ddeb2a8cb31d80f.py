"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We seek min n such that 120 | phi(n).
    120 = 2^3 * 3 * 5.
    
    Let's check small n:
    phi(1) = 1
    phi(2) = 1
    phi(3) = 2
    phi(4) = 2
    phi(5) = 4
    phi(6) = 2
    phi(7) = 6
    phi(8) = 4
    phi(9) = 6
    phi(10) = 4
    phi(11) = 10
    phi(12) = 4
    phi(13) = 12
    phi(14) = 6
    phi(15) = 8
    phi(16) = 8
    phi(17) = 16
    phi(18) = 6
    phi(19) = 18
    phi(20) = 8
    phi(21) = 12
    phi(22) = 10
    phi(23) = 22
    phi(24) = 8
    phi(25) = 20
    phi(26) = 12
    phi(27) = 18
    phi(28) = 12
    phi(29) = 28
    phi(30) = 8
    phi(31) = 30
    phi(32) = 16
    phi(33) = 20
    phi(34) = 16
    phi(35) = 24
    phi(36) = 12
    phi(37) = 36
    phi(38) = 18
    phi(39) = 24
    phi(40) = 16
    phi(41) = 40
    phi(42) = 12
    phi(43) = 42
    phi(44) = 20
    phi(45) = 24
    phi(46) = 22
    phi(47) = 46
    phi(48) = 16
    phi(49) = 42
    phi(50) = 20
    phi(51) = 32
    phi(52) = 24
    phi(53) = 52
    phi(54) = 18
    phi(55) = 40
    phi(56) = 24
    phi(57) = 36
    phi(58) = 28
    phi(59) = 58
    phi(60) = 16
    phi(61) = 60
    phi(62) = 30
    phi(63) = 36
    phi(64) = 32
    phi(65) = 48
    phi(66) = 20
    phi(67) = 66
    phi(68) = 32
    phi(69) = 44
    phi(70) = 24
    phi(71) = 70
    phi(72) = 24
    phi(73) = 72
    phi(74) = 36
    phi(75) = 40
    phi(76) = 36
    phi(77) = 60
    phi(78) = 24
    phi(79) = 78
    phi(80) = 32
    phi(81) = 54
    phi(82) = 40
    phi(83) = 82
    phi(84) = 24
    phi(85) = 64
    phi(86) = 42
    phi(87) = 56
    phi(88) = 40
    phi(89) = 88
    phi(90) = 24
    phi(91) = 72
    phi(92) = 44
    phi(93) = 60
    phi(94) = 46
    phi(95) = 72
    phi(96) = 32
    phi(97) = 96
    phi(98) = 42
    phi(99) = 60
    phi(100) = 40
    
    ...
    
    Let's check candidates where phi(n) could be 120.
    If n is prime, phi(n) = n-1. So n-1 = 120 => n=121 (not prime).
    If n = p*q (distinct primes), phi(n) = (p-1)(q-1) = 120.
    Possible pairs (p-1, q-1) that multiply to 120:
    (1, 120) -> p=2, q=121 (not prime)
    (2, 60) -> p=3, q=61 (prime). n = 3*61 = 183. phi(183) = 2*60 = 120.
    (3, 40) -> p=4 (not prime)
    (4, 30) -> p=5, q=31 (prime). n = 5*31 = 155. phi(155) = 4*30 = 120.
    (5, 24) -> p=6 (not prime)
    (6, 20) -> p=7, q=21 (not prime)
    (8, 15) -> p=9 (not prime)
    (10, 12) -> p=11, q=13 (prime). n = 11*13 = 143. phi(143) = 10*12 = 120.
    
    So n=143 is a candidate.
    Are there smaller ones?
    
    Check n < 143.
    We need phi(n) to be a multiple of 120.
    Since n < 143, phi(n) <= n-1 < 142.
    So phi(n) must be exactly 120 (since 240 > 142).
    
    So we just need to find if there is any n < 143 such that phi(n) = 120.
    
    Let's list n with phi(n) = 120.
    We found 143, 155, 183.
    Are there others?
    
    Case 1: n is prime. n-1=120 => n=121 (not prime).
    Case 2: n = p^k. phi(n) = p^{k-1}(p-1) = 120.
    p-1 must divide 120.
    If p=2, 2^{k-1} = 120 (impossible, 120 not power of 2).
    If p=3, 3^{k-1}(2) = 120 => 3^{k-1} = 60 (impossible).
    If p=5, 5^{k-1}(4) = 120 => 5^{k-1} = 30 (impossible).
    If p=7, 7^{k-1}(6) = 120 => 7^{k-1} = 20 (impossible).
    If p=11, 11^{k-1}(10) = 120 => 11^{k-1} = 12 (impossible).
    If p=13, 13^{k-1}(12) = 120 => 13^{k-1} = 10 (impossible).
    If p=31, 31^{k-1}(30) = 120 => 31^{k-1} = 4 (impossible).
    If p=61, 61^{k-1}(60) = 120 => 61^{k-1} = 2 (impossible).
    
    Case 3: n = p*q (distinct primes). (p-1)(q-1) = 120.
    We found (11,13) -> 143, (5,31) -> 155, (3,61) -> 183.
    Other factor pairs of 120:
    (1, 120) -> 2, 121(no)
    (2, 60) -> 3, 61 -> 183
    (3, 40) -> 4(no)
    (4, 30) -> 5, 31 -> 155
    (5, 24) -> 6(no)
    (6, 20) -> 7, 21(no)
    (8, 15) -> 9(no)
    (10, 12) -> 11, 13 -> 143
    
    Case 4: n = p^a * q^b ... more complex.
    If n has 3 prime factors, say p,q,r.
    phi(n) = p^{a-1}q^{b-1}r^{c-1}(p-1)(q-1)(r-1) = 120.
    Minimal product of (p-1)(q-1)(r-1) for distinct primes 2,3,5 is 1*2*4=8.
    120/8 = 15.
    So we could have p=2, q=3, r=5.
    phi(2^a 3^b 5^c) = 2^{a-1} 3^{b-1} 4 * (1) * 2 * 4? No.
    phi(2^a) = 2^{a-1} (for a>=1)
    phi(3^b) = 2*3^{b-1}
    phi(5^c) = 4*5^{c-1}
    Product = 2^{a-1} * 2*3^{b-1} * 4*5^{c-1} = 2^{a+2} * 3^{b-1} * 5^{c-1} = 120 = 2^3 * 3 * 5.
    
    Comparing powers:
    5^{c-1} = 5 => c-1=1 => c=2.
    3^{b-1} = 3 => b-1=1 => b=2.
    2^{a+2} = 2^3 => a+2=3 => a=1.
    So n = 2^1 * 3^2 * 5^2 = 2 * 9 * 25 = 450.
    phi(450) = 120.
    450 > 143.
    
    What if primes are 2,3,7?
    phi(2^a 3^b 7^c) = 2^{a-1} * 2*3^{b-1} * 6*7^{c-1} = 12 * 2^{a-2} * 3^{b-1} * 7^{c-1}.
    120 = 12 * 10.
    2^{a-2} * 3^{b-1} * 7^{c-1} = 10 = 2 * 5.
    7 must divide RHS, impossible.
    
    What if primes are 2,5,3? Same as above.
    
    What if n = 2^a * 3^b?
    phi(n) = 2^{a-1} * 2*3^{b-1} = 2^a * 3^{b-1} = 120 = 2^3 * 3 * 5.
    5 must divide LHS, impossible.
    
    What if n = 2^a * 5^b?
    phi(n) = 2^{a-1} * 4*5^{b-1} = 2^{a+1} * 5^{b-1} = 120 = 2^3 * 3 * 5.
    3 must divide LHS, impossible.
    
    So no solution with 2 or 3 prime factors is smaller than 143?
    Wait, did I miss any 2-prime factor cases?
    (p-1)(q-1) = 120.
    We checked all divisors.
    
    Did I miss any 1-prime factor cases (prime powers)?
    Checked.
    
    So 143 is the smallest.
    
    Let's double check if any n < 143 has phi(n) divisible by 120.
    Since phi(n) < n < 143, phi(n) can only be 120.
    We established that the only solutions to phi(n)=120 are 143, 155, 183, 450, etc.
    The smallest is 143.
    
    Therefore, the answer is 143.
    """
    return 143
# EVOLVE-BLOCK-END
