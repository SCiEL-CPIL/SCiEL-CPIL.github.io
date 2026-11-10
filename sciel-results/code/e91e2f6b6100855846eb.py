import math

def construct():
    max_mod = 128
    primes = []
    for i in range(2, max_mod + 1):
        is_prime = True
        for j in range(2, int(math.isqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    # Use residue 0 for all. Since moduli are pairwise coprime (primes),
    # gcd(p_i, p_j) = 1, so 0 == 0 mod 1 is the only condition, which is trivially satisfied for disjointness?
    # Wait. Disjointness condition: r1 != r2 mod gcd(m1, m2).
    # If gcd = 1, r1 mod 1 = 0 and r2 mod 1 = 0.
    # So r1 == r2 mod 1.
    # The classes are {x | x = r1 + k*m1} and {y | y = r2 + l*m2}.
    # They are disjoint if there is no x such that x = r1 + k*m1 = r2 + l*m2.
    # This implies r1 - r2 = l*m2 - k*m1.
    # If gcd(m1, m2)=1, there is ALWAYS a solution to m1*k - m2*l = r1 - r2.
    # So if gcd(m1, m2)=1, the classes INTERSECT!
    
    # My previous reasoning was WRONG.
    # Let's re-read the task carefully.
    # "Every two infinite residue classes must be disjoint"
    # "verifier uses the exact gcd criterion"
    
    # Two arithmetic progressions $r_1 \pmod{m_1}$ and $r_2 \pmod{m_2}$ are disjoint
    # if and only if $r_1 \not\equiv r_2 \pmod{\gcd(m_1, m_2)}$.
    
    # If $\gcd(m_1, m_2) = 1$, then $r_1 \equiv r_2 \pmod 1$ is ALWAYS true.
    # So they are NEVER disjoint.
    
    # Therefore, we CANNOT use pairwise coprime moduli!
    # We need $\gcd(m_1, m_2) > 1$ for every pair to have a chance of being disjoint?
    # No, if $\gcd(m_1, m_2) = d > 1$, we can choose $r_1, r_2$ such that $r_1 \not\equiv r_2 \pmod d$.
    # If $\gcd(m_1, m_2) = 1$, we CANNOT choose $r_1, r_2$ to be disjoint.
    
    # So, we must select a set of moduli such that for EVERY pair, $\gcd(m_i, m_j) > 1$.
    # This means the set of moduli must be "pairwise non-coprime".
    
    # This is a completely different constraint.
    # We want to maximize the number of moduli $m_1, \dots, m_k \in [2, 128]$
    # such that $\gcd(m_i, m_j) > 1$ for all $i \ne j$.
    # And then we must verify if there exist residues $r_i$ such that
    # $r_i \not\equiv r_j \pmod{\gcd(m_i, m_j)}$ for all $i \ne j$.
    
    # If we pick all even numbers, $\gcd$ is at least 2.
    # Can we pick all even numbers in [2, 128]?
    # Count: 64 numbers.
    # If we pick all even numbers, $\gcd(m_i, m_j) \ge 2$.
    # We need to assign residues $r_i \in [0, m_i-1]$ such that
    # $r_i \not\equiv r_j \pmod{\gcd(m_i, m_j)}$.
    
    # This is a constraint satisfaction problem.
    # If we just pick all even numbers, is it always solvable?
    # Consider $m_1=2, m_2=4, m_3=6$.
    # $\gcd(2,4)=2 \implies r_1 \not\equiv r_2 \pmod 2$.
    # $\gcd(2,6)=2 \implies r_1 \not\equiv r_3 \pmod 2$.
    # $\gcd(4,6)=2 \implies r_2 \not\equiv r_3 \pmod 2$.
    # We need 3 distinct values mod 2? No, only 2 values (0, 1).
    # So we cannot have 3 classes if all pairwise gcds are 2.
    # We need a clique in the graph where edges represent "can be disjoint".
    
    # Actually, the problem is simpler if we look at the structure.
    # If we pick moduli that are all multiples of a common prime $p$,
    # then $\gcd(m_i, m_j)$ is a multiple of $p$.
    # If we set all $r_i = 0$, then $0 \equiv 0 \pmod{\gcd}$. They intersect.
    # We need $r_i \not\equiv r_j \pmod d_{ij}$.
    
    # Let's try a different strategy.
    # Pick moduli $m_i = 2 \cdot p_i$ where $p_i$ are distinct primes?
    # No, let's look at the "pairwise non-coprime" graph.
    # The maximum size of a set of integers in $[2, N]$ such that every pair has gcd > 1 is known.
    # It is the maximum size of a subset of $\{2, \dots, N\}$ with no coprime pairs.
    # This is equivalent to finding the maximum independent set in the coprime graph,
    # or maximum clique in the non-coprime graph.
    # For $N=128$, the set of all even numbers has size 64.
    # Can we add any odd number?
    # If we add an odd number $o$, it must share a factor with ALL even numbers in the set.
    # The even numbers include 2, 4, 6, 8...
    # $o$ must share a factor with 2? No, $o$ is odd, gcd(o, 2)=1.
    # So we cannot have any odd number if 2 is in the set.
    # So if we pick all evens, we can't add odds.
    # Size 64.
    
    # Can we do better?
    # What if we don't pick 2?
    # If we pick all multiples of 3: 3, 6, 9, ..., 126.
    # Count: 42.
    # Smaller than 64.
    
    # So 64 is a strong candidate for the number of moduli.
    # Now, can we assign residues to these 64 even moduli such that they are all pairwise disjoint?
    # Let $S = \{2, 4, 6, \dots, 128\}$.
    # For any $m_i, m_j \in S$, $\gcd(m_i, m_j) \ge 2$.
    # We need $r_i \not\equiv r_j \pmod{\gcd(m_i, m_j)}$.
    
    # Consider $m_i = 2$. $r_2 \in \{0, 1\}$.
    # Consider $m_j = 4$. $r_4 \in \{0, 1, 2, 3\}$.
    # $\gcd(2, 4) = 2$. Condition: $r_2 \not\equiv r_4 \pmod 2$.
    # If $r_2 = 0$, $r_4$ must be odd.
    # If $r_2 = 1$, $r_4$ must be even.
    
    # This looks like a graph coloring problem on the intersection graph.
    # Given the complexity, a simple heuristic might fail.
    # However, 64 is much larger than 7.
    # Even if we can only realize half of them due to residue conflicts, it's better.
    
    # Let's try a specific construction for even moduli.
    # Let $m = 2k$.
    # If we set $r_m = m/2$?
    # $r_{2k} = k$.
    # Check $m_1=2 (k=1, r=1)$ and $m_2=4 (k=2, r=2)$.
    # $\gcd(2,4)=2$. $r_1=1, r_2=2$. $1 \not\equiv 2 \pmod 2$ (1 vs 0). OK.
    # Check $m_1=2 (r=1)$ and $m_3=6 (k=3, r=3)$.
    # $\gcd(2,6)=2$. $r_1=1, r_3=3$. $1 \not\equiv 3 \pmod 2$ (1 vs 1). FAIL.
    
    # So $r_m = m/2$ doesn't work.
    
    # Let's try $r_m = 1$ for all $m$?
    # $m_1=2, r_1=1$. $m_2=4, r_2=1$.
    # $\gcd(2,4)=2$. $1 \equiv 1 \pmod 2$. FAIL.
    
    # Let's try to solve this properly with a small search or a known pattern.
    # Actually, the task says "Maximize the number of classes".
    # If I can't find a valid set of 64, I should find a valid set of, say, 10-20.
    
    # Alternative: Use moduli that are powers of 2?
    # Current program does this: 2, 4, 8, 16, 32, 64, 128.
    # Residues: 1, 2, 4, 8, 16, 32, 64.
    # Count: 7.
    
    # Can we add more?
    # What if we use moduli $2, 4, 8, \dots$ and $3, 9, 27 \dots$?
    # No, $\gcd(2,3)=1$, so they intersect.
    
    # We MUST have $\gcd(m_i, m_j) > 1$ for all pairs.
    # This implies all moduli must share a common prime factor?
    # Not necessarily. $6, 10, 15$.
    # $\gcd(6,10)=2, \gcd(6,15)=3, \gcd(10,15)=5$. All > 1.
    # This is a "triangle" in the non-coprime graph.
    # Can we extend this?
    # $6, 10, 15, 30$.
    # $\gcd(6,30)=6, \gcd(10,30)=10, \gcd(15,30)=15$.
    # This works.
    
    # The set of all divisors of a highly composite number?
    # Let $M = \text{lcm}(1, \dots, K)$.
    # If we pick all divisors of $M$ that are $\ge 2$, they are pairwise non-coprime?
    # No. Divisors of 6: 2, 3, 6.
    # $\gcd(2,3)=1$. Fail.
    
    # So we need a set of numbers where EVERY pair shares a factor.
    # This is a clique in the non-coprime graph.
    # The largest such set in $[2, 128]$ is the set of all even numbers (size 64).
    # Or all multiples of 3 (size 42).
    # Or all multiples of 5 (size 25).
    
    # So 64 is the max possible moduli count.
    # Now, can we assign residues to the 64 even numbers?
    # This is a hard CSP.
    # However, we can try a greedy approach or a simple pattern.
    
    # Let's try $r_m = m \pmod 2$? No, $m$ is even, so $r_m=0$.
    # Then all $r_m=0$. $\gcd(m_i, m_j) \ge 2$. $0 \equiv 0$. Fail.
    
    # Let's try $r_m = 1$ for $m=2$, and $r_m = 0$ for $m>2$?
    # $m=2, r=1$.
    # $m=4, r=0$. $\gcd(2,4)=2$. $1 \not\equiv 0 \pmod 2$. OK.
    # $m=6, r=0$. $\gcd(2,6)=2$. $1 \not\equiv 0 \pmod 2$. OK.
    # $m=4, r=0$ and $m=6, r=0$. $\gcd(4,6)=2$. $0 \equiv 0 \pmod 2$. FAIL.
    
    # It seems difficult to assign residues for ALL 64.
    # But we can pick a SUBSET of even numbers that works.
    # For example, powers of 2 work with specific residues.
    # 2, 4, 8, 16, 32, 64, 128. (7 numbers).
    
    # What if we pick 2, 4, 8, ..., 128 AND 3, 9, 27...?
    # No, gcd(2,3)=1.
    
    # What if we pick 6, 12, 24, 48, 96?
    # And 10, 20, 40, 80?
    # $\gcd(6, 10)=2$.
    # $\gcd(12, 20)=4$.
    # $\gcd(24, 40)=8$.
    # $\gcd(48, 80)=16$.
    # $\gcd(96, 80)=16$.
    # $\gcd(6, 20)=2$.
    # All gcds > 1.
    # This set is pairwise non-coprime.
    # Count: 5 + 4 = 9.
    # Better than 7.
    
    # We can construct a larger set by taking multiples of 2 and 3?
    # No, 2 and 3 are coprime.
    # We need a set of moduli $M$ such that $\forall m_i, m_j \in M, \gcd(m_i, m_j) > 1$.
    # This is equivalent to $M$ being a subset of the union of multiples of some primes, such that every pair intersects.
    # The largest such set is the multiples of a single prime.
    # Multiples of 2: 64 numbers.
    # Can we find a valid residue assignment for a large subset of multiples of 2?
    
    # Let's try to maximize the count within multiples of 2.
    # We found 7 with powers of 2.
    # Can we add 6?
    # Current: 2, 4, 8, 16, 32, 64, 128.
    # Residues: 1, 2, 4, 8, 16, 32, 64.
    # Add 6. Need $r_6$ such that $r_6 \not\equiv r_i \pmod{\gcd(6, m_i)}$.
    # $\gcd(6, 2)=2 \implies r_6 \not\equiv 1 \pmod 2 \implies r_6$ even.
    # $\gcd(6, 4)=2 \implies r_6 \not\equiv 2 \pmod 2 \implies r_6$ odd.
    # Contradiction. So 6 cannot be added to this specific set.
    
    # So we need a different set or different residues.
    
    # Given the time limit and complexity, I will implement a backtracking search for the maximum clique in the "compatible" graph, or just a greedy heuristic that tries to add moduli one by one, checking if a valid residue exists.
    
    # However, a simpler improvement:
    # The current program uses powers of 2.
    # I will use a greedy algorithm that picks moduli from [2, 128] in an order that maximizes flexibility, and assigns residues using a constraint solver (backtracking) for the selected set.
    # To keep it fast, I'll limit the search or use a heuristic.
    
    # Actually, just returning the 31 primes was a trap because they are coprime.
    # I will return the 7 powers of 2 (current) but try to add more.
    # If I can't easily find more, I'll stick to a robust algorithm that finds a maximal set.
    
    # Let's try a simple backtracking for a set of size > 7.
    # Candidates: All numbers 2..128.
    # This is too big for naive backtracking.
    
    # Let's restrict to even numbers.
    # Try to find a large subset of even numbers with compatible residues.
    
    # I will implement a function that tries to build a list of pairs.
    # Start with empty list.
    # Iterate through candidates.
    # If candidate can be added (there exists a residue r such that for all existing pairs, gcd condition holds), add it.
    # To find r, we check all r in [0, m-1].
    
    # This is $O(N^2 \cdot M)$ where N is count, M is max modulus.
    # $128^2 \cdot 128 \approx 2M$ ops. Fast.

def construct():
    max_mod = 128
    candidates = list(range(2, max_mod + 1))
    
    # Sort candidates? Maybe descending or by number of divisors?
    # Let's just iterate.
    
    selected = [] # list of [r, m]
    
    for m in candidates:
        found = False
        for r in range(m):
            valid = True
            for r_j, m_j in selected:
                g = math.gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                found = True
                break
        # If not found, skip m
    
    return selected
