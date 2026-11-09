"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Precompute: for each possible value v, which pairs (b,c) with b<=c have v | (b+c)
    # We'll use this for fast checking when adding elements
    # For a given set S and candidate x, we need to check:
    # 1. x as 'a': no b,c in S with x<b<=c and x|(b+c)
    # 2. x as 'b': no a,c in S with a<x<=c and a|(x+c)
    # 3. x as 'c': no a,b in S with a<b<=x and a|(b+x)

    # Precompute divisibility: divs[d] = set of s values where d|s
    # Actually, let's precompute for each d, the list of residue classes
    # For checking x|(b+c): we need b+c ≡ 0 (mod x), i.e., c ≡ -b (mod x)

    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        size = len(S_list)
        for i in range(size):
            a = S_list[i]
            for j in range(i + 1, size):
                b = S_list[j]
                for k in range(j, size):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def check_add(S_set, S_sorted, x):
        """
        Check if x can be added to S_set while maintaining the property.
        Optimized version with early termination.
        """
        if not S_sorted:
            return True

        size = len(S_sorted)

        # Role 1: x is 'a' (smallest in triple)
        # Need: for all b,c in S with x < b <= c, x does NOT divide b+c
        # Optimize: iterate b from first element > x
        b_start = 0
        while b_start < size and S_sorted[b_start] <= x:
            b_start += 1

        for i in range(b_start, size):
            b = S_sorted[i]
            # For this b, we need c >= b such that (b+c) % x != 0
            # i.e., c % x != (-b) % x
            target_residue = (-b) % x
            for j in range(i, size):
                c = S_sorted[j]
                if c % x == target_residue:
                    return False

        # Role 2: x is 'b' (middle)
        # Need: for all a,c in S with a < x <= c, a does NOT divide x+c
        # a must be < x, c must be >= x
        for i in range(size):
            a = S_sorted[i]
            if a >= x:
                break
            # For this a, check all c >= x in S
            for j in range(size):
                c = S_sorted[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Role 3: x is 'c' (largest)
        # Need: for all a,b in S with a < b <= x, a does NOT divide b+x
        for i in range(size):
            a = S_sorted[i]
            for j in range(i + 1, size):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False

        return True

    best_set = set()
    best_size = 0

    # Strategy 1: Start with a large tail, greedily add smaller elements
    # Key insight: for tail starting at t, the set {t, t+1, ..., n} is valid if
    # for all a < b <= c in the tail, a does not divide b+c.
    # For large enough t, this is automatically satisfied since b+c > 2t > a*n... 
    # Actually need b+c not divisible by a. For a in [t, n], b,c in [t, n], b+c in [2t, 2n].
    # a divides b+c only if b+c is a multiple of a. Since a >= t and b+c <= 2n,
    # the only possible multiple is b+c = a (impossible since b+c >= 2t > a for t > n/2)
    # or b+c = 2a (possible if 2a <= 2n, i.e., a <= n, which is always true).
    # So we need 2a not in [2t, 2n] range... wait, b+c can be 2a.
    # For tail {t,...,n}: a,b,c all in [t,n]. b+c = 2a means a = (b+c)/2.
    # Since b,c >= t, a >= t. And b+c <= 2n, so a <= n.
    # The condition is: no a,b,c in [t,n] with a<b<=c and b+c=2a.
    # b+c=2a means a is the average of b and c. With a<b<=c, we have a < (a+b)/2... 
    # Actually if b+c=2a, then a=(b+c)/2. Since b<=c, a>=b. But we need a<b.
    # So a < b <= c and a = (b+c)/2 implies a < b <= 2a-b, so c <= 2a-b.
    # This is possible. E.g., a=3, b=4, c=2? No, c>=b. a=4, b=5, c=3? No.
    # a=5, b=6, c=4? No, c>=b required.
    # Actually a=(b+c)/2 with a<b<=c: let b=a+d1, c=a+d2 with 0<d1<=d2.
    # Then a=(a+d1+a+d2)/2 = a+(d1+d2)/2, so d1+d2=0. Contradiction since d1>0.
    # So b+c=2a is IMPOSSIBLE when a<b<=c! Because (b+c)/2 > a when b>a.
    # 
    # What about b+c = ka for k>=2? b+c >= 2t, a <= n. ka = b+c.
    # For k=2: shown impossible above.
    # For k>=3: ka >= 3a >= 3t. But b+c <= 2n. So need 3t <= 2n, t <= 2n/3.
    # For t > 2n/3 = 170.67, i.e., t >= 171, no multiples possible.
    # For t in [n/2+1, 170], need to check k=3 case: 3a = b+c.

    # So tails starting at t >= 171 are automatically valid!
    # Let's verify: for t=171, a,b,c in [171,256]. b+c in [342,512]. 
    # a in [171,256]. Multiples of a in [342,512]: 2a in [342,512] (yes, 2*171=342, 2*256=512).
    # But we showed 2a=b+c impossible with a<b<=c.
    # 3a in [513,768]. b+c <= 512. So 3a > 512 >= b+c. No 3a in range.
    # So indeed t>=171 gives valid tails.

    # For t < 171, we need to verify. Let's just check.

    # Strategy 1: Try tails from t=171 down to 100
    for start_threshold in range(171, 99, -1):
        if time.time() > deadline - 5:
            break

        current_set = set(range(start_threshold, n + 1))
        if not current_set:
            continue

        # Verify initial set is valid
        if not is_valid_set(current_set):
            continue

        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

        # Try to add smaller elements in descending order
        candidates = list(range(start_threshold - 1, 0, -1))
        random.shuffle(candidates)

        S_sorted = sorted(current_set)
        for x in candidates:
            if time.time() > deadline - 3:
                break

            if check_add(current_set, S_sorted, x):
                current_set.add(x)
                S_sorted = sorted(current_set)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)

    # Strategy 2: Greedy from scratch - add elements in increasing order if valid
    if time.time() < deadline - 10:
        current_set = set()
        S_sorted = []
        for x in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if check_add(current_set, S_sorted, x):
                current_set.add(x)
                S_sorted = sorted(current_set)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

    # Strategy 3: Multiple random restarts with different starting points
    if time.time() < deadline - 15:
        for restart in range(30):
            if time.time() > deadline - 3:
                break

            # Random starting threshold
            start_threshold = random.randint(140, 180)
            current_set = set(range(start_threshold, n + 1))
            if not current_set:
                continue

            if not is_valid_set(current_set):
                continue

            S_sorted = sorted(current_set)

            # Add smaller elements in random order
            candidates = list(range(1, start_threshold))
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > deadline - 3:
                    break
                if check_add(current_set, S_sorted, x):
                    current_set.add(x)
                    S_sorted = sorted(current_set)

            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)

    # Strategy 4: Local search - try to improve by swapping elements
    if time.time() < deadline - 10 and best_set:
        best_list = sorted(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 5:
                    break
                # Try removing best_list[i] and adding a different element
                removed = best_list[i]
                temp_set = set(best_list[:i] + best_list[i+1:])
                S_sorted = sorted(temp_set)

                # Try to add a better element
                for x in range(1, n + 1):
                    if x in temp_set:
                        continue
                    if check_add(temp_set, S_sorted, x):
                        new_set = temp_set | {x}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_list = sorted(best_set)
                            best_size = len(best_set)
                            improved = True
                            break

    # Final verification
    if best_set and not is_valid_set(best_set):
        # Fallback: use a simple valid set
        fallback = set(range(171, n + 1))
        if is_valid_set(fallback):
            best_set = fallback
        else:
            # Try even larger tail
            for t in range(172, n + 1):
                fallback = set(range(t, n + 1))
                if is_valid_set(fallback):
                    best_set = fallback
                    break
            if not best_set:
                best_set = {256}

    return sorted(list(best_set))
# EVOLVE-BLOCK-END
