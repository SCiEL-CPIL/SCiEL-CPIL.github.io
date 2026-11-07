"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product as iproduct

def _is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 60  # safety margin

    # 210 = 2 * 3 * 5 * 7, all squarefree
    # We need distinct ints in (210, 10210] whose product * 210 is a perfect square.
    # This means the squarefree part of (210 * product) must be 1.
    # Equivalently, the XOR (mod 2) of prime exponents across 210 and all selected
    # numbers must be 0 for every prime.
    #
    # Since 210 has primes {2,3,5,7} each to power 1, the selected numbers'
    # squarefree parts must collectively supply each of {2,3,5,7} an odd number
    # of times, and no other primes at all.
    #
    # Strategy: enumerate all set partitions of {2,3,5,7}. Each partition gives
    # a set of squarefree "sf values". For each sf value d, the smallest number
    # > 210 with that sf is d * k^2 where k = floor(sqrt(210/d)) + 1 (adjusted).
    # Pick the best partition minimizing the max of those numbers.
    #
    # Also consider: can we do better by using MORE numbers where some sf values
    # cancel? E.g., if we use sf values {2, 2, 3, 5, 7}, the 2's cancel and we
    # still need 3*5*7=105. But that's 5 numbers vs fewer. The key insight is that
    # adding a pair of identical sf values doesn't change the net sf product but
    # adds numbers, potentially increasing the max. So partitions of {2,3,5,7}
    # into exactly the nonempty subsets are optimal (no redundant pairs).
    #
    # However, we should also check: could using a number whose sf contains a prime
    # NOT in {2,3,5,7} help? No, because that prime would need to appear in another
    # selected number to cancel, but then both numbers would have that extra prime,
    # and we'd need yet more numbers. This can only increase the max.
    #
    # So the optimal solution is among partitions of {2,3,5,7} into nonempty subsets.
    # Bell(4) = 15 partitions. We enumerate all and pick the one minimizing max.

    primes = [2, 3, 5, 7]

    # Generate all set partitions of a small list
    def get_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        for partition in get_partitions(elements[1:]):
            # Add first to each existing subset
            for i in range(len(partition)):
                new_subset = partition[i] + [first]
                new_partition = partition[:i] + [new_subset] + partition[i+1:]
                yield new_partition
            # Add first as new subset
            yield partition + [[first]]

    best_max = 10211
    best_set = None

    for partition in get_partitions(primes):
        # Compute sf value for each subset
        sf_values = []
        for subset in partition:
            sf = 1
            for p in subset:
                sf *= p
            sf_values.append(sf)

        # For each sf value, find the smallest n = sf * k^2 > 210
        # Ensure all numbers are distinct
        numbers = []
        used = set()
        valid = True

        for sf in sf_values:
            # Find smallest k such that sf * k^2 > 210
            k = math.isqrt(210 // sf) + 1
            while sf * k * k <= 210:
                k += 1
            # Search for distinct number
            found = False
            while sf * k * k <= 10210:
                n = sf * k * k
                if n not in used:
                    numbers.append(n)
                    used.add(n)
                    found = True
                    break
                k += 1
            if not found:
                valid = False
                break

        if valid and numbers:
            mx = max(numbers)
            if mx < best_max:
                best_max = mx
                best_set = numbers[:]

        if time.time() > deadline:
            break

    # Fallback: single element 210 * 4 = 840
    if best_set is None:
        best_set = [840]
        best_max = 840

    # Verify
    prod = 210
    for x in best_set:
        prod *= x
    if not _is_perfect_square(prod):
        best_set = [840]

    # Ensure pure Python ints in a list
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
