"""Improved: Incremental GF(2) basis with bitmask index tracking for min-max search."""
import math
import time


# ---------- Prime sieve ----------
def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(range(start, n + 1, step)))
    return [i for i in range(2, n + 1) if is_prime[i]]


MAX_VAL = 10210
_PRIMES = _sieve(MAX_VAL)
_PRIME_INDEX = {p: i for i, p in enumerate(_PRIMES)}


def _bitmask_kernel(n):
    """Compute the bitmask of the squarefree kernel of n."""
    if n <= 1:
        return 0
    result = 0
    f = n
    for i, p in enumerate(_PRIMES):
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result |= (1 << i)
    if f > 1:
        result |= (1 << _PRIME_INDEX[f])
    return result


_KER_MASK = [0] * (MAX_VAL + 1)
for i in range(211, MAX_VAL + 1):
    _KER_MASK[i] = _bitmask_kernel(i)

_TARGET_MASK = _bitmask_kernel(210)


def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n


def _validate(result, N=210):
    if not result or len(result) > 100:
        return False
    if len(set(result)) != len(result):
        return False
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > MAX_VAL:
            return False
    prod = 1
    for x in result:
        prod *= x
    return _is_perfect_square(prod * N)


def _extract_indices_from_bitmask(bm, offset, count):
    """Extract indices from a bitmask, where bit i corresponds to element at offset+i."""
    indices = []
    i = 0
    while bm and len(indices) < count:
        if bm & 1:
            indices.append(offset + i)
        bm >>= 1
        i += 1
    return indices


def construct():
    N = 210
    start_time = time.time()
    deadline = start_time + 85.0

    best_result = None
    best_raw = float('inf')

    # Incremental basis construction
    # basis: list of (pivot_bit, kernel_value, index_bitmask)
    # index_bitmask: Python int where bit j is set if original element j is in this basis vector
    # Elements are numbered 0..(M-211) corresponding to values 211..M
    
    basis = []  # list of [pivot, ker, idx_bm]
    # We'll process elements in order: element i corresponds to value 211+i
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        idx = M - 211  # 0-based index
        ker = _KER_MASK[M]
        
        # Try to add this vector to the basis
        cur = ker
        cur_bm = 1 << idx
        
        # Reduce by existing basis vectors
        for i in range(len(basis)):
            pivot, b_ker, b_bm = basis[i]
            if cur & (1 << pivot):
                cur ^= b_ker
                cur_bm ^= b_bm
        
        if cur != 0:
            pivot = cur.bit_length() - 1
            # Insert maintaining order by pivot descending
            insert_pos = 0
            for i in range(len(basis)):
                if basis[i][0] > pivot:
                    insert_pos = i + 1
                else:
                    break
            basis.insert(insert_pos, [pivot, cur, cur_bm])
        
        # Check if target is in the span of current basis
        t = _TARGET_MASK
        t_bm = 0
        for pivot, b_ker, b_bm in basis:
            if t & (1 << pivot):
                t ^= b_ker
                t_bm ^= b_bm
        
        if t == 0 and t_bm != 0:
            # Extract solution
            indices = _extract_indices_from_bitmask(t_bm, 0, M - 211 + 1)
            subset = [211 + i for i in indices]
            
            if len(subset) >= 1 and len(subset) <= 100:
                if _validate(subset, N):
                    raw = max(subset) - N
                    if raw < best_raw:
                        best_raw = raw
                        best_result = list(subset)
                    # Since we're going M upward, first valid solution gives min raw
                    break
    
    # If not found, try fallback
    if best_result is None:
        # Fallback: size 1
        for n in range(211, MAX_VAL + 1):
            if _KER_MASK[n] == _TARGET_MASK:
                best_result = [n]
                break

    if best_result is None:
        best_result = [840]

    if _validate(best_result, N):
        return list(best_result)
    return [840]

if __name__ == "__main__":
    construct()
