import time

def construct():
    p = 1009
    N = 1008
    deadline = time.time() + 80  # 80s safety margin for 90s limit

    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Precompute self-inverse check
    is_self_inv = [False] * (p + 1)
    for x in range(1, p):
        if (x * x) % p == 1:
            is_self_inv[x] = True
            
    best_length = 0
    best_start = 1

    # Use a sliding window approach: for each start, find the maximum valid end
    # Instead of using a set (O(n) to add), we can use a boolean array for O(1) lookup
    # But more importantly, we can optimize by noting that once we pass an inverse pair,
    # we can't go back. Let's use the two-pointer technique.
    
    # Actually, the problem is: find the longest contiguous subarray [s, e] such that
    # for all x, y in [s, e], x*y % p != 1.
    # This is equivalent to: the interval [s, e] must not contain any pair (x, inv[x])
    # where x != inv[x], and must not contain any self-inverse element.
    
    # So we need the longest interval that avoids:
    # 1. Self-inverse elements (1 and 1008)
    # 2. Any pair (x, inv[x]) where x < inv[x] (we only need to check one direction)
    
    # Let's precompute for each position i, what is the earliest position j >= i 
    # such that [j, ...] would be invalid if we started at some point before j.
    # Actually, let's think differently.
    
    # For a fixed start s, the constraint is that for every x in [s, e], inv[x] must not be in [s, e].
    # This means e < min(inv[x] for x in [s, e] if inv[x] > x) and also e < min(inv[x] for x in [s, e] if inv[x] < s is false since we need inv[x] in [s,e]).
    
    # Wait, the condition is: for all x, y in [s, e], x*y != 1 mod p.
    # This is equivalent to: for all x in [s, e], inv[x] is NOT in [s, e].
    
    # So for a fixed s, as we increase e, the first time we add a value x such that inv[x] is already in [s, x-1] or inv[x] == x, we must stop.
    
    # Let's use the two-pointer (sliding window) approach:
    # Maintain a window [left, right] and a count of how many elements in the window have their inverse also in the window.
    # But actually, it's simpler: for each right, find the smallest left such that [left, right] is valid.
    
    # Let's try a different approach: for each right endpoint, find the maximum left such that [left, right] is valid.
    # Then the answer is max over right of (right - left + 1).
    
    # For a fixed right, [left, right] is valid iff:
    # 1. No self-inverse element in [left, right]
    # 2. For no x in [left, right] is inv[x] also in [left, right]
    
    # Let's precompute for each position i, the "conflict" with other positions.
    # Actually, let's just use the two-pointer with a frequency array.
    
    # For each x in [1, N], inv[x] is its modular inverse.
    # A window [L, R] is valid iff for all x in [L, R], inv[x] is not in [L, R].
    
    # Let's use two pointers:
    # right = 0, left = 1
    # We'll maintain a count array: count[x] = number of times x appears in [left, right]
    # But since all values are distinct in [1, N], count[x] is either 0 or 1.
    
    # Actually, let's think about it this way:
    # When we add right to the window, we need to check if inv[right] is already in the window.
    # If inv[right] is in [left, right-1], then the window becomes invalid.
    # Also, if right is self-inverse, the window becomes invalid.
    
    # So we can do:
    # For right from 1 to N:
    #   If right is self-inverse, or inv[right] is in [left, right-1], then we need to move left forward.
    #   Specifically, if inv[right] >= left, then we must move left to inv[right] + 1.
    #   Also, if right is self-inverse, we must move left to right + 1.
    
    # Let's implement this:
    
    # We need to know if inv[right] is in [left, right-1].
    # Since all values are distinct, inv[right] is in [left, right-1] iff left <= inv[right] <= right-1.
    
    # So the algorithm is:
    # left = 1
    # for right in 1 to N:
    #   if is_self_inv[right]:
    #     left = right + 1
    #   else:
    #     if left <= inv[right] <= right - 1:
    #       left = inv[right] + 1
    #   # Now [left, right] is valid
    #   if right - left + 1 > best_length:
    #     best_length = right - left + 1
    #     best_start = left
    
    # Let's verify this logic with a small example.
    # Suppose p = 7, N = 6.
    # Inverses: 1->1, 2->4, 3->5, 4->2, 5->3, 6->6
    # Self-inverse: 1, 6
    # 
    # right=1: self-inv, left=2. Window [2,1] is empty.
    # right=2: inv[2]=4. Is 2<=4<=1? No. Window [2,2] valid. Length 1.
    # right=3: inv[3]=5. Is 2<=5<=2? No. Window [2,3] valid. Length 2.
    # right=4: inv[4]=2. Is 2<=2<=3? Yes. left=3. Window [3,4] valid. Length 2.
    # right=5: inv[5]=3. Is 3<=3<=4? Yes. left=4. Window [4,5] valid. Length 2.
    # right=6: self-inv, left=7. Window [7,6] empty.
    # 
    # Best: [2,3] or [3,4] or [4,5], length 2.
    # Let's check [2,3]: 2*2=4!=1, 2*3=6!=1, 3*3=2!=1. Valid.
    # [3,4]: 3*3=2!=1, 3*4=5!=1, 4*4=2!=1. Valid.
    # [4,5]: 4*4=2!=1, 4*5=6!=1, 5*5=4!=1. Valid.
    # [2,4]: 2*4=1=1 mod 7. Invalid.
    # So the algorithm seems correct.
    
    left = 1
    for right in range(1, N + 1):
        if time.time() > deadline:
            break
            
        if is_self_inv[right]:
            left = right + 1
        else:
            inv_right = inv[right]
            # Check if inv_right is in [left, right-1]
            if left <= inv_right <= right - 1:
                left = inv_right + 1
        
        current_length = right - left + 1
        if current_length > best_length:
            best_length = current_length
            best_start = left

    return [best_start, best_length]
