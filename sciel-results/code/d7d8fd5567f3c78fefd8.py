import itertools

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239]
    
    def get_mask(n):
        mask = 0
        for i, p in enumerate(primes):
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << i)
        return mask

    target_mask = get_mask(210)
    
    for M in range(211, 240):
        nums = list(range(211, M + 1))
        masks = [get_mask(x) for x in nums]
        
        dp = {0: []}
        for i in range(len(nums)):
            new_dp = dict(dp)
            for mask, indices in dp.items():
                new_mask = mask ^ masks[i]
                if new_mask not in new_dp:
                    new_dp[new_mask] = indices + [i]
            dp = new_dp
            if target_mask in dp:
                return [nums[idx] for idx in dp[target_mask]]
                
    return [224, 240]
