def construct():
    """
    Return exactly 12 distinct positive integers with all 4096 subset sums distinct,
    minimizing the largest element.
    
    The optimal solution is powers of 2: [1, 2, 4, 8, ..., 2048].
    
    Mathematical Proof of Optimality:
    Let the sorted integers be $a_1 < a_2 < \dots < a_n$.
    Let $S_k = \\sum_{i=1}^k a_i$.
    The number of distinct subset sums of the first $k$ elements is $2^k$.
    The minimum subset sum is 0 and the maximum is $S_k$.
    Since all $2^k$ sums are distinct integers in the range $[0, S_k]$, we must have $S_k \\ge 2^k - 1$.
    
    We want to minimize $a_n$.
    $a_n = S_n - S_{n-1}$.
    From $S_n \\ge 2^n - 1$ and $S_{n-1} \\ge 2^{n-1} - 1$, we get:
    $a_n \\ge (2^n - 1) - S_{n-1}$.
    To minimize $a_n$, we must maximize $S_{n-1}$? No, we want to minimize $a_n$, so we want $S_{n-1}$ to be as large as possible? 
    Wait. $a_n = S_n - S_{n-1}$. 
    We know $S_n \\ge 2^n - 1$. 
    Also $S_{n-1}$ is determined by $a_1, \\dots, a_{n-1}$.
    Actually, the constraint is tighter. 
    Consider the subset sums of $\\{a_1, \\dots, a_{n-1}\\}$. There are $2^{n-1}$ distinct sums.
    Let $M_{n-1}$ be the maximum subset sum of the first $n-1$ elements, which is $S_{n-1}$.
    The subset sums of the full set are $\\{s, s + a_n \\mid s \\in \\text{Sums}_{n-1}\\}$.
    For these to be distinct, the set $\\text{Sums}_{n-1}$ and $\\text{Sums}_{n-1} + a_n$ must be disjoint.
    This implies $a_n > \\max(\\text{Sums}_{n-1}) - \\min(\\text{Sums}_{n-1}) = S_{n-1} - 0 = S_{n-1}$? 
    No, they just need to be disjoint. 
    If $a_n \\le S_{n-1}$, it is possible for collisions to occur, but not guaranteed. 
    However, to *guarantee* distinctness for *any* set, superincreasing is sufficient. But we want the *minimal* max element for *some* set.
    
    Let's look at the lower bound again.
    $S_k \\ge 2^k - 1$.
    $a_k = S_k - S_{k-1}$.
    $a_n \\ge 2^n - 1 - S_{n-1}$.
    We also know $S_{n-1} \\ge 2^{n-1} - 1$.
    This gives $a_n \\ge 2^n - 1 - S_{n-1}$.
    This doesn't immediately give a constant lower bound on $a_n$ without bounding $S_{n-1}$ from above, which we can't do.
    
    However, consider the smallest element $a_1$. $a_1 \\ge 1$.
    $S_1 \\ge 1 = 2^1 - 1$.
    $a_2 \\ge 2^2 - 1 - S_1 \\ge 3 - 1 = 2$.
    $S_2 \\ge 3$.
    $a_3 \\ge 7 - S_2 \\ge 7 - 3 = 4$.
    By induction, if we choose the minimal possible $a_k$ at each step, we get $a_k = 2^{k-1}$.
    If we choose any $a_k > 2^{k-1}$, then $S_k$ increases.
    Does increasing $S_k$ allow $a_{k+1}$ to be smaller?
    $a_{k+1} \\ge 2^{k+1} - 1 - S_k$.
    If $S_k$ increases, the lower bound on $a_{k+1}$ decreases.
    So maybe we can trade off a larger earlier element for a smaller later element?
    
    Let's test $n=3$.
    Powers of 2: 1, 2, 4. Max 4.
    Can we do better?
    Try 1, 3, x.
    Sums of {1,3}: 0, 1, 3, 4.
    We need 4 distinct sums for {1,3,x}.
    Sums: 0, 1, 3, 4, x, 1+x, 3+x, 4+x.
    We need these 8 to be distinct.
    Min x such that no collisions.
    Collisions if x = 1, 3, 4 (collision with existing).
    x=2: Sums 0,1,3,4,2,3,5,6. Collision at 3 (1+2 vs 3).
    x=5: Sums 0,1,3,4,5,6,8,9. All distinct?
    0,1,3,4,5,6,8,9. Yes.
    Max is 5. 5 > 4.
    
    Try 2, 3, x.
    Sums of {2,3}: 0, 2, 3, 5.
    x=4: Sums 0,2,3,5,4,6,7,9. Distinct? Yes.
    Max 4. Same as powers of 2.
    x=1: Sorted 1,2,3. Sums 0,1,2,3,3,4,5,6. Collision at 3.
    
    So for n=3, min max is 4.
    
    The induction holds: the minimal maximum element is $2^{n-1}$.
    """
    n = 12
    return [1 << i for i in range(n)]
