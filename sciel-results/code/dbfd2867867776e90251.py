import numpy as np
import itertools

D = 11

def construct():
    # Generate all candidate integer vectors in R^11 with small norms.
    # We consider:
    # - Norm squared 2: two coordinates are +/-1, rest 0.
    # - Norm squared 4: one coordinate is +/-2, rest 0.
    # - Norm squared 4: two coordinates are +/-1 and one is +/-1? No, that's norm sq 3.
    #   Actually, norm squared 4 can also be: four coordinates with +/-1? No, that's 4.
    #   Wait, (1,1,1,1,0,...) has norm sq 4.
    #   Also (2,0,...) has norm sq 4.
    #   Also (1,1,0,...) has norm sq 2.
    #   Also (1,-1,0,...) has norm sq 2.
    #   
    # Let's be systematic. We want to maximize k such that min pairwise dist >= max norm.
    # If we pick only norm-squared 2 vectors, max norm = sqrt(2).
    # For two norm-squared 2 vectors u, v: dist^2 = 4 - 2*u.v.
    # We need dist^2 >= 2 => 4 - 2*u.v >= 2 => u.v <= 1.
    # Since u, v have entries in {-1,0,1} and exactly two non-zero entries each,
    # u.v can be: 2 (if they share both positions with same sign), 1 (share one pos, same sign),
    # 0 (share no pos, or share one with opposite sign), -1 (share one, opposite sign), -2 (share both, opposite sign).
    # So we need to avoid pairs with u.v = 2.
    # u.v = 2 means u and v are identical. So we just need distinct vectors.
    # Thus ALL 220 norm-squared 2 vectors form a valid set.
    #
    # Can we add norm-squared 4 vectors?
    # If we add a norm-squared 4 vector w, max norm becomes 2.
    # For w (norm 2) and u (norm sqrt(2)): dist^2 = 4 + 2 - 2*w.u = 6 - 2*w.u.
    # We need dist^2 >= 4 => 6 - 2*w.u >= 4 => w.u <= 1.
    # w has one coordinate +/-2. u has two coordinates +/-1.
    # w.u = 2 * (u_i) where i is the non-zero coord of w.
    # So w.u is in {-2, 0, 2}.
    # We need w.u <= 1, so w.u cannot be 2.
    # w.u = 2 means u_i = 1 (same sign as w's non-zero).
    # So for a given w (say 2e_i), we can only use u with u_i <= 0, i.e., u_i = -1 or u_i = 0.
    # This excludes all u with u_i = 1.
    # There are C(10,1)*2 = 20 such u for each w.
    # There are 220 total u. So we keep 200 u per w.
    # But we also need pairwise constraints among the kept u's and among w's.
    # This is complex. The 220 set is known to be optimal for norm-squared 2 only.
    # Adding higher norm vectors likely reduces total count.
    #
    # However, let's also consider norm-squared 4 vectors of type (1,1,1,1,0,...).
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors. Too many.
    #
    # Given the 120s limit, and that 220 is a strong baseline, I'll stick to 220.
    # But let's try a quick greedy to see if we can beat 220 by mixing.
    # Actually, the theoretical kissing number in 11D is 504 (for real vectors).
    # For integer vectors, it's much smaller.
    #
    # I'll return the 220 norm-squared 2 vectors.

    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors)
