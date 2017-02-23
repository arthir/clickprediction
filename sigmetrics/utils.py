import numpy as np
from decimal import Decimal

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

#The score is the probability P[Bin(n,alpha) >= m]
#where:
#m = number or URLs user sent that have CTR better than median/95th_percentile
#alpha = 0.5 for median, 0.05 for 95th percentile
#n = total number of URLs a user sent
def calc_score(alpha, m, n):
    try:
        return 1.0 - float(sum([Decimal(alpha ** l) * Decimal((1.0 - alpha) ** (n - l)) * Decimal(choose(n, l)) for l in xrange(m)]))
    except OverflowError:
        print 'OverflowError', alpha, m, n
        return 'NaN'

# alpha = 50 for median; 95 for 95th percentile
def get_percentile_url(urls, alpha): # [urls] -> ctr
    return np.percentile(urls.values(), alpha)


