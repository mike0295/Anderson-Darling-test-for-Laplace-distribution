import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
import json
import pandas as pd

"""
This Anderson-Darling test for Laplacian distribution assumes that 
we do not know the location and scale parameters of the original distribution.

Null hypothesis: The sample data is drawn from a Laplacian distribution.
If result Anderson Statistic is larger than the critical value, 
the null hypothesis is rejected at the corresponding significance level.

Source for critical values:
https://www.tandfonline.com/doi/pdf/10.1081/SAC-9687287

Example code is at the bottom. Simply run the whole file to run example code
"""

def laplace_ad_test(data):
    data = np.array(data)
    loc = np.median(data)
    n = data.shape[0]
    scale = np.sum(np.abs(data-loc))/n

    z_i = np.zeros_like(data)

    for i, d in enumerate(data):
        if d < loc:
            z_i[i] = np.exp((d-loc)/scale)/2
        else:
            z_i[i] = 1 - np.exp((-d+loc)/scale)/2


    rank = np.arange(1, n+1)
    z_i = np.sort(z_i)
    rev_z_i = np.flip(z_i)
    s = (2*rank-1)/n
    s *= (np.log(z_i)+np.log(1-rev_z_i))
    s = np.sum(s)
    A_sq = -n-s

    print("Anderson statistic: {}\n".format(A_sq))
    fetch_crit_value(n)

    x = np.linspace(np.min(data), np.max(data), 1000)
    pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
    plt.plot(x, pdf, color='red', label='laplace')
    plt.hist(data, bins=85, density=True)
    plt.legend()
    plt.show()

    return A_sq


def fetch_crit_value(n):
    if n%2 == 1:
        table = np.array([[0.5,     -0.928269,  0.2916],
                          [0.75,    -0.569497,  0.2637],
                          [0.8,     -0.477586,  0.2636],
                          [0.85,    -0.367685,  0.2600],
                          [0.9,     -0.227272,  0.2525],
                          [0.95,    -0.017972,  0.2053],
                          [0.975,   0.163424,   -0.0755],
                          [0.990,   0.365898,   -0.2964],
                          [0.995,   0.503329,   -0.6978],
                          [0.9975,  0.623224,   -0.8344],
                          [0.999,   0.760512,   -1.0847]])
    else:
        table = np.array([[0.5,     -0.932288,  3.2310],
                          [0.75,    -0.571021,  3.2644],
                          [0.8,     -0.477644,  3.2895],
                          [0.85,    -0.368555,  3.3703],
                          [0.9,     -0.228027,  3.4067],
                          [0.95,    -0.018564,  0.2053],
                          [0.975,   0.162622,   3.2346],
                          [0.990,   0.365076,   3.0296],
                          [0.995,   0.500015,   2.7197],
                          [0.9975,  0.616555,   2.4751],
                          [0.999,   0.759613,   2.1387]])

    crit_val = {'Significance':[], 'Critical value':[]}
    for row in table:
        crit_val['Significance'].append(row[0])
        crit_val['Critical value'].append(np.exp(row[1]+row[2]/n))

    crit_val = pd.DataFrame.from_dict(crit_val)
    print("Critical value table\n", crit_val.to_string(index=False))


# Example run ###

# data = sst.laplace.rvs(size=1000)
# laplace_ad_test(data)

#################