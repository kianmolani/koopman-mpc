""" 
...


Created by: Kian Molani
Last updated: Mar. 07, 2024

"""

import numpy as np


class Monomial:
    def __init__(self, m: int):
        """
        ...

        m (int) :: highest degree in monomial basis

        """

        self.m = m

    def forward(self, x):
        """
        ...

        x (np.array) :: eDMD input data matrices of dimension N x M

        return (np.array) :: lifted eDMD data matrices of dimension N x (M x K), where K = m + 1

        """

        res = np.empty((x.shape[0], 0))
        for degree in range(self.m + 1):
            res = np.hstack((res, np.power(x, degree)))

        return res
    
class Radial:
    def __init__(self):
        """
        ...

        """

        pass

    def forward(self, x):
        """
        ...

        """

        pass