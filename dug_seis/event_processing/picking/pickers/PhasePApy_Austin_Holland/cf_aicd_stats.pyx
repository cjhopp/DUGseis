import numpy as np
import math
cimport numpy as np
cimport cython
from libc.stdio cimport printf

ctypedef np.double_t DOUBLE_t

cdef extern from "math.h":
    double sqrt(double m)
    double log10(double m)
    double fabs(double m)

@cython.boundscheck(False)
@cython.wraparound (False)
def cyOptStdDev(np.ndarray[DOUBLE_t, ndim=1] a, int n):
    cdef Py_ssize_t i
    cdef double m = 0.0
    for i in range(n):
        m += a[i]
    m /= n
    cdef double v = 0.0
    for i in range(n):
        v += (a[i] - m)**2

    #printf("%d\n", n)
    return sqrt(v / n)
# end func

def stats(np.ndarray[DOUBLE_t, ndim=1] data,
          np.ndarray[DOUBLE_t, ndim = 1] AIC,
          np.ndarray[DOUBLE_t, ndim = 1] AIC_deriv,
          int npts):

    # reverse indexing to remove the nan, np.std(data[:0]) is nan, starting index need to
    # be npts-2, if data array only has 1 sample, the std is 0, the log10(0) is inf
    cdef int k, i
    cdef double a, b
    for k in range(npts-2,0,-1):

      #a = k*np.log10(np.std(data[:k])**2)+(npts-k-1)*np.log10(np.std(data[k:])**2)
      stdb = cyOptStdDev(data[:k], k)
      stde = cyOptStdDev(data[k:], npts - k)
      a = k * log10(stdb*stdb) + (npts - k - 1) * log10(stde*stde)

      #print a,np.log10(np.std(data[k:]))
      if a == -float('inf'):
        a = AIC[k+1]
      AIC[k] = a
    AIC[0] = AIC [1]
    AIC[-1] = AIC[-2]

    for i in range(npts-1):
      b = fabs(AIC[i+1]-AIC[i])
      AIC_deriv[i+1] = b

    AIC_deriv[0] = AIC_deriv[1]

    return AIC, AIC_deriv
# end func
