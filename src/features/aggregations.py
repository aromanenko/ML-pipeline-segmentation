from scipy.signal import cwt, find_peaks_cwt, ricker, welch

def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def percentile(n):
    """
    Calculate n - percentile of data
    """
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'perc%s' % n
    return percentile_


def variation(x):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    mean = x.mean()
    if mean != 0:
        return x.std() / mean
    else:
        return np.nan
    
def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """    
    return np.nansum(x * x)

def absolute_sum_of_changes(x):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.nansum(np.abs(np.diff(x)))
    
def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = x.mean()
    return np.where(x > m)[0].size

def count_below_mean(x):
    """
    Returns the number of values in x that are lower than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = x.mean()
    return np.where(x < m)[0].size


def count_above(t):
    """
    Returns the percentage of values in x that are higher than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    def count_above_(x):
        return np.sum(x >= t) / len(x)

    count_above_.__name__ = 'count_above%s' % t
    return count_above_

def count_below(t):
    """
    Returns the percentage of values in x that are lower than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    def count_below_(x):
        return np.sum(x <= t) / len(x)

    count_above_.__name__ = 'count_above%s' % t
    return count_above_mean

def first_location_of_minimum(x):
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def longest_strike_below_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(_get_length_sequences_where(x < x.mean())) if x.size > 0 else 0

def longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(_get_length_sequences_where(x > x.mean())) if x.size > 0 else 0


def mean_second_derivative_central(x):
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN

def number_crossing_m(x):
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than mean and the next is greater, or vice-versa.
    crossings.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: int
    """
    m = x.mean()
    positive = x > m
    return np.where(np.diff(positive))[0].size

def number_cwt_peaks(n):  
    """
    Number of different peaks in x.

    To estimamte the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """  
    def number_cwt_peaks_(x):

        return len(
            find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker)
        )

    return number_cwt_peaks_

def range_count(x):
    """
    Count observed values within the interval [mean - 3*std, max + 3*std).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the count of values within the range
    :rtype: int
    """
    min_ = x.mean() - 3 * x.std()
    max_ = x.mean() + 3 * x.std()
    return np.sum((x >= min_) & (x < max_))


def ratio_beyond_r_sigma(r):
    """
    Ratio of values that are more than r * std(x) (so r times sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :param r: the ratio to compare with
    :type r: float
    :return: the value of this feature
    :return type: float
    """
    def ratio_beyond_r_sigma_(x):
        return np.sum(np.abs(x - x.mean()) > r * x.std()) / x.size

    return ratio_beyond_r_sigma_

def root_mean_square(x):
    """
    Returns the root mean square (rms) of the time series.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sqrt(np.square(x).mean()) if len(x) > 0 else np.NaN

def symmetry_looking(r):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :r: is the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    def symmetry_looking_(x):
        mean_median_difference = np.abs(x.mean() - x.median())
        max_min_difference = x.max() - x.min()
        return mean_median_difference < (r * max_min_difference)

    return symmetry_looking_

def benford_correlation(x):
    """
     Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.array(
        [int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))]
    )

    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    return np.corrcoef(benford_distribution, data_distribution)[0, 1]

def cid_ce(x):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    x = np.diff(x)
    return np.sqrt(np.nansum(x * x))

