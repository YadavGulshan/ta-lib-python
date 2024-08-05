from datetime import datetime
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import pytest

import talib
from talib import func


def test_talib_version():
    assert talib.__ta_version__[:5] == b'0.4.0'


def test_num_functions():
    assert len(talib.get_functions()) == 158


def test_input_wrong_type():
    a1 = np.arange(10, dtype=int)
    with pytest.raises(Exception):
        func.MOM(a1)


def test_input_lengths():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(11, dtype=float)
    with pytest.raises(Exception):
        func.BOP(a2, a1, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a2, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a2, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a1, a2)


def test_input_allnans():
    a = np.arange(20, dtype=float)
    a[:] = np.nan
    r = func.RSI(a)
    assert np.all(np.isnan(r))


def test_input_nans():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(10, dtype=float)
    a2[0] = np.nan
    a2[1] = np.nan
    r1, r2 = func.AROON(a1, a2, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])
    r1, r2 = func.AROON(a2, a1, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])


def test_unstable_period():
    a = np.arange(10, dtype=float)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_unstable_period('EMA', 5)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, 7, 8])
    talib.set_unstable_period('EMA', 0)


def test_compatibility():
    a = np.arange(10, dtype=float)
    talib.set_compatibility(0)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_compatibility(1)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1.25, 2.125, 3.0625, 4.03125, 5.015625, 6.0078125, 7.00390625, 8.001953125])
    talib.set_compatibility(0)


def test_MIN(series):
    result = func.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 1] == 93.780
    assert result[i + 2] == 93.780
    assert result[i + 3] == 92.530
    assert result[i + 4] == 92.530
    values = np.array([np.nan, 5., 4., 3., 5., 7.])
    result = func.MIN(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, 4, 3, 3, 5])


def test_MAX(series):
    result = func.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 2] == 95.090
    assert result[i + 3] == 95.090
    assert result[i + 4] == 94.620
    assert result[i + 5] == 94.620


def test_MOM():
    values = np.array([90.0, 88.0, 89.0])
    result = func.MOM(values, timeperiod=1)
    assert_array_equal(result, [np.nan, -2, 1])
    result = func.MOM(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, -1])
    result = func.MOM(values, timeperiod=3)
    assert_array_equal(result, [np.nan, np.nan, np.nan])
    result = func.MOM(values, timeperiod=4)
    assert_array_equal(result, [np.nan, np.nan, np.nan])


def test_BBANDS(series):
    upper, middle, lower = func.BBANDS(
        series,
        timeperiod=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=talib.MA_Type.EMA
    )
    i = np.where(~np.isnan(upper))[0][0]
    assert len(upper) == len(middle) == len(lower) == len(series)
    # assert abs(upper[i + 0] - 98.0734) < 1e-3
    assert abs(middle[i + 0] - 92.8910) < 1e-3
    assert abs(lower[i + 0] - 87.7086) < 1e-3
    # assert abs(upper[i + 13] - 93.674) < 1e-3
    assert abs(middle[i + 13] - 87.679) < 1e-3
    assert abs(lower[i + 13] - 81.685) < 1e-3


def test_DEMA(series):
    result = func.DEMA(series)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert abs(result[i + 1] - 86.765) < 1e-3
    assert abs(result[i + 2] - 86.942) < 1e-3
    assert abs(result[i + 3] - 87.089) < 1e-3
    assert abs(result[i + 4] - 87.656) < 1e-3


def test_EMAEMA(series):
    result = func.EMA(series, timeperiod=2)
    result = func.EMA(result, timeperiod=2)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert i == 2


def test_CDL3BLACKCROWS():
    o = np.array(
        [39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 39.00, 40.32, 40.51,
         38.09, 35.00, 27.66, 30.80])
    h = np.array(
        [40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 40.84, 41.69, 40.84,
         38.12, 35.50, 31.74, 32.51])
    l = np.array(
        [35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 35.80, 39.26, 36.73,
         33.37, 30.03, 27.03, 28.31])
    c = np.array(
        [40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.29, 40.46, 37.08,
         33.37, 30.03, 31.46, 28.31])

    result = func.CDL3BLACKCROWS(o, h, l, c)
    assert_array_equal(result, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0])


def test_RSI():
    a = np.array([0.00000024, 0.00000024, 0.00000024,
                  0.00000024, 0.00000024, 0.00000023,
                  0.00000024, 0.00000024, 0.00000024,
                  0.00000024, 0.00000023, 0.00000024,
                  0.00000023, 0.00000024, 0.00000023,
                  0.00000024, 0.00000024, 0.00000023,
                  0.00000023, 0.00000023], dtype='float64')
    result = func.RSI(a, 10)
    assert_array_equal(result,
                       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0])
    result = func.RSI(a * 100000, 10)
    assert_array_almost_equal(result, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                       33.333333333333329, 51.351351351351347, 39.491916859122398, 51.84807024709005,
                                       42.25953803191981, 52.101824405061215, 52.101824405061215, 43.043664867691085,
                                       43.043664867691085, 43.043664867691085])


def test_MAVP():
    a = np.array([1, 5, 3, 4, 7, 3, 8, 1, 4, 6], dtype=float)
    b = np.array([2, 4, 2, 4, 2, 4, 2, 4, 2, 4], dtype=float)
    result = func.MAVP(a, b, minperiod=2, maxperiod=4)
    assert_array_equal(result, [np.nan, np.nan, np.nan, 3.25, 5.5, 4.25, 5.5, 4.75, 2.5, 4.75])
    sma2 = func.SMA(a, 2)
    assert_array_equal(result[4::2], sma2[4::2])
    sma4 = func.SMA(a, 4)
    assert_array_equal(result[3::2], sma4[3::2])
    result = func.MAVP(a, b, minperiod=2, maxperiod=3)
    assert_array_equal(result, [np.nan, np.nan, 4, 4, 5.5, 4.666666666666667, 5.5, 4, 2.5, 3.6666666666666665])
    sma3 = func.SMA(a, 3)
    assert_array_equal(result[2::2], sma2[2::2])
    assert_array_equal(result[3::2], sma3[3::2])


def test_MAXINDEX():
    import talib as func
    import numpy as np
    a = np.array([1., 2, 3, 4, 5, 6, 7, 8, 7, 7, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 15])
    b = func.MA(a, 10)
    c = func.MAXINDEX(b, 10)
    assert_array_equal(c, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 21])
    d = np.array([1., 2, 3])
    e = func.MAXINDEX(d, 10)
    assert_array_equal(e, [0, 0, 0])


def test_XBAR():
    import talib as func
    import numpy as np
    a = np.array([10., 9, 4, 5, 6, 7, 8, 9, 8, 8, 4, 5, 6, 7, 8, 9, 10])
    xhigh, xlow = func.XBAR(a, a, timeperiod=5)
    nan = np.NAN
    assert_array_equal(xhigh, [nan, nan, nan, nan, 10., 9., 8., 9., 9., 9., 9., 9., 8., 8., 8., 9., 10.])
    assert_array_equal(xlow, [nan, nan, nan, nan, 4., 4., 4., 5., 6., 7., 4., 4., 4., 4., 4., 5., 6., ])


def test_AVGXBAR():
    import talib as func
    import numpy as np

    def rolling_average(arr, window_size):
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    a = np.array([10., 9, 4, 5, 6, 7, 8, 9, 8, 8, 4, 5, 6, 7, 8, 9, 10])
    xhigh, xlow, xvol = func.AVGXBAR(a, a, a, timeperiod=5)
    xhigh: np.ndarray
    assert_array_equal(xlow[4:], rolling_average(a, 5))


def test_CDLWICK():
    import talib as func
    import numpy as np

    o = np.array([100.00, 101.50, 103.25, 102.75, 104.00, 103.50, 105.25, 106.00, 107.50, 108.75])
    h = np.array([102.25, 104.00, 105.50, 104.25, 106.75, 105.00, 107.75, 108.50, 110.00, 111.25])
    low = np.array([99.25, 100.75, 102.00, 101.50, 103.25, 102.00, 104.50, 105.25, 106.75, 107.50])
    c = np.array([101.75, 103.50, 102.50, 104.25, 103.75, 105.50, 106.25, 107.75, 109.00, 110.50])

    def calculate_wick_size():
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - low

        return lower_wick + upper_wick

    result: np.ndarray = func.CDLWICK(o, h, low, c)
    assert len(result) == len(o)
    assert_array_equal(calculate_wick_size(), result)


def test_CDLWICKPERCENT():
    import talib as func
    import numpy as np

    o = np.array([100.00, 101.50, 103.25, 102.75, 104.00, 103.50, 105.25, 106.00, 107.50, 108.75])
    h = np.array([102.25, 104.00, 105.50, 104.25, 106.75, 105.00, 107.75, 108.50, 110.00, 111.25])
    low = np.array([99.25, 100.75, 102.00, 101.50, 103.25, 102.00, 104.50, 105.25, 106.75, 107.50])
    c = np.array([101.75, 103.50, 102.50, 104.25, 103.75, 105.50, 106.25, 107.75, 109.00, 110.50])

    def calculate_wick_percent():
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - low
        bar_size = h - low

        return (lower_wick + upper_wick) / bar_size

    result: np.ndarray = func.CDLWICKPERCENT(o, h, low, c)
    assert len(result) == len(o)
    assert_array_equal(calculate_wick_percent(), result)


def test_CDLMAXBAR():
    import talib as func
    import numpy as np

    h = np.array([1., 2, 7, 3, 7, 6, 11, 9, 8, 2])
    low = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_period = 2

    result: np.ndarray = func.CDLMAXBAR(h, low, timeperiod=time_period)
    assert_array_equal(result, [np.nan, 1, 5, 5, 3, 3, 5, 5, 2, 7])


def test_ADR():
    import talib as func
    import numpy as np

    h = np.array([1., 2, 7, 3, 7, 6, 11, 9, 8, 2])
    low = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_period = 2

    result: np.ndarray = func.ADR(h, low, timeperiod=time_period)
    assert_array_equal(result, [np.nan, 1., 3., 2.5, 1.5, 2., 3., 3.5, 1., 3.5])


def test_ABR():
    import talib as func
    import numpy as np

    o = np.array([1., 2, 7, 3, 7, 6, 11, 9, 8, 2])
    c = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_period = 2

    result: np.ndarray = func.ABR(o, c, timeperiod=time_period)
    assert_array_equal(result, [np.nan, 1., 3., 2.5, 1.5, 2., 3., 3.5, 1., 3.5])


def test_PIVOT_POINTS():
    import talib as func
    import numpy as np

    h = np.array([102.25, 104.00, 105.50, 104.25, 106.75, 105.00, 107.75, 108.50, 110.00, 111.25])
    low = np.array([99.25, 100.75, 102.00, 101.50, 103.25, 102.00, 104.50, 105.25, 106.75, 107.50])
    c = np.array([101.75, 103.50, 102.50, 104.25, 103.75, 105.50, 106.25, 107.75, 109.00, 110.50])

    result: Tuple[np.ndarray] = func.PIVOTPOINTS(h, low, c)
    pp, r1, s1, r2, s2 = result

    expected_pp = np.full_like(h, np.nan, dtype=float)
    expected_r1 = np.full_like(h, np.nan, dtype=float)
    expected_r2 = np.full_like(h, np.nan, dtype=float)
    expected_s1 = np.full_like(h, np.nan, dtype=float)
    expected_s2 = np.full_like(h, np.nan, dtype=float)

    expected_pp[1:] = (h[:-1] + low[:-1] + c[:-1]) / 3
    expected_r1[1:] = 2 * expected_pp[1:] - low[:-1]
    expected_r2[1:] = expected_pp[1:] + (h[:-1] - low[:-1])
    expected_s1[1:] = 2 * expected_pp[1:] - h[:-1]
    expected_s2[1:] = expected_pp[1:] - h[:-1] + low[:-1]

    assert_array_equal(r1, expected_r1)
    assert_array_equal(r2, expected_r2)
    assert_array_equal(s1, expected_s1)
    assert_array_equal(s2, expected_s2)
    assert_array_equal(pp, expected_pp)


def test_VWAP():
    import talib as func
    import numpy as np
    import pandas as pd

    def calculate_vwap(group):
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        return (typical_price * group['volume']).cumsum() / group['volume'].cumsum()

    data = {
        "timestamp": pd.date_range(start="2024-04-01 08:30:00", periods=20, freq="5h"),
        "open": np.random.uniform(100, 110, 20),
        "high": np.random.uniform(110, 120, 20),
        "low": np.random.uniform(90, 100, 20),
        "close": np.random.uniform(100, 110, 20),
        "volume": np.random.randint(100, 1000, 20, dtype="int32")
    }
    df = pd.DataFrame(data)
    df['timestamp_unix'] = df['timestamp'].astype(int) // 10 ** 9
    df['date'] = df['timestamp'].dt.date
    df['vwap'] = df.groupby('date', group_keys=False).apply(calculate_vwap)

    vwap = func.VWAP(df['high'], df['low'], df['close'], df['volume'], df['timestamp_unix'].values.astype(np.int32))

    assert_array_almost_equal(vwap, df['vwap'])


if __name__ == "__main__":
    test_VWAP()
