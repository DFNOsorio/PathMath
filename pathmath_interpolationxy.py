import numpy.random
from numpy import *
from pylab import *
from scipy import interpolate
import random
import os
from numpy.testing import assert_almost_equal, assert_approx_equal

import seaborn
import matplotlib.pyplot as plt



def get_path_smooth_s(t, s, x, y, begin_interp=0, end_interp=-1, stol=0.1, smoothing_factor=0.1):
    """ This function computes a spatial interpolation.

    Parameters
    ----------
    t: array
      float timestamp (seconds)
    s: array
      float mouse distance from original x and y
    x: array
      int mouse cursor x position.
    y: array
      int mouse cursor y position.
    begin_interp: int
      index in array to begin interpolation (if we want just a part).
    end_interp: int
      index in array to end interpolation (if we want just a part).
    stol: float
      tolerance (samples/pixel)
    smoothing_factor: int
      value for the smoothing factor  

    Returns
    -------
    ss: array
      float distance values in interpolation.
    xs: array
      float mouse cursor x position spatial interpolated.
    ys: array
      float mouse cursor y position spatial interpolated.
    vs: array
      float spatial velocity
    angle_value: array
      float angular results
    curvature: array
      float curvature results
    tt: array
      list of ajusted times
    """
    ss = arange(s[begin_interp], s[-1]+stol, stol)
    # TODO: evaluate the usage of weights

    splxs = interpolate.UnivariateSpline(s, x)
    splxs.set_smoothing_factor(smoothing_factor)
    xs = splxs(ss)
    splys = interpolate.UnivariateSpline(s, y)
    splys.set_smoothing_factor(smoothing_factor)
    ys = splys(ss)

    splts = interpolate.UnivariateSpline(s, t)

    # TODO: Assert that dss is constant

    if len(ss) > 1:
        vs = diff(ss)[0]/splts(ss, nu=1)
    else:
        vs = -1

    # angle = atan(dy/dx)
    # unwrap removes descontinuities.
    angle_value = unwrap(arctan2(splys.derivative()(ss), splxs.derivative()(ss)))

    # c = (dx * ddy - dy * ddx) / ((dx^2+dy^2)^(3/2))
    curvature_top = splxs.derivative()(ss) * splys.derivative(2)(ss) - \
        splys.derivative()(ss) * splxs.derivative(2)(ss)

    curvature_bottom = (splxs.derivative()(ss)**2 + splys.derivative()(ss)**2) ** (3/2.0)

    curvature = curvature_top/curvature_bottom

    tt = arange(t[begin_interp], t[-1]+stol, stol)

    return ss, xs, ys, vs, angle_value, curvature, tt


def get_path_smooth_t(t, x, y, begin_interp=0, end_interp=-1, ttol=0.1, smoothing_factor=0.1):
    """ This function computes a temporal interpolation.

    Parameters
    ----------
    t: array
      float timestamp (seconds)
    x: array
      int mouse cursor x position.
    y: array
      int mouse cursor y position.
    begin_interp: int
      index in array to begin interpolation (if we want just a part).
    end_interp: int
      index in array to end interpolation (if we want just a part).
    ttol: float
      tolerance (samples/second)
    smoothing_factor: int
      value for the smoothing factor

    Returns
    -------
    tt: array
      float time values in interpolation.
    xt: array
      float mouse cursor x position spatial interpolated.
    yt: array
      float mouse cursor y position spatial interpolated.
    """

    t = t[begin_interp:end_interp]
    x = x[begin_interp:end_interp]
    y = y[begin_interp:end_interp]

    if len(find(diff(x) == 0)) > 0 and len(find(diff(y) == 0)) > 0:
        for i in find(diff(x) == 0):
            if i in find(diff(y) == 0):
                _x = x[:i]
                x_ = x[i+1:]
                x = concatenate((_x, x_))
                _y = y[:i]
                y_ = y[i+1:]
                y = concatenate((_y, y_))

    t = t[:len(x)]
    _tt = arange(t[0], t[-1], ttol)

    # TODO: evaluate the usage of weights
    if len(x) > 4:
        splxt = interpolate.UnivariateSpline(t, x)
        splxt.set_smoothing_factor(smoothing_factor)
        xt = splxt(_tt)
        splyt = interpolate.UnivariateSpline(t, y)
        splyt.set_smoothing_factor(smoothing_factor)
        yt = splyt(_tt)
    else:
        print x
        _tt = []
        xt = []
        yt = []

    return _tt, xt, yt


def join_and_sort(x, y):
    x_y = [[x[i], y[i], i] for i in arange(0, len(x))]
    x_y_sorted = array(sorted(x_y, key=lambda h: h[0]))

    new_x = x_y_sorted[:, 0]
    new_y = x_y_sorted[:, 1]
    original_indexs = x_y_sorted[:, 2]

    return new_x, new_y, x_y_sorted


def get_windows(x, y, scanning_window = 1):
    def list_populator(indexes):
        points = []
        index_on_window.append(indexes)
        [[points.append([new_x[i], new_y[i]]), points_total.append([new_x[i], new_y[i]])] for i in indexes]
        points_on_window.append(points)


    new_x, new_y, x_y_sorted = join_and_sort(x, y)
    window_end = new_x[0]
    points_on_window = []
    points_total = []
    index_on_window = []

    if scanning_window != 1:
        scanning_window = mean(abs(diff(new_x)))

    plt.xticks(arange(min(x), max(x) + scanning_window, scanning_window))
    while window_end < new_x[len(new_x)-1]:

        window_start = window_end
        window_end = window_start + scanning_window

        indexes = list(where(logical_and(new_x >= window_start, new_x < window_end))[0])

        if len(indexes) != 0 and len(indexes) <= 2:
            list_populator(indexes)

        elif len(indexes) > 2:
            plt.plot(new_x[indexes], new_y[indexes], 'rd')
            y_vec = new_y[indexes]
            max_y = find(y_vec == max(y_vec))[0]
            min_y = find(y_vec == min(y_vec))[0]
            indexes = array(indexes)
            indexes = list(indexes[[max_y, min_y]])
            list_populator(indexes)

    points_total = array(points_total)
    points_on_window = array(points_on_window)
    index_on_window = array(index_on_window)

    plt.plot(points_total[:, 0], points_total[:, 1],'-go')

    return scanning_window, points_total, points_on_window, index_on_window


def get_area(x, y, scanning_window=1):
    scanning_window, points_total, points_on_window, index_on_window = get_windows(x, y, scanning_window)
    return scanning_window


def get_s(x, y):
    """ This function calculates the distance traveled.

    Parameters
    ----------
    x: array
      int mouse cursor x position.
    y: array
      int mouse cursor y position.

    Returns
    -------
    s: array
      float cumulative distance traveled.
    """

    ds = sqrt(diff(x)**2+diff(y)**2)
    s = cumsum(concatenate(([0], ds)))

    return s


def get_v(t, s):
    """ This function calculates the velocity in time.

    Parameters
    ----------
    t: array
      float timestamp (seconds)
    s: array
      float cumulative sum of distance traveled.

    Returns
    -------
    diff(s)/diff(t[:len(s)]): array
      velocity (pixeis/sec)
    """

    return diff(s)/diff(t[:len(s)])

############################
#                          #
#      Test Functions      #
#                          #
############################

def generate_circle(r=1.0, s=0.01):
    """ This function generate a circle to test functions.

    Parameters
    ----------
    r: array
      radius of circle
    s: float
      step for time.

    Returns
    -------
    t: array
      time
    x: array
      values of x position
    y: array
      values of y position
    """

    t = arange(0, 2*pi, s)
    x = sin(t)*r
    y = cos(t)*r

    return t, x, y


def generate_random_circle():
    """ This function generate a random circle (r=1) to test functions.

    Returns
    -------
    t: array
      time
    x: array
      values of x position
    y: array
      values of y position
    """

    exp_range = numpy.random.exponential(scale=0.1, size=100)
    d = cumsum(exp_range[exp_range > 0.05])
    t = d[d < 2*pi]

    x = sin(t)
    y = cos(t)

    return t, x, y


def check_incremental_s(s):
    """ This function verifies if s is gradually increasing.

    Parameters
    ----------
    s: array
      values to analyse.

    Returns
    -------
    all(diff(s) > 0.0): bool
      True if array is gradually increasing.
    """

    return all(diff(s) > 0.0)


############################
#                          #
#     Print Functions      #
#                          #
############################

def regularPlot(x,y,title,xlabel,ylabel,fontsize = 20):
  fig = plt.figure()
  plt.plot(x,y,'o')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title, fontsize = fontsize)

  return fig

def twoPlots(xx,yy,titles,xlabels,ylabels,fontsizes = [20,20]):
  fig = plt.figure()
  for i in xrange(0,2):
    plt.subplot(2,1,i+1)
    plt.plot(xx[i],yy[i])
    plt.xlabel(xlabels[i])
    plt.ylabel(ylabels[i])
    plt.title(titles[i], fontsize = fontsizes[i])
  
  return fig

def overlap(xx,yy,title,xlabel,ylabel,legend,fontsize = 20):
  fig = plt.figure()
  for i in xrange(0,2):
    plt.plot(xx[i],yy[i])
  
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title, fontsize = fontsize)
  plt.legend(legend)
  
  return fig

############################
#                          #
#      To Be Deleted       #
#                          #
############################

#x = [-3.7,-3.4,-2.7,-2.3,-2.2,-2.1,-2.5,-3.0,-3.5,-4.0]
#y = [3.1,3.5,3.2,3.0,2.3,1.3,0.5,0.0,1.0,2.0]
x = []
y = []
[[x.append(random.uniform(-2, 2)), y.append(random.uniform(-2, 2))] for _ in arange(0, 100)]

figure1 = regularPlot(x, y, "Circle", "x", "y")

window_size = get_area(x, y, scanning_window=0 )




plt.show()


# [t,x,y] =  generate_circle(r=2)
#
#
#
# #figure1 = regularPlot(x,y,"Circle","x","y")
#
# s = get_s(x,y)
# v = get_v(s,t)
#
#
#
# #figure2 = regularPlot(t,s,"Path","t","s")
#
#
# ss, xs, ys, vs, angle_value, curvature, tt = get_path_smooth_s(t, s, x, y, begin_interp=0, end_interp=-1, stol=0.1, smoothing_factor=0.01)
#
#
#
# #figure3 = twoPlots([x,xs],[y,ys],["normal","interpolated"],["x","xs"],["y","ys"],fontsizes = [20,20])
# #figure4 = overlap([x,xs],[y,ys],"overlap","x","y",["normal","interpolated"])
#
# splxs = interpolate.UnivariateSpline(x, y)
#
#
# plt.show()