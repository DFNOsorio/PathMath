import numpy.random
from numpy import *
from pylab import *
from scipy import interpolate, spatial
import random
import os
from numpy.testing import assert_almost_equal, assert_approx_equal

import seaborn
import matplotlib.pyplot as plt
import bokeh as bk
from bokeh.io import hplot
from bokeh.plotting import figure, output_file, show
from COPs import COPy, COPx, smooth




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

#####


def join_and_sort(x, y):

    """ This function joins x and y arrays and sorts array elements using the x axis

    Parameters
    ----------
    x: array
      x axis coordinates
    y: array
      y axis coordinates

    Returns
    -------
    new_x: array
        x axis coordinates sorted

    new_y: array
        y axis coordinates sorted

    x_y_sorted: array
        joint sorted x and y coordinates
    """

    x_y = [[x[i], y[i], i] for i in arange(0, len(x))]
    x_y_sorted = array(sorted(x_y, key=lambda h: h[0]))

    new_x = x_y_sorted[:, 0]
    new_y = x_y_sorted[:, 1]

    return new_x, new_y, x_y_sorted


def get_windows(x, y, scanning_window=0):

    """ This function gets the contour points for scattered data. It segments the data along the x axis using
    predetermined window size and finds the maximum and minimum y for that interval.

    Parameters
    ----------
    x: array
        x axis coordinates

    y: array
        y axis coordinates

    scanning_window: float
        Value for the segmentation window. 0 to use the mean distance between points.

    Returns
    -------
    scanning_window: float
        Value for the segmentation window.

    points_total: array
        Array containing all the contour points

    points_on_window: array
        Array containing all the contour points, stored by scanning window.

    index_on_window: array
        Array containing the indexes for the contour points, stored by scanning window.

        """

    def list_populator(indexes):
        indexes = array(indexes)
        points = []

        dup_x = find(diff(new_x[indexes]) == 0)
        dup_y = find(diff(new_y[indexes]) == 0)

        index_equal = find(dup_x == dup_y)
        if len(index_equal) != 0:

            index_2_del = dup_x[index_equal]
            indexes = delete(indexes, index_2_del)

        index_on_window.append(indexes)
        [[points.append([new_x[i], new_y[i]]), points_total.append([new_x[i], new_y[i]])] for i in indexes]
        points_on_window.append(points)

    new_x, new_y, x_y_sorted = join_and_sort(x, y)

    window_end = new_x[0]
    points_on_window = []
    points_total = []
    index_on_window = []

    if scanning_window == 0:
        scanning_window = mean(abs(diff(new_x)))

    while window_end < new_x[len(new_x)-1] + scanning_window:


        window_start = window_end
        window_end = window_start + scanning_window

        indexes = list(where(logical_and(new_x >= window_start, new_x < window_end))[0])

        if len(indexes) != 0 and len(indexes) <= 2:
            list_populator(indexes)

        elif len(indexes) > 2:
            y_vec = new_y[indexes]
            max_y = find(y_vec == max(y_vec))[0]
            min_y = find(y_vec == min(y_vec))[0]
            indexes = array(indexes)
            indexes = list(indexes[[max_y, min_y]])
            list_populator(indexes)

    points_total = array(points_total)
    points_on_window = array(points_on_window)
    index_on_window = array(index_on_window)

    return scanning_window, points_total, points_on_window, index_on_window


def area_contour(points_total, show_plot=False):

    """ This function sorts the contour data, using first all the positive y points and afterwards adds all the negative
    points. It also adds the first point again in order to close the contour.

        Parameters
        ----------
        points_total: array
            x axis coordinates

        show_plot: boolean
            option to plot the contour

        Returns
        -------
        points_reshape: array
            contour path

        """

    points_positive = points_total[(points_total[:, 1] >= 0), :]
    points_negative = points_total[(points_total[:, 1] < 0), :]

    points_reshape = points_positive
    if len(points_positive) > 0:
        points_reshape = concatenate([points_reshape, flipud(points_negative), [points_positive[0, :]]])
    else:
        points_negative = flipud(points_negative)
        points_reshape = concatenate([points_negative, [points_negative[0, :]]])

    if show_plot:

        plt.plot(points_reshape[:, 0], points_reshape[:, 1], 'r--')
        regular_plot(points_reshape[:, 0], points_reshape[:, 1], 'Area Contour', 'x', 'y', plot_line='r-o')
        plt.fill_between(points_reshape[:, 0], min(points_reshape[:, 1]), points_reshape[:, 1])

    return points_reshape


def uncrosser_shortest_dist(contour_array):

    new_path = array([contour_array[0]])
    index_available = arange(1, len(contour_array))

    index, sense_of_direction = uncrosser_start(contour_array)

    while len(index_available) > 0:
        get_next_index(index, index_available, contour_array, sense_of_direction, "1")
        new_path = concatenate([new_path, [contour_array[index]]])

    new_path = concatenate([new_path, [contour_array[0]]])
    plt.plot(new_path[0, 0], new_path[0, 1], 'go')
    plt.plot(new_path[:, 0], new_path[:, 1], 'g--')
    regular_plot(new_path[:, 0], new_path[:, 1], 'Area Contour', 'x', 'y', plot_line='r-o')
    plt.fill_between(new_path[:, 0], min(new_path[:, 1]), new_path[:, 1])


def uncrosser_start(contour_array):
    index = 0
    min_index = where(contour_array[:, 0] == min(contour_array[:, 0]))[0][0]
    max_index = where(contour_array[:, 0] == max(contour_array[:, 0]))[0][0]
    if index != max_index:
        direction = True  # From min to max
    else:
        direction = False

    sense_of_direction = [min_index, max_index, direction]

    return index, sense_of_direction


def direction_filter(current_index, available_indexes, points_array, direction):

    available_x = points_array[available_indexes, 0]
    current_x = points_array[current_index, 0]

    if direction:
        return available_indexes[where(available_x >= current_x)[0]]
    else:
        return available_indexes[where(available_x <= current_x)[0]]


def get_next_index(current_index, index_available, contour_array, sense_of_direction, trend):
    # Filter using the direction of the contour

    filtered_indexes = direction_filter(current_index, index_available, contour_array, sense_of_direction[2])

    # Get distances from the current index to the available ones

    distance = zeros((len(filtered_indexes), 2))

    for i in range(0, len(filtered_indexes)):
        distance[i][0] = sqrt((contour_array[current_index][0] - contour_array[filtered_indexes[i]][0]) ** 2.0 +
                              (contour_array[current_index][1] - contour_array[filtered_indexes[i]][1]) ** 2.0)
        distance[i][1] = filtered_indexes[i]

    distance = array(sorted(distance, key=lambda h: h[0]))

    if current_index == 0 and distance[0, 1] == 12:
        distance = distance[1:]

    print distance
    # Select the x closer points


    # Score from 0 to x based on the distance, being x the furthest


    # Score based on the trend (if from the last and current index the x went up or went down)


    # Sum 2 scores


    # 2+ same scores?

        # Yes

            # Only one point not following the trend?

                # Yes

                    # Get the closest

                # No

                    # Get new index using the lowest trend score

        # No

            # Get new index using the lowest trend score


    # Get new trend


    # Check direction


    # Update available indexes


# index_available = delete(index_available, find(index_available == i))



def area_calc(contour_array):
    uncrosser_shortest_dist(contour_array)
    """ This function uses the contour path to calculate the area, using Green's theorem.

        Parameters
        ----------
        contour_array: array
            contour path

        Returns
        -------
        area: float
            value for the area within the contour

    """

    x = contour_array[:, 0]
    y = contour_array[:, 1]

    area = 0

    for i in arange(1, len(y) - 1, 1):
        area += (y[i - 1] * x[i] - x[i - 1] * y[i])
    return area / 2.0


def get_area(x, y, scanning_window=1, show_plot=False):

    """ This function calculates the area of a set of scattered points.

        Parameters
        ----------
        x: array
            x axis coordinates

        y: array
            y axis coordinates

        scanning_window: float
            Value for the segmentation window. 0 to use the mean distance between points.

        Returns
        -------
        area: float
            value for the area within the contour

     """

    scanning_window, points_total, points_on_window, index_on_window = get_windows(x, y, scanning_window)

    contour_array = area_contour(points_total, show_plot)

    area = area_calc(contour_array)

    return area, contour_array

###


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


def generate_random_data(xlim, ylim, number_points):
    x = []
    y = []
    [[x.append(random.uniform(xlim[0], xlim[1])), y.append(random.uniform(ylim[0], ylim[1]))] for _ in arange(0, number_points)]

    return x, y


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


def regular_plot(x, y, title, xlabel, ylabel, fontsize=20, plot_line='-'):

    fig = plt.figure()
    plt.plot(x, y, plot_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=fontsize)

    return fig


def two_plots(xx, yy, titles, xlabels, ylabels, fontsizes=[20,20], plot_line=['-', '-']):
    fig = plt.figure()
    for i in xrange(0, 2):
        plt.subplot(2, 1, i+1)
        plt.plot(xx[i], yy[i], plot_line[i])
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
        plt.title(titles[i], fontsize=fontsizes[i])
  
    return fig


def overlap(xx, yy, title, xlabel, ylabel, legend, fontsize=20, plot_line=['-', '-']):
    fig = plt.figure()
    for i in xrange(0, 2):
        plt.plot(xx[i], yy[i], plot_line[i])
  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=fontsize)
    plt.legend(legend)
  
    return fig

############################
#                          #
#      Bokeh Plotting      #
#                          #
############################


def print_simple_bokeh(x, y, title, xlabel, ylabel):

    bk.plotting.output_file("scatter_data.html")
    p = bk.plotting.figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel)
    p.line(x, y)
    bk.plotting.show(p)


def bokeh_subplot(x, y, title, xlabel, ylabel):
    bk.plotting.output_file("subplot_data.html")
    s1 = bk.plotting.figure(title=title[0], x_axis_label=xlabel[0], y_axis_label=ylabel[0])
    s1.circle(x[0], y[0], size=5, color='firebrick', alpha=0.5)
    s1.line(x[1], y[1], alpha=0.5, line_width=2, line_dash="dashed")

    s2 = bk.plotting.figure(title=title[1], x_axis_label=xlabel[1], y_axis_label=ylabel[1])
    s2.circle(x[1], y[1], size=5, color='olive', alpha=0.5)
    s2.patch(x[1], y[1], alpha=0.5, line_width=2)

    p = hplot(s1, s2)

    bk.plotting.show(p)
############################
#                          #
#      To Be Deleted       #
#                          #
############################

# Random scatter data



#x, y = generate_random_data([-20, 20], [-20, 20], 100)


area, contour_array = get_area(COPx[0:20], COPy[0:20], scanning_window=0, show_plot=True)

#print area

#print_simple_bokeh(contour_array[:, 0], contour_array[:, 1], "Generated Data", "x", "y")

figure1 = regular_plot(contour_array[:, 0], contour_array[:, 1], "Generated Data", "x", "y", plot_line='o--')





#bokeh_subplot([x, contour_array[:, 0]], [y, contour_array[:, 1]], ["Generated Data", "Contour Plot"], ["x", "x"], ["y", "y"])


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