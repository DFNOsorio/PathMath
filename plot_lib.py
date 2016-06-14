import bokeh as bk
from bokeh.io import hplot
from bokeh.plotting import figure, output_file, show

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