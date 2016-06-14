from numpy import *

def point_checker(point_i, point_i1, point_2_check):

    m = (point_i1[1] - point_i[1]) / (point_i1[0] - point_i[0])

    b = point_i1[1] - m * point_i1[0]

    point_y = m * point_2_check[0] + b

    if point_y >= point_2_check[1]:
        return True
    else:
        return False


def new_contour(points):
    up = []
    down = []
    beging = 0

    if len(points[0]) == 1:
        beging = 1
        up.append(points[0][0])
    else:
        up.append(points[0][0])
        down.append(points[0][1])

    for i in arange(0, 4):
    #for i in arange(0, len(points)-1):
        up_point_on_index = up[-1]

        next_points = points[i+1]

        if len(next_points) > 1:

            up_point_next_index = next_points[0]
            down_point_next_index = next_points[1]

            good = point_checker(up_point_on_index, up_point_next_index, down_point_next_index)

            if good:
                up.append(up_point_next_index)
                down.append(down_point_next_index)
            else:
                up.append(down_point_next_index)
                down.append(up_point_next_index)

        if len(next_points) == 1:
            down.append(next_points[0])

    contour_final = join_arrays(array(up), array(down))
    return [array(up), array(down), contour_final]


def join_arrays(up, down):

    return concatenate([up, flipud(down), [up[0, :]]])
