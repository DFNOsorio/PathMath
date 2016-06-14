from numpy import *


def line(point_i, point_i1):
    m = (point_i1[1] - point_i[1]) / (point_i1[0] - point_i[0])

    b = point_i1[1] - m * point_i1[0]

    return m, b


def point_checker(point_i, point_i1, point_2_check):

    m, b = line(point_i, point_i1)

    point_y = m * point_2_check[0] + b

    if point_y >= point_2_check[1]:
        return True
    else:
        return False


def add_point(up, down, up_point_on_index, up_point_next_index, down_point_next_index):

    good = point_checker(up_point_on_index, up_point_next_index, down_point_next_index)

    if good:
        up.append(up_point_next_index)
        down.append(down_point_next_index)
    else:
        up.append(down_point_next_index)
        down.append(up_point_next_index)
    return up, down


def add_missing_points(up, down, point1, point2, point2add):
    good = point_checker(point1, point2, point2add)

    if good:
        up = up.append(point2add)
    elif not good:
        down = down.append(point2add)
    return up, down


def join_arrays(up, down):

    return concatenate([up, flipud(down), [up[0, :]]])


def uncrosser(up, down):
    new_up = []
    new_down = [down[0]]

    for i in range(0, len(down) - 1):
        p_1 = down[i]
        p_2 = down[i+1]

        indexes = where(logical_and(up[:, 0] >= p_1[0], up[:, 0] <= p_2[0]))[0]

        if len(indexes) != 0:
            for j in indexes:
                if p_1[1] >= up[j, 1] or p_2[1] >= up[j, 1]:
                    new_down.append(up[j])
                else:
                    new_up.append(up[j])
        new_down.append(p_2)

    index = where(up[:, 0] > down[-1][0])[0]
    if len(index) > 0:
        for i in indexes:
            new_up.append(up[i])

    return new_up, new_down


def new_contour(points):
    up = []
    down = []
    indexes = []
    can_compare = False

    if len(points[0]) == 1:
        up.append(points[0][0])

    else:
        up.append(points[0][0])
        down.append(points[0][1])

    for i in arange(0, len(points)-1):
        up_point_on_index = up[-1]
        next_points = points[i+1]

        if len(next_points) > 1:

            up_point_next_index = next_points[0]
            down_point_next_index = next_points[1]

            up, down = add_point(up, down, up_point_on_index, up_point_next_index, down_point_next_index)

            if i != 0 and (len(up) >= 2 or len(down) >= 2):
                can_compare = True

        if len(next_points) == 1:
                indexes.append(i+1)

        if can_compare and len(indexes) > 1:

            if len(up) >= 2:
                point_1 = up[-2]
                point_2 = up[-1]

            else:
                point_1 = down[-2]
                point_2 = down[-1]

            m, b = line(point_1, point_2)
            new_indexes = []
            for j in indexes:
                if point_2[0] < points[j][0][0]:
                    new_indexes.append(j)
                else:
                    if points[j][0][0] * m + b < points[j][0][1]:
                        up.append(points[j][0])
                    if points[j][0][0] * m + b > points[j][0][1]:
                        down.append(points[j][0])
                    up = sorted(up, key=lambda h: h[0])
                    down = sorted(down, key=lambda h: h[0])
            indexes = new_indexes

    new_up, new_down = uncrosser(array(up), array(down))

    contour_final = join_arrays(array(new_up), array(new_down))

    return [array(new_up), array(new_down), contour_final]



