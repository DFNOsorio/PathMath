from numpy import *
from scipy import spatial


#path to max

#path to min

#dirivadas


def initial_conditions_contour(contour_path):

    min_index_x = where(contour_path[:, 0] == min(contour_path[:, 0]))[0][0]
    max_index_x = where(contour_path[:, 0] == max(contour_path[:, 0]))[0][0]

    return [min_index_x, max_index_x, "L2R"]


def direction_filter(contour_path, current_index, available_indexes, direction):

    available_x = contour_path[available_indexes, 0]
    current_x = contour_path[current_index, 0]

    if direction == "L2R":
        return available_indexes[where(available_x >= current_x)[0]]
    else:
        return available_indexes[where(available_x <= current_x)[0]]


def direction_update(current_index, sense_of_direction):

    if current_index == sense_of_direction[0]:
        sense_of_direction[2] = "L2R"
    elif current_index == sense_of_direction[1]:
        sense_of_direction[2] = "R2L"

    return sense_of_direction


def get_scores(decision_array):


    print "hey"


def get_next_index(contour_path, current_index, available_indexes):

    decision_array = zeros((len(available_indexes), 6))
    dist_zero = -1

    for i in range(0, len(available_indexes)):
        decision_array[i][0] = (contour_path[available_indexes[i]][0] - contour_path[current_index][0])
        decision_array[i][1] = (contour_path[available_indexes[i]][1] - contour_path[current_index][1])

        decision_array[i][2] = spatial.distance.euclidean(contour_path[current_index], contour_path[available_indexes[i]])
        if decision_array[i][2] == 0:
            dist_zero = i

        decision_array[i][3] = decision_array[i][1] / decision_array[i][0]

        decision_array[i][4] = 1 if decision_array[i][1] >= 0 else -1
        decision_array[i][5] = available_indexes[i]

    if dist_zero != -1:
        decision_array = delete(decision_array, dist_zero, 0)

    decision = get_scores(decision_array)

    print decision

    return decision[0, 6]


def new_contour(contour_path):

    sense_of_direction = initial_conditions_contour(contour_path)

    current_index = sense_of_direction[0]

    new_path = [int(current_index)]

    index_available = arange(1, len(contour_path))

    while len(index_available) > 39:
        print len(index_available)
        dir_indexes = direction_filter(contour_path, current_index, index_available, sense_of_direction[2])
        current_index = get_next_index(contour_path, current_index, dir_indexes)
        sense_of_direction = direction_update(current_index, sense_of_direction)
        new_path = concatenate([new_path, [int(current_index)]])
        index_available = delete(index_available, where(index_available == current_index))

    return new_path