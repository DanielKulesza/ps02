"""
CS6476: Problem Set 2 Experiment file

This script contains a series of function calls that run your ps2
implementation and output images so you can verify your results.
"""


import cv2

import ps2


def draw_tl_center(image_in, center, state):
    """Marks the center of a traffic light image and adds coordinates
    with the state of the current image

    Use OpenCV drawing functions to place a marker that represents the
    traffic light center. Additionally, place text using OpenCV tools
    that show the numerical and string values of the traffic light
    center and state. Use the following format:

        ((x-coordinate, y-coordinate), 'color')

    See OpenCV's drawing functions:
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    Make sure the font size is large enough so that the text in the
    output image is legible.
    Args:
        image_in (numpy.array): input image.
        center (tuple): center numeric values.
        state (str): traffic light state values can be: 'red',
                     'yellow', 'green'.

    Returns:
        numpy.array: output image showing a marker representing the
        traffic light center and text that presents the numerical
        coordinates with the traffic light state.
    """
    cv2.circle(image_in,center,2,(0,0,0),3)
    cv2.putText(image_in, ("((%d, %d), %s)" % (center[0], center[1], state)), (center[0]+5,center[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2, cv2.CV_AA)
    cv2.putText(image_in, ("((%d, %d), %s)" % (center[0], center[1], state)), (center[0]+5,center[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.CV_AA)

    return image_in

def mark_traffic_signs(image_in, signs_dict):
    #print signs_dict
    """Marks the center of a traffic sign and adds its coordinates.

    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}

    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', 'warning', and 'traffic_light'.

    Use cv2.putText to place the coordinate values in the output
    image.

    Args:
        image_in (numpy.array): the image to draw on.
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.

    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """
    for key in signs_dict: 
        cv2.circle(image_in,signs_dict[key],2,(0,0,0),3)
        cv2.putText(image_in, ("(%d, %d)" % (signs_dict[key][0], signs_dict[key][1])), (signs_dict[key][0]-30,signs_dict[key][1]-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3, cv2.CV_AA)
        cv2.putText(image_in, ("(%d, %d)" % (signs_dict[key][0], signs_dict[key][1])), (signs_dict[key][0]-30,signs_dict[key][1]-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.CV_AA)

        cv2.putText(image_in, ("%s" %  key), (signs_dict[key][0]-25,signs_dict[key][1]+30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3, cv2.CV_AA)
        cv2.putText(image_in, ("%s" %  key), (signs_dict[key][0]-25,signs_dict[key][1]+30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, cv2.CV_AA)


    return image_in

def part_1():

    input_images = ['simple_tl', 'scene_tl_1', 'scene_tl_2', 'scene_tl_3']#'test_images/tl_green_299_287_background']#'scene_tl_3']
    output_labels = ['ps2-1-a-1', 'ps2-1-a-2', 'ps2-1-a-3', 'ps2-1-a-4']

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("output/{}.png".format(label), img_out)

        #tl = cv2.imread("input_images/{}.png".format(img_in))
        #cv2.imwrite("output/{}.png".format(label), ps2.traffic_light_detection(tl, radii_range))


def part_2():

    input_images = ['scene_dne_1', 
                    'scene_stp_1', 
                    'scene_constr_1',
                    'scene_wrng_1', 
                    'scene_yld_1']

    output_labels = ['ps2-2-a-1', 
                     'ps2-2-a-2', 
                     'ps2-2-a-3', 
                     'ps2-2-a-4',
                     'ps2-2-a-5']

    sign_fns = [ps2.do_not_enter_sign_detection, 
                ps2.stop_sign_detection,
                ps2.construction_sign_detection, 
                ps2.warning_sign_detection,
                ps2.yield_sign_detection]

    sign_labels = ['no_entry','stop','construction','warning','yield']#['no_entry', 'stop', 'construction', 'warning', 'yield']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite("output/{}.png".format(label), img_out)

        #sign_img = cv2.imread("input_images/{}.png".format(img_in))
        #cv2.imwrite("output/{}.png".format(label), fn(sign_img))


def part_3():

    input_images = ['scene_some_signs', 'scene_all_signs']
    output_labels = ['ps2-3-a-1', 'ps2-3-a-2']

    for img_in, label in zip(input_images, output_labels):

        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_4():
    input_images = ['scene_some_signs_noisy', 'scene_all_signs_noisy']
    output_labels = ['ps2-4-a-1', 'ps2-4-a-2']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_noisy(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_5a():
    input_images = ['img-5-a-1', 'img-5-a-2', 'img-5-a-3']
    output_labels = ['ps2-5-a-1', 'ps2-5-a-2', 'ps2-5-a-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_5b():
    input_images = ['img-5-b-1', 'img-5-b-2', 'img-5-b-3']
    output_labels = ['ps2-5-b-1', 'ps2-5-b-2', 'ps2-5-b-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def sign_test():
    input_images = [#'scene_dne_1']
                    #'test_images/stop_blank_top_left']
                    #'test_images/stop_blank_top_right']
                    #'test_images/stop_blank_bot_right']
                    #'scene_stp_1']
                    #'test_images/stop_249_149_blank']
                    #'test_images/stop_249_149_background']
                    #'scene_constr_1']
                    #'test_images/construction_150_200_blank']
                    #'scene_wrng_1']
                    #'test_images/warning_250_300_blank']
                    #'test_images/yield_173_358_blank']
                    #'scene_yld_1']
                    #'test_images/yield_173_358_background']
                    #'test_images/scene_all_signs']
                    #'test_images/yield_bot_left_blank']
                    #'scene_some_signs']
                    #'scene_all_signs']
                    #'scene_some_signs_noisy']
                    'scene_all_signs_noisy']

    output_labels = ['test_scene']
                     #'ps2-2-a-2'] 
                     #'ps2-2-a-3']
                     #'ps2-2-a-4']
                     #'ps2-2-a-5']

    sign_fns = [#ps2.do_not_enter_sign_detection]
                #ps2.stop_sign_detection]
                #ps2.construction_sign_detection]
                ps2.warning_sign_detection]
                #ps2.yield_sign_detection]

    sign_labels = ['construction'] #['no_entry', 'stop', 'construction', 'warning', 'yield']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        cv2.imwrite("output/{}.png".format(label), fn(sign_img))



def traffic_light_test():
    name = [#'scene_all_signs']
            #'scene_some_signs']
            #'scene_some_signs_noisy']
            'scene_all_signs_noisy']

    radii_range = range(10,30,1)
    img_in = cv2.imread("input_images/{}.png".format(name[0]))
    cv2.imwrite("output/{}.png".format('test_scene_tl'), ps2.noisy_traffic_light_detection(img_in))

if __name__ == '__main__':
    part_1()
    part_2()
    #sign_test()
    part_3()
    #traffic_light_test()
    part_4()
    part_5a()
    part_5b()
