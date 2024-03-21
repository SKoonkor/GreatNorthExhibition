import numpy as np

def offsets(x, y, c):
    x_offset = (x/60)*1920 - 640*c
    y_offset = (y/33.5)*1080
    return np.array([int(x_offset), int(y_offset)])

def scale(item_actual_size, item_screen_size):
    '''
    This function calculate the scale factor for the captured video based on the object size in the background image
        item_actual_size: the size of the object in real life in the unit of m
        item_screen_size: 
    '''
    return (item_screen_size*1.6)/(item_actual_size*18)
