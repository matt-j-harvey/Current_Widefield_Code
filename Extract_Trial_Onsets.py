import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec


def load_ai_recorder_file(ai_recorder_file_location):
    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    print("Number of seconds", number_of_seconds)

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    return data_matrix


def get_frame_indexes(frame_stream):
    frame_indexes = {}
    state = 1
    threshold = 2
    count = 0

    for timepoint in range(0, len(frame_stream)):

        if frame_stream[timepoint] > threshold:
            if state == 0:
                state = 1
                frame_indexes[timepoint] = count
                count += 1

        else:
            if state == 1:
                state = 0
            else:
                pass

    return frame_indexes


def get_step_onsets(trace, threshold=0.24, window=10):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times, onset_line


def split_visual_onsets_by_context(onsets_list):

    visual_1_onsets = onsets_list[0]
    visual_2_onsets = onsets_list[1]
    odour_1_onsets  = onsets_list[2]
    odour_2_onsets  = onsets_list[3]

    following_window_size = 7000
    combined_odour_onsets = odour_1_onsets + odour_2_onsets

    visual_block_stimuli_1, odour_block_stimuli_1 = split_stream_by_context(visual_1_onsets, combined_odour_onsets,
                                                                            following_window_size)
    visual_block_stimuli_2, odour_block_stimuli_2 = split_stream_by_context(visual_2_onsets, combined_odour_onsets,
                                                                            following_window_size)

    return [visual_block_stimuli_1, visual_block_stimuli_2, odour_block_stimuli_1, odour_block_stimuli_2]


def split_stream_by_context(stimuli_onsets, context_onsets, context_window):
    context_negative_onsets = []
    context_positive_onsets = []

    # Iterate Through Visual 1 Onsets
    for stimuli_onset in stimuli_onsets:
        context = False
        window_start = stimuli_onset
        window_end = stimuli_onset + context_window

        for context_onset in context_onsets:
            if context_onset >= window_start and context_onset <= window_end:
                context = True

        if context == True:
            context_positive_onsets.append(stimuli_onset)
        else:
            context_negative_onsets.append(stimuli_onset)

    return context_negative_onsets, context_positive_onsets


def visualise_onsets():

    plt.title("Visual 1")
    plt.plot(ai_recorder_data[visual_1_channel])
    plt.plot(visual_1_line)
    plt.show()

    plt.title("Visual 2")
    plt.plot(ai_recorder_data[visual_2_channel])
    plt.plot(visual_2_line)
    plt.show()

    plt.title("Odour 1")
    plt.plot(ai_recorder_data[odour_1_channel])
    plt.plot(odour_1_line)
    plt.show()

    plt.title("Odour 2")
    plt.plot(ai_recorder_data[odour_2_channel])
    plt.plot(odour_2_line)
    plt.show()

    plt.title("Frames")
    plt.plot(ai_recorder_data[frame_channel])
    frame_line = np.zeros(len(ai_recorder_data[frame_channel]))
    frame_times = list(frame_onsets.keys())
    for time in frame_times:
        frame_line[time] = 1
    plt.plot(frame_line)
    plt.show()


def get_nearest_frame(stimuli_onsets, frame_onsets):
    frame_times = frame_onsets.keys()
    nearest_frames = []
    window_size = 50

    for onset in stimuli_onsets:
        smallest_distance = 1000
        closest_frame = None

        window_start = onset - window_size
        window_stop  = onset + window_size

        for timepoint in range(window_start, window_stop):

            #There is a frame at this time
            if timepoint in frame_times:
                distance = abs(onset - timepoint)

                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_frame = frame_onsets[timepoint]

        nearest_frames.append(closest_frame)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Reward": 0,
        "Lick": 1,
        "Visual 1": 2,
        "Visual 2": 3,
        "Odour 1": 4,
        "Odour 2": 5,
        "Irrelevance": 6,
        "Running": 7,
        "Trial End": 8,
        "Camera Trigger": 9,
        "Camera Frames": 10,
        "LED 1": 11,
        "LED 2": 12
    }

    return channel_index_dictionary


if __name__ == '__main__':

    # Load Data
    home_directory              = r"/home/matthew/Documents/2020_03_12/1"
    ai_recorder_file_location   = home_directory + "/20200312-180413.h5"
    save_location               = home_directory + "/Stimuli_Onsets"
    ai_recorder_data            = load_ai_recorder_file(ai_recorder_file_location)
    channel_index_dictionary    = create_stimuli_dictionary()

    # Get Visual and Odour Onsets
    visual_1_channel = channel_index_dictionary["Visual 1"]
    visual_2_channel = channel_index_dictionary["Visual 2"]
    odour_1_channel  = channel_index_dictionary["Odour 1"]
    odour_2_channel  = channel_index_dictionary["Odour 2"]
    frame_channel    = channel_index_dictionary["LED 1"]

    #Get Frame Onsets - "Blue LED Onsets
    frame_onsets = get_frame_indexes(ai_recorder_data[frame_channel])

    #Get Stimuli Onsets
    visual_1_onsets, visual_1_line = get_step_onsets(ai_recorder_data[visual_1_channel])
    visual_2_onsets, visual_2_line = get_step_onsets(ai_recorder_data[visual_2_channel])
    odour_1_onsets, odour_1_line  = get_step_onsets(ai_recorder_data[odour_1_channel])
    odour_2_onsets, odour_2_line  = get_step_onsets(ai_recorder_data[odour_2_channel])

    #Visualise These Onsets To Manualy Check They Seem Ok
    visualise_onsets()

    #Split Onsets By Context
    onsets_list = [visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets]
    context_onset_list = split_visual_onsets_by_context(onsets_list)
    visual_block_stimuli_1 = context_onset_list[0]
    visual_block_stimuli_2 = context_onset_list[1]
    odour_block_stimuli_1  = context_onset_list[2]
    odour_block_stimuli_2  = context_onset_list[3]

    #Get Nearest Frames To Each Onset
    visual_block_stimuli_1_onset_frames = get_nearest_frame(visual_block_stimuli_1, frame_onsets)
    visual_block_stimuli_2_onset_frames = get_nearest_frame(visual_block_stimuli_2, frame_onsets)
    odour_block_stimuli_1_onset_frames  = get_nearest_frame(odour_block_stimuli_1,  frame_onsets)
    odour_block_stimuli_2_onset_frames  = get_nearest_frame(odour_block_stimuli_2,  frame_onsets)

    #Save Onsets
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    np.save(save_location + "/Visual_Block_Rewarded_Onset_Frames",   visual_block_stimuli_1_onset_frames)
    np.save(save_location + "/Visual_Block_Unrewarded_Onset_Frames", visual_block_stimuli_2_onset_frames)
    np.save(save_location + "/Odour_Block_Rewarded_Onset_Frames",    odour_block_stimuli_1_onset_frames)
    np.save(save_location + "/Odour_Block_Unrewarded_Onset_Frames",  odour_block_stimuli_2_onset_frames)
    np.save(save_location + "/Frame_Times", frame_onsets)

    print("Finished Extrating Trial Onsets")

