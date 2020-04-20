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


def reconstruct_video(data):

    # Load Mask
    mask = np.load(home_directory + "/Mask.npy")
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    number_of_frames = np.shape(data)[0]

    plt.ion()
    for frame in range(number_of_frames):
        template = np.zeros((600, 608))
        frame_data = data[frame]
        print(frame_data)
        np.put(template, indicies, frame_data)
        template = np.nan_to_num(template)
        template = ndimage.gaussian_filter(template, 1)

        plt.imshow(template, vmin=0, cmap='inferno', vmax=1.5)

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    plt.close()


def get_stimuli_average(preprocessed_data, stimuli_onsets):

    #Get Data From All Trials
    all_trials = []
    for onset in stimuli_onsets:
        trial_data = get_single_trial_trace(onset, preprocessed_data)
        #reconstruct_video(trial_data)
        all_trials.append(trial_data)

    #Get Mean From Each Trial
    number_of_timepoints = np.shape(all_trials[0])[0]
    number_of_pixels = np.shape(all_trials[0])[1]
    trial_average = np.zeros((number_of_timepoints, number_of_pixels))

    for timepoint in range(number_of_timepoints):
        timepoint_data_list = []

        for trial in all_trials:
            timepoint_data_list.append(trial[timepoint])

        timepoint_data_list = np.array(timepoint_data_list)
        timepoint_mean = np.mean(timepoint_data_list, axis=0)
        trial_average[timepoint] = timepoint_mean


    reconstruct_video(trial_average)
    return trial_average


def get_single_trial_trace(onset, preprocessed_data):

    trial_data = []

    #Get Baseline Mean and SD
    baseline_data = preprocessed_data[onset + trial_start:onset]
    baseline_means                  = np.mean(baseline_data, axis=0)
    baseline_standard_deviations    = np.std(baseline_data, axis=0)

    window_start = onset + trial_start
    window_stop = onset + trial_end
    for timepoint in range(window_start, window_stop):
        window_data = preprocessed_data[timepoint - window_size : timepoint + window_size]
        window_mean = np.mean(window_data, axis=0)
        window_difference = np.subtract(window_mean, baseline_means)
        window_z_score = np.divide(window_difference, baseline_standard_deviations)
        #window_z_score = window_difference
        trial_data.append(window_z_score)
    trial_data = np.array(trial_data)


    return trial_data


def save_evoked_responses(name, matrix):

    #Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    #Get File Name
    filename = save_directory + "/" + name + "_Activity_Matrix.npy"
    np.save(filename, matrix)

    print("Save Directory", save_directory)
    print("Filename", filename)



def save_trial_details(name, stimuli_onsets):

    #0 - Window Start
    #1 - Window End
    #2 - Window Size

    number_of_trials = np.shape(stimuli_onsets)[0]
    print("Number of trials", number_of_trials)

    trial_details = np.zeros((number_of_trials, 3), dtype=int)

    for trial in range(number_of_trials):
        onset = stimuli_onsets[trial]
        window_start = onset + trial_start
        window_stop = onset + trial_end

        trial_details[trial, 0] = int(window_start)
        trial_details[trial, 1] = int(window_stop)
        trial_details[trial, 2] = int(window_size)

    # Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get File Name
    filename = save_directory + "/" + name + "_Trial_Details.npy"
    np.save(filename, trial_details)





if __name__ == '__main__':

    #Load Data
    home_directory = r"/home/matthew/Documents/2020_03_12/1"
    preprocessed_data_file_location = home_directory + "/Heamocorrected_Data_Masked_Tables.h5"
    stimuli_onsets_location         = home_directory + "/Stimuli_Onsets"
    responses_save_location         = home_directory + "/Stimuli_Evoked_Responses"

    #Load Processed Data
    table = tables.open_file(preprocessed_data_file_location, mode='r')
    preprocessed_data = table.root.Data

    #Calculate Average Responses
    trial_start = -15
    trial_end = 150
    window_size = 3

    visual_block_visual_1_onsets = np.load(stimuli_onsets_location + "/Visual_Block_Rewarded_Onset_Frames.npy")
    visual_block_visual_2_onsets = np.load(stimuli_onsets_location + "/Visual_Block_Unrewarded_Onset_Frames.npy")
    odour_block_visual_1_onsets  = np.load(stimuli_onsets_location + "/Odour_Block_Rewarded_Onset_Frames.npy")
    odour_block_visual_2_onsets  = np.load(stimuli_onsets_location + "/Odour_Block_Unrewarded_Onset_Frames.npy")

    #Save Average Responses
    if not os.path.exists(responses_save_location):
        os.mkdir(responses_save_location)

    """
    visual_block_visual_1_average = get_stimuli_average(preprocessed_data, visual_block_visual_1_onsets)
    visual_block_visual_2_average = get_stimuli_average(preprocessed_data, visual_block_visual_2_onsets)
    odour_block_visual_1_average  = get_stimuli_average(preprocessed_data, odour_block_visual_1_onsets)
    odour_block_visual_2_average  = get_stimuli_average(preprocessed_data, odour_block_visual_2_onsets)



    save_evoked_responses("Visual_Block_Visual_1", visual_block_visual_1_average)
    save_evoked_responses("Visual_Block_Visual_2", visual_block_visual_2_average)
    save_evoked_responses("Odour_Block_Visual_1",  odour_block_visual_1_average)
    save_evoked_responses("Odour_Block_Visual_2",  odour_block_visual_2_average)
    """

    save_trial_details("Visual_Block_Visual_1", visual_block_visual_1_onsets)
    save_trial_details("Visual_Block_Visual_2", visual_block_visual_2_onsets)
    save_trial_details("Odour_Block_Visual_1",  odour_block_visual_1_onsets)
    save_trial_details("Odour_Block_Visual_2",  odour_block_visual_2_onsets)