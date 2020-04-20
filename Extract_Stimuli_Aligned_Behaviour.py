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


def create_stimuli_dictionary():

    channel_index_dictionary = {
    "Reward"            :0,
    "Lick"              :1,
    "Visual 1"          :2,
    "Visual 2"          :3,
    "Odour 1"           :4,
    "Odour 2"           :5,
    "Irrelevance"       :6,
    "Running"           :7,
    "Trial End"         :8,
    "Camera Trigger"    :9,
    "Camera Frames"     :10,
    "LED 1"             :11,
    "LED 2"             :12
    }

    return channel_index_dictionary


def load_trial_details(name):

    file_directory = home_directory + "/Stimuli_Evoked_Responses/" + name + "/" + name + "_Trial_Details.npy"
    print("File directory: ", file_directory)

    trial_details = np.load(file_directory)
    print("Trial details", np.shape(trial_details))

    trial_starts    = []
    max_duration    = 0

    number_of_trials = np.shape(trial_details)[0]
    for trial in range(number_of_trials):
        frame_start = trial_details[trial][0]
        frame_stop  = trial_details[trial][1]

        start_time = time_frame_dict[frame_start]
        stop_time = time_frame_dict[frame_stop]

        trial_starts.append(start_time)
        duration = stop_time - start_time
        if duration > max_duration:
            max_duration = duration

    return trial_starts, max_duration



def get_trial_average_trace(trial_starts, max_duration, channel_of_interest):

    channel_trace = ai_recorder_data[channel_index_dictionary[channel_of_interest]]

    number_of_trials = len(trial_starts)
    trial_matrix = np.zeros((number_of_trials, max_duration))

    for trial in range(number_of_trials):
        start = trial_starts[trial]
        stop  = start + max_duration
        trial_trace = channel_trace[start:stop]
        trial_matrix[trial] = trial_trace

    mean_trace = np.mean(trial_matrix, axis=0)
    standard_deviation = np.std(trial_matrix, axis=0)


    upper_bound = np.add(mean_trace, standard_deviation)
    lower_bound = np.subtract(mean_trace, standard_deviation)

    x_cords = list(range(len(mean_trace)))
    #plt.plot(x_cords, mean_trace, c='b')
    #plt.fill_between(x=x_cords, y1=mean_trace, y2=upper_bound, color='b', alpha = 0.2)
    #plt.fill_between(x=x_cords, y1=mean_trace, y2=lower_bound, color='b', alpha = 0.2)

    plt.show()

    return mean_trace, upper_bound, lower_bound


def get_average_behaviour_trace(name):

    # Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    trial_starts, max_duration = load_trial_details(name)

    behaviour_matrix = []
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Running"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Lick"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Reward"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Odour 1"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Odour 2"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Visual 1"))
    behaviour_matrix.append(get_trial_average_trace(trial_starts, max_duration, "Visual 2"))
    behaviour_matrix = np.array(behaviour_matrix)

    # Get File Name
    filename = save_directory + "/" + name + "_Behaviour_Matrix.npy"
    np.save(filename, behaviour_matrix)


if __name__ == '__main__':

    # File Directories
    home_directory = r"/home/matthew/Documents/2020_03_12/1"
    processed_file_location = home_directory + "Heamocorrected_Data_Masked.hdf5"
    ai_recorder_file_location = home_directory + "/20200312-180413.h5"
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"

    #Load Data
    ai_recorder_data = load_ai_recorder_file(ai_recorder_file_location)
    channel_index_dictionary = create_stimuli_dictionary()
    frame_time_dict = np.load(home_directory + "/Stimuli_Onsets/Frame_Times.npy", allow_pickle=True)
    frame_time_dict = frame_time_dict[()]
    time_frame_dict = {v: k for k, v in  frame_time_dict.items()} #Invert The Dictionary

    get_average_behaviour_trace("Visual_Block_Visual_1")
    get_average_behaviour_trace("Visual_Block_Visual_2")
    get_average_behaviour_trace("Odour_Block_Visual_1")
    get_average_behaviour_trace("Odour_Block_Visual_2")
