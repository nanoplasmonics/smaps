import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import chisquare
import scipy.stats as ss

# Custom Style Parameters
custom_style = {
    'axes.spines.left': True,  
    'axes.spines.bottom': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.edgecolor': 'grey',
}

# Apply Seaborn Style
plt.style.use("seaborn-white")
plt.rcParams.update(custom_style)

# Additional Plot Settings
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')


def load_data():
    """Function to load data using a Tkinter file dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])

    if not file_path:
        print("No file selected. Exiting...")
        return None

    try:
        with open(file_path, 'r') as file:
            data = [list(map(float, line.strip().split())) for line in file]
    except Exception as e:
        print("Error reading file:", e)
        return None

    return data


def post_processing(data, time_step, state_corrections, start_time=None, end_time=None):
    """Post-process data by identifying and correcting states based on event durations."""
    last_column = np.array(data)[:, -1]
    unique_values = np.unique(last_column)

    if len(unique_values) < 2 or len(unique_values) > 5:
        print("Error: Last column does not have 2-5 distinct values.")
        return None, None, None

    print(unique_values)
    sorted_values = sorted(unique_values)

    # State Classification
    if len(sorted_values) == 2:
        high_states = [sorted_values[-1]]
        low_states = [sorted_values[0]]
    elif len(sorted_values) == 3:
        high_states = sorted_values[1:]
        low_states = sorted_values[:2]

    high_state_durations, low_state_durations = [], []
    current_state, previous_state, duration, current_time = None, None, 0, 0
    high_event_count, low_event_count, start_index = 0, 0, 0
    mean_high_low_state = (sorted_values[-1] + sorted_values[0]) / 2

    # Iterate through data points
    for time_index, value in enumerate(last_column):
        if (start_time is None or current_time >= start_time) and (end_time is None or current_time <= end_time):
            if value in high_states or value in low_states:
                if current_state != value:
                    if current_state is not None:
                        if duration < 0.05:
                            mean_segment = np.mean(np.array(data)[:, -2][start_index:time_index])
                            if abs(mean_segment - mean_high_low_state) / mean_high_low_state < 0.1:
                                if previous_state is not None:
                                    state_corrections[start_index:time_index] = [previous_state] * (time_index - start_index)
                                    current_state = previous_state
                            else:
                                if current_state in high_states:
                                    high_state_durations.append(duration)
                                    high_event_count += 1
                                elif current_state in low_states:
                                    low_state_durations.append(duration)
                                    low_event_count += 1
                        else:
                            if current_state in high_states:
                                high_state_durations.append(duration)
                                high_event_count += 1
                            elif current_state in low_states:
                                low_state_durations.append(duration)
                                low_event_count += 1
                    previous_state = current_state
                    current_state = value
                    duration = 0
                    start_index = time_index
                duration += time_step

        current_time += time_step

    # Handle final state duration
    if current_state in high_states:
        high_state_durations.append(duration)
    elif current_state in low_states:
        low_state_durations.append(duration)

    return state_corrections


def calculate_state_durations(data, time_step, start_time=None, end_time=None):
    """Calculate durations for high and low states."""
    unique_values = np.unique(data)
    if len(unique_values) != 2:
        print("Error: Data does not have exactly 2 distinct values.")
        return None, None

    sorted_values = sorted(unique_values)
    high_state, low_state = sorted_values[-1], sorted_values[0]
    high_state_durations, low_state_durations = [], []
    current_state, duration, current_time = None, 0, 0
    high_event_count, low_event_count = 0, 0

    for value in data:
        if (start_time is None or current_time >= start_time) and (end_time is None or current_time <= end_time):
            if value == high_state:
                if current_state != value:
                    if current_state is not None:
                        if current_state == high_state:
                            high_state_durations.append(duration)
                            high_event_count += 1
                        elif current_state == low_state:
                            low_state_durations.append(duration)
                            low_event_count += 1
                    current_state = value
                    duration = 0
                duration += time_step
            elif value == low_state:
                if value != current_state:
                    if current_state is not None:
                        if current_state == high_state:
                            high_state_durations.append(duration)
                            high_event_count += 1
                        elif current_state == low_state:
                            low_state_durations.append(duration)
                            low_event_count += 1
                    current_state = value
                    duration = 0
                duration += time_step

        current_time += time_step

    if current_state == high_state:
        high_state_durations.append(duration)
        high_event_count += 1
    elif current_state == low_state:
        low_state_durations.append(duration)
        low_event_count += 1

    print("Low event count:", low_event_count)
    print("High event count:", high_event_count)
    return high_state_durations, low_state_durations


def plot_state_durations(high_state_durations, low_state_durations):
    """Plot histograms of state durations."""
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.hist(high_state_durations, bins='auto', color='blue', alpha=0.7, label='High State')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Duration Histogram of High State')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(low_state_durations, bins='auto', color='red', alpha=0.7, label='Low State')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Duration Histogram of Low State')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to load data, process it, and plot state durations."""
    data = load_data()
    if data is None:
        return None

    time_step = 0.00001
    times = [i * time_step for i in range(len(data))]
    last_column = [row[-1] for row in data]
    state_corrections = last_column.copy()

    state_corrections = post_processing(data, time_step, state_corrections)
    high_state_durations, low_state_durations = calculate_state_durations(state_corrections, time_step)
    if high_state_durations is None or low_state_durations is None or state_corrections is None:
        return None

    plot_state_durations(high_state_durations, low_state_durations)

    plt.figure(figsize=(16, 5))
    plt.plot(times, state_corrections, linewidth=2, color='deeppink', label='Corrected Fit')
    plt.plot(times, [row[-2] for row in data], linewidth=1, color='grey', alpha=0.5, label='3 Hz filtered')
    plt.xlabel("Time (s)")
    plt.ylabel("Transmission (V)")
    plt.show()

    return state_corrections


if __name__ == "__main__":
    state_corrections = main()
    if state_corrections is not None:
        print("State Corrections complete")
