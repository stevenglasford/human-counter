
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_signal_timing(
    output_dir="output",
    signal_cycle=90,
    green_time=50,
    red_time=None,
    intersections=5,
    speed_mps=13.41,
    distance_between_lights=400,
    num_vehicles=100,
    simulation_time=900,
    random_seed=42
):
    if red_time is None:
        red_time = signal_cycle - green_time

    np.random.seed(random_seed)
    vehicle_arrivals = np.sort(np.random.uniform(0, simulation_time, num_vehicles))

    stops_per_vehicle = []
    delay_per_vehicle = []

    for arrival_time in vehicle_arrivals:
        vehicle_stops = 0
        vehicle_delay = 0
        current_time = arrival_time

        for _ in range(intersections):
            travel_time = distance_between_lights / speed_mps
            current_time += travel_time
            time_into_cycle = current_time % signal_cycle

            if time_into_cycle > green_time:
                vehicle_stops += 1
                wait_time = signal_cycle - time_into_cycle
                vehicle_delay += wait_time
                current_time += wait_time

        stops_per_vehicle.append(vehicle_stops)
        delay_per_vehicle.append(vehicle_delay)

    df = pd.DataFrame({
        'Vehicle': range(num_vehicles),
        'Stops': stops_per_vehicle,
        'Delay (s)': delay_per_vehicle
    })

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1
    plt.figure()
    plt.plot(df['Vehicle'], df['Stops'], label='Stops per Vehicle')
    plt.xlabel('Vehicle Index')
    plt.ylabel('Stops')
    plt.title('Stops per Vehicle Across Corridor')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'stops_per_vehicle.png'))
    plt.close()

    # Plot 2
    plt.figure()
    plt.plot(df['Vehicle'], df['Delay (s)'], label='Delay per Vehicle', color='orange')
    plt.xlabel('Vehicle Index')
    plt.ylabel('Delay (seconds)')
    plt.title('Delay per Vehicle Across Corridor')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'delay_per_vehicle.png'))
    plt.close()

    # Summary analysis
    average_stops = np.mean(stops_per_vehicle)
    average_delay = np.mean(delay_per_vehicle)
    arrival_on_green = np.sum(np.array(stops_per_vehicle) == 0) / len(stops_per_vehicle) * 100
    red_light_prob = np.sum(np.array(stops_per_vehicle) > 0) / (len(stops_per_vehicle) * intersections) * 100

    analysis = f"""TRAFFIC SIGNAL TIMING ANALYSIS REPORT

Signal Cycle Length: {signal_cycle} seconds
Green Time: {green_time} seconds
Red Time: {red_time} seconds
Number of Intersections: {intersections}
Distance Between Lights: {distance_between_lights} meters

Total Vehicles Simulated: {num_vehicles}

--- Results ---
Average Stops per Vehicle: {average_stops:.2f}
Average Delay per Vehicle: {average_delay:.2f} seconds
Arrival on Green Rate: {arrival_on_green:.2f}%
Red Light Encounter Probability per Intersection: {red_light_prob:.2f}%

Interpretation:
- A higher Arrival on Green rate (>85%) indicates good coordination.
- A low average delay (<30s) means good flow efficiency.
- High red light probability suggests poor signal coordination.
"""

    with open(os.path.join(output_dir, 'signal_timing_analysis.txt'), 'w') as f:
        f.write(analysis)

    print("Analysis complete. Output saved to:", output_dir)

if __name__ == "__main__":
    analyze_signal_timing()
