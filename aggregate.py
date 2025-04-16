
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta

def parse_timestamp_from_filename(filename):
    base = os.path.basename(filename)
    try:
        timestamp_str = base.split("__")[1].replace(".csv", "")
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except:
        return None

def load_and_expand_csv(csv_file):
    start_time = parse_timestamp_from_filename(csv_file)
    if not start_time:
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    df["time_of_day"] = [start_time + timedelta(seconds=int(s)) for s in df["second"]]
    df["time_of_day"] = df["time_of_day"].dt.time
    return df

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

def aggregate_all_data():
    files = glob.glob("*.csv")
    all_data = pd.concat([load_and_expand_csv(f) for f in files if "__" in f], ignore_index=True)
    all_data["seconds_of_day"] = all_data["time_of_day"].apply(time_to_seconds)
    return all_data.sort_values("seconds_of_day")

def plot_line(x, y, ylabel, title, filename, labels=None, colors=None):
    plt.figure(figsize=(14, 6))
    if isinstance(y, pd.DataFrame):
        for col, color in zip(y.columns, colors or ['black']*len(y.columns)):
            plt.plot(x, y[col], label=col, color=color)
    else:
        plt.plot(x, y, color="black")
    plt.xlabel("Time of Day (seconds from midnight)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if labels or (isinstance(y, pd.DataFrame) and y.shape[1] > 1):
        plt.legend(loc="upper right")
    plt.savefig(filename)
    plt.close()

def main():
    df = aggregate_all_data()

    if df.empty:
        print("No valid CSV data found.")
        return

    # Total human count
    plot_line(
        df["seconds_of_day"],
        df["total_count"],
        "Total Humans",
        "Total Number of Humans Throughout the Day",
        "aggregate_total_humans.png"
    )

    # Walking vs idle
    plot_line(
        df["seconds_of_day"],
        df[["walking_count", "idle_count"]],
        "People Count",
        "Walking vs Idle People Throughout the Day",
        "aggregate_walk_vs_idle.png",
        labels=["Walking", "Idle"],
        colors=["blue", "brown"]
    )

    # Time stopped at red light
    red_light_seconds = df[df["light"] == "red"].groupby("seconds_of_day").size()
    all_seconds = pd.Series(0, index=range(0, 86400))
    all_seconds.update(red_light_seconds)
    plot_line(
        all_seconds.index,
        all_seconds.values,
        "Red Light Duration (sec)",
        "Amount of Time Stopped at Red Lights Throughout the Day",
        "aggregate_red_light_duration.png"
    )

    # Buses stopped at red
    bus_red = df[df["bus_waiting"] == True].groupby("seconds_of_day").size()
    all_bus_seconds = pd.Series(0, index=range(0, 86400))
    all_bus_seconds.update(bus_red)
    plot_line(
        all_bus_seconds.index,
        all_bus_seconds.values,
        "Buses at Red Light",
        "Buses Stuck at Red Lights Throughout the Day",
        "aggregate_buses_stopped.png"
    )

    # Green light presence
    green_count = df[df["light"] == "green"].groupby("seconds_of_day").size()
    all_green = pd.Series(0, index=range(0, 86400))
    all_green.update(green_count)
    plot_line(
        all_green.index,
        all_green.values,
        "Green Light Presence",
        "Green Light Appearances Throughout the Day",
        "aggregate_green_light_rate.png"
    )

    print("Aggregate graphs saved.")

if __name__ == "__main__":
    main()