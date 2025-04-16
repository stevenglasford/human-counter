# human-counter
A program using open AI tools and python to create graphs used to improve traffic flows for buses, bikes and pedestrians

The outputs of the human_counter.py creates a png file and a csv after looking at the videos. This script is designed to be able to look at a large number of videos simultaneously in a background and nightly job style. The processing of the videos is CPU intensive and requires a large number of time to run.

After running this script the script will output the events happening in the video. In particular it looks at the following:

-- counts the total number of humans and plots the number of unique humans per second of the video. 
-- counts the number of people walking per second in the video
-- counts the number of people waiting at bus stops or otherwise "idling"
-- counts the number of vehicles on the roadways
-- measures the speed of vehicles
-- counts the number of buses (if the bus has clear windows the people on board the bus are detected and counted as "walking", this error is not present in the Minneapolis area as all buses have window tints for Metro Transit
-- displays the amount of time spent at red lights
-- displays the time spent at green lights
-- detects when a bus is waiting at a red light, as opposed to loading passengers at a bus stop
-- detects when a red light violation occurs. As defined by at least 5 or more people walking across a red light (this indicates that the traffic light needs to be retimed)

All of this is on a single busy chart. This data can then be referenced by looking at the corresponding .csv file from with the .png file was created

The `aggregate.py` file then combines all of the csv files into several graphs that spread all of the videos per time of day in which they were made. This will produce the following new charts:

-- A day spread graph of the number of people counted per day
-- A day spread graph of the amount of time spent at red lights
-- A day spread graph of the number of green lights able to be passed through given time of day
-- a day graph of the total speed of cars moving
-- a day graph of the probability of a bus stopping at a red light and waiting 
-- a day spread graph of the number of people walking or moving
-- a day spread graph of the number of people idling
-- a day spread graph of the number of "red light violations" as defined by 5 or more pedestrians walking through a red light
