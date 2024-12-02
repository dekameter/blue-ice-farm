# Minecraft Ice Farm Simulator

Simulates the generation of a Minecraft ice farm and outputs a comma-separated line for each size, minimum duration, maximum duration, average duration, and median duration (all in ticks). If the size is greater than 7, than an X pattern is used for the water sources; otherwise, only a single diagnal is used.

At the time I wrote this in 2021, the goal was to better optimize ice farming for blue ice based on both the size of the ice farm and at what percentage yield to clear out the farm.

## Modes

For the sake of analysis, I've developed three distinct modes based on open-ended questions I had about the nature of ice farm generaton, such as when would it be best to clear out the farm if not when it's completely frozen over.

### Standard Generation

![A top-down, terminal-based representation of a Minecraft ice farm forming from the borders toward the center until completely full.](https://dekameter.github.io/img/ice_farm.gif)

The default mode simulates ice formation in the farm until a given yield threshold is met. The threshold is a percentage represented by the range 0.0 to 1.0 (aka. 0% to 100%). Based on data collected through this script, the default threshold was chosen to be 0.95 (95%) (see [`--moments` mode](#moments-mode) for more details).

Default mode collects the number of Minecraft ticks it takes to reach the yield threshold. The following data is saved to the given output path:
`{farm size},{threshold},{effective yield,{yield},{time (in ticks)}`

### Center Reached Mode

![A top-down, terminal-based representation of a Minecraft ice farm forming from the borders toward the center. The generation stops once the center has been reached by a path.](https://dekameter.github.io/img/ice_farm_center.gif)

The `--center` mode still follows the same ice generation rules. However, once any ice has reached the center, the generation stops. Originally this was explored as an alternative strategy to determine when to clear out the ice farm

Unlike the default generation, the center reaching mode records how much of the ice farm has been filled as a percentage (0.0 to 1.0). Since the threshold parameter makes no sense here, the threshold is ignored. The following data is saved to the given output path:
`{farm size},{effective yield},{yield},{yield ratio},{time (in ticks)}`

### Moments Mode

![A line graph plotting the area filled (in percentage) against the time duration (in percentage), separated by the farm sizes 3 through 17.](https://dekameter.github.io/img/ice_farm_plot.png)

The `--moment` mode also keeps the same Minecraft generation logic, but instead it tracks any moment (tick update) new ice is formed and records the yield in that moment.Both the maximum possible yield changes relative to farm size and takes longer to freeze over completely, making it to compare and contrast the data across different sizes. The yield and time are normalized relative to the effective yield and total time taken to make comparisons possible.

As shown in the plot above, the farm reaches near full capacity sooner the larger the farm size is. Anecdotally, it seems that as the farm size tends toward infinity, the farm will be roughly 95% full by around 60% of the time it takes to fill the entire plot. _That means 40% of the time would be wasted to generate a measely 5%!_

Since this mode is more concerned with tracking every moment of change, the threshold defaults to 1.0 (100%). The following data is saved to the given output path:
`{farm size},{threshold},{effective yield},{yield},{yield ratio},{total time (in ticks)},{time (in ticks)},{time ratio}`