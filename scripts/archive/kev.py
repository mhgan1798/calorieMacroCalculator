#### Calorie tracker app
#### 07-06-2020

# %% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go


# %% Initialise variables
height = 170
weight = 75
age = 22
gender = "M"  # or "F"
activityLevel = "Active"  # out of 5
goal = "aggressive cut"

# %% Calculations / Equations for BMR - Harris Benedict Equation
menBMR = 13.397 * weight + 4.799 * height + 5.677 * age + 88.362
womenBMR = 9.274 * weight + 3.098 * height + 4.330 * age + 447.593

# %% Activity levels calories
levels = ["Sedentary", "Lightly Active", "Active", "Very Active", "Extremely Active"]
level_calories = [1.1, 1.2, 1.3, 1.5, 1.7]

multiplier = level_calories[np.where(np.array(activityLevel) == np.array(levels))[0][0]]

maintenanceCalories_men = menBMR * multiplier
maintenanceCalories_women = womenBMR * multiplier

maintenanceCalories = maintenanceCalories_men

# %% Macro suggester 4:4:9 calorie per gram, 1:2:1 macros
goal_array = np.array(["aggressive cut", "cut", "maintain", "bulk", "heavy bulk"])
additiveTerm_array = np.array([-500, -300, 0, 300, 500])

# Macros according to the goal
macro_cut = np.array([0.3, 0.45, 0.25])
macro_maintain = np.array([0.25, 0.5, 0.25])
macro_bulk = macro_maintain
macro_df = pd.DataFrame(
    {
        "macro_heavycut": macro_cut,
        "macro_cut": macro_cut,
        "macro_maintain": macro_maintain,
        "macro_bulk": macro_bulk,
        "macro_heavybulk": macro_bulk,
    }
)

# Match the final arrays to the goal
goal_idx = np.where(goal == goal_array)[0][0]
additiveTerm = additiveTerm_array[goal_idx]
macros = macro_df.iloc[:, goal_idx].values

# Calculate the final calories based on the goal
finalCalories = maintenanceCalories + additiveTerm

# Compute macros
protein = (finalCalories * macros[0]) / 4
carbs = (finalCalories * macros[1]) / 4
fat = (finalCalories * macros[2]) / 9

# Summarise nutrition
nutrition_df = pd.DataFrame(
    data={
        "Calories": finalCalories,
        "Protein (g)": protein,
        "Carbs (g)": carbs,
        "Fat (g)": fat,
    },
    index=[1],
).T.rename(columns={1: goal})

nutrition_table = nutrition_df.reset_index().rename(columns={"index": ""}).T
nutrition_table.columns = nutrition_table.iloc[0, :]
nutrition_table = pd.DataFrame(nutrition_table.iloc[1, :]).T

# Convert all to numeric then round to 1 dp
cols = nutrition_table.columns
nutrition_table[cols] = nutrition_table[cols].apply(pd.to_numeric, errors="coerce")
nutrition_table = nutrition_table.round(1)

nutrition_table

# %% Visualise the dinner plate
plotData = nutrition_df.iloc[1:,].T
plotData = (
    plotData.assign(NV=plotData["Carbs (g)"] + plotData["Fat (g)"])
    .rename({"NV": "Starchy and fibrous carbs"}, axis=1)
    .T
)
plotData = plotData.drop(["Carbs (g)", "Fat (g)"])
plotData = plotData.rename({"Protein (g)": "Lean Protein"})

colname = plotData.columns[0]

fig_dinner_plate = px.pie(
    plotData,
    values=str(colname),
    names=plotData.index,
    title="Dinner plate of " + str(int(finalCalories)) + " calories",
)

fig_dinner_plate

# %% Calculate weight loss rate
onePound = 3500  # calories
oneKG = onePound * 2.20  # calories

# Calculate how many days it takes to lose a Kg
daysPerKg = abs(oneKG / additiveTerm)
kgPerDay = 1 / daysPerKg * np.sign(additiveTerm)

# %% Simulate weight loss for 3 months
# Initialise variables
time = np.arange(1, 91)
weightArray = np.array([])
weight_i = weight
fatArray = np.array([])
fat_i = 0.15 * weight

# Assume the strategy is followed perfectly
for i in time:
    weightArray = np.append(weightArray, weight_i)
    fatArray = np.append(fatArray, fat_i)

    weight_i = weight_i + kgPerDay

    # 1/4 of weight loss is fat-free. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3970209/
    fat_i = fat_i + 0.75 * kgPerDay

# Plot the simulation space
simulatedSpace = pd.DataFrame(
    {"day": time, "Weight (kg)": weightArray, "Fat (kg)": fatArray}
)
simulatedSpace = simulatedSpace.melt("day")

fig_sim = px.line(
    data_frame=simulatedSpace,
    x="day",
    y="value",
    color="variable",
    labels={"value": ""},
    title="Weight after 90 days (assume 15% body fat at start)",
    range_y=[0, max(simulatedSpace["value"]) * 1.1],
).update_layout({"margin": {"l": 40}})

# Add annotation for weight
labels = np.array(
    [weightArray[0], weightArray[29], weightArray[59], weightArray[89]]
).round(2)

fig_sim.add_trace(
    go.Scatter(
        x=[1, 30, 60, 90],
        y=labels + 0.2,
        mode="text",
        text=labels,
        textposition="top center",
        showlegend=False,
    )
)

# Add annotation for fat weight
labels = np.array([fatArray[0], fatArray[29], fatArray[59], fatArray[89]]).round(2)

fig_sim.add_trace(
    go.Scatter(
        x=[1, 30, 60, 90],
        y=labels + 0.2,
        mode="text",
        text=labels,
        textposition="top center",
        showlegend=False,
    )
)

fig_sim

# %%


# %%
