#### Calorie tracker app
#### Howard Gan
#### 07-06-2020

# %% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


# %% Initialise variables
height = 170
weight = 75
age = 22
gender = "M"  # or "F"
activityLevel = "Active"  # "Sedentary", "Lightly Active", "Active", "Very Active", "Extremely Active"
goal = "aggressive cut"  # "aggressive cut", "cut", "maintain", "bulk", "heavy bulk"

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

#  %% Initialise the app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.DARKLY],
)

server = app.server
#  %% Develop the UI/layout
app.layout = html.Div(
    style={"padding": "20px 20px 20px 20px"},
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                html.H1(
                    "Calories and macronutrient calculator",
                    style={"text-decoration": "none", "color": "inherit"},
                )
            ],
        ),
        html.Br(),
        html.H3(children="Type in your stats"),
        dcc.Input(id="height", type="text", placeholder="Height (cm)"),
        html.Br(),
        dcc.Input(id="weight", type="text", placeholder="Weight (kg)"),
        html.Br(),
        dcc.Input(id="age", type="number", placeholder="Age"),
        html.Br(),
        dcc.Input(id="gender", type="text", placeholder="M or F?"),
        html.Br(),
        html.Br(),
        html.H3(children="Describe your activity level"),
        dcc.RadioItems(
            id="activity",
            options=[
                {"label": " Sedentary (you don't exercise)", "value": "Sedentary"},
                {
                    "label": " Lightly Active (work out 1-2 times/week)",
                    "value": "Lightly Active",
                },
                {"label": " Active (work out 2-3 times/week)", "value": "Active"},
                {
                    "label": " Very Active (work out 4-5 times a week)",
                    "value": "Very Active",
                },
                {
                    "label": " Extremely Active (work out 6-7 times a week)",
                    "value": "Extremely Active",
                },
            ],
            value="Sedentary",
            labelStyle={"display": "block"},
        ),
        html.Br(),
        html.H3(children="What are your goals?"),
        dcc.RadioItems(
            id="goals",
            options=[
                {"label": " Lose weight quickly", "value": "aggressiveCut"},
                {"label": " Lose weight", "value": "cut",},
                {"label": " Body recomposition", "value": "recomp"},
                {"label": " Gain weight", "value": "bulk",},
                {"label": " Gain weight quickly", "value": "aggressiveBulk",},
            ],
            value="recomp",
            labelStyle={"display": "block"},
        ),
        html.Br(),
        html.H3(children="Here's a strategy for you..."),
        dbc.Table.from_dataframe(
            nutrition_table,
            bordered=True,
            dark=False,
            size="lg",
            hover=True,
            # style={"width": "60%"},
        ),
        html.Br(),
        dcc.Graph(id="dinner_plate", figure=fig_dinner_plate),
        html.Br(),
        dcc.Graph(id="sim_space", figure=fig_sim),
    ],
)
#  %% Write callbacks
# @app.callback(
#     Output("out-all-types", "children"),
#     [Input("input_{}".format(_), "value")],
# )

# %% Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
