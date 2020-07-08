#### Calorie tracker app
#### Howard Gan
#### 07-06-2020

# %% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly
import plotly.graph_objects as go

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


# %% Initialise variables
# height = 170
# weight = 75
# age = 22
# gender = "M"  # or "F"
# activityLevel = "Active"  # "Sedentary", "Lightly Active", "Active", "Very Active", "Extremely Active"
# goal = "aggressive cut"  # "aggressive cut", "cut", "maintain", "bulk", "heavy bulk"

# %% Calculations / Equations for BMR - Harris Benedict Equation
def calcBMR(weight, height, age, gender):

    # Error handling
    if any(v is None for v in [weight, height, age, gender]):
        BMR = 0
    else:
        # Handle raw inputs (strings)
        weight = float(weight)
        height = float(height)
        age = int(age)
        gender = str(gender)

    # Conditions for equations
    if gender == "M":
        BMR = 13.397 * weight + 4.799 * height - 5.677 * age + 88.362

    if gender == "F":
        BMR = 9.274 * weight + 3.098 * height - 4.330 * age + 447.593

    return BMR


# %% Activity levels calories
def calcMaintenanceCalories(activityLevel, BMR):

    if any(v is None for v in [activityLevel, BMR]):
        maintenanceCalories = 0
    else:
        levels = [
            "Sedentary",
            "Lightly Active",
            "Active",
            "Very Active",
            "Extremely Active",
        ]
        level_calories = [1.1, 1.3, 1.45, 1.6, 1.75]

        multiplier = level_calories[
            np.where(np.array(activityLevel) == np.array(levels))[0][0]
        ]

        maintenanceCalories = BMR * multiplier

    return maintenanceCalories


# %% Macro suggester 4:4:9 calorie per gram, 1:2:1 macros
def calcNutrition(maintenanceCalories, goal):

    if any(v is None for v in [maintenanceCalories, goal]):
        nutrition_table = pd.DataFrame(
            {"Calories": [0], "Protein (g)": [0], "Carbs (g)": [0], "Fat (g)": [0]}
        )
        additiveTerm = 0
    else:
        goal_array = np.array(
            ["aggressiveCut", "cut", "recomp", "bulk", "aggressiveBulk"]
        )
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
        nutrition_table[cols] = nutrition_table[cols].apply(
            pd.to_numeric, errors="coerce"
        )
        nutrition_table = nutrition_table.round(1)

    return nutrition_table, additiveTerm


# %% Visualise the dinner plate
def plotDinnerPlate(nutrition_table, goal):
    # Prepare the data for plotting
    plotData = nutrition_table.iloc[:, 1:]
    plotData = (
        plotData.assign(NV=plotData["Carbs (g)"] + plotData["Fat (g)"])
        .rename({"NV": "Starchy and fibrous carbs"}, axis=1)
        .T
    )
    plotData = plotData.drop(["Carbs (g)", "Fat (g)"])
    plotData = plotData.rename({"Protein (g)": "Lean Protein"})

    # Split the starchy and fibrous carbs into two distinct columns
    plotData = plotData.T
    plotData["Starchy carbs"] = plotData["Starchy and fibrous carbs"] * 2 / 5
    plotData["Fibrous carbs"] = plotData["Starchy and fibrous carbs"] * 3 / 5
    plotData = plotData.drop(plotData.columns[1], axis=1)
    plotData = plotData.T

    # Get the column names
    colname = plotData.columns[0]

    # Get final calories
    finalCalories = nutrition_table.Calories.values[0]

    # condition for plotting
    if finalCalories != 0:
        # Plot the dinner table
        fig_dinner_plate = px.pie(
            plotData,
            values=str(colname),
            names=plotData.index,
            title="Plate of " + str(int(finalCalories * 0.4)) + " calories",
            color_discrete_sequence=["#5db02c", "#f0886e", "#f2e6bf"],
        )
        fig_dinner_plate.update_layout(
            {"margin": {"t": 90, "r": 100, "b": 20, "l": 50}}
        )
    else:
        fig_dinner_plate = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["Please fill out all of the inputs."]),
                    cells=dict(values=["Thank you."]),
                )
            ]
        )
        fig_dinner_plate.update_layout(height=85, margin={"t": 10, "b": 0})

    return fig_dinner_plate


# %% Calculate weight loss rate
def calcWeightLossRate(additiveTerm):
    onePound = 3500  # calories
    oneKG = onePound * 2.20  # calories

    # Calculate how many days it takes to lose a Kg
    if additiveTerm != 0:
        daysPerKg = abs(oneKG / additiveTerm)
        kgPerDay = 1 / daysPerKg * np.sign(additiveTerm)
    else:
        daysPerKg = 0
        kgPerDay = 0

    return kgPerDay


# %% Simulate weight loss for 3 months
def simulateWeightLoss(weight, weightLossRate, goal):

    # If there are no errors, run the simulation
    if weight is not None:
        weight = float(weight)

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

            weight_i = weight_i + weightLossRate

            # 1/4 of weight loss is fat-free. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3970209/
            fat_i = fat_i + 0.75 * weightLossRate

        # Plot the simulation space
        if any(goal == np.array(["aggressiveCut", "cut", "bulk", "aggressiveBulk"])):
            simulatedSpace = pd.DataFrame(
                {"day": time, "Weight (kg)": weightArray, "Fat (kg)": fatArray}
            )
        else:
            simulatedSpace = pd.DataFrame({"day": time, "Weight (kg)": weightArray})

        simulatedSpace = simulatedSpace.melt("day")

        fig_sim = px.line(
            data_frame=simulatedSpace,
            x="day",
            y="value",
            color="variable",
            labels={"value": ""},
            title="Weight after 90 days (assume 15% body fat at start)",
            range_y=[
                min(simulatedSpace["value"]) * 0.8,
                max(simulatedSpace["value"]) * 1.2,
            ],
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

        # Add annotation for fat weight if bulking or cutting
        if any(goal == np.array(["aggressiveCut", "cut", "bulk", "aggressiveBulk"])):
            labels = np.array(
                [fatArray[0], fatArray[29], fatArray[59], fatArray[89]]
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

            fig_sim.update_layout(
                {"yaxis": {"range": [0, max(simulatedSpace["value"]) * 1.2]}}
            )

    # otherwise, handle errors
    else:
        weight = 0
        weightLossRate = 0
        fig_sim = go.Figure().update_layout(height=10)

    return fig_sim


# %% Function to generate table from dataframe
def generate_table(dataframe, max_rows=26):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])]
        +
        # Body
        [
            html.Tr([html.Td(dataframe.iloc[i][col]) for col in dataframe.columns])
            for i in range(min(len(dataframe), max_rows))
        ]
    )


#  %% Initialise the app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.DARKLY],
)
app.title = "Calorie and macronutrient calculator by HG"

server = app.server


#  %% Develop the UI/layout
app.layout = html.Div(
    style={"padding": "20px 30px 30px 30px"},
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
        # html.Br(),
        html.H3(children="Type in your stats"),
        html.H6("Height (cm), Weight (kg), Age and Sex:"),
        dcc.Input(id="height", type="number", placeholder="Height (cm)", value=170),
        # html.Br(),
        dcc.Input(id="weight", type="number", placeholder="Weight (kg)", value=70),
        # html.Br(),
        dcc.Input(id="age", type="number", placeholder="Age", value=20),
        # html.Br(),
        dcc.Input(id="gender", type="text", placeholder="M or F?"),
        # dcc.Dropdown(
        #     id="gender",
        #     placeholder="M or F?",
        #     options=[
        #         {"label": "Male", "value": "M"},
        #         {"label": "Female", "value": "F"},
        #     ],
        # ),
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
            # value="Sedentary",
            labelStyle={"display": "block"},
        ),
        html.Br(),
        html.H3(children="What are your goals?"),
        dcc.RadioItems(
            id="goal",
            options=[
                {"label": " Lose fat quickly", "value": "aggressiveCut"},
                {"label": " Lose fat", "value": "cut",},
                {"label": " Body recomposition", "value": "recomp"},
                {"label": " Gain mass", "value": "bulk",},
                {"label": " Gain mass quickly", "value": "aggressiveBulk",},
            ],
            # value="recomp",
            labelStyle={"display": "block"},
        ),
        html.Hr(),
        # Hide this until all inputs have been selected
        html.Div(id="strategy-content"),
        # html.H3(children="Here's a strategy for you..."),
        # dash_table.DataTable(
        #     id="nut-table",
        #     style_cell={"padding": "5px", "color": "black"},
        #     style_header={
        #         "backgroundColor": "white",
        #         "fontWeight": "bold",
        #         "color": "black",
        #     },
        #     style_cell_conditional=[
        #         {"if": {"column_id": c}, "textAlign": "left"}
        #         for c in ["Date", "Region"]
        #     ],
        # ),
        # html.Br(),
        # dcc.Graph(id="dinner_plate"),
        # html.Br(),
        # dcc.Graph(id="sim_space"),
        # Hidden divs inside the app that stores the intermediate value
        html.Div(id="bmr", style={"display": "none"}),
        html.Div(id="maintenanceCals", style={"display": "none"}),
        html.Div(id="additiveTerm", style={"display": "none"}),
    ],
)

#  %% Write callbacks

#### CHAIN THE CALCULATIONS FOR THE NUTRITION TABLE
# Calculate the BMR
@app.callback(
    Output("bmr", "children"),
    [
        Input("weight", "value"),
        Input("height", "value"),
        Input("age", "value"),
        Input("gender", "value"),
    ],
)
def calcBMR_app(weight, height, age, gender):
    BMR = calcBMR(weight, height, age, gender)
    return BMR


# Calculate the maintenance calories
@app.callback(
    Output("maintenanceCals", "children"),
    [Input("activity", "value"), Input("bmr", "children")],
)
def calcMaintenanceCalories_app(activityLevel, BMR):
    maintenanceCals = calcMaintenanceCalories(activityLevel, BMR)
    return maintenanceCals


# Produce the nutrition table DataTable object
@app.callback(
    [
        Output("nut-table", "data"),
        Output("nut-table", "columns"),
        Output("additiveTerm", "children"),
    ],
    [Input("maintenanceCals", "children"), Input("goal", "value")],
)
def calcNutrition_app(maintenanceCals, goal):
    # if all(v is not None in locals() for v in [maintenanceCals, goal]):
    nutrition_table, additiveTerm = calcNutrition(maintenanceCals, goal)
    columns = [{"name": i, "id": i} for i in nutrition_table.columns]

    return nutrition_table.to_dict("records"), columns, additiveTerm


# Produce the dinner plate
@app.callback(
    Output("dinner_plate", "figure"),
    [Input("maintenanceCals", "children"), Input("goal", "value")],
)
def plotDinnerPlate_app(maintenanceCals, goal):
    nutrition_table = calcNutrition(maintenanceCals, goal)[0]

    dinnerplate = plotDinnerPlate(nutrition_table, goal)
    return dinnerplate


# Produce the simulation space
@app.callback(
    Output("sim_space", "figure"),
    [
        Input("weight", "value"),
        Input("maintenanceCals", "children"),
        Input("goal", "value"),
    ],
)
def simulateWeightLoss_app(weight, maintenanceCals, goal):
    additiveTerm = calcNutrition(maintenanceCals, goal)[1]
    wlr = calcWeightLossRate(additiveTerm)

    return simulateWeightLoss(weight, wlr, goal)


# Render the strategy if all inputs are filled
@app.callback(
    Output("strategy-content", "children"),
    [
        Input("height", "value"),
        Input("weight", "value"),
        Input("age", "value"),
        Input("gender", "value"),
        Input("activity", "value"),
        Input("goal", "value"),
    ],
)
def render_content(height, weight, age, gender, activity, goal):
    # If any error
    if any(v is None for v in [height, weight, age, gender, activity, goal]):
        return html.Div()
    else:
        return html.Div(
            [
                html.H3(children="Here's a strategy for you..."),
                html.H5(children="Daily nutritional requirements:"),
                html.Div(
                    style={"padding": "0px 30px 0px 20px"},
                    children=dash_table.DataTable(
                        id="nut-table",
                        style_cell={
                            "textAlign": "center",
                            "color": "black",
                            "minWidth": "20%",
                            "width": "20%",
                            "maxWidth": "20%",
                        },
                        style_header={
                            "backgroundColor": "white",
                            "fontWeight": "bold",
                            "color": "black",
                        },
                        style_table={"width": "85%"},
                    ),
                ),
                html.Br(),
                html.H5(
                    children="Try to lay out your lunch and dinner plates with the following calories and types of foods:"
                ),
                dcc.Graph(id="dinner_plate"),
                html.Br(),
                html.H6(
                    children="If in doubt, just follow the rule of thirds! (1:1:1 for each of the food groups)"
                ),
                html.Br(),
                html.H5(
                    children="How much will you weigh in 90 days if you follow this strategy?:"
                ),
                dcc.Graph(id="sim_space"),
            ]
        )


# %% Run the app
if __name__ == "__main__":
    app.run_server(debug=False)

