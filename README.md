# Diet Optimization Model for Cost Minimization under Nutritional Constraints

## Authors  
Matheus Nogueira  
João Pedro Martinez  

## About this Project

This project was developed as the final assignment for the course **ENG1467 – Optimization** at PUC-Rio. The objective was to formulate and solve an optimization problem that minimizes the total cost of a weekly food plan, while ensuring that nutritional and dietary requirements are satisfied.

Using a dataset of food items with associated prices and nutritional information (calories, protein, carbohydrates, and fat), the model constructs a **daily and weekly dietary plan** based on an individual's physical profile (weight, height, age, sex, and activity level).

### Model Highlights

- **Objective**: Minimize the total weekly cost of a diet that satisfies caloric and macronutrient constraints.
- **Constraints**:
  - Daily intake bounds for protein, carbohydrates, fat, and total calories.
  - Weekly minimum and maximum purchase quantities for each product to encourage diversity.
  - Specific allocation patterns for food items based on “active” and “rest” days to promote variety and realistic consumption behavior.
- **User Profile**: Macronutrient and calorie needs are dynamically computed using the **Harris-Benedict formula** adjusted for physical activity level.
- **Solution Methods**:
  - The problem is solved using **GLPK’s interior point method** via JuMP.
  - Results are compared to a custom **Interior Point algorithm**, implemented from scratch, to verify solution quality and performance.

### Results

The model was tested on three individuals with varying characteristics. In all cases, it generated personalized meal plans that respected nutritional targets and minimized cost. The GLPK and custom interior point implementations returned equivalent results, with expected differences in runtime performance.

This project demonstrates how mathematical programming can be effectively applied to real-world dietary planning, combining optimization theory, numerical methods, and data analysis.
