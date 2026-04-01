# Predicting Urban Bike-Sharing Demand: A Data Mining Study for STEM Education

---

## Cover Page

**Project Title:** Predicting Urban Bike-Sharing Demand: A Data Mining Study for STEM Education

**Team Name:** Data Mining Group

**Submission Date:** March 15, 2026

---

### Team Members

| Name | Student ID | Role |
|------|-----------|------|
| [Member A] | [XXXXXXXX] | Team Representative |
| [Member B] | [XXXXXXXX] | Member |
| [Member C] | [XXXXXXXX] | Member |
| [Member D] | [XXXXXXXX] | Member |

---

### Contribution Breakdown

| Section | Member A | Member B | Member C | Member D |
|---------|----------|----------|----------|----------|
| (b) Objectives | 40% | 20% | 20% | 20% |
| (c) Data Sources | 20% | 40% | 20% | 20% |
| (d) Preliminary Findings | 20% | 20% | 40% | 20% |
| (e) Future Plans | 20% | 20% | 20% | 40% |

---
---

## (b) Project Objectives

### Background

Every morning, millions of urban commuters make a quiet calculation: *Is it worth cycling today?* They weigh the weather, the time, the season, and whether today is a workday. What feels like an instinctive, personal decision is, in aggregate, a highly predictable pattern — one that generates enormous quantities of data and presents a rich opportunity for data mining.

Washington D.C.'s Capital Bikeshare system, one of the largest bike-sharing networks in the United States, records every rental transaction with a precise timestamp, linking it to hourly weather observations from the city's meteorological station. Over the two-year period from 2011 to 2012, this system accumulated 17,379 hourly observations, capturing how ridership responded to everything from a crisp autumn morning to a humid summer afternoon to a snowy January commute. The result is a dataset that is simultaneously rich enough to support serious machine learning research and intuitive enough to be understood by a middle school student.

This project builds on that dataset to pursue two interconnected goals. The first is scientific: to construct and rigorously evaluate a suite of machine learning models capable of accurately forecasting hourly bike-sharing demand from temporal and meteorological inputs. The second is pedagogical: to translate the entire data mining workflow — from raw data to live prediction — into an interactive, bilingual teaching platform that makes the power and logic of machine learning accessible to students with no programming background.

The relevance to STEM education is immediate and compelling. Students who use this platform are not passively reading about algorithms; they are actively running models, comparing outputs, examining why one method outperforms another, and testing predictions against their own real-world intuitions. The platform transforms abstract concepts — regularisation, ensemble learning, feature importance — into observable phenomena with interpretable outcomes.

### Specific Objectives

This project is structured around four specific, measurable objectives:

**Objective 1: Develop and Compare Six Regression Models**

We will train six supervised learning regression algorithms — Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest, Gradient Boosting, and XGBoost — on the processed dataset and evaluate each using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the coefficient of determination (R²). The models are deliberately ordered from simplest to most complex, creating a natural learning progression: the linear models establish interpretable baselines; the decision tree bridges the gap to nonlinear modelling; the ensemble methods demonstrate what is achievable when many models are combined.

**Objective 2: Identify and Interpret the Key Demand Drivers**

Accurate prediction is not the only goal. We also aim to understand *what drives ridership* — which features carry the most predictive weight, and whether the model's learned patterns align with our domain intuitions. Tree-based models provide feature importance scores that make this analysis tractable. We expect hour of day, perceived temperature, and workday status to emerge as the dominant factors, but the relative rankings and interaction effects are non-trivial and worth investigating.

**Objective 3: Execute a Rigorous, End-to-End Data Preparation Pipeline**

Raw data rarely arrives in a form suitable for machine learning. Our preprocessing pipeline will demonstrate a principled approach to data cleaning (removing leakage variables and redundant features), feature engineering (cyclic encoding of temporal variables, construction of interaction terms), and target transformation (log(1+cnt) to address right skew). Each decision will be documented and justified — modelling the kind of systematic, explainable reasoning that STEM education aims to cultivate.

**Objective 4: Build and Deploy an Interactive STEM Teaching Platform**

We will develop a bilingual (Chinese/English) WebUI using Streamlit, containerised with Docker for one-command deployment. The platform consists of five modules: a Dashboard for exploratory data visualisation, a Data Preprocessing page explaining each cleaning and engineering step, a Deep Learning module where students run six algorithms and compare their metrics, a Summary Report comparing all trained models side by side, and a Test Model page where students input custom conditions and receive instant predictions. The platform is designed to operate entirely through visual interaction — no code, no command line — making it suitable for middle school classrooms.

### Connection to STEM Lesson Design

The data mining workflow maps precisely onto the scientific inquiry cycle that underpins STEM education:

- **Observe:** Students examine the dashboard charts and notice patterns — ridership peaks at 17:00, plummets in rain, surges in autumn.
- **Hypothesise:** Students form predictions about which model will perform best and why certain features matter more than others.
- **Experiment:** Students run each algorithm, recording metrics and comparing outcomes.
- **Analyse:** Students examine feature importance charts and residual distributions to understand model behaviour.
- **Communicate:** Students use the summary report to articulate which model they would recommend for a real deployment, and why.

This cycle repeats naturally within the platform, creating a self-guided inquiry experience that does not require teacher intervention at every step.

---

## (c) Data Sources

### The Dataset

The data underpinning this project is the **UCI Bike Sharing Dataset**, compiled by Hadi Fanaee-T and João Gama from the University of Porto and published in 2014 alongside their research on ensemble-based event detection. The dataset is publicly available from the UCI Machine Learning Repository and has been used in hundreds of academic studies, making it one of the most thoroughly validated regression benchmarks in the machine learning literature.

We use the hourly version of the dataset (`hour.csv`), which contains **17,379 records** spanning exactly two full calendar years: January 1, 2011 through December 31, 2012. Each record represents a single hour of operation of the Capital Bikeshare system in Washington D.C., and captures the total number of bicycle rentals alongside the meteorological and calendar conditions at that time. The dataset contains **17 original columns** and is notable for its exceptional cleanliness — there are no missing values and no duplicate rows anywhere in the dataset, an unusual quality that allows us to focus our preparation effort entirely on feature engineering rather than imputation.

> **[FIGURE 1 — Place here: Daily total rentals trend line chart (2011–2012), showing the two-year upward trend and the repeating seasonal waves. The x-axis should be date, y-axis total daily rentals, ideally with a smoothed trend line overlaid. Source: app.py Dashboard page.]**

### Variable Description

The 17 columns fall into three natural groups.

The **temporal and calendar features** record the conditions of the day and the hour. The year (encoded as 0 for 2011, 1 for 2012) captures the platform's growth trajectory. The month (1–12), hour (0–23), and season (1=Spring, 2=Summer, 3=Autumn, 4=Winter) capture cyclical time patterns at different granularities. Three binary flags — `holiday`, `workingday`, and `weekday` — distinguish the nature of each day. Together, these features encode the rhythms of human urban life: the commuter who rides to the office on weekday mornings, the family that rents bikes on a sunny Saturday afternoon, the solitary night rider at 2 a.m.

The **weather features** capture environmental conditions. The weather situation (`weathersit`) is a categorical variable on a four-point scale from clear skies to heavy rain. Temperature is recorded twice — once as actual air temperature (`temp`, restorable to °C by multiplying by 41) and once as perceived temperature (`atemp`, multiplied by 50). Relative humidity (`hum`) and wind speed (`windspeed`) complete the picture. All four continuous weather variables were normalised to [0,1] by the dataset authors; we restore them to natural units for analysis and visualisation.

The **target variables** are `casual` (rentals by unregistered users), `registered` (rentals by registered users), and `cnt` (their sum, our prediction target). The `cnt` variable ranges from 1 to 977, with a mean of 189.5 and a median of 142.0 — the gap between mean and median is the first visible sign of right skew, which we address in preprocessing.

### Data Quality Assessment

The quality of this dataset is genuinely exceptional by real-world standards, but not without issues that a careful analyst must address.

The most significant problem is **data leakage**. The columns `casual` and `registered` are not independent predictors of `cnt` — they are its literal arithmetic components: `cnt = casual + registered`. Including them as features would allow any model to "predict" `cnt` by simple addition, achieving near-perfect accuracy with zero genuine learning. This is one of the most common and dangerous errors in applied machine learning, and identifying it here provides an important teaching moment for students.

The second problem is **severe multicollinearity** between `temp` (air temperature) and `atemp` (perceived temperature). The Pearson correlation between these two variables is **0.9877** — among the highest feature-feature correlations one typically encounters in a real dataset. When two features are this highly correlated, including both in a linear model destabilises the coefficient estimates, and including both in any model adds redundancy without benefit. We retain `atemp` because perceived temperature more directly represents the thermal comfort that influences a rider's decision to cycle.

A third, minor issue involves the variable `weathersit`. The fourth category — heavy rain or hail — appears only **3 times** across the entire two-year dataset, representing 0.02% of records. With only three training examples, no model can learn a meaningful pattern for this category. We merge it with category 3 (light rain/snow) to create a stable three-level weather severity scale.

Finally, the `instant` column is a simple row-number index with no predictive meaning, and `dteday` is a date string whose information is already fully encoded in `yr`, `mnth`, and `hr`. Both are removed before modelling.

> **[FIGURE 2 — Place here: Two-panel figure. Left: scatter plot of temp vs atemp for all 17,379 records, with r=0.987 annotated, demonstrating the multicollinearity. Right: bar chart showing the four weathersit categories and their record counts (11,413 / 4,544 / 1,419 / 3), clearly showing the class imbalance of category 4. Source: app.py Data Preprocessing page.]**

### Data Preparation Pipeline

Our preparation pipeline transforms the raw 17-column dataset into a clean, 23-feature matrix ready for modelling. Each step is driven by a clear rationale.

**Cleaning** removes five columns: `instant`, `dteday`, `casual`, `registered`, and `temp`. It also recodes the three records with `weathersit = 4` to `weathersit = 3`. After cleaning, the feature matrix contains 11 columns.

**Cyclic encoding** addresses a subtle but important problem with temporal features. Hour, month, weekday, and season are all inherently circular: hour 23 is only one step away from hour 0, but numerically they appear to be 23 units apart. If we pass raw integer hours to a linear model, it will treat midnight and 11 p.m. as nearly opposite extremes. We resolve this by mapping each circular variable onto the unit circle using sine and cosine:

$$hr\_sin = \sin\!\left(\frac{2\pi \times hr}{24}\right), \quad hr\_cos = \cos\!\left(\frac{2\pi \times hr}{24}\right)$$

This approach, applied to all four periodic variables (hr, mnth, weekday, season), adds 8 new columns while preserving temporal continuity.

**Time period segmentation** creates a categorical variable (`time_period`) dividing the 24-hour day into five behaviourally meaningful segments — late night (0–6), morning rush (7–9), daytime (10–16), evening rush (17–19), and night (20–23). This captures the coarse structure of daily demand without relying solely on the fine-grained hour variable.

**Interaction features** capture the joint effects of two variables that cannot be represented by either alone. Working day and hour interact strongly: on working days, rentals form a pronounced double peak at 8 a.m. and 5–6 p.m.; on weekends, they form a single broad peak around midday. The product `hr × workingday` encodes this difference. Temperature and humidity interact because discomfort on hot, humid days is multiplicatively worse than either factor alone: `atemp × (1 − hum)` approximates a simple thermal comfort index. Season and hour interact because the active riding window shifts across the year: in summer, ridership extends well into the evening; in winter, it concentrates around midday. The product `season × hr` captures this shift.

**Target transformation** is the final and most important step. The raw `cnt` distribution is right-skewed with a skewness coefficient of +1.277 — the interquartile range spans only 40 to 281, yet the distribution has a long tail reaching 977. Training regression models on this distribution causes them to over-prioritise high-demand outliers and systematically underestimate low-demand hours. Applying log(1 + cnt) compresses the tail and approximately symmetrises the distribution (skewness: −0.818), leading to more balanced and stable training. All model outputs are inverse-transformed using exp(ŷ) − 1 before reporting metrics in original rental units.

> **[FIGURE 3 — Place here: Side-by-side histogram of raw cnt distribution (left) vs log(1+cnt) distribution (right). Both should show a KDE curve. Annotate the skewness value on each panel (+1.277 and -0.818). This is the clearest visual argument for the log transform. Source: app.py Data Preprocessing page.]**

The final dataset contains **17,379 rows** and **23 input features**, with `cnt_log` as the target variable for training and `cnt` retained for reporting.

---

## (d) Preliminary Findings

### Exploratory Data Analysis

Before building any model, we conducted a thorough exploratory analysis to understand the structure of demand, identify dominant patterns, and develop intuitions about which features should matter most.

**The platform grew dramatically between 2011 and 2012.** Total rentals increased from 1,243,103 in 2011 to 2,049,576 in 2012 — a 65% increase in a single year. Mean hourly ridership grew from 143.8 to 234.7. This growth is not merely a background trend; it means that a model trained on 2011 data must somehow generalise to a 2012 population that is substantially larger. It also makes the year feature (`yr`) a strong predictor — not because the year itself causes demand, but because it serves as a proxy for the platform's accumulated user base.

**The hour of day is the single most powerful predictor.** Hourly ridership follows a radically different pattern on working days compared to weekends, and this difference is one of the most striking features of the dataset.

On working days, demand forms a sharp double peak: a morning spike reaching an average of **359.0 rentals at 8 a.m.**, a midday trough, then an even larger evening surge peaking at **525.3 rentals at 5 p.m.** This pattern unmistakably reflects the commuting behaviour of registered users — the same people riding to work in the morning and home in the evening. On non-working days, the pattern is entirely different: a single broad peak of **372.7 rentals at 1 p.m.**, characteristic of leisure ridership spread across the afternoon.

> **[FIGURE 4 — Place here: Line chart of mean hourly rentals (0–23) with two lines — one for working days (blue) and one for non-working days (orange). This is the most visually compelling chart in the entire analysis. The double-peak vs single-peak contrast is immediately striking. Source: app.py Dashboard page.]**

This distinction is further reinforced by the user type breakdown: on working days, registered users account for an average of **167.6 rentals per hour** versus only **25.6 from casual users** — a ratio of 6.5 to 1. On weekends, the ratio narrows to 2.2 to 1 (124.0 vs 57.4), reflecting the higher proportion of tourists and occasional riders. This finding suggests that registered and casual users are responding to entirely different sets of incentives, a nuance that motivates the interaction feature `hr × workingday`.

**Weather has a large, quantifiable, and intuitive impact on demand.** Mean hourly rentals under clear or partly cloudy conditions are **204.9**. Mist and cloud reduce this to **175.2** — a 14.5% drop. Light rain or snow cuts demand nearly in half, to **111.6** — a 45.5% reduction from fair weather. These numbers are large enough to matter operationally and intuitive enough to need no statistical training to understand. They also motivate careful weather feature engineering: a linear model might assume that the effect of rain is constant across all hours and seasons, but reality is likely more complex.

**Temperature is the most strongly correlated continuous feature with demand**, with a Pearson r of **+0.401**. Humidity shows a meaningful negative correlation (r = −0.323): high humidity, especially in combination with high temperatures, appears to discourage cycling. Wind speed has a weaker and perhaps surprising positive correlation (r = +0.093), which may partly reflect the fact that dry, breezy days tend to be preferred over still, humid ones. These correlations are linear summaries of what are almost certainly nonlinear relationships — which is precisely why tree-based models are likely to outperform linear ones here.

> **[FIGURE 5 — Place here: Bar chart of mean hourly rentals by season (Spring=111.1, Summer=208.3, Autumn=236.0, Winter=198.9), with a second grouped bar chart showing the same breakdown split by working day. The autumn peak and the fact that summer is not the highest season (surprising to most students) is the key takeaway.]**

**The seasonal pattern is counterintuitive in an instructive way.** The highest average demand occurs not in summer but in **autumn (236.0 rentals/hour)**, followed by summer (208.3), winter (198.9), and spring (111.1). The suppression of summer demand in Washington D.C. — a city known for hot, humid summers — relative to the mild, low-humidity autumn is a finding that generates productive discussion in the classroom: it demonstrates that "more pleasant weather = more cycling" is too simple, and that the specific combination of temperature and humidity matters.

**The target variable's distribution reveals the need for transformation.** The raw `cnt` distribution has a mean of 189.5 but a median of only 142.0, and the interquartile range (Q1=40, Q3=281) sits well below the mean. The distribution extends to a maximum of 977 — more than five times the median — creating a long right tail driven by a relatively small number of peak hours (summer weekday evenings in good weather). Training a regression model on this distribution as-is would cause it to over-invest in predicting the rare high-demand hours at the expense of the much more common low-to-moderate demand hours. The log(1+cnt) transformation reduces skewness from +1.277 to −0.818, producing a distribution that is far more amenable to regression learning.

### Preliminary Modelling Results

To establish performance baselines and verify the viability of our approach, we conducted an initial round of modelling using a random 80/20 train-test split. While we intend to replace this with a rigorous temporal split for the final analysis, the preliminary results already reveal the essential story of this dataset.

> **[FIGURE 6 — Place here: Predicted vs actual scatter plot for XGBoost (the best model), showing points clustered tightly around the y=x diagonal line. This visually communicates model quality better than any metric table alone. Source: app.py Deep Learning > XGBoost page.]**

| Model | RMSE (rentals/hr) | MAE (rentals/hr) | R² |
|-------|-------------------|-------------------|----|
| Linear Regression | ~134 | ~93 | ~0.39 |
| Ridge Regression | ~133 | ~92 | ~0.40 |
| Decision Tree Regressor | ~92 | ~58 | ~0.74 |
| Random Forest | ~52 | ~34 | ~0.92 |
| Gradient Boosting | ~55 | ~37 | ~0.91 |
| XGBoost | ~48 | ~32 | ~0.93 |

*Note: These are preliminary estimates from initial runs without hyperparameter tuning. Final figures will be reported after optimisation.*

The results tell a consistent and pedagogically valuable story. The two linear models — despite being the simplest and most interpretable — achieve only R² ≈ 0.40. This is not a failure of implementation; it reflects a genuine structural limitation. Bike-sharing demand is not a linear function of its inputs. The hour-of-day effect alone is highly nonlinear (near-zero at 4 a.m., over 500 at 5 p.m.), and the interaction between hour and workday status is complex in a way that no linear model can represent without extensive manual feature crafting.

The Decision Tree Regressor dramatically improves performance to R² ≈ 0.74 — a jump of 34 percentage points — by capturing these nonlinear relationships through recursive partitioning. However, a single decision tree is vulnerable to overfitting: it can memorise specific training patterns that do not generalise well. This limitation is precisely what ensemble methods address.

Random Forest and Gradient Boosting both achieve R² above 0.90 by combining hundreds of individual trees, cancelling out the idiosyncratic errors of any single tree. XGBoost — which adds second-order gradient approximation and built-in regularisation to the boosting framework — achieves the best result at R² ≈ 0.93 and RMSE ≈ 48 rentals per hour. To contextualise this: with a mean demand of 189.5 rentals per hour, an RMSE of 48 represents an average proportional error of roughly 25%, which for operational bike-sharing management (where rough demand forecasts are sufficient to guide rebalancing decisions) is quite practical.

> **[FIGURE 7 — Place here: R² comparison bar chart for all six models, ordered from left to right by increasing complexity (LR → Ridge → DT → RF → GB → XGB). Use a horizontal reference line at R²=0.90 labelled "excellent threshold". This is the single most important summary figure for the modelling section. Source: app.py Summary Report page.]**

### Proposed STEM Classroom Activities

The findings above directly inform four structured classroom activities, each designed to engage students with a specific concept through the interactive platform.

**Activity 1 — "The Commuter and the Tourist"**

Students open the Dashboard and examine the working day vs. weekend hourly demand chart (Figure 4 above). The instructor asks: *Why do you think the shapes are so different?* Students discuss in pairs, identifying the double-peak commuter signature, and then use the Test Model page to verify their intuitions — predicting rentals at 8 a.m. on a working day versus 8 a.m. on a weekend, holding all other variables constant. This activity introduces the concept of user segmentation and the idea that the same hour can mean something very different depending on context.

**Activity 2 — "Weather or Not"**

Students compare the mean rentals across the three weather categories from the Dashboard, noting the 45.5% drop under rain. They then role-play as a bike-sharing operator who must decide how many bicycles to deploy at each station at 8 a.m. — but the weather forecast is uncertain. This activity introduces the concept of prediction under uncertainty and motivates the need for quantitative demand forecasting rather than guesswork.

**Activity 3 — "The Model Tournament"**

Students run all six models in sequence, recording RMSE and R² in a shared table. They then work in groups to answer: *Why did XGBoost beat Linear Regression by such a large margin?* Groups are given a hint: *Think about whether the relationship between "hour" and "rentals" could possibly be described by a straight line.* This activity introduces nonlinearity, model complexity, and the bias-variance trade-off — arguably the three most important concepts in applied machine learning.

**Activity 4 — "Design Your Prediction"**

Students use the Test Model page, inputting their own commuting scenario (their actual time of departure, the current month and weather, whether today is a school day). They receive a predicted rental count from their chosen model and compare it with the other models' predictions. They then discuss: *Which model's prediction would you trust most? Which would you use if you were the city's transport manager?* This activity makes the modelling exercise personal and forces students to confront the practical meaning of model evaluation metrics.

---

## (e) Future Plans

### Remaining Tasks

The work completed to date establishes a solid foundation: the data is clean, the platform is deployed, initial models are running, and preliminary results confirm that our research questions are well-posed. What remains is to deepen, refine, and formalise each component.

**Temporal train-test split (highest priority).** Our current random 80/20 split violates the temporal structure of the data — a model trained on random samples from both 2011 and 2012 is implicitly "shown the future" during training, inflating all performance metrics. We will replace this with a strict forward-looking split: the model will be trained on data up to a fixed cutoff date and tested on the subsequent weeks. This more accurately simulates real deployment, where a model trained today must predict tomorrow's demand. We anticipate that reported R² values will decrease slightly under this stricter evaluation — and that this decrease will itself be a valuable teaching point about the difference between in-sample and out-of-sample performance.

**Hyperparameter optimisation.** The preliminary models use default or hand-picked hyperparameters. For Random Forest and XGBoost — the two models with the most tunable parameters — we will conduct a systematic grid search or random search over key parameters (number of trees, maximum depth, learning rate for XGBoost). This is expected to reduce RMSE by 5–15% relative to the preliminary results and will provide additional material for the STEM lesson on model optimisation.

**STEM lesson plan development.** The four classroom activities described in section (d) will be formalised into a complete lesson plan covering 5–6 sessions of 45 minutes each. Each session will have defined learning objectives aligned with STEM curriculum standards, a structured sequence of guided and open-ended tasks, and formative assessment questions to check student understanding. A teacher's guide will accompany the student-facing materials.

**Final report and presentation preparation.** The project findings will be synthesised into a comprehensive final report, including updated model evaluation results, detailed interpretation of feature importance rankings, and a reflective discussion of lessons learned. A presentation slide deck summarising the key findings and the STEM platform will be prepared for the group submission.

### Anticipated Challenges

**Challenge 1: Temporal leakage and performance expectations.** Switching to a temporal train-test split will likely reduce reported R² from ~0.93 to somewhere between 0.85 and 0.91, depending on the split point. This is a feature of honest evaluation, not a flaw in our models, and we will document this transparently. The pedagogical value — showing students that evaluation methodology matters as much as model choice — is itself worthwhile.

**Challenge 2: Balancing technical depth with classroom accessibility.** The mathematical machinery behind log-transformations, gradient descent, and regularisation is non-trivial for middle school students. Our strategy is to use the platform as a visual intermediary: students observe the *effects* of these techniques (a more symmetric distribution, a lower RMSE, a more stable training curve) before encountering the formulas. Concepts that cannot be shown visually within the platform will be explained through analogy — for instance, gradient boosting as "learning from your mistakes, one step at a time."

**Challenge 3: Decision tree overfitting as a teaching moment.** The Decision Tree Regressor's R² of ~0.74 — substantially below the ensemble models' ~0.92 — raises a natural question: can we make it better by growing a deeper tree? The answer is: yes on training data, no on test data. We will build a brief demonstration into the platform showing how train RMSE decreases monotonically with tree depth while test RMSE eventually worsens, making overfitting not just a concept students read about but something they can observe and manipulate in real time.

**Challenge 4: Generalising findings to diverse student backgrounds.** The platform is designed in both Chinese and English, and the visual-first interface (sliders, bar charts, gauge dials) is intended to be accessible regardless of mathematical background. However, different teachers may have different levels of comfort with the underlying concepts, and the lesson plan will need to include differentiated guidance for instructors.

---

## (f) References

Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2(2–3), 113–127. https://doi.org/10.1007/s13748-013-0040-3

UCI Machine Learning Repository. (2013). *Bike Sharing Dataset*. University of California, Irvine. Retrieved from https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56–61).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*Submitted by Data Mining Group — March 2026*
