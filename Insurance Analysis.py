from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt


def get_stats(dataframe, column):
    """
    Prints the average and standard deviation for the data in column
    :param dataframe: DataFrame containing the column to be analyzed
    :param column: String name of the column on which to report stats
    :return: None
    """
    stat_avg = round(dataframe[column].mean(), 2)
    stat_std = round(dataframe[column].std(), 2)
    print(f'{column}--Average: {stat_avg}, Standard Deviation: {stat_std}')


def proportions(dataframe, column):
    """
    Prints the proportion of each unique value in column
    :param dataframe: DataFrame containing the columns to be analyzed
    :param column: Column for which to calculate proportions of unique values
    :return: None
    """
    print(f'Proportions for {column}')
    total = len(dataframe[column])
    for val in dataframe[column].unique():
        num_columns = list(dataframe[column]).count(val)
        proportion = num_columns / total * 100
        proportion = round(proportion, 2)
        print(f'{val}: {proportion}%')
    print('')


def subplot_size(lst=None, num=None):
    """
    Function for calculating the dimensions for subplots,
            making it as square as possible. Takes either a list
            or an int as parameters. If both are given, prefers list.
            If none are given, raises an exception.
    :param lst: List of columns for which to make subplots.
    :param num: Number of subplots to make, if known.
    :return: Int number of rows and int number of columns
            for the figure of subplots.
    """
    if lst:
        length = len(lst)
    elif num:
        length = num
    else:
        raise ValueError
    rows = int(length ** 0.5)
    columns = rows
    while rows * columns < length:
        if columns % 2 == 0:
            rows += 1
        else:
            columns += 1
    return rows, columns


def graph_histograms(dataframe, *columns):
    """
    Makes and shows a figure of subplots, graphing histograms
            from any number of columns of data in dataframe.
    :param dataframe: The dataframe containing columns
    :param columns: String names of any number of columns
    :return: None
    """
    rows, cols = subplot_size(lst=columns)
    plt.figure(num='Histograms')
    for i, col in enumerate(columns):
        plt.subplot(rows, cols, i+1)
        plt.hist(dataframe[col])
        plt.title(col)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def graph_features(dataframe, x_data, y_data, *features_lst):
    """
    Makes and shows a figure of subplots. Each graph
            is a scatter plot showing only the data for a subgroup
            For example, if 'sex' is one of the features, the function
            will plot X_data vs Y_data for only men, then only for women.
    :param dataframe: The dataframe containing the X_data, Y_data, and features
    :param x_data: String name of the column of data for the X-axis
    :param y_data: String name of the column of data for the Y-axis
    :param features_lst: String names of the features to plot. E.g. 'Smoker'
    :return: None
    """
    i = 1
    uniques = 0
    for feature in features_lst:
        uniques += dataframe[feature].nunique()
    rows, columns = subplot_size(num=uniques)
    fig = plt.figure(num='Features', figsize=(rows * 4, columns * 4))
    for feature in features_lst:
        featureframe = dataframe[[x_data, y_data, feature]]
        for val in dataframe[feature].unique():
            plt.subplot(rows, columns, i)
            subframe = featureframe[(featureframe[feature] == val)]
            plt.scatter(subframe[x_data], subframe[y_data])
            if type(val) is str:
                val = val.capitalize()
            plt.title(f'{feature.capitalize()}: {val}')
            i += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(f'{x_data.capitalize()} vs {y_data.capitalize()} for Subgroups')
    plt.show()


def lin_reg(dataframe, label=None):
    """
    Performs multiple linear regression on a dataframe
    with 'charges' as the labels, and the rest of the data
    as features. Then prints the resulting R2 value and coefficients.
    :param dataframe: The dataframe containing the data to be used.
    :param label: String description of the DataFrame, e.g. 'All Data', or 'Non-Smokers'
    :return: None
    """
    # Separate the data into dataframes for features and labels
    labels = dataframe['charges']
    features = dataframe.drop('charges', axis=1)

    # Split data into training and test
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    # Data is linear; construct LinearRegression object
    linear = LinearRegression()

    # Train model
    linear.fit(features_train, labels_train)

    print('Multiple Linear Regression Insights', end='')
    if label:
        print(f' for {label}:')
    else:
        print(':')
    # Find R2 value with test data
    scr = linear.score(features_test, labels_test)
    scr = round(scr, 4)
    print(f'R2 value: {scr}')

    # Print the coefficients
    for feat, coef in zip(features.columns, linear.coef_):
        coef = str(round(coef, 3))
        print(f'{feat} coef: {coef}')
    print('')


# Import the data
df = pd.read_csv('insurance.csv')

# Get a few stats from the data
print('Stats:')
get_stats(df, 'bmi')
get_stats(df, 'age')
get_stats(df, 'children')
get_stats(df, 'charges')
print('')
proportions(df, 'sex')
proportions(df, 'smoker')
proportions(df, 'region')

# Make histograms to view the shape of the data
graph_histograms(df, 'bmi', 'age', 'children', 'charges')
# Mean and standard deviation aren't all that helpful here because
# the 'charges' data aren't a normal distribution; it's skewed left.

# Look at the data as a whole, age vs. charges
plt.figure(num='All Data')
plt.scatter(df.age, df.charges)
plt.title('All Data: Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()
# There are three tiers of charges! Why?

# Lets graph age vs. charges for each unique feature
graph_features(df, 'age', 'charges', 'sex', 'children', 'smoker', 'region')
# The 'yes' smokers are all on the middle- and highest-tier charges.

nonsmokers = df[(df['smoker'] == 'no')]
graph_features(nonsmokers, 'age', 'charges', 'sex', 'children', 'smoker', 'region')
# Graphing only non-smoker data removes the highest-tier charges,
# so 'yes' smokers account for the highest tier.
# But the lower- and middle-tier lines still exist among non-smokers.
# People with 5 kids have lower charges, but the rest of the data
# looks fairly uniform. Meaning, no other unique feature seems to
# account for the tiers; all features partake of the two tiers
# pretty equally. We can't look at the data for any one feature
# to explain the other two tiers. Is there perhaps a factor
# affecting the charges which is excluded from our data?
# Perhaps different tiered plans?

# Clean the Data for Machine Learning
# Make dummy variables from the categorical region data
for reg in df.region.unique():
    name = 'region_' + reg
    df[name] = df.region.apply(lambda row: 1 if row == reg else 0)

# We won't need the region column anymore
df = df.drop('region', axis=1)

# Convert female to 1 and male to 0, as dichotomous data
df['sex'] = df.sex.apply(lambda row: 1 if row == 'female' else 0)

# Convert smoker 'yes' to 1 and 'no' to 0, as dichotomous data
df['smoker'] = df.smoker.apply(lambda row: 1 if row == 'yes' else 0)

lin_reg(df, 'All Data')

# With an R2 value of around 0.75, we can say that
# about 75% of the 'charges' data variance is explained
# by our features (independent variables).
# There's plenty of room to improve our R2 value;
# could this be more evidence of a factor missing from
# our data? Perhaps the tiered medical insurance plans?
# Perhaps the R2 value was not as high as it could be because
# I am trying to fit a single trend line to what is basically
# three lines of data (the three tiers). What if I isolate the
# lines by selecting ranges of charges? Perhaps coefficients
# Could be telling in this case. Let's find out.

# Create a new dataframe isolating the tier lines.
# With (age, charges) being a data point, I used
# (18, 7500) and (65, 21000) to find a line that separates
# the tiers. The equation is: y = 287.23x + 2329.79
# Lets look at only the bottom two tiers by excluding smokers,
# and separate the lower and middle tiers using this line.
# Make dummy variables where False is under the line (bottom tier)
# and True is above the line (middle tier). Then select
# Dataframes for each tier. Let's also run it separately for
# the highest tier, the smokers.

smokers = df[(df['smoker'] == 1)]
df['tier'] = df.age.apply(lambda x: 287.23 * x + 2329.79)
df['tier'] = df.charges > df.tier
low_df = df[(df['smoker'] == 0) & (df['tier'] == False)]
mid_df = df[(df['smoker'] == 0) & (df['tier'] == True)]
low_df = low_df.drop('tier', axis=1)
mid_df = mid_df.drop('tier', axis=1)

plt.figure('Tiered Data Separated')
plt.scatter(low_df.age, low_df.charges, label='Low-Tier Non-Smokers', c='limegreen', alpha=0.4)
plt.scatter(mid_df.age, mid_df.charges, label='Mid-Tier Non-Smokers', c='forestgreen', marker='v', alpha=0.4)
# plt.scatter(smokers.age, smokers.charges, label='Smokers')
#plt.show()
# Ah! So it's not actually three tiers; it's rather that
# Smokers and non-smokers each have two tiers.

# I'm updating the scatter plot for the discovery of the two
# sets of tiers, but I'm leaving the code commented out as
# a record of how I got there.
smokers_low = smokers[(smokers['charges'] < 30000)]
smokers_high = smokers[(smokers['charges'] >= 30000)]
plt.scatter(smokers_low.age, smokers_low.charges, label='Low-tier Smokers', c='red', alpha=0.4)
plt.scatter(smokers_high.age, smokers_high.charges, label='High-tier Smokers', c='maroon', marker='v', alpha=0.4)
plt.legend()
plt.title('Split Tiers, Age vs Charges for Smokers and Non-Smokers')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Now lets run Regression on each to see if we can find any
# significant difference between the sets' coefficients.
lin_reg(smokers_high, 'High-tier Smoker Data')
lin_reg(smokers_low, 'Low-tier Smoker Data')
lin_reg(mid_df, 'High-tier Non-smoker Data')
lin_reg(low_df, 'Low-tier Non-smoker Data')

# With R2 values greater than 0.9 for the low-tier data, we can
# say that the features (independent variables) in our data account
# for 90%+ of the variance in their 'charges' data. But R2 values
# in the 0.3 - 0.4 range for the high-tier data shows that our
# features don't sufficiently account for the charges data. Is there
# indeed some factor or feature we're missing to explain the split in data?

# In any case, being a smoker is a significant factor in the costs.
# Our low-tier data with good good models show that sex is a
# fairly significant factor in determining charges. Age is consistent
# between the two low-tier models, but I'd be curious to find out why the
# other features aren't as comparable between the two.
