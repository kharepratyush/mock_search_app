import itertools
import pandas as pd

# Define logical dish-query pairs for matches
logical_pairs = [
    ("burger", "cheeseburger"),
    ("pizza", "thin crust pizza"),
    ("pasta", "spaghetti"),
    ("sushi", "california roll"),
    ("taco", "chicken taco"),
    ("salad", "caesar salad"),
    ("steak", "grilled steak"),
    ("sandwich", "club sandwich"),
    ("fries", "crispy fries"),
    ("wrap", "chicken wrap"),
]

# Generate matching pairs with label 1
matching_pairs = [
    {"dish": dish, "query": query, "label": 1} for dish, query in logical_pairs
]

# Generate mismatching pairs (dish-query combinations that don't logically match) with label 0
all_dishes = [pair[0] for pair in logical_pairs]
all_queries = [pair[1] for pair in logical_pairs]

mismatching_pairs = []
for dish, query in itertools.product(all_dishes, all_queries):
    if (dish, query) not in logical_pairs:
        mismatching_pairs.append({"dish": dish, "query": query, "label": 0})
    if len(mismatching_pairs) + len(matching_pairs) >= 10000:
        break

# Combine matching and mismatching pairs
labeled_data = matching_pairs + mismatching_pairs[: 10000 - len(matching_pairs)]

# Create a DataFrame and save it as CSV
labeled_df = pd.DataFrame(labeled_data)
labeled_output_path = "labeled_dish_query_dataset.csv"
labeled_df.to_csv(labeled_output_path, index=False)
