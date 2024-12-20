# Define base logical dish-query pairs
import pandas as pd

base_pairs = [
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

# Generate variations for each base pair
variations = []
for dish, query in base_pairs:
    variations.extend(
        [
            (dish, query),
            (dish, f"best {query}"),
            (dish, f"spicy {query}"),
            (f"vegan {dish}", query),
            (f"vegan {dish}", f"healthy {query}"),
            (f"grilled {dish}", f"barbecue {query}"),
            (f"fried {dish}", f"crispy {query}"),
            (f"double {dish}", f"extra {query}"),
            (dish, f"homemade {query}"),
            (f"classic {dish}", f"traditional {query}"),
        ]
    )

# Ensure the dataset contains exactly 10,000 unique pairs
unique_pairs = list(set(variations))  # Remove duplicates
if len(unique_pairs) < 10000:
    # Add further variations until we reach 10,000 samples
    while len(unique_pairs) < 10000:
        for dish, query in base_pairs:
            unique_pairs.append((f"special {dish}", f"gourmet {query}"))
            if len(unique_pairs) >= 10000:
                break

# Create a DataFrame from the unique pairs
final_data = unique_pairs[:10000]  # Ensure we have exactly 10,000 samples
final_df = pd.DataFrame(final_data, columns=["dish", "query"])

# Save to CSV
innovative_output_path = "unlabeled_dataset.csv"
final_df.to_csv(innovative_output_path, index=False)
