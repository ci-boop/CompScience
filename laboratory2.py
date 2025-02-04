# Weather conditions and whether to bring an umbrella dataset
dataset = [
    # (Windy, Rainy, Cloudy, Bring Umbrella)
    ("Yes",  "Yes",  "Yes",  "Yes"),
    ("Yes",  "No",   "Yes",  "No"),
    ("No",   "Yes",  "Yes",  "Yes"),
    ("No",   "No",   "No",   "No"),
    ("Yes",  "Yes",  "No",   "Yes"),
    ("No",   "Yes",  "No",   "Yes"),
    ("Yes",  "No",   "No",   "No"),
    ("No",   "No",   "Yes",  "No"),
]

# Count occurrences
feature_counts = {"Yes": {}, "No": {}}
class_counts = {"Yes": 0, "No": 0}
vocab = set()

for windy, rainy, cloudy, label in dataset:
    class_counts[label] += 1

    for feature in (windy, rainy, cloudy):
        vocab.add(feature)
        if feature in feature_counts[label]:
            feature_counts[label][feature] += 1
        else:
            feature_counts[label][feature] = 1

# Compute prior probabilities
total_samples = sum(class_counts.values())
prior_yes = class_counts["Yes"] / total_samples
prior_no = class_counts["No"] / total_samples

# Function to calculate feature probabilities
def feature_probability(feature, label, alpha=1):
    feature_freq = feature_counts[label].get(feature, 0) + alpha
    total_features = sum(feature_counts[label].values()) + alpha * len(vocab)
    return feature_freq / total_features

# Classify new weather conditions
def classify(windy, rainy, cloudy): 
    yes_prob = prior_yes
    no_prob = prior_no
    
    for feature in (windy, rainy, cloudy):
        yes_prob *= feature_probability(feature, "Yes")
        no_prob *= feature_probability(feature, "No")
    
    return "Yes" if yes_prob > no_prob else "No"

# User Input
print("Enter the weather conditions (Yes/No):")
windy = input("Is it windy? (Yes/No): ").strip().capitalize()
rainy = input("Is it rainy? (Yes/No): ").strip().capitalize()
cloudy = input("Is it cloudy? (Yes/No): ").strip().capitalize()

# Make prediction
prediction = classify(windy, rainy, cloudy)
print(f"\nWeather: (Windy: {windy}, Rainy: {rainy}, Cloudy: {cloudy}) -> Bring Umbrella? {prediction}")