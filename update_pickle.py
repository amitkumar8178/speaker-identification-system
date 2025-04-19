import pickle

file = 'data.pkl'

# Load
with open(file, 'rb') as f:
    data = pickle.load(f)

# Modify (example for dict)
data['new_key'] = 'new_value'

# Save
with open(file, 'wb') as f:
    pickle.dump(data, f)

print("Pickle file updated.")
