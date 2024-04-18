import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Convert to NumPy array
array = df.values
print(array.shape[0])