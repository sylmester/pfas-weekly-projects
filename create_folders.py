import os

# Names of the subfolders to create inside each week folder
SUBFOLDERS = ["Elias", "Irene", "Alberto", "Sylvester", "Weekly Project"]

# Number of week folders
NUM_WEEKS = 9

# Base directory where the week folders will be created
# Change this to wherever you want the weekN folders to live,
# e.g. "." for the current directory, or an absolute path.
BASE_DIR = "."

for week_num in range(1, NUM_WEEKS + 1):
    week_folder = os.path.join(BASE_DIR, f"week{week_num}")
    for sub in SUBFOLDERS:
        path = os.path.join(week_folder, sub)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

print("\nDone! All week folders and subfolders are set up.")