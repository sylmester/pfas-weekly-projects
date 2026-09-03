import os

SUBFOLDERS = ["Elias", "Irene", "Alberto", "Sylvester", "Weekly Project"]
NUM_WEEKS = 9
BASE_DIR = "."

for week_num in range(1, NUM_WEEKS + 1):
    week_folder = os.path.join(BASE_DIR, f"week{week_num}")
    for sub in SUBFOLDERS:
        path = os.path.join(week_folder, sub)
        os.makedirs(path, exist_ok=True)
        # add a placeholder so git tracks the empty folder
        gitkeep_path = os.path.join(path, ".gitkeep")
        open(gitkeep_path, "a").close()
        print(f"Created: {path}")

print("\nDone!")