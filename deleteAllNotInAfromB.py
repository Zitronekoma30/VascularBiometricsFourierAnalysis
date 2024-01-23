import os
import shutil

def get_file_names(folder):
    file_names = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_names.append(file)
    return file_names

def delete_files_not_in_list(folder_a, folder_b):
    # Get the list of filenames in folder A
    filenames_a = set(get_file_names(folder_a))

    # Iterate over files in subfolders of folder B
    for root_b, _, files_b in os.walk(folder_b):
        for filename_b in files_b:
            if filename_b not in filenames_a:
                # Build the full path to the file in folder B
                file_path_b = os.path.join(root_b, filename_b)

                # Delete the file in folder B
                try:
                    os.remove(file_path_b)
                    print(f"Deleted: {file_path_b}")
                except Exception as e:
                    print(f"Error deleting {file_path_b}: {e}")

if __name__ == "__main__":
    # Example usage:
    folder_path_a = input("Enter the path to folder A: ")
    folder_path_b = input("Enter the path to folder B: ")

    # Check if the provided paths are valid directories
    if os.path.isdir(folder_path_a) and os.path.isdir(folder_path_b):
        delete_files_not_in_list(folder_path_a, folder_path_b)
    else:
        print("Invalid folder paths. Please make sure both paths are valid directories.")
