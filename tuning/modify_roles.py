import json
import os

# Function to update role descriptions within JSON data
def update_role_descriptions(file_path, role_updates):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Temporary dictionary to hold updated roles
    updated_data = {}

    # Iterate through the items in the JSON data
    for key, value in data.items():
        # Check if the key matches the pattern "role(description)"
        for original_role, new_description in role_updates.items():
            if key.startswith(original_role + "(") and key.endswith(")"):
                # Replace the role description
                new_key = f"{original_role}({new_description})"
                updated_data[new_key] = value
                break
        else:
            # If the key doesn't match the pattern, keep it as is
            updated_data[key] = value
    
    # Write the modified data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(updated_data, file, indent=4)

# Define your role updates here (example)
role_updates = {
    "agent_0": "NewRoleForAgent0",
    "agent_1": "NewRoleForAgent1",
    "adversary_0": "NewRoleForAdversary0"
}

# Example usage
directory = "./data/tuning_sfsa/role_tuning"


# Loop through all JSON files in the specified directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        update_role_descriptions(file_path, role_updates)
