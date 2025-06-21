import json

def toolace_to_jsonl():
    # Load the toolace.json file
    with open('toolace.json', 'r') as file:
        toolace_data = json.load(file)

    # Change "system" field to "id" in all entries
    for entry in toolace_data:
        if "system" in entry:
            entry["id"] = entry.pop("system")

    # Reorder fields to put "id" before "conversations" in all entries
    for entry in toolace_data:
        if "id" in entry and "conversations" in entry:
            # Create a new ordered dictionary with "id" first
            reordered_entry = {}
            
            # Add "id" field first
            reordered_entry["id"] = entry["id"]
            
            # Add all other fields except "id" and "conversations"
            for key, value in entry.items():
                if key not in ["id", "conversations"]:
                    reordered_entry[key] = value
            
            # Add "conversations" field last
            reordered_entry["conversations"] = entry["conversations"]
            
            # Replace the original entry with the reordered one
            entry.clear()
            entry.update(reordered_entry)


    # Save the modified data back to a new JSON file
    with open('toolace_modified.json', 'w') as output_file:
        json.dump(toolace_data, output_file, indent=2)

def mt_5k_to_json():
    # Load the mt_5k.json file
    with open('apigen-mt_5k.json', 'r') as file:
        mt_5k_data = json.load(file)

    # Change "system" field to "id" in all entries
    for entry in mt_5k_data:
        if "system" in entry:
            entry["id"] = entry.pop("system")
    
    # Remove "tools" field from all entries
    for entry in mt_5k_data:
        if "tools" in entry:
            del entry["tools"]
        
    # Reorder fields to put "id" before "conversations" in all entries
    for entry in mt_5k_data:
        if "id" in entry and "conversations" in entry:
            # Create a new ordered dictionary with "id" first
            reordered_entry = {}
            
            # Add "id" field first
            reordered_entry["id"] = entry["id"]
            
            # Add all other fields except "id" and "conversations"
            for key, value in entry.items():
                if key not in ["id", "conversations"]:
                    reordered_entry[key] = value
            
            # Add "conversations" field last
            reordered_entry["conversations"] = entry["conversations"]
            
            # Replace the original entry with the reordered one
            entry.clear()
            entry.update(reordered_entry)
    
    # Save the modified data back to a new JSON file
    with open('mt_5k_modified.json', 'w') as output_file:
        json.dump(mt_5k_data, output_file, indent=2)

if __name__ == "__main__":
    #toolace_to_jsonl()
    mt_5k_to_json()