import json
import pandas as pd
import ast  # local import to avoid polluting global namespace unnecessarily

def toolace_to_jsonl():
    # Load the toolace.json file
    with open('toolace.json', 'r') as file:
        toolace_data = json.load(file)

    # Read "system" field and add it as first conversation item
    for entry in toolace_data:
        if "system" in entry:
            system_value = entry["system"]
            # Create the system message
            system_message = {
                "from": "system",
                "value": system_value
            }
            # Insert as first item in conversations
            entry["conversations"].insert(0, system_message)
    
    
    # Change "system" field to "id" in all entries
    for entry in toolace_data:
        if "system" in entry:
            entry["id"] = entry.pop("system")
    
    # Replace the id value with the value of the second item in conversations
    for entry in toolace_data:
        if "id" in entry and "conversations" in entry and len(entry["conversations"]) >= 2:
            entry["id"] = entry["conversations"][1]["value"]

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
    
    for entry in mt_5k_data:
        if "id" in entry and "conversations" in entry:
            entry["conversations"].insert(0, {
                "from": "system",
                "value": entry["id"]
            })
    
    # Remove "id" field from all entries
    for entry in mt_5k_data:
        if "id" in entry:
            del entry["id"]
    
    
    # Save the modified data back to a new JSON file
    with open('mt_5k_modified.json', 'w') as output_file:
        json.dump(mt_5k_data, output_file, indent=2)

def bitagent_to_json():
    # Read the parquet file
    bitagent_data = pd.read_parquet('bitagent_tool_calling_shuffle.parquet')

    # Convert the DataFrame to a list of dictionaries
    bitagent_list = bitagent_data.to_dict('records')

    # Ensure each conversation entry is parsed into a list[dict]
    def _safe_parse(raw):
        """Try to parse a raw string (JSON-like / python-repr) into a python object.
        Returns the original value on total failure."""
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw  # give up – return as-is
            

    
    for entry in bitagent_list:
        conv_raw = entry.get("conversation")

        # Case 1: the whole field is a single string representing the convo list
        if isinstance(conv_raw, str):
            conv_parsed = _safe_parse(conv_raw)
            # Make sure it's a list
            if isinstance(conv_parsed, dict):
                conv_parsed = [conv_parsed]
            entry["conversations"] = conv_parsed if isinstance(conv_parsed, list) else []
            continue  # move to next entry; it'll be processed in later loop

        # Case 2: it's already an iterable (list/tuple)
        parsed_list = []
        for item in conv_raw or []:  # handle None
            parsed_item = _safe_parse(item)
            # If the parsed item is itself a list, flatten one level
            if isinstance(parsed_item, list):
                parsed_list.extend(parsed_item)
            else:
                parsed_list.append(parsed_item)
        entry["conversations"] = parsed_list
    
    # Remove the "conversation" field from every entry
    for entry in bitagent_list:
        if "conversation" in entry:
            del entry["conversation"]
    

    # Process each entry to change field names
    for entry in bitagent_list:
        for conversation in entry["conversations"]:
            # Skip malformed items
            if not isinstance(conversation, dict):
                continue
            # Change "role" to "from"
            if "role" in conversation:
                conversation["from"] = conversation.pop("role")
            # Change "content" to "value"
            if "content" in conversation:
                conversation["value"] = conversation.pop("content")

    # Remove the "tools" field from every entry
    for entry in bitagent_list:
        if "tools" in entry:
            del entry["tools"]
    
    # Convert all "value" fields in conversations to strings
    for entry in bitagent_list:
        for conversation in entry["conversations"]:
            # Skip malformed items
            if not isinstance(conversation, dict):
                continue
            # Convert "value" field to string if it exists
            if "value" in conversation:
                if not isinstance(conversation["value"], str):
                    conversation["value"] = str(conversation["value"])


    # Save the modified data to a new JSON file
    with open('bitagent_modified.json', 'w') as output_file:
        json.dump(bitagent_list, output_file, indent=2)

def coalm_to_json():
    
    # Read the parquet files
    coalm1_df = pd.read_parquet('coalm1.parquet')
    coalm2_df = pd.read_parquet('coalm2.parquet')
    
    # Convert DataFrames to lists of dictionaries
    coalm1_list = coalm1_df.to_dict('records')
    coalm2_list = coalm2_df.to_dict('records')
    
    # Change "instruction" field to "id" in both datasets
    for entry in coalm1_list:
        if "instruction" in entry:
            entry["id"] = entry.pop("instruction")
    
    for entry in coalm2_list:
        if "instruction" in entry:
            entry["id"] = entry.pop("instruction")

    # Add conversations field to coalm1_list
    for entry in coalm1_list:
        entry["conversations"] = []
    # Add conversations field to coalm2_list
    for entry in coalm2_list:
        entry["conversations"] = []
    
    # Process "input" fields in coalm1_list
    for entry in coalm1_list:
        if "input" in entry and isinstance(entry["input"], str):
            parts = entry["input"].split(":", 1)  # Split on first occurrence only
            if len(parts) == 2:
                entry["conversations"].append({
                    "from": parts[0].strip(),
                    "value": parts[1].strip()
                })
            else:
                entry["conversations"].append({
                    "from": "",
                    "value": ""
                })
    
    # Process "input" fields in coalm2_list
    for entry in coalm2_list:
        if "input" in entry and isinstance(entry["input"], str):
            parts = entry["input"].split(":", 1)  # Split on first occurrence only
            if len(parts) == 2:
                entry["conversations"].append({
                    "from": parts[0].strip(),
                    "value": parts[1].strip()
                })
            else:
                entry["conversations"].append({
                    "from": "",
                    "value": ""
                })
    


    # Process "output" fields in coalm1_list
    for entry in coalm1_list:
        if "output" in entry and isinstance(entry["output"], str):
            parts = entry["output"].split(":", 1)  # Split on first occurrence only
            if len(parts) == 2 and parts[0] == "System":
                entry["conversations"].append({
                    "from": parts[0].strip(),
                    "value": parts[1].strip()
                })
            else:
                entry["conversations"].append({
                    "from": "",
                    "value": entry["output"]
                })
    


    # Process "output" fields in coalm2_list
    for entry in coalm2_list:
        if "output" in entry and isinstance(entry["output"], str):
            parts = entry["output"].split(":", 1)  # Split on first occurrence only
            if len(parts) == 2 and parts[0] == "System":
                entry["conversations"].append({
                    "from": parts[0].strip(),
                    "value": parts[1].strip()
                })
            else:
                entry["conversations"].append({
                    "from": "",
                    "value": entry["output"]
                })
    
    # Add system message with id value as first element in conversations for coalm1_list
    for entry in coalm1_list:
        if "id" in entry and entry["conversations"]:
            entry["conversations"].insert(0, {
                "from": "System",
                "value": entry["id"]
            })
    
    # Add system message with id value as first element in conversations for coalm2_list
    for entry in coalm2_list:
        if "id" in entry and entry["conversations"]:
            entry["conversations"].insert(0, {
                "from": "System",
                "value": entry["id"]
            })
    
    # Remove "input" and "output" fields from all entries
    for entry in coalm1_list:
        entry.pop("input", None)
        entry.pop("output", None)
        entry.pop("id", None)
    
    for entry in coalm2_list:
        entry.pop("input", None)
        entry.pop("output", None)
        entry.pop("id", None)
    
    # Save to JSON files
    with open('coalm1.json', 'w') as f1:
        json.dump(coalm1_list, f1, indent=2)
    
    with open('coalm2.json', 'w') as f2:
        json.dump(coalm2_list, f2, indent=2)
    
def glaive_to_json():
    # Read the glaive function calling dataset
    with open('glaive-function-calling-v2.json', 'r') as f:
        glaive_data = json.load(f)
    
    # Convert to unified format
    glaive_list = []
    for entry in glaive_data:
        conversations = []
        
        if "system" in entry and isinstance(entry["system"], str):
            parts = entry["system"].split(":", 1)  # Split on first occurrence only
            if len(parts) == 2:
                conversations.append({
                    "from": parts[0].strip(),
                    "value": parts[1].strip()
                })
        
        if "chat" in entry and isinstance(entry["chat"], str):
            chat_parts = entry["chat"].split("\n\n\n")
            for chunk in chat_parts:
                text = chunk.strip()
                if not text:
                    continue  # skip empty segments
                parts = text.split(":", 1)  # only split on the first ':'
                if len(parts) == 2:
                    from_field, value_field = parts[0].strip(), parts[1].strip()
                else:
                    # If no ':' present, treat whole chunk as value with unknown sender
                    from_field, value_field = "", text
                conversations.append({
                    "from": from_field,
                    "value": value_field
                })


        
        glaive_list.append({"conversations": conversations})
    
    # Save to JSON file
    with open('glaive_modified.json', 'w') as f:
        json.dump(glaive_list, f, indent=2)

def hammer2_to_json():
    # Read the multi-turn.json file
    with open('multi-turn.json', 'r') as f:
        multi_turn_data = json.load(f)

    # read the single-turn.json file
    with open('single-turn.json', 'r') as f:
        single_turn_data = json.load(f)

    # combine the two datasets
    hammer2_list = multi_turn_data + single_turn_data

    
    # Change "messages" field to "conversations" in all entries
    for entry in hammer2_list:
        if "messages" in entry:
            entry["conversations"] = entry.pop("messages")
            
    # Change "role" to "from" and "content" to "value" in conversations
    for entry in hammer2_list:
        if "conversations" in entry:
            for conversation in entry["conversations"]:
                if "role" in conversation:
                    conversation["from"] = conversation.pop("role")
                if "content" in conversation:
                    conversation["value"] = conversation.pop("content")

    # Convert non-string "value" fields to strings
    for entry in hammer2_list:
        if "conversations" in entry:
            for conversation in entry["conversations"]:
                if "value" in conversation and not isinstance(conversation["value"], str):
                    conversation["value"] = json.dumps(conversation["value"])
    
    # Remove "id" and "tools" fields from all entries
    for entry in hammer2_list:
        if "id" in entry:
            del entry["id"]
        if "tools" in entry:
            del entry["tools"]
    
    # Save to JSON file
    with open('hammer2_modified.json', 'w') as f:
        json.dump(hammer2_list, f, indent=2)

def gorilla_to_json():
    import os
    import json
    
    # Get all JSON files in the BFCL directory
    bfcl_dir = 'BFCL'
    json_files = [f for f in os.listdir(bfcl_dir) if f.endswith('.json')]
    
    # Combine all JSON data
    gorilla_list = []
    
    for json_file in json_files:
        file_path = os.path.join(bfcl_dir, json_file)
        try:
            # First, attempt to load the whole file as standard JSON (either an object or an array).
            with open(file_path, 'r') as f:
                data = json.load(f)

                # If data is a list, extend gorilla_list; else append as single entry
                if isinstance(data, list):
                    gorilla_list.extend(data)
                else:
                    gorilla_list.append(data)

        except json.JSONDecodeError:
            # Fallback: treat the file as newline-delimited JSON (one JSON object per line).
            with open(file_path, 'r') as f:
                f.seek(0)  # Rewind to the beginning of the file
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # skip empty lines
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            gorilla_list.extend(obj)
                        else:
                            gorilla_list.append(obj)
                    except json.JSONDecodeError as inner_err:
                        # Warn and skip malformed lines rather than aborting the entire conversion
                        print(f"Skipping malformed JSON line {line_num} in {json_file}: {inner_err}")
        except Exception as e:
            # Catch-all for other IO related errors
            print(f"Unexpected error processing {json_file}: {e}")
    
    # Rename "question" field to "conversations" in all entries
    for entry in gorilla_list:
        if "question" in entry:
            entry["conversations"] = entry.pop("question")
    
    # Flatten the double nested conversations list
    for entry in gorilla_list:
        if "conversations" in entry and isinstance(entry["conversations"], list):
            # If conversations is a list of lists, flatten it
            if entry["conversations"] and isinstance(entry["conversations"][0], list):
                entry["conversations"] = entry["conversations"][0]
    
    # Convert function field to string in all entries
    for entry in gorilla_list:
        if "function" in entry and isinstance(entry["function"], list):
            entry["function"] = json.dumps(entry["function"])
    
    # Convert "role" to "from" and "content" to "value" in conversations
    for entry in gorilla_list:
        if "conversations" in entry and isinstance(entry["conversations"], list):
            for conversation in entry["conversations"]:
                if isinstance(conversation, dict):
                    if "role" in conversation:
                        conversation["from"] = conversation.pop("role")
                    if "content" in conversation:
                        conversation["value"] = conversation.pop("content")
    
    # Convert function field to conversation format and append to conversations
    for entry in gorilla_list:
        if "function" in entry:
            function_conversation = {
                "from": "function",
                "value": entry["function"]
            }
            if "conversations" in entry:
                entry["conversations"].append(function_conversation)
            else:
                entry["conversations"] = [function_conversation]
    
    # Remove "id" and "function" fields from all entries
    for entry in gorilla_list:
        entry.pop("id", None)
        entry.pop("function", None)
        entry.pop("initial_config", None)
    
    # Save combined data to JSON file
    with open('gorilla_modified.json', 'w') as f:
        json.dump(gorilla_list, f, indent=2)

if __name__ == "__main__":
    #toolace_to_jsonl()
    #mt_5k_to_json()
    #bitagent_to_json()
    #coalm_to_json()
    #glaive_to_json()
    #hammer2_to_json()
    gorilla_to_json()