import json

with open(r"D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject\model_visuals\features_with_metadata.json", "r") as f:
    metadata = json.load(f)

print("Number of entries:", len(metadata))
print("First 2 entries:", metadata[:2])
