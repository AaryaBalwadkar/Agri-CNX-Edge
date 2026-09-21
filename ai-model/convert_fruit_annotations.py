import json
from pathlib import Path

fruit_dir = Path("data/fruit")
json_files = list(fruit_dir.glob("**/*.json"))

print(f"Found {len(json_files)} JSON annotation files in {fruit_dir}...")

converted_count = 0
for json_path in json_files:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    img_w = data.get("imageWidth", 512)
    img_h = data.get("imageHeight", 512)
    
    yolo_lines = []
    for shape in data.get("shapes", []):
        if shape.get("label") == "Healthy_Apple":
            points = shape.get("points", [])
            if not points:
                continue
            
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Normalize to YOLO format [class_id cx cy w h]
            cx = ((x_min + x_max) / 2.0) / img_w
            cy = ((y_min + y_max) / 2.0) / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h
            
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    if yolo_lines:
        txt_path = json_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
        converted_count += 1

print(f"Successfully generated {converted_count} .txt annotation files!")