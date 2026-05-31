def estimate_share(detections):
    total_area = 0
    brand_area = {}
    for item in detections:
        x1,y1,x2,y2 = item['bbox']
        area = (x2-x1)*(y2-y1)
        total_area += area
        brand_area[item['brand']] = brand_area.get(item['brand'],0)+area

    if total_area == 0:
        return {}

    return {k: round(v*100/total_area,2) for k,v in brand_area.items()}
